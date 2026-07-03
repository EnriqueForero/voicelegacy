"""voicelegacy — XTTS-v2 inference wrapper.

Thin wrapper around `coqui-tts` (the Idiap fork) that:
- Validates that CUDA / VRAM are available before model load.
- Sets the COQUI_TOS_AGREED env var explicitly (user must accept CPML).
- Caches the loaded model on the module level to avoid re-loading on every call.
- Caches speaker conditioning latents when the lower-level XTTS API is exposed.
- Frees VRAM on demand.

Why the env var: XTTS-v2 weights are released under the Coqui Public Model
License (CPML), which requires explicit acceptance. The library checks
COQUI_TOS_AGREED=1 before loading. See: https://coqui.ai/cpml
"""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voicelegacy.config import XTTS_OUTPUT_SR, SynthesisConfig
from voicelegacy.logging_config import get_logger
from voicelegacy.text_strategy import plan_text_synthesis

logger = get_logger()

# Module-level caches. The model is large (~2 GB on disk, 2-3 GB VRAM) and
# expensive to instantiate. Speaker conditioning latents are also expensive and
# deterministic for a fixed reference set, so cache them across synthesis calls.
_MODEL_CACHE: dict[str, Any] = {}
_CONDITIONING_LATENTS_CACHE: dict[str, tuple[Any, Any]] = {}


def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch  # local import — avoid forcing torch on test rigs without it

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_xtts_model(config: SynthesisConfig, accept_tos: bool) -> Any:
    """Load (or fetch from cache) an XTTS-v2 model.

    Args:
        config: SynthesisConfig with model_name and device.
        accept_tos: Must be True. Setting it sets COQUI_TOS_AGREED=1.

    Returns:
        Initialized TTS object ready for inference.

    Raises:
        RuntimeError: If accept_tos is False.
        ImportError: If coqui-tts is not installed.
    """
    if not accept_tos:
        raise RuntimeError(
            "You must accept the Coqui Public Model License (CPML) before loading XTTS-v2. "
            "Read it at https://coqui.ai/cpml then pass accept_tos=True."
        )

    os.environ["COQUI_TOS_AGREED"] = "1"

    cache_key = f"{config.model_name}|{config.device}"
    if cache_key in _MODEL_CACHE:
        logger.info("Using cached model: {}", cache_key)
        return _MODEL_CACHE[cache_key]

    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise ImportError(
            "coqui-tts is not installed. Install with: pip install coqui-tts"
        ) from exc

    device = _resolve_device(config.device)
    logger.info("Loading XTTS model '{}' on device '{}'", config.model_name, device)

    tts = TTS(config.model_name)
    tts = tts.to(device)

    _MODEL_CACHE[cache_key] = tts
    logger.info("Model loaded and cached.")
    return tts


def release_model() -> None:
    """Evict cached model(s), cached conditioning latents, and free VRAM."""
    global _MODEL_CACHE, _CONDITIONING_LATENTS_CACHE
    _MODEL_CACHE.clear()
    _CONDITIONING_LATENTS_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Model and conditioning-latent caches released; VRAM freed.")
    except ImportError:
        pass


def release_conditioning_latents() -> None:
    """Evict cached XTTS conditioning latents without unloading the model."""
    _CONDITIONING_LATENTS_CACHE.clear()
    gc.collect()
    logger.info("Conditioning-latent cache released.")


def _apply_seed(seed: int | None) -> None:
    """Set torch / numpy / random seeds before inference.

    Centralized so the logic is testable without instantiating TTS. With
    `seed=None` the function is a no-op (legacy non-deterministic behavior).
    """
    if seed is None:
        return

    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Force deterministic algorithms where supported. This costs ~10-20%
        # performance but guarantees byte-identical outputs across runs.
        # Some XTTS-v2 operations may still be non-deterministic at the CUDA
        # kernel level — those are out of our control.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed — seeding numpy/random only")

    logger.debug("Seeded inference with seed={}", seed)


def _conditioning_cache_key(reference_wavs: list[str], config: SynthesisConfig) -> str:
    """Build a stable key from reference paths, file metadata and conditioning knobs.

    The conditioning knobs are part of the key because different
    ``gpt_cond_len`` / ``max_ref_len`` / etc. produce *different* latents for
    the same reference files — omitting them would silently serve stale
    latents after a config change.
    """
    h = hashlib.sha256()
    for raw in sorted(reference_wavs):
        p = Path(raw)
        stat = p.stat()
        h.update(str(p.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
    knobs = (
        f"gcl={config.gpt_cond_len}|gccl={config.gpt_cond_chunk_len}"
        f"|mrl={config.max_ref_len}|snr={config.sound_norm_refs}"
    )
    h.update(knobs.encode("ascii"))
    return h.hexdigest()


def _extract_xtts_model(tts: Any) -> Any | None:
    """Return lower-level XTTS model object if the high-level wrapper exposes it."""
    model = getattr(tts, "synthesizer", None)
    model = getattr(model, "tts_model", None)
    if model is None:
        return None
    if not (hasattr(model, "get_conditioning_latents") and hasattr(model, "inference")):
        return None
    return model


def _get_conditioning_latents_cached(
    model: Any, reference_wavs: list[str], config: SynthesisConfig
) -> tuple[Any, Any]:
    """Get or compute XTTS speaker conditioning latents for a reference set.

    Passes the conditioning knobs explicitly. Before v0.7.1 this call used
    the raw ``get_conditioning_latents`` defaults (``gpt_cond_len=6``,
    ``max_ref_length=30``) while the ``tts_to_file`` fallback used
    XttsConfig's (12 / 10) — the same corpus produced two different voices
    depending on which path ran. Both paths now honor ``SynthesisConfig``.
    """
    key = _conditioning_cache_key(reference_wavs, config)
    if key in _CONDITIONING_LATENTS_CACHE:
        logger.info("Using cached XTTS conditioning latents: {}", key[:8])
        return _CONDITIONING_LATENTS_CACHE[key]

    logger.info(
        "Computing XTTS conditioning latents for {} reference file(s) "
        "(gpt_cond_len={}s, chunk={}s, max_ref_len={}s, sound_norm={})...",
        len(reference_wavs),
        config.gpt_cond_len,
        config.gpt_cond_chunk_len,
        config.max_ref_len,
        config.sound_norm_refs,
    )
    latents = model.get_conditioning_latents(
        audio_path=reference_wavs,
        gpt_cond_len=config.gpt_cond_len,
        gpt_cond_chunk_len=config.gpt_cond_chunk_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )
    if not isinstance(latents, tuple) or len(latents) != 2:
        raise RuntimeError("XTTS get_conditioning_latents returned an unexpected value")
    _CONDITIONING_LATENTS_CACHE[key] = latents
    return latents


def _tensor_to_numpy_1d(wav: Any) -> np.ndarray:
    """Convert XTTS output audio to a 1-D float32 numpy array."""
    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.size == 0:
        raise RuntimeError("XTTS manual inference produced empty/non-1D audio")
    return arr


def _try_synthesize_with_conditioning_cache(
    tts: Any,
    text: str,
    speaker_wav_list: list[str],
    output_path: Path,
    config: SynthesisConfig,
) -> bool:
    """Best-effort XTTS lower-level inference using cached conditioning latents.

    Returns True when manual inference succeeded and wrote the WAV. Returns False
    when the high-level TTS wrapper does not expose the model API. Exceptions from
    an exposed but failing model are logged and also fall back to ``tts_to_file``.
    """
    if not config.cache_conditioning_latents:
        return False

    model = _extract_xtts_model(tts)
    if model is None:
        logger.debug("XTTS model API not exposed; falling back to tts_to_file.")
        return False

    try:
        gpt_cond_latent, speaker_embedding = _get_conditioning_latents_cached(
            model, speaker_wav_list, config
        )
        text_plan = plan_text_synthesis(text, config)
        if text_plan.warning:
            logger.warning("Text synthesis warning: {}", text_plan.warning)
        out = model.inference(
            text,
            config.language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=config.temperature,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            speed=config.speed,
            enable_text_splitting=text_plan.xtts_split_sentences,
        )
        if not isinstance(out, dict) or "wav" not in out:
            raise RuntimeError("XTTS manual inference did not return {'wav': ...}")
        wav = _tensor_to_numpy_1d(out["wav"])
        sf.write(str(output_path), wav, XTTS_OUTPUT_SR, subtype="PCM_16")
        logger.info("Wrote {} with cached XTTS conditioning latents", output_path)
        return True
    except Exception as exc:
        logger.warning("Cached-latent XTTS inference failed; falling back to tts_to_file: {}", exc)
        return False


def synthesize_to_file(
    tts: Any,
    text: str,
    speaker_wav: Path | list[Path],
    output_path: Path,
    config: SynthesisConfig,
) -> Path:
    """Run XTTS-v2 inference and save the result to a WAV file.

    Args:
        tts: Loaded TTS object (from load_xtts_model).
        text: Text to synthesize. Will be auto-split at sentence boundaries
            if config.enable_text_splitting is True.
        speaker_wav: Path (or list of paths) to reference audio for voice
            conditioning. Passing a list averages the speaker latents — useful
            when you have multiple short clean segments.
        output_path: Where to write the output WAV.
        config: SynthesisConfig with language and decoder hyperparameters.
            If config.seed is not None, applies it to torch / numpy / random
            before inference for reproducible output.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If text is empty or reference files don't exist.
    """
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    # Normalize speaker_wav to a list of str (TTS API accepts either)
    if isinstance(speaker_wav, str | Path):
        speaker_wav_list = [str(speaker_wav)]
    else:
        speaker_wav_list = [str(p) for p in speaker_wav]

    for p in speaker_wav_list:
        if not Path(p).exists():
            raise ValueError(f"Reference audio not found: {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Seed before inference — reproducibility for legacy outputs.
    _apply_seed(config.seed)

    text_plan = plan_text_synthesis(text, config)
    if text_plan.warning:
        logger.warning("Text synthesis warning: {}", text_plan.warning)

    logger.info(
        "Synthesizing {} char(s) to {} with {} ref file(s)  seed={}  split_sentences={}",
        len(text),
        output_path.name,
        len(speaker_wav_list),
        config.seed,
        text_plan.xtts_split_sentences,
    )

    used_latent_cache = _try_synthesize_with_conditioning_cache(
        tts=tts,
        text=text,
        speaker_wav_list=speaker_wav_list,
        output_path=output_path,
        config=config,
    )
    if not used_latent_cache:
        tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=speaker_wav_list if len(speaker_wav_list) > 1 else speaker_wav_list[0],
            language=config.language,
            split_sentences=text_plan.xtts_split_sentences,
            # XTTS-v2 supports these decoder kwargs via the API:
            temperature=config.temperature,
            length_penalty=config.length_penalty,
            repetition_penalty=config.repetition_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
            speed=config.speed,
            # Conditioning knobs — model.synthesize pops exactly these names,
            # so the fallback now conditions identically to the cached path.
            gpt_cond_len=config.gpt_cond_len,
            gpt_cond_chunk_len=config.gpt_cond_chunk_len,
            max_ref_len=config.max_ref_len,
            sound_norm_refs=config.sound_norm_refs,
        )

    logger.info("Wrote {} ({} Hz)", output_path, XTTS_OUTPUT_SR)
    return output_path
