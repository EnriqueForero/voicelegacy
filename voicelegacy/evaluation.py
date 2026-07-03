"""voicelegacy — Evaluation harness: turn "sounds right" into numbers.

Context: Google Colab Free (T4, ~12 GB RAM). Designed to be driven once per
corpus to produce a frozen baseline, then re-run after every precision change.

The single golden rule of the improvement plan is: *no precision change ships
without a number from this harness*. This module is that number-maker.

Design — why synthesis is injected, not imported:
    `run_benchmark` does NOT import coqui-tts. It receives a ``synthesize``
    callable ``(text) -> (waveform, sample_rate)``. In production the CLI wires
    that callable to the real XTTS model; in tests a mock returns synthetic
    audio. This keeps the heavy GPU coupling at the edge (the CLI) and leaves
    the harness pure, fast, and unit-testable without a GPU or model weights.

Metrics (every one is optional and degrades to ``skipped`` when its dependency
is absent, exactly like ``voicelegacy.similarity``):
    * SECS (speaker similarity) via ECAPA-TDNN (SpeechBrain) — the TTS-paper
      standard. https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
      Resemblyzer is kept as a cheap second opinion (already in the package).
      WARNING: ECAPA and Resemblyzer live on DIFFERENT scales. The 0.60/0.75/
      0.85 bands documented for Resemblyzer do NOT transfer to ECAPA. Calibrate
      ECAPA bands against your own corpus before trusting them.
    * WER / CER round-trip: transcribe the synthetic audio with faster-whisper
      and compare against the requested text. This is the only cheap detector
      of the dominant long-text failure mode — silent truncation / hallucination
      / eaten words. https://github.com/SYSTRAN/faster-whisper
    * MOS proxy via TorchAudio-SQUIM objective (no clean reference needed):
      PESQ / STOI / SI-SDR. A consistent proxy, not ground truth.
      https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html
    * RTF (compute seconds / audio seconds) and peak VRAM, from
      ``voicelegacy.telemetry``.

The WER/CER computation itself is implemented here (edit distance) so it needs
no extra dependency — only the ASR step needs faster-whisper.

Author: Néstor Enrique Forero Herrera   Date: 2026-06-10   Version: 0.5.0
"""

from __future__ import annotations

import json
import re
import statistics
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voicelegacy.logging_config import get_logger
from voicelegacy.telemetry import runtime_snapshot

logger = get_logger()

# A synthesis backend: text in, (mono float waveform, sample_rate) out.
SynthesizeFn = Callable[[str], tuple[np.ndarray, int]]

DEFAULT_GOLDEN_RESOURCE = "golden_texts_es.txt"
DEFAULT_ASR_MODEL = "large-v3"
DEFAULT_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
VALID_CATEGORIES = ("short", "medium", "long", "chapter")

# Module-level model caches. Loading any of these is slow and downloads weights
# on first use; we never want to pay that per sample.
_ECAPA_CACHE: dict[str, Any] = {}
_ASR_CACHE: dict[str, Any] = {}
_SQUIM_CACHE: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BenchmarkConfig:
    """Knobs for a benchmark run. Zero magic numbers live in the logic.

    Attributes:
        compute_secs_ecapa: Compute ECAPA-TDNN speaker similarity (needs
            ``speechbrain``).
        compute_secs_resemblyzer: Also compute the Resemblyzer second opinion
            (needs ``resemblyzer``).
        compute_wer: Transcribe with faster-whisper and compute WER/CER
            (needs ``faster-whisper``).
        compute_mos: Compute the SQUIM objective MOS proxy (needs ``torchaudio``).
        asr_model: faster-whisper model id used for the WER round-trip.
        asr_language: ISO language passed to the ASR model.
        ecapa_source: Hugging Face id of the ECAPA-TDNN checkpoint.
        wer_fold_accents: Strip accents before WER (more lenient; off by default
            because Spanish accents are phonemically meaningful).
        wer_expand_numbers: Expand digit runs to Spanish words before WER, if
            ``num2words`` is installed (handles "2024" vs "dos mil veinticuatro").
        seed: Recorded in the report for traceability; the injected synthesizer
            owns the actual seeding.
    """

    compute_secs_ecapa: bool = True
    compute_secs_resemblyzer: bool = True
    compute_wer: bool = True
    compute_mos: bool = True
    asr_model: str = DEFAULT_ASR_MODEL
    asr_language: str = "es"
    ecapa_source: str = DEFAULT_ECAPA_SOURCE
    wer_fold_accents: bool = False
    wer_expand_numbers: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.asr_model.strip():
            raise ValueError("asr_model must be a non-empty model id.")
        if not self.asr_language.strip():
            raise ValueError("asr_language must be a non-empty ISO code.")
        if not self.ecapa_source.strip():
            raise ValueError("ecapa_source must be a non-empty HF id.")

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {
            "compute_secs_ecapa": self.compute_secs_ecapa,
            "compute_secs_resemblyzer": self.compute_secs_resemblyzer,
            "compute_wer": self.compute_wer,
            "compute_mos": self.compute_mos,
            "asr_model": self.asr_model,
            "asr_language": self.asr_language,
            "ecapa_source": self.ecapa_source,
            "wer_fold_accents": self.wer_fold_accents,
            "wer_expand_numbers": self.wer_expand_numbers,
            "seed": self.seed,
        }


# ═══════════════════════════════════════════════════════════════════
# Value objects
# ═══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Metric:
    """One metric outcome, including the case where it could not run.

    Attributes:
        value: Numeric result, or None when ``status`` is not ``"ok"``.
        status: ``"ok"`` | ``"skipped"`` | ``"error"``.
        detail: Human-readable reason (dependency missing, exception text...).
    """

    value: float | None
    status: str
    detail: str = ""

    @classmethod
    def ok(cls, value: float, detail: str = "") -> Metric:
        return cls(value=float(value), status="ok", detail=detail)

    @classmethod
    def skipped(cls, detail: str) -> Metric:
        return cls(value=None, status="skipped", detail=detail)

    @classmethod
    def error(cls, detail: str) -> Metric:
        return cls(value=None, status="error", detail=detail)

    def to_dict(self) -> dict[str, object]:
        """Rich serialization (keeps ``detail``) for the JSON report."""
        out: dict[str, object] = {"status": self.status}
        out["value"] = round(self.value, 4) if self.value is not None else None
        if self.detail:
            out["detail"] = self.detail
        return out

    def flat(self, prefix: str) -> dict[str, object]:
        """Flat two-column serialization for tabular (parquet) rows."""
        value = round(self.value, 4) if self.value is not None else None
        return {prefix: value, f"{prefix}_status": self.status}


@dataclass(frozen=True)
class GoldenText:
    """A single benchmark stimulus.

    Attributes:
        text_id: Stable id (e.g. ``"short-01"``).
        category: One of ``VALID_CATEGORIES``.
        text: The text to synthesize.
    """

    text_id: str
    category: str
    text: str

    @property
    def char_count(self) -> int:
        """Number of characters in the stimulus text."""
        return len(self.text)


@dataclass(frozen=True)
class SampleEvaluation:
    """All measurements for one synthesized stimulus."""

    text_id: str
    category: str
    char_count: int
    audio_seconds: float
    synth_seconds: float
    rtf: float | None
    peak_vram_mb: float | None
    metrics: dict[str, Metric]
    asr_transcript: str | None
    audio_path: Path | None

    def to_dict(self) -> dict[str, object]:
        """Rich serialization (nested metrics) for the JSON report."""
        return {
            "text_id": self.text_id,
            "category": self.category,
            "char_count": self.char_count,
            "audio_seconds": round(self.audio_seconds, 3),
            "synth_seconds": round(self.synth_seconds, 3),
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
            "peak_vram_mb": self.peak_vram_mb,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "asr_transcript": self.asr_transcript,
            "audio_path": str(self.audio_path) if self.audio_path else None,
        }

    def to_row(self) -> dict[str, object]:
        """Flat serialization for a tabular (parquet/JSONL) row."""
        row: dict[str, object] = {
            "text_id": self.text_id,
            "category": self.category,
            "char_count": self.char_count,
            "audio_seconds": round(self.audio_seconds, 3),
            "synth_seconds": round(self.synth_seconds, 3),
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
            "peak_vram_mb": self.peak_vram_mb,
        }
        for name, metric in self.metrics.items():
            row.update(metric.flat(name))
        return row


@dataclass(frozen=True)
class BenchmarkReport:
    """Full result of a benchmark run: per-sample rows plus aggregates."""

    run_id: str
    created_at: str
    voicelegacy_version: str
    config: dict[str, object]
    runtime: dict[str, object]
    samples: tuple[SampleEvaluation, ...]
    n_references: int = 0
    notes: dict[str, object] = field(default_factory=dict)

    def _metric_names(self) -> list[str]:
        names: list[str] = []
        for sample in self.samples:
            for name in sample.metrics:
                if name not in names:
                    names.append(name)
        return names

    def summary(self) -> dict[str, object]:
        """Aggregate stats per metric (over ``ok`` samples) and per category."""

        def _agg(values: Sequence[float]) -> dict[str, float] | None:
            if not values:
                return None
            return {
                "n": len(values),
                "mean": round(statistics.fmean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "stdev": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            }

        metric_summary: dict[str, object] = {}
        for name in self._metric_names():
            ok_values = [
                s.metrics[name].value
                for s in self.samples
                if name in s.metrics
                and s.metrics[name].status == "ok"
                and s.metrics[name].value is not None
            ]
            statuses = [s.metrics[name].status for s in self.samples if name in s.metrics]
            metric_summary[name] = {
                "stats": _agg(ok_values),
                "n_ok": statuses.count("ok"),
                "n_skipped": statuses.count("skipped"),
                "n_error": statuses.count("error"),
            }

        rtf_values = [s.rtf for s in self.samples if s.rtf is not None]
        category_counts: dict[str, int] = {}
        for sample in self.samples:
            category_counts[sample.category] = category_counts.get(sample.category, 0) + 1

        return {
            "n_samples": len(self.samples),
            "by_category": category_counts,
            "rtf": _agg(rtf_values),
            "metrics": metric_summary,
        }

    def to_dict(self) -> dict[str, object]:
        """Full JSON-friendly report including the per-sample detail."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "voicelegacy_version": self.voicelegacy_version,
            "n_references": self.n_references,
            "config": self.config,
            "runtime": self.runtime,
            "notes": self.notes,
            "summary": self.summary(),
            "samples": [s.to_dict() for s in self.samples],
        }

    def write_json(self, path: Path) -> Path:
        """Write the full report to ``path`` (creating parents)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Benchmark report written: {}", path)
        return path

    def rows(self) -> list[dict[str, object]]:
        """Flat rows (one per sample) for tabular accumulation."""
        return [{"run_id": self.run_id, **s.to_row()} for s in self.samples]


# ═══════════════════════════════════════════════════════════════════
# Golden-set loading
# ═══════════════════════════════════════════════════════════════════
def load_golden_texts(path: Path | None = None) -> list[GoldenText]:
    """Load the benchmark stimulus set.

    Args:
        path: A ``golden_texts_es.txt``-formatted file. If None, the version
            packaged with voicelegacy is used.

    Returns:
        Parsed stimuli with per-category sequential ids.

    Raises:
        FileNotFoundError: If an explicit ``path`` is given but does not exist.
        ValueError: If the file is empty or has an invalid category/format.
    """
    if path is None:
        raw = (
            resources.files("voicelegacy.data").joinpath(DEFAULT_GOLDEN_RESOURCE).read_text("utf-8")
        )
        origin = f"packaged:{DEFAULT_GOLDEN_RESOURCE}"
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Golden texts file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        origin = str(path)

    texts: list[GoldenText] = []
    per_category: dict[str, int] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(
                f"{origin}:{lineno}: expected <category><TAB><text>, got: {stripped[:60]!r}"
            )
        category, text = line.split("\t", 1)
        category = category.strip()
        text = text.strip()
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"{origin}:{lineno}: invalid category {category!r} (allowed: {VALID_CATEGORIES})"
            )
        if not text:
            raise ValueError(f"{origin}:{lineno}: empty text for category {category!r}")
        per_category[category] = per_category.get(category, 0) + 1
        texts.append(
            GoldenText(
                text_id=f"{category}-{per_category[category]:02d}", category=category, text=text
            )
        )

    if not texts:
        raise ValueError(f"{origin}: no stimuli found (file is empty or all comments).")
    logger.info("Loaded {} golden stimuli from {}", len(texts), origin)
    return texts


# ═══════════════════════════════════════════════════════════════════
# Text normalization + WER / CER (no extra dependency)
# ═══════════════════════════════════════════════════════════════════
_PUNCT_RE = re.compile(r"[^0-9a-záéíóúñü\s]", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\d+")


def _strip_accents(text: str) -> str:
    """Remove combining accents, keeping ñ → n folding explicit-free."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _expand_numbers_es(text: str) -> str:
    """Expand digit runs to Spanish words, if ``num2words`` is installed.

    Returns the text unchanged (and warns once) when num2words is missing, so
    the WER still works — at the cost of over-penalizing number tokens when the
    ASR spells them out.
    """
    try:
        from num2words import num2words
    except ImportError:
        logger.debug("num2words not installed; digits left as-is for WER normalization.")
        return text

    def _sub(match: re.Match[str]) -> str:
        return num2words(int(match.group()), lang="es")

    return _DIGITS_RE.sub(_sub, text)


def normalize_text(
    text: str,
    *,
    fold_accents: bool = False,
    expand_numbers: bool = True,
) -> str:
    """Canonicalize text for a fair word/character error rate comparison.

    The same normalizer must be applied to BOTH the reference and the ASR
    hypothesis; what matters is that the transform is deterministic and applied
    identically to both sides.

    Steps: NFC → optional number expansion → lowercase → optional accent fold →
    strip punctuation (keeps Spanish letters and digits) → collapse whitespace.
    """
    out = unicodedata.normalize("NFC", text)
    if expand_numbers:
        out = _expand_numbers_es(out)
    out = out.lower()
    if fold_accents:
        out = _strip_accents(out)
    out = _PUNCT_RE.sub(" ", out)
    out = _WS_RE.sub(" ", out).strip()
    return out


def _edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Levenshtein edit distance between two token sequences (O(n·m), O(m) RAM)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, *, normalizer: Callable[[str], str]) -> float:
    """Word error rate ∈ [0, ∞). 0 = perfect; > 1 means more edits than words.

    A reference that normalizes to empty returns 0.0 when the hypothesis is also
    empty, else 1.0 (everything is an insertion).
    """
    ref_tokens = normalizer(reference).split()
    hyp_tokens = normalizer(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def character_error_rate(
    reference: str, hypothesis: str, *, normalizer: Callable[[str], str]
) -> float:
    """Character error rate ∈ [0, ∞) over normalized, space-free strings."""
    ref_chars = normalizer(reference).replace(" ", "")
    hyp_chars = normalizer(hypothesis).replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ═══════════════════════════════════════════════════════════════════
# Audio helpers
# ═══════════════════════════════════════════════════════════════════
def _load_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    """Load a WAV as mono float32 resampled to ``target_sr``."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(audio, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
# SECS — ECAPA-TDNN (primary) + Resemblyzer (second opinion)
# ═══════════════════════════════════════════════════════════════════
def _load_ecapa_encoder(source: str) -> Any:
    """Load (or fetch from cache) a SpeechBrain ECAPA-TDNN encoder.

    Raises:
        ImportError: If speechbrain (and torch) are not installed.
    """
    if source in _ECAPA_CACHE:
        return _ECAPA_CACHE[source]
    try:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0 layout
            from speechbrain.pretrained import EncoderClassifier
    except ImportError as exc:
        raise ImportError(
            "speechbrain is not installed. To enable ECAPA-TDNN speaker "
            "similarity run: pip install speechbrain. This is an OPTIONAL "
            "dependency; the rest of the harness works without it."
        ) from exc

    logger.info("Loading ECAPA-TDNN encoder {} (first call downloads weights)...", source)
    savedir = Path.home() / ".cache" / "voicelegacy" / "ecapa" / source.replace("/", "__")
    encoder = EncoderClassifier.from_hparams(source=source, savedir=str(savedir))
    _ECAPA_CACHE[source] = encoder
    return encoder


def release_ecapa_encoder() -> None:
    """Evict the cached ECAPA encoder and free CUDA memory if possible."""
    _ECAPA_CACHE.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _embed_ecapa(encoder: Any, wav_path: Path) -> np.ndarray:
    """Embed one WAV with ECAPA at 16 kHz; returns a 1-D float vector."""
    import torch

    audio = _load_wav_mono(wav_path, target_sr=16000)
    tensor = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        embedding = encoder.encode_batch(tensor)
    return embedding.squeeze().detach().cpu().numpy().astype(np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity guarded against zero-norm vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def secs_ecapa_metric(output_wav: Path, reference_wavs: Sequence[Path], source: str) -> Metric:
    """Cosine SECS between the output and the centroid of references (ECAPA)."""
    if not reference_wavs:
        return Metric.error("no reference WAVs provided")
    try:
        encoder = _load_ecapa_encoder(source)
        ref_embeddings = [_embed_ecapa(encoder, Path(r)) for r in reference_wavs]
        centroid = np.mean(np.stack(ref_embeddings, axis=0), axis=0)
        out_embedding = _embed_ecapa(encoder, Path(output_wav))
        return Metric.ok(max(0.0, _cosine(out_embedding, centroid)), detail="ecapa-tdnn")
    except ImportError as exc:
        return Metric.skipped(str(exc).split(".")[0])
    except Exception as exc:
        return Metric.error(f"{type(exc).__name__}: {exc}")


def secs_resemblyzer_metric(output_wav: Path, reference_wavs: Sequence[Path]) -> Metric:
    """Resemblyzer SECS (cheap second opinion). Skips if resemblyzer absent."""
    if not reference_wavs:
        return Metric.error("no reference WAVs provided")
    try:
        from voicelegacy.similarity import compute_similarity

        report = compute_similarity(Path(output_wav), [Path(r) for r in reference_wavs])
        return Metric.ok(report.score, detail="resemblyzer")
    except ImportError as exc:
        return Metric.skipped(str(exc).split(".")[0])
    except Exception as exc:
        return Metric.error(f"{type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════
# WER round-trip — faster-whisper ASR
# ═══════════════════════════════════════════════════════════════════
def _load_asr_model(model_id: str) -> Any:
    """Load (or fetch from cache) a faster-whisper model.

    Raises:
        ImportError: If faster-whisper is not installed.
    """
    if model_id in _ASR_CACHE:
        return _ASR_CACHE[model_id]
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. To enable the WER round-trip run: "
            "pip install faster-whisper. This is an OPTIONAL dependency."
        ) from exc

    device, compute_type = "cpu", "int8"
    try:
        import torch

        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
    except ImportError:
        pass

    logger.info(
        "Loading faster-whisper {} on {} (first call downloads weights)...", model_id, device
    )
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    _ASR_CACHE[model_id] = model
    return model


def transcribe(wav_path: Path, model_id: str, language: str) -> str:
    """Transcribe a WAV with faster-whisper. Raises ImportError if absent."""
    model = _load_asr_model(model_id)
    segments, _info = model.transcribe(str(wav_path), language=language, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


def wer_cer_metrics(
    reference_text: str, output_wav: Path, config: BenchmarkConfig
) -> tuple[Metric, Metric, str | None]:
    """Return (WER metric, CER metric, ASR transcript) for one sample."""

    def _normalizer(text: str) -> str:
        return normalize_text(
            text,
            fold_accents=config.wer_fold_accents,
            expand_numbers=config.wer_expand_numbers,
        )

    try:
        transcript = transcribe(Path(output_wav), config.asr_model, config.asr_language)
    except ImportError as exc:
        msg = str(exc).split(".")[0]
        return Metric.skipped(msg), Metric.skipped(msg), None
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        return Metric.error(msg), Metric.error(msg), None

    wer = word_error_rate(reference_text, transcript, normalizer=_normalizer)
    cer = character_error_rate(reference_text, transcript, normalizer=_normalizer)
    return Metric.ok(wer), Metric.ok(cer), transcript


# ═══════════════════════════════════════════════════════════════════
# MOS proxy — TorchAudio-SQUIM (objective, no clean reference)
# ═══════════════════════════════════════════════════════════════════
def _load_squim_model() -> Any:
    """Load (or fetch from cache) the SQUIM objective model.

    Raises:
        ImportError: If torchaudio is not installed.
    """
    if "objective" in _SQUIM_CACHE:
        return _SQUIM_CACHE["objective"]
    try:
        from torchaudio.pipelines import SQUIM_OBJECTIVE
    except ImportError as exc:
        raise ImportError(
            "torchaudio is not installed. To enable the SQUIM MOS proxy run: "
            "pip install torchaudio. This is an OPTIONAL dependency."
        ) from exc

    logger.info("Loading TorchAudio-SQUIM objective model (first call downloads weights)...")
    model = SQUIM_OBJECTIVE.get_model()
    _SQUIM_CACHE["objective"] = model
    return model


def squim_metrics(output_wav: Path) -> dict[str, Metric]:
    """Return {squim_pesq, squim_stoi, squim_sisdr} from one SQUIM forward pass."""
    names = ("squim_pesq", "squim_stoi", "squim_sisdr")
    try:
        import torch

        model = _load_squim_model()
        audio = _load_wav_mono(Path(output_wav), target_sr=16000)
        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            stoi, pesq, si_sdr = model(tensor)
        return {
            "squim_pesq": Metric.ok(float(pesq.item()), detail="squim-objective"),
            "squim_stoi": Metric.ok(float(stoi.item()), detail="squim-objective"),
            "squim_sisdr": Metric.ok(float(si_sdr.item()), detail="squim-objective"),
        }
    except ImportError as exc:
        msg = str(exc).split(".")[0]
        return {name: Metric.skipped(msg) for name in names}
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        return {name: Metric.error(msg) for name in names}


# ═══════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════
def _reset_peak_vram() -> None:
    """Reset CUDA peak-memory accounting so per-sample VRAM is isolated."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def evaluate_sample(
    stimulus: GoldenText,
    synthesize: SynthesizeFn,
    reference_wavs: Sequence[Path],
    audio_dir: Path,
    config: BenchmarkConfig,
) -> SampleEvaluation:
    """Synthesize one stimulus, persist the audio, and compute every metric."""
    _reset_peak_vram()
    start = time.perf_counter()
    waveform, sample_rate = synthesize(stimulus.text)
    synth_seconds = time.perf_counter() - start

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    audio_seconds = len(waveform) / sample_rate if sample_rate > 0 else 0.0
    rtf = (synth_seconds / audio_seconds) if audio_seconds > 0 else None
    peak_vram_mb = runtime_snapshot().cuda_max_allocated_mb

    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{stimulus.text_id}.wav"
    sf.write(str(audio_path), waveform, sample_rate, subtype="PCM_16")

    metrics: dict[str, Metric] = {}
    transcript: str | None = None

    if config.compute_secs_ecapa:
        metrics["secs_ecapa"] = secs_ecapa_metric(audio_path, reference_wavs, config.ecapa_source)
    if config.compute_secs_resemblyzer:
        metrics["secs_resemblyzer"] = secs_resemblyzer_metric(audio_path, reference_wavs)
    if config.compute_wer:
        wer, cer, transcript = wer_cer_metrics(stimulus.text, audio_path, config)
        metrics["wer"] = wer
        metrics["cer"] = cer
    if config.compute_mos:
        metrics.update(squim_metrics(audio_path))

    return SampleEvaluation(
        text_id=stimulus.text_id,
        category=stimulus.category,
        char_count=stimulus.char_count,
        audio_seconds=audio_seconds,
        synth_seconds=synth_seconds,
        rtf=rtf,
        peak_vram_mb=peak_vram_mb,
        metrics=metrics,
        asr_transcript=transcript,
        audio_path=audio_path,
    )


def run_benchmark(
    synthesize: SynthesizeFn,
    texts: Iterable[GoldenText],
    reference_wavs: Sequence[Path],
    *,
    audio_dir: Path,
    config: BenchmarkConfig | None = None,
    run_id: str | None = None,
    notes: dict[str, object] | None = None,
) -> BenchmarkReport:
    """Run the full benchmark over a stimulus set.

    Args:
        synthesize: Backend callable ``(text) -> (waveform, sample_rate)``. In
            production this wraps the real XTTS model; in tests, a mock. The
            harness never imports a TTS engine itself.
        texts: Stimuli to evaluate (e.g. from ``load_golden_texts``).
        reference_wavs: The speaker's reference corpus WAVs (for SECS).
        audio_dir: Directory where synthesized WAVs are written (for the
            file-based metrics and for human audit).
        config: Benchmark configuration; defaults to ``BenchmarkConfig()``.
        run_id: Identifier for this run; defaults to a UTC timestamp.
        notes: Free-form metadata to embed in the report (e.g. engine name).

    Returns:
        A ``BenchmarkReport`` with per-sample rows and aggregates.

    Raises:
        ValueError: If ``texts`` is empty or no reference WAVs are usable.
        FileNotFoundError: If a reference WAV path does not exist.
    """
    config = config or BenchmarkConfig()
    stimuli = list(texts)
    if not stimuli:
        raise ValueError("No stimuli to benchmark. Pass a non-empty `texts`.")

    # Pre-flight (Fail Fast): references must exist before any synthesis.
    refs = [Path(r) for r in reference_wavs]
    if not refs:
        raise ValueError("At least one reference WAV is required for SECS.")
    missing = [str(r) for r in refs if not r.exists()]
    if missing:
        raise FileNotFoundError(f"Reference WAV(s) not found: {missing}")

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audio_dir = Path(audio_dir) / run_id

    try:
        from voicelegacy import __version__ as version
    except ImportError:  # pragma: no cover - defensive
        version = "unknown"

    samples: list[SampleEvaluation] = []
    total = len(stimuli)
    logger.info("Benchmark {} — {} stimuli, {} reference WAV(s)", run_id, total, len(refs))
    for index, stimulus in enumerate(stimuli, start=1):
        logger.info("[{}/{}] {} ({} chars)", index, total, stimulus.text_id, stimulus.char_count)
        samples.append(evaluate_sample(stimulus, synthesize, refs, audio_dir, config))

    report = BenchmarkReport(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        voicelegacy_version=version,
        config=config.to_dict(),
        runtime=runtime_snapshot().to_dict(),
        samples=tuple(samples),
        n_references=len(refs),
        notes=notes or {},
    )
    logger.info("Benchmark {} complete: {} samples evaluated.", run_id, len(samples))
    return report


# ═══════════════════════════════════════════════════════════════════
# Cumulative accumulation across runs
# ═══════════════════════════════════════════════════════════════════
def append_to_cumulative(report: BenchmarkReport, parquet_path: Path) -> Path:
    """Append this run's flat rows to a cumulative table for cross-run trends.

    Prefers parquet (needs pandas + a parquet engine). Falls back to a JSONL
    sibling when those are unavailable, so accumulation never hard-fails on a
    bare install. Returns the path actually written.
    """
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = report.rows()

    try:
        import pandas as pd

        df_new = pd.DataFrame(new_rows)
        if parquet_path.exists():
            df_old = pd.read_parquet(parquet_path)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_parquet(parquet_path, index=False)
        logger.info("Appended {} rows to {}", len(new_rows), parquet_path)
        return parquet_path
    except ImportError:
        fallback = parquet_path.with_suffix(".jsonl")
        with fallback.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.warning(
            "pandas/parquet engine unavailable; appended {} rows to {} instead.",
            len(new_rows),
            fallback,
        )
        return fallback


# ═══════════════════════════════════════════════════════════════════
# Availability probes (graceful skip helpers)
# ═══════════════════════════════════════════════════════════════════
def ecapa_available() -> bool:
    """True if speechbrain is importable."""
    try:
        import speechbrain  # noqa: F401

        return True
    except ImportError:
        return False


def asr_available() -> bool:
    """True if faster-whisper is importable."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def squim_available() -> bool:
    """True if torchaudio is importable."""
    try:
        import torchaudio  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "GoldenText",
    "Metric",
    "SampleEvaluation",
    "SynthesizeFn",
    "append_to_cumulative",
    "asr_available",
    "character_error_rate",
    "ecapa_available",
    "evaluate_sample",
    "load_golden_texts",
    "normalize_text",
    "run_benchmark",
    "secs_ecapa_metric",
    "secs_resemblyzer_metric",
    "squim_available",
    "squim_metrics",
    "transcribe",
    "wer_cer_metrics",
    "word_error_rate",
]
