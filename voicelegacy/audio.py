"""voicelegacy — Audio preprocessing utilities.

Functions to load, clean, normalize, resample, and segment audio files so
they meet XTTS-v2 input expectations (mono, 22050 Hz, clean, properly
normalized).

Why these specific steps:
- Resample to 22050 Hz: XTTS-v2 conditioning encoder was trained at this rate.
- Mono: XTTS-v2 ignores stereo info; converting saves memory.
- Loudness normalize to -23 LUFS: EBU R128 broadcast standard; avoids
  the conditioning network seeing wildly varying signal levels.
- Spectral denoise: XTTS-v2 will faithfully clone background noise if it's
  in the reference. Better to remove it.
- Trim silences: pad-silence in the reference makes the model think pauses
  are part of the speaker's style.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view

from voicelegacy.config import XTTS_INPUT_SR
from voicelegacy.logging_config import get_logger

logger = get_logger()

# Suppress librosa/soundfile deprecation noise during batch processing
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Container formats ffmpeg can transcode to WAV for the pipeline. Kept as a
# module constant so corpus.py and the CLI use a single source of truth.
# soundfile/librosa cannot decode .mp4, .mkv, .webm, .mov, .aac directly —
# ffmpeg is the only viable bridge.
EXTENSIONS_CONVERTIBLE_TO_WAV: frozenset[str] = frozenset(
    {".mp4", ".mp3", ".m4a", ".mkv", ".ogg", ".flac", ".aac", ".mov", ".webm"}
)


@dataclass(frozen=True)
class AudioStats:
    """Lightweight container for audio analysis results."""

    duration_s: float
    sample_rate: int
    n_channels: int
    rms_db: float
    peak_db: float
    snr_db: float
    spectral_rolloff_hz: float = 0.0


# ─── Loading ───────────────────────────────────────────────────────
def load_audio_mono(path: Path, target_sr: int = XTTS_INPUT_SR) -> tuple[np.ndarray, int]:
    """Load audio as mono float32, resampled to target_sr.

    Args:
        path: Path to input audio file (any format soundfile/librosa supports).
        target_sr: Target sampling rate in Hz. Defaults to XTTS-v2's 22050.

    Returns:
        Tuple of (audio_array, original_sample_rate).

        - audio_array: 1-D float32 numpy array at target_sr, values in [-1, 1].
        - original_sample_rate: The sample rate of the file BEFORE resampling.
          This is what callers must use to detect phone-codec / low-fidelity
          sources, since librosa.load always resamples to target_sr and would
          otherwise mask the original rate.

    Raises:
        FileNotFoundError: If `path` does not exist.
        RuntimeError: If the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Read original sample rate BEFORE librosa resamples, so phone-codec
    # detection downstream actually works. soundfile.info is metadata-only,
    # does not decode the audio body.
    try:
        info = sf.info(str(path))
        original_sr = int(info.samplerate)
    except Exception as exc:
        raise RuntimeError(f"Failed to read header of {path.name}: {exc}") from exc

    try:
        # Prefer soundfile for WAV/FLAC/OGG and other libsndfile-backed formats:
        # it is deterministic and avoids librosa/numba first-call latency in CI.
        y_raw, sr_read = sf.read(str(path), dtype="float32", always_2d=True)
        if y_raw.size == 0:
            raise RuntimeError(f"Empty audio after load: {path}")
        y = np.mean(y_raw, axis=1).astype(np.float32)
        if int(sr_read) != int(target_sr):
            from math import gcd

            from scipy.signal import resample_poly

            factor = gcd(int(sr_read), int(target_sr))
            y = resample_poly(y, int(target_sr) // factor, int(sr_read) // factor).astype(
                np.float32
            )
    except Exception as sf_exc:
        try:
            # Fallback for formats soundfile cannot decode. Container formats
            # should normally be converted by ffmpeg pre-flight first.
            y, _sr = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode {path.name}: {exc}") from sf_exc

    if y.size == 0:
        raise RuntimeError(f"Empty audio after load: {path}")

    return y.astype(np.float32, copy=False), original_sr


def compute_stats(y: np.ndarray, sr: int, original_sr: int | None = None) -> AudioStats:
    """Compute basic acoustic statistics for a mono audio array.

    Args:
        y: Mono audio, float32 in [-1, 1].
        sr: Sample rate at which `y` is currently sampled (after any resampling).
            Used for time-domain measurements like duration and SNR framing.
        original_sr: Sample rate of the SOURCE file before resampling, if known.
            Recorded in `AudioStats.sample_rate` so downstream quality gates can
            detect phone-codec / low-fidelity sources. If None, falls back to
            `sr` (back-compat path for callers that pass synthetic arrays).

    Returns:
        AudioStats with duration, levels, SNR estimate, and the most
        meaningful sample rate (original if provided, else `sr`).
    """
    duration_s = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    peak_db = 20.0 * np.log10(peak)
    snr_db = _estimate_dynamic_range_db(y)
    rolloff_hz = _spectral_rolloff_hz(y, sr)

    return AudioStats(
        duration_s=duration_s,
        sample_rate=original_sr if original_sr is not None else sr,
        n_channels=1,
        rms_db=rms_db,
        peak_db=peak_db,
        snr_db=snr_db,
        spectral_rolloff_hz=rolloff_hz,
    )


def _spectral_rolloff_hz(y: np.ndarray, sr: int, roll_percent: float = 0.85) -> float:
    """Estimate the frequency below which ``roll_percent`` of energy lives.

    This measures the signal's *actual* usable bandwidth, unlike the container
    sample rate. Telephone-band audio (~3.4 kHz) that was upsampled to 44.1 kHz
    still has a low rolloff here, so a rolloff gate catches what the header gate
    cannot. Returns the median rolloff across frames (0.0 for silence/too-short).
    """
    if y.size < 2048 or sr <= 0:
        return 0.0
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-9:
        return 0.0
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)
    except Exception as exc:
        logger.warning("Spectral rolloff failed ({}); reporting 0.", exc)
        return 0.0
    finite = rolloff[np.isfinite(rolloff)]
    return float(np.median(finite)) if finite.size else 0.0


def _estimate_dynamic_range_db(
    y: np.ndarray, frame_length: int = 2048, hop_length: int = 512
) -> float:
    """Estimate speech/noise dynamic range from frame-energy percentiles.

    This is NOT a laboratory SNR estimator. It compares the loudest 10% of
    frames against the quietest 10% and is therefore a practical speech-cleanliness
    proxy for ranking reference clips. The public field remains named ``snr_db``
    for backward compatibility, but reports should interpret it as an estimated
    dynamic range until a true WADA-SNR implementation is introduced.

    Args:
        y: Mono audio array.
        frame_length: STFT-equivalent frame size.
        hop_length: Hop between frames.

    Returns:
        Estimated dynamic range in dB. Higher generally means cleaner speech.
    """
    if len(y) < frame_length:
        return 0.0

    # Frame-wise RMS energy. Framing is done with an explicit numpy sliding
    # window instead of librosa.util.frame: with 1-D input and ``axis=0``,
    # librosa >= 0.10 returns shape (n_frames, frame_length), so the previous
    # ``np.mean(frames**2, axis=0)`` averaged ACROSS frames — collapsing 300+
    # frames into 2048 near-identical per-position values. Top-10%/bottom-10%
    # of those is ~1.0, i.e. ~0 dB for ANY real recording, which made the
    # ``snr >= 15 dB`` gate reject every candidate and forced
    # ``denoise_only_if_noisy`` to always denoise. The window view makes the
    # sample axis unambiguous regardless of librosa version.
    windows = sliding_window_view(y, frame_length)[::hop_length]
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
    rms_sorted = np.sort(rms)

    n = len(rms_sorted)
    if n < 10:
        return 0.0

    # Top 10% = speech, bottom 10% = noise
    noise_rms = float(np.mean(rms_sorted[: max(1, n // 10)]))
    signal_rms = float(np.mean(rms_sorted[-max(1, n // 10) :]))

    if noise_rms <= 1e-9:
        return 60.0  # essentially noiseless

    return float(20.0 * np.log10(signal_rms / noise_rms))


def _estimate_snr_db(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Backward-compatible alias for the dynamic-range estimator.

    Deprecated: use ``_estimate_dynamic_range_db``. Kept so external notebooks
    that imported the private helper do not break immediately.
    """
    return _estimate_dynamic_range_db(y, frame_length=frame_length, hop_length=hop_length)


# ─── Cleaning ──────────────────────────────────────────────────────
def apply_preemphasis(y: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Apply a simple speech pre-emphasis filter.

    Pre-emphasis can improve intelligibility for muffled archival recordings by
    lifting high-frequency speech cues before noise reduction. It is deliberately
    optional because overuse can make sibilants harsh.
    """
    if not 0.0 <= coefficient < 1.0:
        raise ValueError(f"coefficient must be in [0, 1); got {coefficient}")
    if y.size == 0 or coefficient == 0.0:
        return y.astype(np.float32, copy=False)
    out = np.empty_like(y, dtype=np.float32)
    out[0] = y[0]
    out[1:] = y[1:] - coefficient * y[:-1]
    return out.astype(np.float32)


def apply_bandpass(
    y: np.ndarray,
    sr: int,
    low_hz: float = 80.0,
    high_hz: float = 10000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a conservative Butterworth band-pass filter for speech cleanup.

    The defaults remove rumble below speech fundamentals and ultrasonic/hash
    content above the useful bandwidth for XTTS-v2 reference conditioning.
    """
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Band-pass requires 0 < low_hz < high_hz")
    nyquist = sr / 2.0
    high = min(high_hz, nyquist * 0.95)
    if high <= low_hz:
        return y.astype(np.float32, copy=False)
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(order, [low_hz / nyquist, high / nyquist], btype="bandpass", output="sos")
        return np.asarray(sosfiltfilt(sos, y), dtype=np.float32)
    except Exception as exc:
        logger.warning("Band-pass filter failed ({}), returning original.", exc)
        return y.astype(np.float32, copy=False)


def denoise(
    y: np.ndarray,
    sr: int,
    prop_decrease: float = 0.85,
    stationary: bool = False,
    time_constant_s: float = 2.0,
) -> np.ndarray:
    """Apply spectral-gating noise reduction.

    Uses the noisereduce library (stationary noise model). Removes background
    hiss, fan noise, and broadband floor without distorting speech if
    prop_decrease is kept moderate.

    Args:
        y: Mono audio array.
        sr: Sample rate.
        prop_decrease: How aggressively to attenuate noise (0-1).
            0.85 is a good balance for speech.

    Returns:
        Denoised audio, same length as input.
    """
    try:
        y_clean = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=stationary,
            time_constant_s=time_constant_s,
        )
    except Exception as exc:
        logger.warning("Denoise failed ({}), returning original.", exc)
        return y
    return np.asarray(y_clean, dtype=np.float32)


def loudness_normalize(
    y: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
    peak_ceiling_dbfs: float = -3.0,
) -> np.ndarray:
    """Normalize audio to target loudness with guaranteed peak headroom.

    Two-stage normalization:
        1. LUFS-normalize to target_lufs (EBU R128 / BS.1770).
        2. If the resulting signal peaks above peak_ceiling_dbfs, apply a
           uniform linear gain so the new peak sits exactly at the ceiling.

    Stage 2 is what fixes the circular clipping bug: pyloudnorm applies a
    scalar gain to hit the LUFS target, which can push transient peaks above
    1.0. Naive np.clip(-1, 1) hard-clips them (peak = 0.0 dBFS exact), and the
    downstream quality gate then rejects the file for "clipping risk" — a
    failure mode the pipeline itself created. With a peak limiter to
    -3 dBFS, we leave headroom and never re-introduce clipping artifacts.

    Args:
        y: Mono audio.
        sr: Sample rate.
        target_lufs: Target integrated loudness. -23 LUFS = EBU broadcast standard.
        peak_ceiling_dbfs: Max allowed peak after normalization, in dBFS.
            -3.0 dBFS leaves comfortable headroom for XTTS-v2 reference audio.
            Must be < 0.0.

    Returns:
        Normalized audio, guaranteed `max(abs(y)) <= 10**(peak_ceiling_dbfs/20)`.

    Raises:
        ValueError: If peak_ceiling_dbfs >= 0 (would not protect against clipping).
    """
    if peak_ceiling_dbfs >= 0.0:
        raise ValueError(
            f"peak_ceiling_dbfs must be < 0 to guarantee headroom; got {peak_ceiling_dbfs}"
        )

    ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)  # e.g. -3 dBFS → 0.7079

    meter = pyln.Meter(sr)  # BS.1770 meter
    try:
        loudness = meter.integrated_loudness(y)
        y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
    except ValueError:
        # Signal too short for gated LUFS measurement → fall back to peak.
        logger.warning("Loudness measurement failed (signal too short); using peak normalization.")
        peak = float(np.max(np.abs(y))) + 1e-9
        y_norm = y * (ceiling / peak)
        return y_norm.astype(np.float32)

    # Peak limiter: scale down (never up) to enforce the ceiling. This is a
    # linear gain on the whole signal, not a hard clip — preserves shape.
    peak = float(np.max(np.abs(y_norm))) + 1e-9
    if peak > ceiling:
        y_norm = y_norm * (ceiling / peak)
        logger.debug(
            "Peak limiter applied: {:.3f} → {:.3f} (ceiling {:.1f} dBFS)",
            peak,
            ceiling,
            peak_ceiling_dbfs,
        )

    return y_norm.astype(np.float32)


def trim_silence(
    y: np.ndarray, top_db: float = 35.0, frame_length: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Trim leading and trailing silence using librosa.effects.trim.

    Args:
        y: Mono audio.
        top_db: Silence threshold below the peak in dB.
        frame_length: Analysis frame size.
        hop_length: Frame hop.

    Returns:
        Trimmed audio (may be empty if everything was below threshold).
    """
    if y.size == 0:
        return y.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y)))
    if peak <= 1e-12:
        return np.asarray([], dtype=np.float32)

    threshold = peak * (10.0 ** (-top_db / 20.0))
    indices = np.flatnonzero(np.abs(y) > threshold)
    if indices.size == 0:
        return np.asarray([], dtype=np.float32)

    # Expand to approximate librosa's frame-aware trim behavior without paying
    # the librosa/numba first-call cost in CI and notebooks.
    start = max(0, int(indices[0]) - frame_length)
    end = min(len(y), int(indices[-1]) + frame_length + hop_length)
    return np.asarray(y[start:end], dtype=np.float32)


# ─── Persistence ───────────────────────────────────────────────────
def save_wav(path: Path, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Save mono audio array to a WAV file.

    Args:
        path: Output path. Parent directory is created if missing.
        y: Mono audio in [-1, 1].
        sr: Sample rate.
        subtype: soundfile subtype. PCM_16 is universal; PCM_24 for higher fidelity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)


# ─── Slicing ───────────────────────────────────────────────────────
def slice_segment(
    y: np.ndarray, sr: int, start_s: float, end_s: float, pad_s: float = 0.0
) -> np.ndarray:
    """Extract a segment by time, with optional symmetric padding.

    Args:
        y: Full audio array.
        sr: Sample rate.
        start_s: Segment start in seconds.
        end_s: Segment end in seconds.
        pad_s: Extra context to include on each side (clamped to bounds).

    Returns:
        Sliced audio array.
    """
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be > start_s ({start_s})")

    n = len(y)
    i0 = max(0, int((start_s - pad_s) * sr))
    i1 = min(n, int((end_s + pad_s) * sr))
    return y[i0:i1]


# ─── Assembly (long-form joins) ────────────────────────────────────
def silence(duration_ms: float, sr: int) -> np.ndarray:
    """Return mono float32 silence of ``duration_ms`` at ``sr`` (>= 0 ms)."""
    if duration_ms < 0:
        raise ValueError(f"duration_ms must be >= 0; got {duration_ms}")
    return np.zeros(round(duration_ms / 1000.0 * sr), dtype=np.float32)


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Join ``a`` and ``b`` with an equal-power crossfade of ``fade_ms``.

    Equal-power (constant-power) uses cos/sin curves so the perceived loudness
    stays constant across the fade. A naive linear crossfade dips about 3 dB at
    the midpoint, which is audible as a little dip at every seam.

    The fade length is clamped to the shorter clip. If either clip is empty or
    ``fade_ms`` is non-positive, the clips are simply concatenated.

    Args:
        a: First mono clip (float32, [-1, 1]).
        b: Second mono clip.
        sr: Sample rate of both clips.
        fade_ms: Crossfade duration in milliseconds.

    Returns:
        ``a`` and ``b`` joined; length is ``len(a) + len(b) - overlap``.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    fade_len = round(fade_ms / 1000.0 * sr)
    overlap = min(fade_len, len(a), len(b))
    if overlap <= 0:
        return np.concatenate([a, b]).astype(np.float32)

    t = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    fade_out = np.cos(t * (np.pi / 2.0))
    fade_in = np.sin(t * (np.pi / 2.0))
    blended = a[-overlap:] * fade_out + b[:overlap] * fade_in
    return np.concatenate([a[:-overlap], blended, b[overlap:]]).astype(np.float32)


def concatenate_audio(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate mono clips, ignoring empty ones. Empty input → empty array."""
    pieces = [np.asarray(p, dtype=np.float32).reshape(-1) for p in parts if np.asarray(p).size]
    if not pieces:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


# ─── Pipeline shortcut ─────────────────────────────────────────────
def clean_segment(
    y: np.ndarray,
    sr: int,
    *,
    apply_denoise: bool = True,
    denoise_stationary: bool = False,
    denoise_only_if_noisy: bool = False,
    denoise_snr_threshold_db: float = 25.0,
    apply_bandpass_filter: bool = False,
    apply_preemphasis_filter: bool = False,
    target_lufs: float = -23.0,
    min_duration_s: float = 0.0,
) -> np.ndarray:
    """Canonical reference-segment cleaning chain — the single source of truth.

    Order matters and is fixed here so it can never drift between callers:
    denoise (optionally conditional) → band-pass → pre-emphasis → trim silence →
    drop-if-too-short → loudness-normalize. Band-passing before denoise would
    starve noisereduce of the bands it needs to model the noise floor, so denoise
    runs first.

    Conditional denoise: when ``denoise_only_if_noisy`` is True, denoise runs only
    if the segment's estimated dynamic range is below ``denoise_snr_threshold_db``.
    Denoising already-clean speech adds artifacts, so skipping it on clean clips is
    safer.

    Args:
        y: Mono audio in [-1, 1].
        sr: Sample rate of ``y``.
        apply_denoise: Master switch for spectral denoising.
        denoise_stationary: Use the stationary noise model instead of adaptive.
        denoise_only_if_noisy: Gate denoise on measured dynamic range.
        denoise_snr_threshold_db: Threshold consulted only when the gate is on.
        apply_bandpass_filter: Apply the conservative speech band-pass.
        apply_preemphasis_filter: Apply speech pre-emphasis.
        target_lufs: Loudness target for the final normalization.
        min_duration_s: If > 0 and the trimmed audio is shorter than this, return
            an empty array (the caller treats that as "drop this segment").

    Returns:
        Cleaned, normalized mono audio — or an empty array if it fell below
        ``min_duration_s`` after trimming.
    """
    if apply_denoise:
        do_denoise = True
        if denoise_only_if_noisy:
            dynamic_range = _estimate_dynamic_range_db(y)
            do_denoise = dynamic_range < denoise_snr_threshold_db
            if not do_denoise:
                logger.debug(
                    "Skipping denoise: dynamic range {:.1f}dB >= threshold {:.1f}dB",
                    dynamic_range,
                    denoise_snr_threshold_db,
                )
        if do_denoise:
            y = denoise(y, sr, stationary=denoise_stationary)
    if apply_bandpass_filter:
        y = apply_bandpass(y, sr)
    if apply_preemphasis_filter:
        y = apply_preemphasis(y)

    y = trim_silence(y)
    if min_duration_s > 0.0 and y.size < int(sr * min_duration_s):
        return np.asarray([], dtype=np.float32)

    return loudness_normalize(y, sr, target_lufs=target_lufs)


def preprocess_full(
    path: Path,
    target_sr: int = XTTS_INPUT_SR,
    apply_denoise: bool = True,
    target_lufs: float = -23.0,
    denoise_stationary: bool = False,
    apply_bandpass_filter: bool = False,
    apply_preemphasis_filter: bool = False,
) -> tuple[np.ndarray, AudioStats]:
    """One-shot preprocessing for a single file.

    Loads → denoises → trims silence → loudness-normalizes → returns clean audio
    plus before/after stats. This is what you call when you have a single clean
    reference recording and just want it cleaned up.

    Args:
        path: Path to input file.
        target_sr: Target sample rate.
        apply_denoise: Whether to apply spectral denoising.
        target_lufs: Target loudness.
        denoise_stationary: Use stationary denoise instead of adaptive non-stationary mode.
        apply_bandpass_filter: Apply conservative speech band-pass cleanup.
        apply_preemphasis_filter: Apply speech pre-emphasis before denoise.

    Returns:
        Tuple of (clean_audio, stats_after_cleanup).
    """
    y, original_sr = load_audio_mono(path, target_sr=target_sr)
    # Delegate to the canonical chain so this one-shot path and the corpus
    # extraction path can never use different cleaning logic or order.
    y = clean_segment(
        y,
        target_sr,
        apply_denoise=apply_denoise,
        denoise_stationary=denoise_stationary,
        apply_bandpass_filter=apply_bandpass_filter,
        apply_preemphasis_filter=apply_preemphasis_filter,
        target_lufs=target_lufs,
    )
    # Use the ORIGINAL sample rate so quality gates can detect phone-codec
    # sources. compute_stats records sample_rate verbatim; downstream gates
    # (e.g. score_segment) check `stats.sample_rate < MIN_SAMPLING_RATE_HZ`.
    stats = compute_stats(y, target_sr, original_sr=original_sr)
    return y, stats


# ─── Container conversion (ffmpeg bridge) ──────────────────────────
def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is in PATH. Used by pre-flight checks."""
    return shutil.which("ffmpeg") is not None


def convert_to_wav(
    src: Path,
    dst: Path | None = None,
    target_sr: int = XTTS_INPUT_SR,
    overwrite: bool = False,
) -> Path:
    """Transcode a container (mp4/mkv/m4a/...) to mono WAV at target_sr.

    soundfile and librosa cannot decode many container formats directly
    (ITU container ≠ codec); ffmpeg is the standard bridge. This used to
    live as a hand-edited cell in the Colab notebook (Cell 17). Moved
    here so the pipeline, the CLI, and the notebook all share one
    implementation.

    Args:
        src: Source file (any format ffmpeg can read).
        dst: Output WAV path. Defaults to `src.with_suffix(".wav")`.
        target_sr: Output sample rate. XTTS-v2's 22050 by default.
        overwrite: If False and `dst` exists, skip and return its path.
            If True, force re-encode.

    Returns:
        Path to the produced (or already-existing) WAV file.

    Raises:
        FileNotFoundError: If `src` does not exist or ffmpeg is not on PATH.
        RuntimeError: If ffmpeg fails to decode the input.
    """
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Source audio not found: {src}")
    if not _ffmpeg_available():
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install it before running this pipeline. "
            "On Colab: !apt -qq install ffmpeg. Locally: apt/brew/choco install ffmpeg."
        )

    dst = Path(dst) if dst is not None else src.with_suffix(".wav")
    if dst.exists() and not overwrite:
        logger.info("Already exists, skipping conversion: {}", dst.name)
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(src),
            "-ar",
            str(target_sr),
            "-ac",
            "1",  # mono
            "-y",  # overwrite (we already gated above)
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Keep only the tail of stderr — ffmpeg dumps a lot of unrelated info.
        tail = result.stderr[-800:]
        raise RuntimeError(f"ffmpeg failed for {src.name}: {tail}")

    size_mb = dst.stat().st_size / 1e6
    logger.info("Converted: {} → {} ({:.1f} MB)", src.name, dst.name, size_mb)
    return dst


def convert_directory_to_wav(
    directory: Path,
    target_sr: int = XTTS_INPUT_SR,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every non-WAV file in `directory` to mono WAV at target_sr.

    Iterates the top level of `directory` (NOT recursive), filtering by
    `EXTENSIONS_CONVERTIBLE_TO_WAV`. Each file is converted in place to
    `<stem>.wav` next to the original. Original files are NOT deleted.

    Args:
        directory: Folder containing source media files.
        target_sr: Target sample rate for the WAV outputs.
        overwrite: If False, skip files whose .wav twin already exists.

    Returns:
        List of paths to the produced (or pre-existing) WAV files,
        in the order they were processed. Files that failed to convert
        are NOT included in the list — failures are logged but the
        function continues, so one bad input doesn't abort a batch.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    candidates = sorted(
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS_CONVERTIBLE_TO_WAV
    )
    if not candidates:
        logger.info("No convertible files found in {}", directory)
        return []

    produced: list[Path] = []
    for src in candidates:
        try:
            wav = convert_to_wav(src, target_sr=target_sr, overwrite=overwrite)
            produced.append(wav)
        except (FileNotFoundError, RuntimeError) as exc:
            # Log but don't abort the batch: one bad file shouldn't kill the run.
            logger.warning("Conversion skipped for {}: {}", src.name, exc)

    logger.info("Converted {}/{} files in {}", len(produced), len(candidates), directory)
    return produced
