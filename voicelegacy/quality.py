"""voicelegacy — Quality scoring for reference-audio candidates.

The single most important driver of clone quality is the reference audio.
This module scores candidate segments so we can keep the top N and discard
the rest, instead of feeding XTTS-v2 noisy or short fragments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicelegacy.audio import AudioStats, compute_stats, load_audio_mono
from voicelegacy.config import (
    MAX_REF_DURATION_S,
    MIN_REF_DURATION_S,
    MIN_SAMPLING_RATE_HZ,
    MIN_SNR_DB,
    XTTS_INPUT_SR,
    ReferenceConfig,
)
from voicelegacy.logging_config import get_logger

logger = get_logger()


@dataclass(frozen=True)
class QualityReport:
    """Result of evaluating a candidate reference audio segment."""

    path: Path
    stats: AudioStats
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict for reporting."""
        return {
            "path": str(self.path),
            "duration_s": round(self.stats.duration_s, 3),
            "sample_rate": self.stats.sample_rate,
            "rms_db": round(self.stats.rms_db, 2),
            "peak_db": round(self.stats.peak_db, 2),
            "snr_db": round(self.stats.snr_db, 2),
            "spectral_rolloff_hz": round(self.stats.spectral_rolloff_hz, 1),
            "score": round(self.score, 4),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def score_segment(
    stats: AudioStats,
    min_duration_s: float = MIN_REF_DURATION_S,
    max_duration_s: float = MAX_REF_DURATION_S,
    min_snr_db: float = MIN_SNR_DB,
    min_sr_hz: int = MIN_SAMPLING_RATE_HZ,
    min_spectral_rolloff_hz: float = 0.0,
) -> tuple[float, bool, tuple[str, ...]]:
    """Score an audio segment against quality gates.

    Scoring rubric (higher = better):
        score = snr_normalized * duration_factor

    Where:
        - snr_normalized in [0, 1] from min_snr_db (0) to 40 dB (1).
        - duration_factor in [0, 1] peaks at the midpoint of the allowed range
          and falls off symmetrically outside it.

    A segment is considered passing if it clears all hard gates:
        - sample_rate >= min_sr_hz
        - min_duration_s <= duration <= max_duration_s
        - snr_db >= min_snr_db
        - peak_db not clipping (> -1 dB FS)

    Args:
        stats: AudioStats from compute_stats.
        min_duration_s: Hard floor on segment length, seconds.
        max_duration_s: Hard ceiling on segment length, seconds.
        min_snr_db: Hard floor for cleanliness in dB. Defaults to the
            package-wide constant ``MIN_SNR_DB``; if you call this from a
            pipeline with a ``ReferenceConfig`` you should pass
            ``config.min_snr_db`` instead so there is a single source of truth.
        min_sr_hz: Hard floor — reject phone-codec audio.

    Returns:
        Tuple (score, passed, list_of_failure_reasons).
    """
    reasons: list[str] = []

    # Hard gates
    if stats.sample_rate < min_sr_hz:
        reasons.append(f"sample_rate {stats.sample_rate}Hz < {min_sr_hz}Hz (phone-codec audio)")
    if min_spectral_rolloff_hz > 0.0 and stats.spectral_rolloff_hz < min_spectral_rolloff_hz:
        # Catches telephone-band audio that was upsampled: the header sample
        # rate looks fine but the measured bandwidth is narrow.
        reasons.append(
            f"spectral rolloff {stats.spectral_rolloff_hz:.0f}Hz < "
            f"{min_spectral_rolloff_hz:.0f}Hz (narrowband/telephone audio)"
        )
    if stats.duration_s < min_duration_s:
        reasons.append(f"duration {stats.duration_s:.2f}s < {min_duration_s}s")
    if stats.duration_s > max_duration_s:
        reasons.append(f"duration {stats.duration_s:.2f}s > {max_duration_s}s")
    if stats.snr_db < min_snr_db:
        reasons.append(f"snr {stats.snr_db:.1f}dB < {min_snr_db}dB")
    if stats.peak_db > -1.0:
        reasons.append(f"clipping risk: peak {stats.peak_db:.1f}dBFS")

    passed = len(reasons) == 0

    # Soft score (always computed for ranking, even when failing)
    snr_norm = float(np.clip((stats.snr_db - min_snr_db) / (40.0 - min_snr_db), 0.0, 1.0))

    # Duration factor: triangle function peaking at the middle of the range
    mid = (min_duration_s + max_duration_s) / 2
    span = (max_duration_s - min_duration_s) / 2
    if span > 0:
        dur_factor = max(0.0, 1.0 - abs(stats.duration_s - mid) / span)
    else:
        dur_factor = 1.0

    score = snr_norm * dur_factor
    return float(score), passed, tuple(reasons)


def evaluate_file(
    path: Path,
    config: ReferenceConfig | None = None,
    *,
    min_duration_s: float | None = None,
    max_duration_s: float | None = None,
    min_snr_db: float | None = None,
    min_sr_hz: int = MIN_SAMPLING_RATE_HZ,
    min_spectral_rolloff_hz: float | None = None,
    target_sr: int = XTTS_INPUT_SR,
) -> QualityReport:
    """Load an audio file and produce a full quality report.

    Args:
        path: Path to audio file.
        config: Optional :class:`ReferenceConfig`. When provided, the gating
            thresholds (``min_segment_duration_s``, ``max_segment_duration_s``,
            ``min_snr_db``) are read from it — a single source of truth shared
            with the rest of the pipeline. Explicit keyword overrides still win,
            which keeps the function usable from ad-hoc scripts that pass only
            the thresholds they care about.
        min_duration_s: Optional override of ``config.min_segment_duration_s``.
            Defaults to :data:`MIN_REF_DURATION_S` when no config is given.
        max_duration_s: Optional override of ``config.max_segment_duration_s``.
            Defaults to :data:`MAX_REF_DURATION_S` when no config is given.
        min_snr_db: Optional override of ``config.min_snr_db``. Defaults to
            :data:`MIN_SNR_DB` when no config is given. There is NO literal
            default inside this function — that is intentional to avoid the
            three-source-of-truth bug the original code shipped with.
        min_sr_hz: Hard floor for sample rate (rejects phone audio).
        target_sr: Sample rate to load at (analysis happens at this rate).

    Returns:
        QualityReport with score, passed flag, and reasons.
    """
    path = Path(path)
    if config is not None:
        effective_min_duration = (
            min_duration_s if min_duration_s is not None else config.min_segment_duration_s
        )
        effective_max_duration = (
            max_duration_s if max_duration_s is not None else config.max_segment_duration_s
        )
        effective_min_snr = min_snr_db if min_snr_db is not None else config.min_snr_db
        effective_min_rolloff = (
            min_spectral_rolloff_hz
            if min_spectral_rolloff_hz is not None
            else config.min_spectral_rolloff_hz
        )
    else:
        effective_min_duration = (
            min_duration_s if min_duration_s is not None else MIN_REF_DURATION_S
        )
        effective_max_duration = (
            max_duration_s if max_duration_s is not None else MAX_REF_DURATION_S
        )
        effective_min_snr = min_snr_db if min_snr_db is not None else MIN_SNR_DB
        effective_min_rolloff = (
            min_spectral_rolloff_hz if min_spectral_rolloff_hz is not None else 0.0
        )

    y, original_sr = load_audio_mono(path, target_sr=target_sr)
    # Pass original_sr so AudioStats.sample_rate reflects the SOURCE rate, not
    # the resampled rate. This is what makes the phone-codec hard gate
    # (sample_rate < min_sr_hz) actually fire on 8 kHz input.
    stats = compute_stats(y, target_sr, original_sr=original_sr)
    score, passed, reasons = score_segment(
        stats,
        min_duration_s=effective_min_duration,
        max_duration_s=effective_max_duration,
        min_snr_db=effective_min_snr,
        min_sr_hz=min_sr_hz,
        min_spectral_rolloff_hz=effective_min_rolloff,
    )
    return QualityReport(path=path, stats=stats, score=score, passed=passed, reasons=reasons)


def rank_candidates(reports: list[QualityReport], top_n: int = 10) -> list[QualityReport]:
    """Return the top-N reports that passed, sorted by score descending.

    Args:
        reports: List of QualityReport objects from evaluate_file.
        top_n: Maximum number to return.

    Returns:
        Sorted list, length <= top_n, only including passing reports.
    """
    passing = [r for r in reports if r.passed]
    if not passing:
        logger.warning("No candidates passed quality gates. Returning empty list.")
        return []

    passing.sort(key=lambda r: r.score, reverse=True)
    return passing[:top_n]


def select_reference_wavs(
    corpus_dir: Path,
    config: ReferenceConfig | None = None,
    *,
    top_n: int | None = None,
) -> list[Path]:
    """Pick the reference WAVs to feed XTTS conditioning, best-quality first.

    Rationale: ``run_reference_phase`` writes *every* filtered segment to
    ``reference_corpus/`` and only records its top-N ranking in a JSON
    report. Until v0.7.1 the synthesis CLIs then globbed the directory
    alphabetically — i.e. chronologically by ``{source}_{idx}_{start}`` —
    so (a) segments that FAILED the SNR/clipping gate were still used for
    conditioning, and (b) XTTS's GPT-style latent, computed from the
    leading ``gpt_cond_len`` seconds, was dominated by whichever segments
    sorted first rather than the best ones. This helper closes that gap by
    re-scoring the directory at synthesis time (cheap: stats only, no GPU)
    and returning the ranked top-N.

    Args:
        corpus_dir: Directory containing candidate reference WAVs.
        config: Optional :class:`ReferenceConfig` supplying gate thresholds
            and the default ``top_n_segments``. Defaults are used when None.
        top_n: How many to keep. ``None`` → ``config.top_n_segments``;
            ``0`` → legacy behavior (ALL wavs, alphabetical), kept as an
            explicit escape hatch for A/B comparison against old outputs.

    Returns:
        Ranked list (best first), or the legacy alphabetical list when
        ``top_n=0`` or when *no* candidate passes the gates (fail-open with
        a loud warning: for a family archive, degraded synthesis beats a
        hard stop — the warning tells the user to fix the corpus).
    """
    cfg = config or ReferenceConfig()
    all_wavs = sorted(corpus_dir.glob("*.wav"))
    if not all_wavs:
        return []

    effective_top_n = cfg.top_n_segments if top_n is None else top_n
    if effective_top_n == 0:
        logger.warning(
            "top_n=0: using ALL {} reference WAVs in alphabetical order "
            "(legacy pre-0.7.1 behavior; quality ranking bypassed).",
            len(all_wavs),
        )
        return all_wavs

    # Per-file fault isolation: one corrupt/unreadable WAV in the directory
    # must not kill synthesis. Unreadable files are excluded from BOTH the
    # ranking and the fallback pool — feeding undecodable bytes to XTTS
    # conditioning is never the right outcome.
    reports = []
    loadable: list[Path] = []
    for wav in all_wavs:
        try:
            reports.append(evaluate_file(wav, config=cfg))
            loadable.append(wav)
        except Exception as exc:
            logger.warning("Skipping unreadable reference WAV {}: {}", wav.name, exc)
    if not loadable:
        logger.error(
            "None of the {} reference WAV(s) in {} could be decoded. Rebuild the corpus.",
            len(all_wavs),
            corpus_dir,
        )
        return []
    ranked = rank_candidates(reports, top_n=effective_top_n)
    if not ranked:
        logger.warning(
            "No reference WAV passed the quality gates (SNR >= {} dB, "
            "duration {}–{} s). Falling back to ALL {} candidates — expect "
            "degraded cloning; rebuild the corpus with better source audio.",
            cfg.min_snr_db,
            cfg.min_segment_duration_s,
            cfg.max_segment_duration_s,
            len(loadable),
        )
        return loadable

    selected = [r.path for r in ranked]
    logger.info(
        "Selected top {} of {} reference WAVs by quality score (best first: {}).",
        len(selected),
        len(all_wavs),
        selected[0].name,
    )
    return selected
