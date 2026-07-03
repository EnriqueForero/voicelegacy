"""voicelegacy — Build a reference corpus from speakerscribe outputs.

speakerscribe emits a structured JSON with diarized segments. This module
walks the JSON, extracts segments belonging to the target speaker, slices
them from the original audio, cleans them up, and writes them to disk as
WAVs ready for XTTS-v2 conditioning.

Expected speakerscribe JSON shape (the relevant subset):
    {
      "source_audio": "interview_01.mp3",
      "language_detected": "es",
      "segments": [
        {"start": 12.34, "end": 18.91, "speaker": "SPEAKER_00", "text": "..."},
        ...
      ]
    }

Anything else in the JSON is ignored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from voicelegacy.audio import (
    clean_segment,
    load_audio_mono,
    save_wav,
    slice_segment,
)
from voicelegacy.config import XTTS_INPUT_SR, ReferenceConfig, WorkspacePaths
from voicelegacy.logging_config import get_logger
from voicelegacy.speakerscribe_schema import load_and_validate_speakerscribe_document

logger = get_logger()


@dataclass(frozen=True)
class SegmentRef:
    """One diarized segment from speakerscribe, after light validation."""

    source_audio: Path
    start_s: float
    end_s: float
    speaker: str
    text: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class F0OutlierResult:
    """Pitch-based diarization sanity check for one segment.

    Attributes:
        segment: Original diarized segment.
        median_f0_hz: Median voiced fundamental frequency in Hz. ``None`` means
            the estimator could not find enough voiced frames.
        robust_z: Robust z-score around the speaker median using MAD.
        is_outlier: Whether the segment should be dropped before extraction.
    """

    segment: SegmentRef
    median_f0_hz: float | None
    robust_z: float | None
    is_outlier: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict for audit reports."""
        return {
            "source_audio": str(self.segment.source_audio),
            "start_s": self.segment.start_s,
            "end_s": self.segment.end_s,
            "speaker": self.segment.speaker,
            "text": self.segment.text,
            "duration_s": round(self.segment.duration_s, 3),
            "median_f0_hz": None if self.median_f0_hz is None else round(self.median_f0_hz, 2),
            "robust_z": None if self.robust_z is None else round(self.robust_z, 3),
            "is_outlier": self.is_outlier,
        }


# ─── JSON parsing ──────────────────────────────────────────────────
def load_speakerscribe_json(json_path: Path, audio_root: Path | None = None) -> list[SegmentRef]:
    """Parse a speakerscribe .json and return diarized segments.

    Args:
        json_path: Path to a `.json` file produced by speakerscribe.
        audio_root: Directory where the source audio lives. If None, defaults
            to the same directory as json_path's parent's parent (matching
            the speakerscribe workspace convention).

    Returns:
        List of SegmentRef objects. May be empty if no segments parse.

    Raises:
        FileNotFoundError: If json_path or referenced source audio is missing.
        ValueError: If JSON is malformed.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"speakerscribe JSON not found: {json_path}")

    document = load_and_validate_speakerscribe_document(json_path)

    source_name = document.source_name or json_path.stem
    if audio_root is None:
        # speakerscribe convention: transcripts/<file>.json → data/<file>
        audio_root = json_path.parent.parent / "data"

    source_audio = Path(audio_root) / source_name
    # If extension was stripped or differs from the actual file on disk, try
    # the formats the pipeline can handle. .wav first (canonical), then the
    # set ffmpeg can transcode (see audio.EXTENSIONS_CONVERTIBLE_TO_WAV).
    if not source_audio.exists():
        for ext in (
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".mp4",
            ".mkv",
            ".webm",
            ".aac",
            ".mov",
        ):
            candidate = source_audio.with_suffix(ext)
            if candidate.exists():
                source_audio = candidate
                break

    out: list[SegmentRef] = [
        SegmentRef(
            source_audio=source_audio,
            start_s=segment.start,
            end_s=segment.end,
            speaker=segment.speaker,
            text=segment.text,
        )
        for segment in document.segments
    ]

    logger.info("Parsed {} segments from {}", len(out), json_path.name)
    return out


# ─── Filtering ─────────────────────────────────────────────────────
def filter_segments(
    segments: list[SegmentRef],
    target_speaker: str,
    min_duration_s: float,
    max_duration_s: float,
) -> list[SegmentRef]:
    """Filter to segments of the target speaker within the duration window.

    Args:
        segments: All segments from one or more speakerscribe JSONs.
        target_speaker: Speaker label to keep (e.g. 'SPEAKER_00').
        min_duration_s: Minimum segment length in seconds.
        max_duration_s: Maximum segment length in seconds.

    Returns:
        Filtered list, sorted by source then by start time.
    """
    out = [
        s
        for s in segments
        if s.speaker == target_speaker and min_duration_s <= s.duration_s <= max_duration_s
    ]
    out.sort(key=lambda s: (str(s.source_audio), s.start_s))
    logger.info(
        "Kept {}/{} segments for speaker '{}' (duration {}-{}s)",
        len(out),
        len(segments),
        target_speaker,
        min_duration_s,
        max_duration_s,
    )
    return out


# ─── F0 outlier detection ──────────────────────────────────────────
def estimate_median_f0_hz(
    y: np.ndarray,
    sr: int,
    *,
    fmin_hz: float = 50.0,
    fmax_hz: float = 500.0,
) -> float | None:
    """Estimate median voiced F0 for a speech segment.

    The result is intentionally conservative: it returns ``None`` when too few
    voiced frames are detected. We use this only as a diarization sanity check,
    not as a biometric classifier.
    """
    if y.size < int(sr * 0.5):
        return None
    try:
        import librosa

        f0 = librosa.yin(
            y.astype(np.float32, copy=False),
            fmin=fmin_hz,
            fmax=fmax_hz,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except Exception as exc:
        logger.warning("F0 estimation failed: {}", exc)
        return None

    valid = f0[np.isfinite(f0)]
    valid = valid[(valid >= fmin_hz) & (valid <= fmax_hz)]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def detect_f0_outliers_from_values(
    segments: list[SegmentRef],
    median_f0_values: list[float | None],
    *,
    min_valid: int = 5,
    mad_threshold: float = 3.5,
) -> list[F0OutlierResult]:
    """Flag robust F0 outliers from pre-computed segment medians.

    This pure helper is easy to test. Runtime extraction uses
    :func:`analyze_f0_outliers`, which first estimates the medians from audio.
    """
    if len(segments) != len(median_f0_values):
        raise ValueError("segments and median_f0_values must have the same length")

    valid_values = np.asarray([v for v in median_f0_values if v is not None], dtype=float)
    if valid_values.size < min_valid:
        return [
            F0OutlierResult(segment=s, median_f0_hz=v, robust_z=None, is_outlier=False)
            for s, v in zip(segments, median_f0_values, strict=True)
        ]

    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    if mad <= 1e-9:
        return [
            F0OutlierResult(segment=s, median_f0_hz=v, robust_z=0.0, is_outlier=False)
            for s, v in zip(segments, median_f0_values, strict=True)
        ]

    results: list[F0OutlierResult] = []
    for seg, value in zip(segments, median_f0_values, strict=True):
        if value is None:
            results.append(F0OutlierResult(seg, None, None, False))
            continue
        robust_z = abs(0.6745 * (float(value) - median) / mad)
        results.append(
            F0OutlierResult(
                segment=seg,
                median_f0_hz=float(value),
                robust_z=float(robust_z),
                is_outlier=bool(robust_z > mad_threshold),
            )
        )
    return results


def analyze_f0_outliers(
    segments: list[SegmentRef],
    config: ReferenceConfig,
    target_sr: int = XTTS_INPUT_SR,
) -> list[F0OutlierResult]:
    """Analyze filtered target-speaker segments for pitch outliers.

    A common speakerscribe failure mode is assigning another speaker's phrase to
    the target label. That contaminated reference then pollutes XTTS conditioning.
    This check estimates median F0 per segment and drops robust outliers only
    when enough valid measurements exist.
    """
    if not segments or not config.enable_f0_outlier_filter:
        return [F0OutlierResult(s, None, None, False) for s in segments]

    cache: dict[Path, np.ndarray] = {}
    f0_values: list[float | None] = []
    for seg in segments:
        if seg.source_audio not in cache:
            try:
                y_src, _ = load_audio_mono(seg.source_audio, target_sr=target_sr)
                cache[seg.source_audio] = y_src
            except Exception as exc:
                logger.warning("F0 preflight could not load {}: {}", seg.source_audio.name, exc)
                f0_values.append(None)
                continue
        y_seg = slice_segment(cache[seg.source_audio], target_sr, seg.start_s, seg.end_s, pad_s=0.0)
        f0_values.append(
            estimate_median_f0_hz(
                y_seg,
                target_sr,
                fmin_hz=config.f0_min_hz,
                fmax_hz=config.f0_max_hz,
            )
        )

    results = detect_f0_outliers_from_values(
        segments,
        f0_values,
        min_valid=config.min_segments_for_f0_filter,
        mad_threshold=config.f0_outlier_mad_threshold,
    )
    outliers = [r for r in results if r.is_outlier]
    if outliers:
        logger.warning(
            "F0 outlier filter rejected {}/{} target-speaker segment(s).",
            len(outliers),
            len(results),
        )
    else:
        logger.info("F0 outlier filter found no target-speaker pitch outliers.")
    return results


def filter_f0_outliers(
    segments: list[SegmentRef],
    config: ReferenceConfig,
    target_sr: int = XTTS_INPUT_SR,
) -> list[SegmentRef]:
    """Return segments after optional F0 outlier rejection."""
    if not config.enable_f0_outlier_filter:
        return segments
    results = analyze_f0_outliers(segments, config, target_sr=target_sr)
    return [r.segment for r in results if not r.is_outlier]


def write_f0_outlier_report(
    report_path: Path,
    results: list[F0OutlierResult],
    config: ReferenceConfig,
) -> None:
    """Persist the pitch-outlier audit report for the reference phase."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rejected = [r for r in results if r.is_outlier]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "enabled": config.enable_f0_outlier_filter,
        "summary": {
            "total_segments_checked": len(results),
            "valid_f0_measurements": sum(1 for r in results if r.median_f0_hz is not None),
            "rejected_as_f0_outliers": len(rejected),
            "mad_threshold": config.f0_outlier_mad_threshold,
            "min_segments_for_filter": config.min_segments_for_f0_filter,
        },
        "rejected_segments": [r.to_dict() for r in rejected],
        "all_segments": [r.to_dict() for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("F0 outlier report → {}", report_path)


# ─── Speaker-overlap (crosstalk) gate ──────────────────────────────
@dataclass(frozen=True)
class OverlapResult:
    """How much a target segment is overlapped by other speakers.

    Attributes:
        segment: The target-speaker segment.
        overlap_s: Seconds of this segment covered by other speakers (union, so
            overlapping interlopers are not double-counted).
        overlap_ratio: ``overlap_s / segment.duration_s``.
        is_overlapping: Whether it exceeded the configured ratio and is dropped.
    """

    segment: SegmentRef
    overlap_s: float
    overlap_ratio: float
    is_overlapping: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict for audit reports."""
        return {
            "source_audio": str(self.segment.source_audio),
            "start_s": self.segment.start_s,
            "end_s": self.segment.end_s,
            "speaker": self.segment.speaker,
            "duration_s": round(self.segment.duration_s, 3),
            "overlap_s": round(self.overlap_s, 3),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "is_overlapping": self.is_overlapping,
        }


def compute_overlap_seconds(target: SegmentRef, others: Iterable[SegmentRef]) -> float:
    """Seconds of ``target`` covered by ``others`` (union of intersections).

    Overlapping interlopers are merged before summing, so two other speakers
    talking over the same instant count once, not twice.
    """
    start, end = target.start_s, target.end_s
    if end <= start:
        return 0.0
    intervals: list[tuple[float, float]] = []
    for other in others:
        lo = max(start, other.start_s)
        hi = min(end, other.end_s)
        if hi > lo:
            intervals.append((lo, hi))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    total += cur_hi - cur_lo
    return total


def analyze_overlap(
    target_segments: list[SegmentRef],
    all_segments: list[SegmentRef],
    max_overlap_ratio: float,
) -> list[OverlapResult]:
    """Measure crosstalk for each target segment against other speakers.

    Only segments from the SAME source audio and a DIFFERENT speaker count as
    interlopers — overlap across different recordings is meaningless.
    """
    by_source: dict[Path, list[SegmentRef]] = defaultdict(list)
    for seg in all_segments:
        by_source[seg.source_audio].append(seg)

    results: list[OverlapResult] = []
    for target in target_segments:
        others = [o for o in by_source.get(target.source_audio, []) if o.speaker != target.speaker]
        overlap_s = compute_overlap_seconds(target, others)
        ratio = overlap_s / target.duration_s if target.duration_s > 0 else 0.0
        results.append(OverlapResult(target, overlap_s, ratio, ratio > max_overlap_ratio))
    return results


def filter_overlapping_segments(
    target_segments: list[SegmentRef],
    all_segments: list[SegmentRef],
    config: ReferenceConfig,
) -> list[SegmentRef]:
    """Drop target segments whose crosstalk exceeds ``config.max_overlap_ratio``.

    No-op unless ``config.enable_overlap_filter`` is set (opt-in, like the F0
    filter), because it changes which references are selected.
    """
    if not config.enable_overlap_filter:
        return target_segments
    results = analyze_overlap(target_segments, all_segments, config.max_overlap_ratio)
    rejected = [r for r in results if r.is_overlapping]
    if rejected:
        logger.warning(
            "Overlap filter rejected {}/{} target-speaker segment(s) as crosstalk.",
            len(rejected),
            len(results),
        )
    else:
        logger.info("Overlap filter found no crosstalk-contaminated segments.")
    return [r.segment for r in results if not r.is_overlapping]


def write_overlap_report(
    report_path: Path,
    results: list[OverlapResult],
    config: ReferenceConfig,
) -> None:
    """Persist the crosstalk audit report for the reference phase."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rejected = [r for r in results if r.is_overlapping]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "enabled": config.enable_overlap_filter,
        "summary": {
            "total_segments_checked": len(results),
            "rejected_as_crosstalk": len(rejected),
            "max_overlap_ratio": config.max_overlap_ratio,
        },
        "rejected_segments": [r.to_dict() for r in rejected],
        "all_segments": [r.to_dict() for r in results],
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Overlap report → {}", report_path)


# ─── Extraction ────────────────────────────────────────────────────
def extract_segments_to_wav(
    segments: list[SegmentRef],
    out_dir: Path,
    config: ReferenceConfig,
    target_sr: int = XTTS_INPUT_SR,
) -> list[Path]:
    """Slice each segment from its source audio and write a clean WAV.

    Caches the loaded source audio across segments from the same file —
    interviews are typically long single-file recordings, so this avoids
    redundant decoding.

    Args:
        segments: Filtered list of SegmentRef objects.
        out_dir: Directory to write extracted WAVs to.
        config: ReferenceConfig controlling cleanup parameters.
        target_sr: Sample rate to load and save at.

    Returns:
        List of paths to written WAV files. Same order as input segments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, np.ndarray] = {}
    written: list[Path] = []

    for idx, seg in enumerate(segments):
        # Load (and cache) the source audio
        if seg.source_audio not in cache:
            try:
                # Source-audio caching: only the resampled array is cached.
                # Original sample rate is not needed here because the generated
                # reference WAVs are always written at target_sr; the
                # phone-codec gate is enforced when those WAVs are later
                # evaluated by quality.evaluate_file on the SOURCE files.
                y_src, _src_sr = load_audio_mono(seg.source_audio, target_sr=target_sr)
                cache[seg.source_audio] = y_src
                logger.info(
                    "Loaded source: {} ({:,} samples)",
                    seg.source_audio.name,
                    len(cache[seg.source_audio]),
                )
            except FileNotFoundError:
                logger.warning("Source audio missing, skipping segment: {}", seg.source_audio)
                continue
            except Exception as exc:
                logger.error("Failed to load {}: {}", seg.source_audio.name, exc)
                continue

        full_y = cache[seg.source_audio]
        try:
            y_seg = slice_segment(full_y, target_sr, seg.start_s, seg.end_s, pad_s=0.05)
        except ValueError as exc:
            logger.warning("Bad segment bounds, skipping: {}", exc)
            continue

        # Canonical cleaning chain (single source of truth in audio.clean_segment):
        # denoise (optionally conditional on SNR) → band-pass → pre-emphasis →
        # trim → drop-if-<1s → loudness-normalize.
        y_seg = clean_segment(
            y_seg,
            target_sr,
            apply_denoise=config.apply_denoise,
            denoise_stationary=config.denoise_stationary,
            denoise_only_if_noisy=config.denoise_only_if_noisy,
            denoise_snr_threshold_db=config.denoise_snr_threshold_db,
            apply_bandpass_filter=config.apply_bandpass_filter,
            apply_preemphasis_filter=config.apply_preemphasis_filter,
            target_lufs=config.target_loudness_lufs,
            min_duration_s=1.0,
        )
        if y_seg.size == 0:  # fell below 1s after trimming
            logger.warning("Segment too short after trim, skipping (idx={})", idx)
            continue

        out_path = out_dir / f"{seg.source_audio.stem}_{idx:04d}_{seg.start_s:08.2f}.wav"
        save_wav(out_path, y_seg, target_sr)
        # Write a transcript sidecar next to the WAV. This makes WAV<->text
        # pairing robust: downstream (fine-tuning dataset prep) reads the
        # sidecar instead of parsing the filename, which previously caused a
        # silent mismatch when the naming scheme drifted. Empty text => no
        # sidecar (segment had no transcription).
        if seg.text and seg.text.strip():
            out_path.with_suffix(".txt").write_text(seg.text.strip(), encoding="utf-8")
        written.append(out_path)

    logger.info("Wrote {} reference WAVs to {}", len(written), out_dir)
    return written


# ─── End-to-end ────────────────────────────────────────────────────
def build_reference_corpus(
    paths: WorkspacePaths,
    config: ReferenceConfig,
) -> list[Path]:
    """End-to-end: walk speakerscribe JSONs → write reference WAVs.

    Args:
        paths: Workspace paths object.
        config: ReferenceConfig.

    Returns:
        List of paths to all written reference WAVs.
    """
    paths.mkdirs()
    json_files = sorted(paths.speakerscribe_out.glob("*.json"))
    if not json_files:
        logger.warning(
            "No speakerscribe JSONs found in {}. Run speakerscribe first.",
            paths.speakerscribe_out,
        )
        return []

    all_segments: list[SegmentRef] = []
    for jf in json_files:
        all_segments.extend(load_speakerscribe_json(jf, audio_root=paths.interviews_raw))

    filtered = filter_segments(
        all_segments,
        target_speaker=config.target_speaker_label,
        min_duration_s=config.min_segment_duration_s,
        max_duration_s=config.max_segment_duration_s,
    )
    if config.enable_overlap_filter:
        overlap_results = analyze_overlap(filtered, all_segments, config.max_overlap_ratio)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_overlap_report(paths.reports / f"overlap_{ts}.json", overlap_results, config)
        filtered = [r.segment for r in overlap_results if not r.is_overlapping]
    if config.enable_f0_outlier_filter:
        f0_results = analyze_f0_outliers(filtered, config)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_f0_outlier_report(paths.reports / f"f0_outliers_{ts}.json", f0_results, config)
        filtered = [r.segment for r in f0_results if not r.is_outlier]

    return extract_segments_to_wav(
        filtered,
        out_dir=paths.reference_corpus,
        config=config,
    )
