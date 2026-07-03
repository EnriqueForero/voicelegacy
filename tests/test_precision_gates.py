"""Tests for the Fase 2 precision levers.

Covers the three correctness additions:
  * the canonical ``audio.clean_segment`` chain, including conditional denoise;
  * the spectral-rolloff bandwidth gate in ``quality.score_segment`` — which
    catches telephone-band audio that was upsampled (and so sails past the
    header sample-rate gate);
  * the speaker-overlap (crosstalk) gate in ``corpus``.

All of these default to off, so they change nothing until explicitly enabled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import voicelegacy.audio as audio
from voicelegacy.audio import AudioStats, clean_segment, compute_stats
from voicelegacy.config import MIN_SAMPLING_RATE_HZ, ReferenceConfig
from voicelegacy.corpus import (
    OverlapResult,
    SegmentRef,
    analyze_overlap,
    compute_overlap_seconds,
    filter_overlapping_segments,
)
from voicelegacy.quality import score_segment

SR = 22050


def _sine(seconds: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _lowpassed_noise(seconds: float, cutoff_hz: float, sr: int = SR) -> np.ndarray:
    """White noise low-passed at cutoff — simulates narrowband (e.g. phone) audio."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(sr * seconds)).astype(np.float32)
    sos = butter(8, cutoff_hz / (sr / 2.0), btype="low", output="sos")
    y = sosfiltfilt(sos, x).astype(np.float32)
    return (0.3 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


# ─── Spectral rolloff + bandwidth gate ─────────────────────────────
class TestSpectralRolloffGate:
    def test_rolloff_is_lower_for_narrowband(self) -> None:
        narrow = compute_stats(_lowpassed_noise(2.0, 3400.0), SR)
        wide = compute_stats(_sine(2.0, 220.0) + 0.2 * _lowpassed_noise(2.0, 10000.0), SR)
        assert narrow.spectral_rolloff_hz < 4500.0
        assert wide.spectral_rolloff_hz > narrow.spectral_rolloff_hz

    def test_gate_off_by_default_passes_narrowband(self) -> None:
        stats = AudioStats(8.0, SR, 1, -20.0, -3.0, 35.0, spectral_rolloff_hz=3000.0)
        _score, passed, reasons = score_segment(stats)  # min_spectral_rolloff_hz defaults to 0
        assert passed
        assert not any("rolloff" in r for r in reasons)

    def test_upsampled_phone_audio_passes_sr_gate_but_fails_rolloff_gate(self) -> None:
        # The whole point: header sample_rate is 22050 (passes the phone-codec
        # gate), yet the measured bandwidth is telephone-grade.
        stats = AudioStats(8.0, SR, 1, -20.0, -3.0, 35.0, spectral_rolloff_hz=3200.0)
        assert stats.sample_rate >= MIN_SAMPLING_RATE_HZ  # sr gate would pass it
        _score, passed, reasons = score_segment(stats, min_spectral_rolloff_hz=6000.0)
        assert not passed
        assert any("rolloff" in r and "narrowband" in r for r in reasons)

    def test_wideband_passes_rolloff_gate(self) -> None:
        stats = AudioStats(8.0, SR, 1, -20.0, -3.0, 35.0, spectral_rolloff_hz=9000.0)
        _score, passed, reasons = score_segment(stats, min_spectral_rolloff_hz=6000.0)
        assert passed
        assert not any("rolloff" in r for r in reasons)


# ─── clean_segment (canonical chain + conditional denoise) ─────────
class TestCleanSegment:
    def test_returns_normalized_audio(self) -> None:
        out = clean_segment(_sine(2.0), SR, apply_denoise=False)
        assert out.dtype == np.float32
        assert out.size > 0
        assert float(np.max(np.abs(out))) <= 10 ** (-3.0 / 20.0) + 1e-3  # peak ceiling

    def test_drops_when_below_min_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(audio, "trim_silence", lambda y, **k: y[: int(SR * 0.3)])
        out = clean_segment(_sine(2.0), SR, apply_denoise=False, min_duration_s=1.0)
        assert out.size == 0

    def test_unconditional_denoise_calls_denoise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def _spy(y, sr, **k):
            calls.append(1)
            return y

        monkeypatch.setattr(audio, "denoise", _spy)
        clean_segment(_sine(2.0), SR, apply_denoise=True, denoise_only_if_noisy=False)
        assert calls == [1]

    def test_conditional_denoise_skips_clean_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        monkeypatch.setattr(audio, "denoise", lambda y, sr, **k: calls.append(1) or y)
        # Force a HIGH dynamic range (clean) → above threshold → no denoise.
        monkeypatch.setattr(audio, "_estimate_dynamic_range_db", lambda y, **k: 40.0)
        clean_segment(
            _sine(2.0),
            SR,
            apply_denoise=True,
            denoise_only_if_noisy=True,
            denoise_snr_threshold_db=25.0,
        )
        assert calls == []  # skipped because audio is clean

    def test_conditional_denoise_runs_on_noisy_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        monkeypatch.setattr(audio, "denoise", lambda y, sr, **k: calls.append(1) or y)
        # Force a LOW dynamic range (noisy) → below threshold → denoise.
        monkeypatch.setattr(audio, "_estimate_dynamic_range_db", lambda y, **k: 8.0)
        clean_segment(
            _sine(2.0),
            SR,
            apply_denoise=True,
            denoise_only_if_noisy=True,
            denoise_snr_threshold_db=25.0,
        )
        assert calls == [1]


# ─── Overlap (crosstalk) gate ──────────────────────────────────────
def _seg(start: float, end: float, speaker: str, src: str = "a.wav") -> SegmentRef:
    return SegmentRef(Path(src), start, end, speaker, "t")


class TestComputeOverlapSeconds:
    def test_no_overlap(self) -> None:
        assert compute_overlap_seconds(_seg(0, 10, "A"), [_seg(11, 15, "B")]) == 0.0

    def test_partial_overlap(self) -> None:
        assert compute_overlap_seconds(_seg(0, 10, "A"), [_seg(8, 12, "B")]) == pytest.approx(2.0)

    def test_full_containment(self) -> None:
        assert compute_overlap_seconds(_seg(0, 10, "A"), [_seg(0, 10, "B")]) == pytest.approx(10.0)

    def test_multiple_overlappers_are_merged_not_double_counted(self) -> None:
        # Two interlopers covering [2,5] and [4,8] → union [2,8] = 6s, not 7s.
        others = [_seg(2, 5, "B"), _seg(4, 8, "C")]
        assert compute_overlap_seconds(_seg(0, 10, "A"), others) == pytest.approx(6.0)

    def test_zero_duration_target(self) -> None:
        assert compute_overlap_seconds(_seg(5, 5, "A"), [_seg(0, 10, "B")]) == 0.0


class TestAnalyzeAndFilterOverlap:
    def test_other_speaker_same_source_counts(self) -> None:
        target = [_seg(0, 10, "A")]
        allsegs = [*target, _seg(0, 5, "B")]  # 5s overlap → ratio 0.5
        results = analyze_overlap(target, allsegs, max_overlap_ratio=0.15)
        assert results[0].overlap_ratio == pytest.approx(0.5)
        assert results[0].is_overlapping is True

    def test_same_speaker_does_not_count(self) -> None:
        target = [_seg(0, 10, "A")]
        allsegs = [*target, _seg(0, 5, "A")]  # same speaker → not crosstalk
        results = analyze_overlap(target, allsegs, max_overlap_ratio=0.15)
        assert results[0].overlap_s == 0.0

    def test_different_source_does_not_count(self) -> None:
        target = [_seg(0, 10, "A", src="a.wav")]
        allsegs = [*target, _seg(0, 5, "B", src="b.wav")]  # other recording
        results = analyze_overlap(target, allsegs, max_overlap_ratio=0.15)
        assert results[0].overlap_s == 0.0

    def test_filter_is_noop_when_disabled(self) -> None:
        cfg = ReferenceConfig(target_speaker_label="A", enable_overlap_filter=False)
        target = [_seg(0, 10, "A"), _seg(0, 10, "A")]
        allsegs = [*target, _seg(0, 9, "B")]
        assert filter_overlapping_segments(target, allsegs, cfg) == target

    def test_filter_drops_crosstalk_when_enabled(self) -> None:
        cfg = ReferenceConfig(
            target_speaker_label="A", enable_overlap_filter=True, max_overlap_ratio=0.15
        )
        clean = _seg(0, 10, "A")
        dirty = _seg(20, 30, "A")
        allsegs = [clean, dirty, _seg(21, 29, "B")]  # 8s over the dirty one
        kept = filter_overlapping_segments([clean, dirty], allsegs, cfg)
        assert clean in kept and dirty not in kept

    def test_overlap_result_to_dict(self) -> None:
        d = OverlapResult(_seg(0, 10, "A"), 5.0, 0.5, True).to_dict()
        assert d["overlap_s"] == 5.0 and d["is_overlapping"] is True
