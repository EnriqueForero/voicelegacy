"""Tests for the audio module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voicelegacy.audio import (
    compute_stats,
    load_audio_mono,
    loudness_normalize,
    preprocess_full,
    save_wav,
    slice_segment,
    trim_silence,
)
from voicelegacy.config import XTTS_INPUT_SR


class TestLoad:
    def test_loads_synthetic_wav(self, synthetic_speech_wav: Path) -> None:
        y, original_sr = load_audio_mono(synthetic_speech_wav)
        assert y.dtype == np.float32
        assert y.ndim == 1
        assert np.max(np.abs(y)) <= 1.0
        assert len(y) > XTTS_INPUT_SR  # at least 1s
        # Fixture writes at XTTS_INPUT_SR, so original_sr must echo it.
        assert original_sr == XTTS_INPUT_SR

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio_mono(tmp_path / "nonexistent.wav")


class TestStats:
    def test_stats_for_known_signal(self, synthetic_speech_wav: Path) -> None:
        y, _ = load_audio_mono(synthetic_speech_wav)
        stats = compute_stats(y, XTTS_INPUT_SR)
        assert stats.sample_rate == XTTS_INPUT_SR
        assert 9.0 < stats.duration_s < 11.0
        assert stats.snr_db > 0  # signal stronger than noise


class TestSlice:
    def test_slice_in_bounds(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.linspace(-1.0, 1.0, sr * 10, dtype=np.float32)  # 10s
        out = slice_segment(y, sr, 2.0, 5.0)
        assert len(out) == sr * 3

    def test_invalid_bounds_raises(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.zeros(sr, dtype=np.float32)
        with pytest.raises(ValueError):
            slice_segment(y, sr, 0.5, 0.5)


class TestNormalize:
    def test_loudness_normalize_does_not_clip(self, synthetic_speech_wav: Path) -> None:
        y, _ = load_audio_mono(synthetic_speech_wav)
        y_norm = loudness_normalize(y, XTTS_INPUT_SR, target_lufs=-20.0)
        assert np.max(np.abs(y_norm)) <= 1.0


class TestTrim:
    def test_trim_silence_removes_leading_padding(self) -> None:
        sr = XTTS_INPUT_SR
        # 1s of silence, 1s of tone, 1s of silence
        silence = np.zeros(sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        y = np.concatenate([silence, tone, silence])
        y_trim = trim_silence(y, top_db=20.0)
        assert len(y_trim) < len(y)
        assert len(y_trim) >= int(sr * 0.5)  # tone should remain


class TestSaveWav:
    def test_roundtrip(self, tmp_path: Path) -> None:
        sr = XTTS_INPUT_SR
        y = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        out = tmp_path / "roundtrip.wav"
        save_wav(out, y, sr)
        assert out.exists()
        y2, _ = load_audio_mono(out)
        # PCM_16 quantization is fine; just confirm length and roughly same content
        assert len(y2) == len(y)
        assert np.corrcoef(y, y2)[0, 1] > 0.99


class TestPreprocessFull:
    def test_end_to_end(self, synthetic_speech_wav: Path) -> None:
        y, stats = preprocess_full(synthetic_speech_wav, apply_denoise=False)
        assert stats.duration_s > 0
        assert stats.sample_rate == XTTS_INPUT_SR
        assert np.max(np.abs(y)) <= 1.0


class TestLoudnessNoClipping:
    """Regression tests for the circular-clipping bug fixed in P0-4.

    Before the fix, pyloudnorm could push transient peaks above 1.0 when the
    target_lufs was loud relative to the source, and the subsequent
    np.clip(-1, 1) hard-clipped them to exactly 1.0. The quality gate then
    flagged the file for "clipping risk: peak 0.0 dBFS" — failure introduced
    by the pipeline itself.

    These tests assert that, regardless of input level, the output never
    sits at the ceiling exact: peak must be strictly below 1.0 with
    measurable headroom.
    """

    @staticmethod
    def _generate_aggressive_signal(sr: int, duration_s: float, peak_target: float) -> np.ndarray:
        """Build a speech-like signal with peaks near `peak_target` (in [-1,1])."""
        n = int(sr * duration_s)
        t = np.arange(n) / sr
        signal = (
            0.6 * np.sin(2 * np.pi * 220 * t)
            + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
        ).astype(np.float32)
        # Scale so max abs == peak_target
        signal = signal * (peak_target / (float(np.max(np.abs(signal))) + 1e-9))
        return signal

    def test_aggressive_input_does_not_clip(self) -> None:
        sr = XTTS_INPUT_SR
        y = self._generate_aggressive_signal(sr, duration_s=5.0, peak_target=0.95)
        y_norm = loudness_normalize(y, sr, target_lufs=-16.0, peak_ceiling_dbfs=-3.0)
        peak = float(np.max(np.abs(y_norm)))
        # -3 dBFS = 0.7079. Allow a tiny epsilon for float32 rounding.
        assert peak < 0.71, f"peak {peak:.4f} above -3 dBFS ceiling — limiter failed"
        # Hard-clip signature: peak landed exactly at 1.0. Must NEVER happen.
        assert peak < 0.999, "Hard clipping detected (peak ≈ 1.0)"

    def test_peak_ceiling_dbfs_validation(self) -> None:
        sr = XTTS_INPUT_SR
        y = self._generate_aggressive_signal(sr, duration_s=3.0, peak_target=0.5)
        with pytest.raises(ValueError, match="peak_ceiling_dbfs"):
            loudness_normalize(y, sr, target_lufs=-23.0, peak_ceiling_dbfs=0.0)

    def test_quiet_input_preserved(self) -> None:
        """When source LUFS == target, the limiter must not engage."""
        sr = XTTS_INPUT_SR
        # Quiet input — already around -23 LUFS, peaks well below ceiling
        y = self._generate_aggressive_signal(sr, duration_s=5.0, peak_target=0.10)
        y_norm = loudness_normalize(y, sr, target_lufs=-23.0, peak_ceiling_dbfs=-3.0)
        peak = float(np.max(np.abs(y_norm)))
        assert peak < 0.71  # below ceiling
        # Result is finite, audible, not silent
        assert peak > 1e-3


class TestP1AudioCleanup:
    """P1 cleanup filters for sub-optimal archival audio."""

    def test_preemphasis_changes_signal_and_preserves_shape(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        out = __import__("voicelegacy.audio", fromlist=["apply_preemphasis"]).apply_preemphasis(y)
        assert out.shape == y.shape
        assert out.dtype == np.float32
        assert not np.allclose(out, y)

    def test_bandpass_preserves_shape(self) -> None:
        from voicelegacy.audio import apply_bandpass

        sr = XTTS_INPUT_SR
        y = np.random.default_rng(42).standard_normal(sr).astype(np.float32) * 0.01
        out = apply_bandpass(y, sr)
        assert out.shape == y.shape
        assert out.dtype == np.float32

    def test_dynamic_range_alias_matches_estimator(self) -> None:
        from voicelegacy.audio import _estimate_dynamic_range_db, _estimate_snr_db

        y = np.concatenate(
            [
                np.zeros(2048, dtype=np.float32),
                np.ones(4096, dtype=np.float32) * 0.2,
            ]
        )
        assert _estimate_snr_db(y) == _estimate_dynamic_range_db(y)


# ─── Assembly primitives (Fase 3 long-form joins) ──────────────────
class TestAssemblyPrimitives:
    SR = 24000

    def test_silence_duration(self) -> None:
        from voicelegacy.audio import silence

        assert len(silence(100.0, self.SR)) == 2400
        assert len(silence(0.0, self.SR)) == 0
        assert silence(50.0, self.SR).dtype == np.float32

    def test_silence_negative_raises(self) -> None:
        from voicelegacy.audio import silence

        with pytest.raises(ValueError, match=">= 0"):
            silence(-1.0, self.SR)

    def test_crossfade_length_is_sum_minus_overlap(self) -> None:
        from voicelegacy.audio import equal_power_crossfade

        a = np.ones(1000, dtype=np.float32)
        b = np.ones(800, dtype=np.float32)
        fade_ms = 10.0
        overlap = round(fade_ms / 1000.0 * self.SR)
        out = equal_power_crossfade(a, b, self.SR, fade_ms)
        assert len(out) == len(a) + len(b) - overlap

    def test_equal_power_curve_bumps_for_identical_signals(self) -> None:
        # For two identical constant signals, an EQUAL-POWER crossfade peaks at
        # amp*sqrt(2) in the middle (a LINEAR crossfade would stay flat at amp).
        # This is what proves the curve is equal-power, not linear.
        from voicelegacy.audio import equal_power_crossfade

        amp = 0.5
        a = np.full(2000, amp, dtype=np.float32)
        b = np.full(2000, amp, dtype=np.float32)
        out = equal_power_crossfade(a, b, self.SR, 40.0)
        overlap = round(40.0 / 1000.0 * self.SR)
        midpoint = out[len(a) - overlap // 2]
        assert midpoint == pytest.approx(amp * np.sqrt(2), abs=0.02)

    def test_no_power_dip_on_uncorrelated_signals(self) -> None:
        # The point of equal-power: joining two UNCORRELATED clips should not dip
        # in power across the fade (a linear crossfade would dip ~3 dB).
        from voicelegacy.audio import equal_power_crossfade

        rng = np.random.default_rng(0)
        a = rng.standard_normal(48000).astype(np.float32)
        b = rng.standard_normal(48000).astype(np.float32)
        out = equal_power_crossfade(a, b, self.SR, 200.0)
        win = 480  # 20 ms windows
        rms = np.array(
            [np.sqrt(np.mean(out[i : i + win] ** 2)) for i in range(0, len(out) - win, win)]
        )
        # No window should fall far below the overall level (no big dip).
        assert rms.min() > 0.80 * np.median(rms)

    def test_crossfade_empty_or_short_falls_back_to_concat(self) -> None:
        from voicelegacy.audio import equal_power_crossfade

        a = np.ones(100, dtype=np.float32)
        empty = np.asarray([], dtype=np.float32)
        assert len(equal_power_crossfade(a, empty, self.SR, 10.0)) == 100
        assert len(equal_power_crossfade(empty, a, self.SR, 10.0)) == 100

    def test_concatenate_audio_ignores_empty(self) -> None:
        from voicelegacy.audio import concatenate_audio

        a = np.ones(10, dtype=np.float32)
        empty = np.asarray([], dtype=np.float32)
        assert len(concatenate_audio([a, empty, a])) == 20
        assert len(concatenate_audio([empty, empty])) == 0


# ─── Dynamic-range estimator regression (v0.7.1) ────────────────────
class TestDynamicRangeEstimator:
    """Guards the per-frame axis fix.

    The previous implementation framed with librosa (axis=0 → shape
    (n_frames, frame_length)) and then averaged over axis=0 — across
    frames — so every real recording scored ~0 dB, the ``snr >= 15 dB``
    quality gate rejected everything and conditional denoise always fired.
    """

    @staticmethod
    def _speechlike(noise: float, sr: int = 22050, dur: float = 8.0) -> np.ndarray:
        n = int(sr * dur)
        t = np.arange(n) / sr
        sig = 0.30 * np.sin(2 * np.pi * 220 * t)
        sig = sig * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))  # syllable envelope
        rng = np.random.default_rng(seed=42)
        return (sig + noise * rng.standard_normal(n)).astype(np.float32)

    def test_clean_speechlike_clears_the_snr_gate(self) -> None:
        from voicelegacy.audio import _estimate_dynamic_range_db
        from voicelegacy.config import MIN_SNR_DB

        dr = _estimate_dynamic_range_db(self._speechlike(noise=0.002))
        assert dr > MIN_SNR_DB, (
            f"clean speech-like signal scored {dr:.2f} dB — the axis bug is back"
        )

    def test_monotonic_with_noise_floor(self) -> None:
        from voicelegacy.audio import _estimate_dynamic_range_db

        clean = _estimate_dynamic_range_db(self._speechlike(noise=0.002))
        noisy = _estimate_dynamic_range_db(self._speechlike(noise=0.05))
        assert clean > noisy + 3.0  # clearly discriminates cleanliness
