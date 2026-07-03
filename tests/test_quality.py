"""Tests for the quality module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voicelegacy.audio import AudioStats
from voicelegacy.quality import evaluate_file, rank_candidates, score_segment


class TestScoreSegment:
    def test_perfect_segment_passes(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=35.0,
        )
        score, passed, reasons = score_segment(stats)
        assert passed is True
        assert reasons == ()
        assert score > 0

    def test_phone_audio_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=8000,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("phone-codec" in r or "sample_rate" in r for r in reasons)

    def test_too_short_rejected(self) -> None:
        stats = AudioStats(
            duration_s=2.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("duration" in r for r in reasons)

    def test_noisy_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=5.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("snr" in r.lower() for r in reasons)

    def test_clipping_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-15.0,
            peak_db=-0.5,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("clip" in r.lower() for r in reasons)


class TestRanking:
    def test_top_n_filters_failures(self) -> None:
        from voicelegacy.quality import QualityReport

        good = QualityReport(
            path=Path("/tmp/a.wav"),
            stats=AudioStats(10.0, 22050, 1, -20, -6, 30),
            score=0.9,
            passed=True,
            reasons=(),
        )
        bad = QualityReport(
            path=Path("/tmp/b.wav"),
            stats=AudioStats(2.0, 22050, 1, -20, -6, 30),
            score=0.1,
            passed=False,
            reasons=("too short",),
        )
        top = rank_candidates([good, bad], top_n=5)
        assert len(top) == 1
        assert top[0].path.name == "a.wav"

    def test_empty_when_nothing_passes(self) -> None:
        from voicelegacy.quality import QualityReport

        bad = QualityReport(
            path=Path("/tmp/x.wav"),
            stats=AudioStats(2.0, 22050, 1, -20, -6, 30),
            score=0.0,
            passed=False,
            reasons=("bad",),
        )
        assert rank_candidates([bad]) == []


class TestEvaluateFile:
    def test_returns_report(self, synthetic_speech_wav: Path) -> None:
        report = evaluate_file(synthetic_speech_wav, min_snr_db=0.0)
        assert report.path == synthetic_speech_wav
        assert report.stats.duration_s > 0
        # Synthetic signal has high SNR, ~10s long → should pass gates
        assert report.passed is True or len(report.reasons) <= 1

    def test_accepts_reference_config_as_single_source_of_truth(
        self, synthetic_speech_wav: Path
    ) -> None:
        """P2-20: evaluate_file should accept a ReferenceConfig directly so the
        thresholds are read from one place (the config), eliminating the
        three-source-of-truth problem (constant + Field default + literal)."""
        from voicelegacy.config import ReferenceConfig

        config = ReferenceConfig(
            target_speaker_label="SPEAKER_00",
            min_segment_duration_s=4.0,
            max_segment_duration_s=15.0,
            min_snr_db=0.0,
        )
        report = evaluate_file(synthetic_speech_wav, config=config)
        assert report.path == synthetic_speech_wav
        # Thresholds came from config, not from a hidden literal
        assert report.passed is True or len(report.reasons) <= 1

    def test_explicit_kwargs_override_config(self, synthetic_speech_wav: Path) -> None:
        """Explicit kwargs always win over ``config`` to keep ad-hoc scripts working."""
        from voicelegacy.config import ReferenceConfig

        config = ReferenceConfig(
            target_speaker_label="SPEAKER_00",
            min_snr_db=0.0,
        )
        # Override min_snr_db to an impossible-to-pass value; reasons must reflect it.
        report = evaluate_file(synthetic_speech_wav, config=config, min_snr_db=120.0)
        assert report.passed is False
        assert any("snr" in r.lower() for r in report.reasons)


class TestPhoneCodecGateEndToEnd:
    """End-to-end test for the phone-codec hard gate (P0-3 fix).

    The previous TestScoreSegment.test_phone_audio_rejected only fabricated
    an AudioStats(sample_rate=8000) and ran score_segment in isolation. That
    bypassed the actual bug: load_audio_mono resampled silently to 22050 Hz,
    AudioStats inherited 22050, and the gate `stats.sample_rate < 16000`
    never fired in real usage.

    This test exercises the full evaluate_file path with a REAL 8 kHz WAV
    on disk. If load_audio_mono ever forgets to propagate the original
    sample rate again, this test fails immediately.
    """

    @staticmethod
    def _write_speech_wav(path: Path, sr: int, duration_s: float = 8.0) -> Path:
        import soundfile as sf

        n = int(sr * duration_s)
        t = np.arange(n) / sr
        # Speech-like harmonics (220 Hz f0 + harmonics) — clean, high SNR
        y = (
            0.30 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 440 * t)
            + 0.07 * np.sin(2 * np.pi * 880 * t)
        )
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllable-like envelope
        y = (y * env).astype(np.float32)
        rng = np.random.default_rng(42)
        y += (0.005 * rng.standard_normal(n)).astype(np.float32)
        sf.write(str(path), y, sr, subtype="PCM_16")
        return path

    def test_real_8khz_wav_rejected_by_phone_codec_gate(self, tmp_path: Path) -> None:
        """A real 8 kHz file must fail evaluate_file with phone-codec reason.

        Pre-fix behavior: stats.sample_rate == 22050, passed == True,
        reasons == (). This is the regression we just closed.
        Post-fix behavior: stats.sample_rate == 8000, passed == False,
        reasons contains either 'phone-codec' or 'sample_rate'.
        """
        wav_path = self._write_speech_wav(tmp_path / "phone.wav", sr=8000, duration_s=8.0)
        report = evaluate_file(wav_path, min_snr_db=0.0)

        assert report.stats.sample_rate == 8000, (
            f"AudioStats.sample_rate must reflect the ORIGINAL rate (8000), "
            f"got {report.stats.sample_rate}. The phone-codec gate is dead."
        )
        assert report.passed is False, "Real 8 kHz audio must NOT pass quality gates"
        assert any("phone-codec" in r or "sample_rate" in r for r in report.reasons), (
            f"Expected phone-codec / sample_rate failure, got {report.reasons}"
        )

    def test_real_22050hz_wav_accepted(self, tmp_path: Path) -> None:
        """Control: a real 22050 Hz file must NOT be rejected by the phone gate."""
        wav_path = self._write_speech_wav(tmp_path / "ok.wav", sr=22050, duration_s=8.0)
        report = evaluate_file(wav_path, min_snr_db=0.0)
        assert report.stats.sample_rate == 22050
        # No reason should mention phone-codec or sample_rate
        assert not any("phone-codec" in r or "sample_rate" in r for r in report.reasons), (
            f"22050 Hz file mis-flagged as phone-codec: {report.reasons}"
        )

    def test_16khz_wav_at_threshold_accepted(self, tmp_path: Path) -> None:
        """The boundary case: 16 kHz is the MIN_SAMPLING_RATE_HZ — must pass."""
        wav_path = self._write_speech_wav(tmp_path / "border.wav", sr=16000, duration_s=8.0)
        report = evaluate_file(wav_path, min_snr_db=0.0)
        assert report.stats.sample_rate == 16000
        assert not any("phone-codec" in r or "sample_rate" in r for r in report.reasons)


# ─── select_reference_wavs (v0.7.1) ─────────────────────────────────
class TestSelectReferenceWavs:
    """Quality-ranked reference selection for synthesis-time conditioning."""

    @staticmethod
    def _speechlike_wav(path: Path, sr: int, duration_s: float, noise: float) -> Path:
        """Harmonic mix + syllable envelope + noise floor (controls SNR)."""
        n = int(sr * duration_s)
        t = np.arange(n) / sr
        sig = (
            0.30 * np.sin(2 * np.pi * 220 * t)
            + 0.15 * np.sin(2 * np.pi * 440 * t)
            + 0.07 * np.sin(2 * np.pi * 880 * t)
        )
        sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        rng = np.random.default_rng(seed=42)
        sig = sig + noise * rng.standard_normal(n)
        sf.write(str(path), sig.astype(np.float32), sr, subtype="PCM_16")
        return path

    def test_ranks_best_snr_first_and_caps_at_top_n(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Alphabetical order is deliberately the INVERSE of quality order:
        # a_ruidoso has the worst SNR, z_limpio the best. The old glob-based
        # selection would have put the noisy one first.
        self._speechlike_wav(corpus / "a_ruidoso.wav", 22050, 8.0, noise=0.02)
        self._speechlike_wav(corpus / "m_medio.wav", 22050, 8.0, noise=0.008)
        self._speechlike_wav(corpus / "z_limpio.wav", 22050, 8.0, noise=0.002)

        selected = select_reference_wavs(corpus, top_n=2)

        assert len(selected) == 2
        assert selected[0].name == "z_limpio.wav"  # best first → leads gpt_cond
        assert "a_ruidoso.wav" not in {p.name for p in selected}

    def test_top_n_zero_is_legacy_alphabetical_all(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        self._speechlike_wav(corpus / "b.wav", 22050, 8.0, noise=0.02)
        self._speechlike_wav(corpus / "a.wav", 22050, 8.0, noise=0.002)

        selected = select_reference_wavs(corpus, top_n=0)

        assert [p.name for p in selected] == ["a.wav", "b.wav"]

    def test_nothing_passing_falls_back_to_all_loadable(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        # Both under the 6 s duration gate → ranking is empty → fail-open.
        self._speechlike_wav(corpus / "corto1.wav", 22050, 1.0, noise=0.002)
        self._speechlike_wav(corpus / "corto2.wav", 22050, 1.0, noise=0.002)

        selected = select_reference_wavs(corpus, top_n=5)

        assert len(selected) == 2

    def test_unreadable_wavs_are_excluded_everywhere(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "basura.wav").write_bytes(b"not a wav at all")
        self._speechlike_wav(corpus / "bueno.wav", 22050, 8.0, noise=0.002)

        selected = select_reference_wavs(corpus, top_n=5)

        assert [p.name for p in selected] == ["bueno.wav"]

    def test_all_unreadable_returns_empty(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "basura.wav").write_bytes(b"nope")

        assert select_reference_wavs(corpus, top_n=5) == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from voicelegacy.quality import select_reference_wavs

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        assert select_reference_wavs(corpus) == []
