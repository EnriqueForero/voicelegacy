"""Tests for the corpus module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from voicelegacy.corpus import filter_segments, load_speakerscribe_json


class TestLoadJSON:
    def test_parses_well_formed(self, speakerscribe_json_factory, tmp_path: Path) -> None:
        # Create the referenced source audio so the path resolves
        audio_dir = tmp_path.parent / "data"
        audio_dir.mkdir(exist_ok=True)
        (audio_dir / "source.wav").touch()

        jp = speakerscribe_json_factory(
            "interview01.json",
            segments=[
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hola"},
                {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01", "text": "Adios"},
            ],
        )
        segs = load_speakerscribe_json(jp, audio_root=tmp_path)
        assert len(segs) == 2
        assert segs[0].speaker == "SPEAKER_00"
        assert segs[0].duration_s == pytest.approx(5.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_speakerscribe_json(tmp_path / "ghost.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not really json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_speakerscribe_json(bad)

    def test_invalid_segment_schema_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "partial.json"
        payload = {
            "source_audio": "x.wav",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "ok"},
                {"start": "not-a-number", "end": 9.0, "speaker": "SPEAKER_00"},  # bad
            ],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid speakerscribe JSON schema"):
            load_speakerscribe_json(p, audio_root=tmp_path)


class TestFilter:
    def test_filters_by_speaker_and_duration(self, tmp_path: Path) -> None:
        from voicelegacy.corpus import SegmentRef

        audio = tmp_path / "x.wav"
        segs = [
            SegmentRef(audio, 0.0, 5.0, "SPEAKER_00", "a"),  # keep
            SegmentRef(audio, 5.0, 5.5, "SPEAKER_00", "b"),  # too short
            SegmentRef(audio, 10.0, 30.0, "SPEAKER_00", "c"),  # too long
            SegmentRef(audio, 30.0, 35.0, "SPEAKER_01", "d"),  # wrong speaker
            SegmentRef(audio, 40.0, 47.0, "SPEAKER_00", "e"),  # keep
        ]
        out = filter_segments(
            segs,
            target_speaker="SPEAKER_00",
            min_duration_s=4.0,
            max_duration_s=15.0,
        )
        kept_texts = [s.text for s in out]
        assert kept_texts == ["a", "e"]


class TestF0OutlierDetection:
    def _segments(self, tmp_path: Path, n: int):
        from voicelegacy.corpus import SegmentRef

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"fake")
        return [SegmentRef(audio, float(i), float(i + 5), "SPEAKER_00", f"s{i}") for i in range(n)]

    def test_detects_single_pitch_outlier_from_values(self, tmp_path: Path) -> None:
        from voicelegacy.corpus import detect_f0_outliers_from_values

        segs = self._segments(tmp_path, 6)
        results = detect_f0_outliers_from_values(
            segs,
            [180.0, 182.0, 179.0, 181.0, 183.0, 310.0],
            min_valid=5,
            mad_threshold=3.5,
        )

        assert [r.is_outlier for r in results] == [False, False, False, False, False, True]
        assert results[-1].robust_z is not None
        assert results[-1].to_dict()["is_outlier"] is True

    def test_does_not_filter_when_too_few_valid_f0_values(self, tmp_path: Path) -> None:
        from voicelegacy.corpus import detect_f0_outliers_from_values

        segs = self._segments(tmp_path, 4)
        results = detect_f0_outliers_from_values(
            segs,
            [180.0, None, None, 320.0],
            min_valid=5,
        )

        assert all(not r.is_outlier for r in results)
        assert all(r.robust_z is None for r in results)

    def test_mismatched_lengths_raise(self, tmp_path: Path) -> None:
        from voicelegacy.corpus import detect_f0_outliers_from_values

        with pytest.raises(ValueError, match="same length"):
            detect_f0_outliers_from_values(self._segments(tmp_path, 2), [180.0])

    def test_write_f0_outlier_report(self, tmp_path: Path) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import detect_f0_outliers_from_values, write_f0_outlier_report

        segs = self._segments(tmp_path, 6)
        results = detect_f0_outliers_from_values(
            segs,
            [180.0, 182.0, 179.0, 181.0, 183.0, 310.0],
            min_valid=5,
            mad_threshold=3.5,
        )
        out = tmp_path / "reports" / "f0.json"
        write_f0_outlier_report(out, results, ReferenceConfig())

        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["summary"]["rejected_as_f0_outliers"] == 1
        assert payload["rejected_segments"][0]["text"] == "s5"

    def test_zero_mad_returns_all_non_outliers(self, tmp_path: Path) -> None:
        """When all F0 values are identical, MAD is 0 and the divide-by-zero
        guard must return everything as non-outlier with robust_z=0."""
        from voicelegacy.corpus import detect_f0_outliers_from_values

        segs = self._segments(tmp_path, 6)
        results = detect_f0_outliers_from_values(segs, [180.0] * 6, min_valid=5, mad_threshold=3.5)
        assert all(not r.is_outlier for r in results)
        assert all(r.robust_z == 0.0 for r in results)

    def test_none_values_preserved_in_results(self, tmp_path: Path) -> None:
        """Segments whose F0 estimation failed (None) must remain non-outliers."""
        from voicelegacy.corpus import detect_f0_outliers_from_values

        segs = self._segments(tmp_path, 6)
        # 5 valid + 1 None — enough to compute thresholds, but the None one
        # must come through cleanly.
        results = detect_f0_outliers_from_values(
            segs,
            [180.0, 182.0, 179.0, 181.0, 183.0, None],
            min_valid=5,
            mad_threshold=3.5,
        )
        assert results[-1].median_f0_hz is None
        assert results[-1].robust_z is None
        assert results[-1].is_outlier is False


# ─── load_speakerscribe_json — audio resolution branches ───────────
class TestLoadSpeakerscribeJsonAudioResolution:
    """The loader resolves source_audio against a set of fallback extensions.
    These tests pin that behavior so the next refactor cannot silently break it.
    """

    def test_resolves_mp3_when_referenced_name_is_wav(
        self, speakerscribe_json_factory, tmp_path: Path
    ) -> None:
        # JSON references "source.wav" but the actual file is "source.mp3"
        (tmp_path / "source.mp3").touch()
        jp = speakerscribe_json_factory(
            "interview.json",
            segments=[{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "ok"}],
            audio_name="source.wav",
        )
        segs = load_speakerscribe_json(jp, audio_root=tmp_path)
        assert len(segs) == 1
        assert segs[0].source_audio.suffix == ".mp3"

    def test_falls_back_to_default_data_dir_when_audio_root_omitted(
        self, speakerscribe_json_factory, tmp_path: Path
    ) -> None:
        # speakerscribe convention: transcripts/<file>.json → data/<file>
        # The factory writes JSONs to tmp_path; with audio_root=None we
        # expect resolution against tmp_path.parent / "data"
        data_dir = tmp_path.parent / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "source.wav").touch()

        jp = speakerscribe_json_factory(
            "interview.json",
            segments=[{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "ok"}],
        )
        segs = load_speakerscribe_json(jp, audio_root=None)
        assert len(segs) == 1
        assert segs[0].source_audio.parent.name == "data"


# ─── estimate_median_f0_hz — real librosa call on synthetic signal ─
class TestEstimateMedianF0:
    """Exercise the real librosa.yin path with a deterministic synthetic
    signal. Keeps the F0-related branches covered (size guard, low-valid
    return, success)."""

    def test_too_short_signal_returns_none(self, tmp_workspace: Path) -> None:
        from voicelegacy.config import XTTS_INPUT_SR
        from voicelegacy.corpus import estimate_median_f0_hz

        # < 0.5 s of signal → must return None before calling librosa
        sr = XTTS_INPUT_SR
        y = np.zeros(int(sr * 0.1), dtype=np.float32)
        assert estimate_median_f0_hz(y, sr) is None

    def test_estimates_median_for_pure_tone(self, tmp_workspace: Path) -> None:
        from voicelegacy.config import XTTS_INPUT_SR
        from voicelegacy.corpus import estimate_median_f0_hz

        sr = XTTS_INPUT_SR
        t = np.arange(int(sr * 2.0)) / sr
        # A pure 200 Hz tone — librosa.yin should return ~200 Hz
        y = (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        median = estimate_median_f0_hz(y, sr, fmin_hz=80.0, fmax_hz=400.0)
        assert median is not None
        # yin is not exact on pure tones; tolerate ±10 Hz
        assert 190.0 <= median <= 210.0

    def test_returns_none_when_librosa_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
    ) -> None:
        """If librosa raises (e.g. version mismatch on bad input), the function
        must log + return None, not propagate."""
        # Force librosa.yin to raise.
        import librosa

        from voicelegacy.config import XTTS_INPUT_SR
        from voicelegacy.corpus import estimate_median_f0_hz

        def raising_yin(*args: object, **kwargs: object) -> np.ndarray:
            raise RuntimeError("simulated yin failure")

        monkeypatch.setattr(librosa, "yin", raising_yin)
        sr = XTTS_INPUT_SR
        y = np.zeros(int(sr * 2.0), dtype=np.float32)
        assert estimate_median_f0_hz(y, sr) is None


# ─── extract_segments_to_wav — true end-to-end with synthetic source ──
class TestExtractSegmentsToWav:
    """End-to-end test of the extract step. Uses the synthetic_speech_wav
    fixture as the source audio so this exercises the real load → slice →
    denoise → trim → save path without any mocks."""

    def _build_segments(self, source: Path) -> list:
        from voicelegacy.corpus import SegmentRef

        # Two valid segments from the same 10s source
        return [
            SegmentRef(source, 1.0, 5.0, "SPEAKER_00", "first"),
            SegmentRef(source, 5.0, 9.0, "SPEAKER_00", "second"),
        ]

    def test_writes_wavs_for_each_segment(self, tmp_path: Path, synthetic_speech_wav: Path) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import extract_segments_to_wav

        out_dir = tmp_path / "out"
        config = ReferenceConfig(
            apply_denoise=False,  # speed up
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
        )
        segs = self._build_segments(synthetic_speech_wav)
        written = extract_segments_to_wav(segs, out_dir, config)

        assert len(written) == 2
        for p in written:
            assert p.exists()
            assert p.suffix == ".wav"

    def test_caches_source_loading_across_segments(
        self,
        tmp_path: Path,
        synthetic_speech_wav: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two segments from the same source must load the audio only once."""
        from voicelegacy.audio import load_audio_mono as real_load
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import extract_segments_to_wav

        load_calls: list[Path] = []

        def counting_load(path: Path, target_sr: int = 22050):
            load_calls.append(Path(path))
            return real_load(path, target_sr=target_sr)

        monkeypatch.setattr("voicelegacy.corpus.load_audio_mono", counting_load)

        config = ReferenceConfig(
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
        )
        segs = self._build_segments(synthetic_speech_wav)
        extract_segments_to_wav(segs, tmp_path / "out", config)
        assert len(load_calls) == 1  # not 2 — caching works

    def test_skips_segment_when_source_audio_missing(
        self, tmp_path: Path, synthetic_speech_wav: Path
    ) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import SegmentRef, extract_segments_to_wav

        segs = [
            SegmentRef(tmp_path / "ghost.wav", 0.0, 5.0, "SPEAKER_00", "ghost"),
            SegmentRef(synthetic_speech_wav, 1.0, 5.0, "SPEAKER_00", "real"),
        ]
        config = ReferenceConfig(
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
        )
        written = extract_segments_to_wav(segs, tmp_path / "out", config)
        # Only the valid segment makes it through
        assert len(written) == 1
        assert written[0].stem.startswith(synthetic_speech_wav.stem)

    def test_skips_segment_with_bad_bounds(
        self, tmp_path: Path, synthetic_speech_wav: Path
    ) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import SegmentRef, extract_segments_to_wav

        # end <= start triggers slice_segment to raise → segment skipped
        segs = [
            SegmentRef(synthetic_speech_wav, 5.0, 4.0, "SPEAKER_00", "bad"),
            SegmentRef(synthetic_speech_wav, 1.0, 5.0, "SPEAKER_00", "good"),
        ]
        config = ReferenceConfig(
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
        )
        written = extract_segments_to_wav(segs, tmp_path / "out", config)
        assert len(written) == 1

    def test_skips_segment_too_short_after_trim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_speech_wav: Path
    ) -> None:
        """Force trim_silence to return a sub-1s array; segment must be dropped."""
        from voicelegacy.config import XTTS_INPUT_SR, ReferenceConfig
        from voicelegacy.corpus import SegmentRef, extract_segments_to_wav

        def aggressive_trim(y: np.ndarray) -> np.ndarray:
            # Return less than 1 second
            return y[: int(XTTS_INPUT_SR * 0.3)]

        # trim_silence now lives in the canonical audio.clean_segment chain (the
        # DRY refactor in Fase 2), so patch it where it is actually called.
        monkeypatch.setattr("voicelegacy.audio.trim_silence", aggressive_trim)
        config = ReferenceConfig(
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
        )
        segs = [SegmentRef(synthetic_speech_wav, 1.0, 5.0, "SPEAKER_00", "x")]
        written = extract_segments_to_wav(segs, tmp_path / "out", config)
        assert written == []

    def test_applies_all_filters_when_configured(
        self, tmp_path: Path, synthetic_speech_wav: Path
    ) -> None:
        """Exercise the bandpass + preemphasis + denoise branches.
        We don't assert audio content (would be brittle); only that the
        pipeline completes and writes WAVs."""
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import extract_segments_to_wav

        config = ReferenceConfig(
            apply_denoise=True,
            denoise_stationary=False,
            apply_bandpass_filter=True,
            apply_preemphasis_filter=True,
        )
        segs = self._build_segments(synthetic_speech_wav)
        written = extract_segments_to_wav(segs, tmp_path / "out", config)
        assert len(written) == 2


# ─── filter_f0_outliers + analyze_f0_outliers — runtime paths ──────
class TestF0OutlierRuntime:
    def test_filter_disabled_returns_all_segments(self, tmp_path: Path) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import SegmentRef, filter_f0_outliers

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"fake")
        segs = [SegmentRef(audio, 0.0, 5.0, "SPEAKER_00", "s")]
        config = ReferenceConfig(enable_f0_outlier_filter=False)
        out = filter_f0_outliers(segs, config)
        assert out == segs  # exact same list returned

    def test_analyze_returns_empty_friendly_when_no_segments(self) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import analyze_f0_outliers

        results = analyze_f0_outliers([], ReferenceConfig(enable_f0_outlier_filter=True))
        assert results == []

    def test_analyze_returns_passthrough_when_filter_disabled(self, tmp_path: Path) -> None:
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import SegmentRef, analyze_f0_outliers

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"fake")
        segs = [SegmentRef(audio, 0.0, 5.0, "SPEAKER_00", "s")]
        results = analyze_f0_outliers(segs, ReferenceConfig(enable_f0_outlier_filter=False))
        assert len(results) == 1
        assert results[0].is_outlier is False
        assert results[0].median_f0_hz is None

    def test_analyze_handles_failed_audio_load(
        self, tmp_path: Path, synthetic_speech_wav: Path
    ) -> None:
        """Mix valid and missing source files; the missing one must produce
        a None F0 measurement but not crash the run."""
        from voicelegacy.config import ReferenceConfig
        from voicelegacy.corpus import SegmentRef, analyze_f0_outliers

        ghost = tmp_path / "missing.wav"
        segs = [
            SegmentRef(synthetic_speech_wav, 1.0, 5.0, "SPEAKER_00", "valid"),
            SegmentRef(ghost, 1.0, 5.0, "SPEAKER_00", "missing"),
        ]
        results = analyze_f0_outliers(
            segs,
            ReferenceConfig(
                enable_f0_outlier_filter=True,
                min_segments_for_f0_filter=3,
            ),
        )
        assert len(results) == 2
        # Missing audio → None median, never flagged outlier
        assert results[1].median_f0_hz is None
        assert results[1].is_outlier is False


# ─── build_reference_corpus — full end-to-end ─────────────────────
class TestBuildReferenceCorpus:
    """Drive the full corpus build: speakerscribe JSON in → WAVs out.
    The whole point of P2-extra in the audit was to plug this gap."""

    def _make_speakerscribe_workspace(
        self,
        tmp_workspace: Path,
        source: Path,
        segments: list[dict],
    ) -> None:
        """Drop the synthetic audio into interviews_raw and a matching JSON
        into speakerscribe_out."""
        # Copy source into interviews_raw under a known name
        import shutil

        target_audio = tmp_workspace / "interviews_raw" / "interview.wav"
        shutil.copy(source, target_audio)

        payload = {
            "source_audio": "interview.wav",
            "language_detected": "es",
            "segments": segments,
        }
        (tmp_workspace / "speakerscribe_out" / "interview.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def test_no_json_files_returns_empty_list(self, tmp_workspace: Path) -> None:
        from voicelegacy.config import ReferenceConfig, WorkspacePaths
        from voicelegacy.corpus import build_reference_corpus

        paths = WorkspacePaths(workspace=str(tmp_workspace))
        config = ReferenceConfig()
        # No JSONs present → returns []
        assert build_reference_corpus(paths, config) == []

    def test_end_to_end_writes_reference_wavs(
        self, tmp_workspace: Path, synthetic_speech_wav: Path
    ) -> None:
        from voicelegacy.config import ReferenceConfig, WorkspacePaths
        from voicelegacy.corpus import build_reference_corpus

        self._make_speakerscribe_workspace(
            tmp_workspace,
            synthetic_speech_wav,
            segments=[
                {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "a"},
                {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_00", "text": "b"},
                # filtered out by speaker
                {"start": 1.0, "end": 5.0, "speaker": "SPEAKER_01", "text": "other"},
            ],
        )

        paths = WorkspacePaths(workspace=str(tmp_workspace))
        config = ReferenceConfig(
            target_speaker_label="SPEAKER_00",
            min_segment_duration_s=3.0,
            max_segment_duration_s=10.0,
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
            enable_f0_outlier_filter=False,  # short synthetic signals not robust
        )
        written = build_reference_corpus(paths, config)

        assert len(written) == 2
        # Other speaker excluded
        for p in written:
            assert "interview" in p.stem

    def test_end_to_end_with_f0_filter_enabled(
        self, tmp_workspace: Path, synthetic_speech_wav: Path
    ) -> None:
        """Runs the full path including F0 analysis. With a pure-tone synthetic
        source all segments will have the same F0 → no outliers, no rejections,
        and the F0 report file is created."""
        from voicelegacy.config import ReferenceConfig, WorkspacePaths
        from voicelegacy.corpus import build_reference_corpus

        # Add a few segments so min_segments_for_f0_filter is satisfied
        self._make_speakerscribe_workspace(
            tmp_workspace,
            synthetic_speech_wav,
            segments=[
                {"start": 0.5, "end": 4.0, "speaker": "SPEAKER_00", "text": f"s{i}"}
                if i == 0
                else {
                    "start": 0.5 + i * 1.5,
                    "end": 4.0 + i * 1.5,
                    "speaker": "SPEAKER_00",
                    "text": f"s{i}",
                }
                for i in range(5)
            ],
        )

        paths = WorkspacePaths(workspace=str(tmp_workspace))
        config = ReferenceConfig(
            target_speaker_label="SPEAKER_00",
            min_segment_duration_s=2.0,
            max_segment_duration_s=10.0,
            apply_denoise=False,
            apply_bandpass_filter=False,
            apply_preemphasis_filter=False,
            enable_f0_outlier_filter=True,
            min_segments_for_f0_filter=3,
        )
        written = build_reference_corpus(paths, config)

        # All segments should pass the F0 filter on a pure-tone source
        assert len(written) >= 3
        # F0 report was created
        f0_reports = list((tmp_workspace / "reports").glob("f0_outliers_*.json"))
        assert len(f0_reports) == 1
