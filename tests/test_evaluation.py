"""Tests for the evaluation harness (``voicelegacy.evaluation``).

The harness is designed to run without a GPU or model weights: synthesis is
injected as a callable and every metric degrades to ``skipped`` when its heavy
optional dependency (speechbrain / faster-whisper / torchaudio) is absent — the
exact situation in CI. These tests exercise the orchestration, the pure WER/CER
math, the text normalizer, report serialization, and the graceful-skip paths.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from voicelegacy.evaluation import (
    BenchmarkConfig,
    BenchmarkReport,
    GoldenText,
    Metric,
    SampleEvaluation,
    _edit_distance,
    _load_wav_mono,
    append_to_cumulative,
    asr_available,
    character_error_rate,
    ecapa_available,
    load_golden_texts,
    normalize_text,
    run_benchmark,
    secs_ecapa_metric,
    secs_resemblyzer_metric,
    squim_available,
    squim_metrics,
    wer_cer_metrics,
    word_error_rate,
)


def _make_synth(sample_rate: int = 24000, seconds: float = 1.0):
    """Return a deterministic mock synthesizer: text -> (waveform, sample_rate)."""

    def _synth(text: str) -> tuple[np.ndarray, int]:
        n = int(sample_rate * seconds)
        rng = np.random.default_rng(len(text))  # variety without randomness across runs
        wav = (0.1 * rng.standard_normal(n)).astype(np.float32)
        return wav, sample_rate

    return _synth


@pytest.fixture
def reference_wavs(tmp_path: Path) -> list[Path]:
    """Three tiny silent-ish reference WAVs on disk."""
    import soundfile as sf

    paths = []
    for i in range(3):
        p = tmp_path / f"ref_{i}.wav"
        sf.write(
            str(p),
            (0.05 * np.random.default_rng(i).standard_normal(8000)).astype(np.float32),
            16000,
        )
        paths.append(p)
    return paths


# ─── Config ────────────────────────────────────────────────────────
class TestBenchmarkConfig:
    def test_defaults_and_to_dict(self) -> None:
        cfg = BenchmarkConfig()
        d = cfg.to_dict()
        assert d["asr_model"] == "large-v3"
        assert d["seed"] == 42
        assert d["compute_wer"] is True

    @pytest.mark.parametrize("field", ["asr_model", "asr_language", "ecapa_source"])
    def test_blank_required_fields_raise(self, field: str) -> None:
        with pytest.raises(ValueError):
            BenchmarkConfig(**{field: "  "})


# ─── Metric value object ───────────────────────────────────────────
class TestMetric:
    def test_ok_skipped_error_constructors(self) -> None:
        assert Metric.ok(0.811234).value == pytest.approx(0.811234)
        assert Metric.ok(1).status == "ok"
        assert Metric.skipped("no dep").value is None
        assert Metric.skipped("no dep").status == "skipped"
        assert Metric.error("boom").status == "error"

    def test_to_dict_rounds_and_keeps_detail(self) -> None:
        d = Metric.ok(0.123456, detail="ecapa").to_dict()
        assert d == {"status": "ok", "value": 0.1235, "detail": "ecapa"}
        assert Metric.skipped("x").to_dict()["value"] is None

    def test_flat_two_columns(self) -> None:
        assert Metric.ok(0.5).flat("secs_ecapa") == {"secs_ecapa": 0.5, "secs_ecapa_status": "ok"}
        assert Metric.skipped("x").flat("wer") == {"wer": None, "wer_status": "skipped"}


# ─── Golden-set loading ────────────────────────────────────────────
class TestLoadGoldenTexts:
    def test_packaged_default_loads_and_counts(self) -> None:
        texts = load_golden_texts()
        cats = {
            c: sum(1 for t in texts if t.category == c)
            for c in ("short", "medium", "long", "chapter")
        }
        assert cats == {"short": 30, "medium": 10, "long": 5, "chapter": 1}
        # ids are sequential per category
        assert texts[0].text_id == "short-01"
        assert any(t.text_id == "chapter-01" for t in texts)
        assert all(t.char_count > 0 for t in texts)

    def test_custom_file(self, tmp_path: Path) -> None:
        f = tmp_path / "g.txt"
        f.write_text(
            "# comment\nshort\tHola mundo.\nmedium\tUn texto medio aquí.\n", encoding="utf-8"
        )
        texts = load_golden_texts(f)
        assert [t.text_id for t in texts] == ["short-01", "medium-01"]
        assert texts[0].text == "Hola mundo."

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_golden_texts(tmp_path / "nope.txt")

    def test_bad_category_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.txt"
        f.write_text("weird\tsome text\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid category"):
            load_golden_texts(f)

    def test_missing_tab_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.txt"
        f.write_text("short no-tab-here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="TAB"):
            load_golden_texts(f)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "g.txt"
        f.write_text("# only comments\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no stimuli"):
            load_golden_texts(f)


# ─── Text normalization ────────────────────────────────────────────
class TestNormalizeText:
    def test_lowercase_punctuation_whitespace(self) -> None:
        assert normalize_text("¡Hola,  Mundo!", expand_numbers=False) == "hola mundo"

    def test_keeps_spanish_letters(self) -> None:
        assert normalize_text("Niño áéíóú", expand_numbers=False) == "niño áéíóú"

    def test_fold_accents(self) -> None:
        assert normalize_text("canción", fold_accents=True, expand_numbers=False) == "cancion"

    def test_expand_numbers_when_available(self) -> None:
        pytest.importorskip("num2words")
        out = normalize_text("tengo 3 gatos", expand_numbers=True)
        assert "tres" in out and "3" not in out

    def test_no_expand_keeps_digits(self) -> None:
        assert "2024" in normalize_text("año 2024", expand_numbers=False)


# ─── WER / CER ─────────────────────────────────────────────────────
class TestErrorRates:
    def _norm(self, s: str) -> str:
        return normalize_text(s, expand_numbers=False)

    def test_edit_distance_basics(self) -> None:
        assert _edit_distance([], []) == 0
        assert _edit_distance([], ["a"]) == 1
        assert _edit_distance(["a", "b"], []) == 2
        assert _edit_distance(list("kitten"), list("sitting")) == 3

    def test_perfect_match_is_zero(self) -> None:
        assert word_error_rate("hola mundo", "Hola, mundo.", normalizer=self._norm) == 0.0
        assert character_error_rate("hola", "Hola", normalizer=self._norm) == 0.0

    def test_one_substitution(self) -> None:
        # 1 wrong word out of 3 → WER 1/3
        assert word_error_rate(
            "uno dos tres", "uno DOS cuatro", normalizer=self._norm
        ) == pytest.approx(1 / 3)

    def test_insertion_and_deletion(self) -> None:
        assert word_error_rate("uno dos", "uno dos tres", normalizer=self._norm) == pytest.approx(
            1 / 2
        )
        assert word_error_rate("uno dos tres", "uno dos", normalizer=self._norm) == pytest.approx(
            1 / 3
        )

    def test_empty_reference_edge_cases(self) -> None:
        assert word_error_rate("", "", normalizer=self._norm) == 0.0
        assert word_error_rate("", "algo", normalizer=self._norm) == 1.0
        assert character_error_rate("", "x", normalizer=self._norm) == 1.0


# ─── Audio helper ──────────────────────────────────────────────────
class TestLoadWavMono:
    def test_resamples_to_target(self, tmp_path: Path) -> None:
        import soundfile as sf

        p = tmp_path / "a.wav"
        sf.write(str(p), np.zeros(22050, dtype=np.float32), 22050)
        out = _load_wav_mono(p, target_sr=16000)
        assert out.dtype == np.float32
        assert abs(len(out) - 16000) <= 2  # ~1 s at 16 kHz

    def test_downmixes_stereo(self, tmp_path: Path) -> None:
        import soundfile as sf

        p = tmp_path / "stereo.wav"
        stereo = np.zeros((16000, 2), dtype=np.float32)
        sf.write(str(p), stereo, 16000)
        out = _load_wav_mono(p, target_sr=16000)
        assert out.ndim == 1


# ─── Metric skip paths (heavy deps absent in CI) ───────────────────
class TestMetricGracefulSkip:
    def test_secs_ecapa_skips_without_speechbrain(self, reference_wavs: list[Path]) -> None:
        if ecapa_available():
            pytest.skip("speechbrain installed; skip-path not exercised here")
        m = secs_ecapa_metric(
            reference_wavs[0], reference_wavs, "speechbrain/spkrec-ecapa-voxceleb"
        )
        assert m.status == "skipped"

    def test_secs_ecapa_errors_without_references(self) -> None:
        m = secs_ecapa_metric(Path("x.wav"), [], "src")
        assert m.status == "error"

    def test_secs_resemblyzer_skips_without_dep(self, reference_wavs: list[Path]) -> None:
        try:
            import resemblyzer  # noqa: F401

            pytest.skip("resemblyzer installed; skip-path not exercised here")
        except ImportError:
            pass
        m = secs_resemblyzer_metric(reference_wavs[0], reference_wavs)
        assert m.status == "skipped"

    def test_wer_skips_without_faster_whisper(self, reference_wavs: list[Path]) -> None:
        if asr_available():
            pytest.skip("faster-whisper installed; skip-path not exercised here")
        wer, cer, transcript = wer_cer_metrics("hola", reference_wavs[0], BenchmarkConfig())
        assert wer.status == "skipped" and cer.status == "skipped" and transcript is None

    def test_squim_skips_without_torchaudio(self, reference_wavs: list[Path]) -> None:
        if squim_available():
            pytest.skip("torchaudio installed; skip-path not exercised here")
        metrics = squim_metrics(reference_wavs[0])
        assert set(metrics) == {"squim_pesq", "squim_stoi", "squim_sisdr"}
        assert all(m.status == "skipped" for m in metrics.values())


# ─── End-to-end orchestration (mock synthesis, no GPU) ─────────────
class TestRunBenchmark:
    def test_preflight_empty_texts(self, reference_wavs: list[Path], tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No stimuli"):
            run_benchmark(_make_synth(), [], reference_wavs, audio_dir=tmp_path)

    def test_preflight_no_references(self, tmp_path: Path) -> None:
        texts = [GoldenText("short-01", "short", "Hola.")]
        with pytest.raises(ValueError, match="reference WAV is required"):
            run_benchmark(_make_synth(), texts, [], audio_dir=tmp_path)

    def test_preflight_missing_reference_file(self, tmp_path: Path) -> None:
        texts = [GoldenText("short-01", "short", "Hola.")]
        with pytest.raises(FileNotFoundError):
            run_benchmark(_make_synth(), texts, [tmp_path / "ghost.wav"], audio_dir=tmp_path)

    def test_runs_metrics_disabled_writes_audio_and_rtf(
        self, reference_wavs: list[Path], tmp_path: Path
    ) -> None:
        texts = [GoldenText(f"short-0{i}", "short", f"Hola numero {i}.") for i in range(1, 4)]
        cfg = BenchmarkConfig(
            compute_secs_ecapa=False,
            compute_secs_resemblyzer=False,
            compute_wer=False,
            compute_mos=False,
        )
        report = run_benchmark(
            _make_synth(seconds=0.5),
            texts,
            reference_wavs,
            audio_dir=tmp_path,
            config=cfg,
            run_id="t1",
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.samples) == 3
        # audio persisted for audit
        for s in report.samples:
            assert s.audio_path is not None and s.audio_path.exists()
            assert s.rtf is not None and s.rtf >= 0.0
            assert s.audio_seconds == pytest.approx(0.5, abs=0.05)
            assert s.metrics == {}  # all disabled
        summary = report.summary()
        assert summary["n_samples"] == 3
        assert summary["by_category"] == {"short": 3}
        assert summary["rtf"]["n"] == 3

    def test_runs_with_metrics_all_skipped_without_deps(
        self, reference_wavs: list[Path], tmp_path: Path
    ) -> None:
        if any((ecapa_available(), asr_available(), squim_available())):
            pytest.skip("a heavy metric dep is installed; this test asserts the all-skipped path")
        texts = [GoldenText("short-01", "short", "Una prueba corta.")]
        report = run_benchmark(
            _make_synth(), texts, reference_wavs, audio_dir=tmp_path, run_id="t2"
        )
        sample = report.samples[0]
        # default config enables ecapa, resemblyzer, wer, cer, and 3 squim metrics
        assert "secs_ecapa" in sample.metrics
        assert "wer" in sample.metrics and "cer" in sample.metrics
        assert {"squim_pesq", "squim_stoi", "squim_sisdr"} <= set(sample.metrics)
        # everything skipped (deps absent), nothing crashed
        assert all(m.status == "skipped" for m in sample.metrics.values())
        smry = report.summary()
        assert smry["metrics"]["wer"]["n_skipped"] == 1
        assert smry["metrics"]["wer"]["n_ok"] == 0


# ─── Report serialization + aggregation ────────────────────────────
def _hand_report() -> BenchmarkReport:
    samples = (
        SampleEvaluation(
            text_id="short-01",
            category="short",
            char_count=10,
            audio_seconds=2.0,
            synth_seconds=1.0,
            rtf=0.5,
            peak_vram_mb=None,
            metrics={"secs_ecapa": Metric.ok(0.80), "wer": Metric.ok(0.10)},
            asr_transcript="hola",
            audio_path=Path("/tmp/a.wav"),
        ),
        SampleEvaluation(
            text_id="short-02",
            category="short",
            char_count=12,
            audio_seconds=4.0,
            synth_seconds=1.0,
            rtf=0.25,
            peak_vram_mb=None,
            metrics={"secs_ecapa": Metric.ok(0.90), "wer": Metric.skipped("no asr")},
            asr_transcript=None,
            audio_path=Path("/tmp/b.wav"),
        ),
    )
    return BenchmarkReport(
        run_id="r1",
        created_at="2026-06-10T00:00:00Z",
        voicelegacy_version="0.5.0",
        config={"seed": 42},
        runtime={"cuda_available": False},
        samples=samples,
        n_references=3,
    )


class TestReport:
    def test_summary_aggregates_over_ok_samples(self) -> None:
        smry = _hand_report().summary()
        assert smry["n_samples"] == 2
        assert smry["metrics"]["secs_ecapa"]["stats"]["mean"] == pytest.approx(0.85)
        assert smry["metrics"]["secs_ecapa"]["n_ok"] == 2
        # wer: one ok, one skipped → mean over the single ok value
        assert smry["metrics"]["wer"]["n_ok"] == 1
        assert smry["metrics"]["wer"]["n_skipped"] == 1
        assert smry["metrics"]["wer"]["stats"]["mean"] == pytest.approx(0.10)
        assert smry["rtf"]["mean"] == pytest.approx(0.375)

    def test_to_dict_and_write_json(self, tmp_path: Path) -> None:
        report = _hand_report()
        out = report.write_json(tmp_path / "rep.json")
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "r1"
        assert loaded["summary"]["n_samples"] == 2
        assert len(loaded["samples"]) == 2
        assert loaded["samples"][0]["metrics"]["secs_ecapa"]["value"] == 0.8

    def test_rows_are_flat(self) -> None:
        rows = _hand_report().rows()
        assert rows[0]["run_id"] == "r1"
        assert rows[0]["secs_ecapa"] == 0.8
        assert rows[0]["secs_ecapa_status"] == "ok"
        assert rows[1]["wer"] is None and rows[1]["wer_status"] == "skipped"


# ─── Cumulative accumulation ───────────────────────────────────────
class TestAppendToCumulative:
    def test_available_backend_writes_and_grows(self, tmp_path: Path) -> None:
        report = _hand_report()
        out1 = append_to_cumulative(report, tmp_path / "bench.parquet")
        assert out1.exists()
        out2 = append_to_cumulative(report, tmp_path / "bench.parquet")
        # second append targets the same file and accumulates
        assert out2 == out1

    def test_jsonl_fallback_without_pandas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force `import pandas` to fail to exercise the JSONL fallback path.
        monkeypatch.setitem(sys.modules, "pandas", None)
        out = append_to_cumulative(_hand_report(), tmp_path / "bench.parquet")
        assert out.suffix == ".jsonl"
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["run_id"] == "r1"
