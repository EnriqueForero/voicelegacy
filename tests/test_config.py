"""Tests for the config module."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voicelegacy.config import (
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)


class TestReferenceConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = ReferenceConfig()
        assert cfg.target_speaker_label == "SPEAKER_00"
        assert cfg.min_segment_duration_s < cfg.max_segment_duration_s

    def test_max_must_exceed_min(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceConfig(min_segment_duration_s=10.0, max_segment_duration_s=5.0)

    def test_out_of_range_snr_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceConfig(min_snr_db=-10.0)


class TestSynthesisConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = SynthesisConfig()
        assert cfg.language == "es"
        assert cfg.device == "auto"

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            SynthesisConfig(temperature=5.0)

    def test_invalid_language(self) -> None:
        with pytest.raises(ValidationError):
            SynthesisConfig(language="klingon")  # type: ignore[arg-type]


class TestWorkspacePaths:
    def test_subdirs_are_constructed(self, tmp_path: Path) -> None:
        ws = WorkspacePaths(workspace=tmp_path)
        assert ws.interviews_raw == tmp_path / "interviews_raw"
        assert ws.speakerscribe_out == tmp_path / "speakerscribe_out"
        assert ws.reference_corpus == tmp_path / "reference_corpus"
        assert ws.synthesis_out == tmp_path / "synthesis_out"
        assert ws.reports == tmp_path / "reports"

    def test_mkdirs_is_idempotent(self, tmp_path: Path) -> None:
        ws = WorkspacePaths(workspace=tmp_path)
        ws.mkdirs()
        ws.mkdirs()  # should not raise
        assert ws.interviews_raw.exists()
        assert ws.reports.exists()


class TestPipelineConfig:
    def test_must_accept_tos(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig()  # accept_coqui_tos defaults to False

    def test_valid_when_tos_accepted(self) -> None:
        cfg = PipelineConfig(accept_coqui_tos=True)
        assert cfg.synthesis.language == "es"
        assert cfg.reference.target_speaker_label == "SPEAKER_00"


def test_p1_reference_cleanup_defaults_are_enabled_safely() -> None:
    cfg = ReferenceConfig()
    assert cfg.apply_denoise is True
    assert cfg.denoise_stationary is False
    assert cfg.apply_bandpass_filter is True
    assert cfg.apply_preemphasis_filter is False


def test_p1_synthesis_defaults_are_reproducible() -> None:
    cfg = SynthesisConfig()
    assert cfg.seed == 42
    assert cfg.compute_similarity is True
    with pytest.raises(ValidationError):
        SynthesisConfig(temperature=1.2)
    with pytest.raises(ValidationError):
        ReferenceConfig(target_loudness_lufs=-12.0)


# ─── Conditioning knobs (v0.7.1) ────────────────────────────────────
def test_conditioning_knob_defaults_match_upstream_xtts_config() -> None:
    """Defaults must mirror upstream XttsConfig (12/4/10/False) so the cached
    and fallback synthesis paths condition identically out of the box."""
    from voicelegacy.config import SynthesisConfig

    cfg = SynthesisConfig()
    assert cfg.gpt_cond_len == 12
    assert cfg.gpt_cond_chunk_len == 4
    assert cfg.max_ref_len == 10
    assert cfg.sound_norm_refs is False


def test_gpt_cond_chunk_len_must_not_exceed_gpt_cond_len() -> None:
    import pytest
    from pydantic import ValidationError

    from voicelegacy.config import SynthesisConfig

    with pytest.raises(ValidationError):
        SynthesisConfig(gpt_cond_len=6, gpt_cond_chunk_len=8)
