"""voicelegacy — Voice cloning pipeline for family legacy.

Builds curated reference corpora from diarized interviews (via speakerscribe)
and synthesizes new speech in the target speaker's voice using XTTS-v2.

Public API:
    >>> from voicelegacy import (
    ...     PipelineConfig, ReferenceConfig, SynthesisConfig, WorkspacePaths,
    ...     run_reference_phase, run_synthesis, run_batch_synthesis,
    ... )
    >>> paths = WorkspacePaths(workspace="/content/drive/MyDrive/Legado")
    >>> config = PipelineConfig(
    ...     reference=ReferenceConfig(target_speaker_label="SPEAKER_00"),
    ...     synthesis=SynthesisConfig(language="es"),
    ...     accept_coqui_tos=True,
    ... )
    >>> corpus = run_reference_phase(paths, config)
    >>> result = run_synthesis("Hola mi nieto.", corpus.top_wavs, paths, config)
"""

from voicelegacy.config import (
    MAX_REF_DURATION_S,
    MIN_REF_DURATION_S,
    MIN_SAMPLING_RATE_HZ,
    MIN_SNR_DB,
    MIN_USABLE_REFERENCE_SEGMENTS,
    XTTS_INPUT_SR,
    XTTS_OUTPUT_SR,
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.evaluation import (
    BenchmarkConfig,
    BenchmarkReport,
    load_golden_texts,
    run_benchmark,
)
from voicelegacy.finetune_dataset import (
    CoherenceResult,
    DatasetBuildResult,
    build_finetune_dataset,
    validate_corpus_coherence,
)
from voicelegacy.finetuned_inference import (
    FineTunedCheckpoint,
    load_finetuned_model,
    release_finetuned_model,
    synthesize_with_finetuned,
)
from voicelegacy.logging_config import configure_logging, get_logger
from voicelegacy.longform import (
    LongFormConfig,
    LongFormResult,
    LongFormSynthesizer,
    segment_text,
)
from voicelegacy.pipeline import (
    CorpusBuildResult,
    SynthesisResult,
    run_batch_synthesis,
    run_reference_phase,
    run_synthesis,
)
from voicelegacy.quality import QualityReport, evaluate_file, rank_candidates
from voicelegacy.similarity import SimilarityReport, compute_similarity
from voicelegacy.synthesis import load_xtts_model, release_model

__version__ = "0.7.1"

__all__ = [
    "MAX_REF_DURATION_S",
    "MIN_REF_DURATION_S",
    "MIN_SAMPLING_RATE_HZ",
    "MIN_SNR_DB",
    "MIN_USABLE_REFERENCE_SEGMENTS",
    "XTTS_INPUT_SR",
    "XTTS_OUTPUT_SR",
    "BenchmarkConfig",
    "BenchmarkReport",
    "CoherenceResult",
    "CorpusBuildResult",
    "DatasetBuildResult",
    "FineTunedCheckpoint",
    "LongFormConfig",
    "LongFormResult",
    "LongFormSynthesizer",
    "PipelineConfig",
    "QualityReport",
    "ReferenceConfig",
    "SimilarityReport",
    "SynthesisConfig",
    "SynthesisResult",
    "WorkspacePaths",
    "build_finetune_dataset",
    "compute_similarity",
    "configure_logging",
    "evaluate_file",
    "get_logger",
    "load_finetuned_model",
    "load_golden_texts",
    "load_xtts_model",
    "rank_candidates",
    "release_finetuned_model",
    "release_model",
    "run_batch_synthesis",
    "run_benchmark",
    "run_reference_phase",
    "run_synthesis",
    "segment_text",
    "synthesize_with_finetuned",
    "validate_corpus_coherence",
]
