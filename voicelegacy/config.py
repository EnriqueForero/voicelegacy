"""voicelegacy — Centralized configuration with Pydantic v2 validation.

Single source of truth for the voice-cloning pipeline. Zero magic numbers in
business logic — every tunable lives here.

Context: Google Colab Free Tier (T4 GPU, ~12 GB system RAM, 15 GB VRAM).
Model: XTTS-v2 (coqui-tts fork) for Spanish zero-shot voice cloning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─── Public constants ──────────────────────────────────────────────
# XTTS-v2 native sampling rates
XTTS_INPUT_SR: int = 22050
"""Sampling rate XTTS-v2 expects for reference audio."""

XTTS_OUTPUT_SR: int = 24000
"""Sampling rate XTTS-v2 generates output at."""

# Reference audio constraints — empirical from XTTS-v2 papers and field tests
MIN_REF_DURATION_S: float = 6.0
"""Below this, the speaker conditioning is too weak — voice drift kicks in."""

MAX_REF_DURATION_S: float = 30.0
"""Above this, no quality gain and Perceiver context window saturates."""

# Quality gating thresholds
MIN_SNR_DB: float = 15.0
"""Below this SNR, the reference is too noisy — clone will sound muffled."""

MIN_SAMPLING_RATE_HZ: int = 16000
"""Anything below 16kHz means phone-codec audio. Reject — see README §Why."""

MIN_USABLE_REFERENCE_SEGMENTS: int = 3
"""Hard floor on top-N references before we'll attempt synthesis.

Below 3 clean reference segments, XTTS-v2 zero-shot drifts noticeably and
the result is usually unsuitable for legacy use. Better to fail loudly and
let the user collect more material than to ship a bad clone."""

SupportedLanguage = Literal[
    "es",
    "en",
    "pt",
    "fr",
    "it",
    "de",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
]
"""Languages supported by XTTS-v2."""

DeviceType = Literal["auto", "cuda", "cpu"]
LongTextStrategy = Literal["auto", "single_pass", "coqui_split"]


# ─── ReferenceConfig ───────────────────────────────────────────────
class ReferenceConfig(BaseModel):
    """Configuration for building the reference-audio corpus.

    The reference corpus is the set of clean speech segments from the target
    speaker that XTTS-v2 will use to condition voice generation. Quality of
    this corpus dominates the final result — garbage in, garbage out.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    target_speaker_label: str = Field(
        default="SPEAKER_00",
        description="Speaker label in the speakerscribe JSON to extract (e.g. 'SPEAKER_00').",
    )
    min_segment_duration_s: float = Field(
        default=6.0,
        ge=1.0,
        le=60.0,
        description=(
            "Drop segments shorter than this (too short = weak signal). "
            "Default 6.0s matches MIN_REF_DURATION_S and the XTTS-v2 recommended "
            "minimum reference length; shorter clips give the conditioning "
            "encoder too little prosodic context."
        ),
    )
    max_segment_duration_s: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="Drop segments longer than this (likely contains pauses or overlap).",
    )
    target_loudness_lufs: float = Field(
        default=-20.0,
        ge=-30.0,
        le=-16.0,
        description=(
            "EBU R128 target loudness for normalization. Acceptable range is "
            "-30 to -16 LUFS. XTTS-v2 was trained near -23 to -18 LUFS — "
            "default -20 sits in that band. Values closer to -12 (commercial "
            "masters) destabilize the conditioning encoder."
        ),
    )
    apply_denoise: bool = Field(
        default=True,
        description="Apply spectral-gating noise reduction (noisereduce).",
    )
    denoise_stationary: bool = Field(
        default=False,
        description=(
            "Use stationary spectral denoise. False enables adaptive non-stationary "
            "denoise, which is usually safer for real interviews with changing noise."
        ),
    )
    apply_bandpass_filter: bool = Field(
        default=True,
        description="Apply conservative 80-7600 Hz speech band-pass cleanup before denoise.",
    )
    apply_preemphasis_filter: bool = Field(
        default=False,
        description="Apply speech pre-emphasis before denoise; useful for muffled archival audio.",
    )
    enable_f0_outlier_filter: bool = Field(
        default=True,
        description=(
            "Detect and drop target-speaker segments whose median fundamental "
            "frequency is a robust outlier. This catches common diarization "
            "mistakes where another speaker leaks into SPEAKER_00."
        ),
    )
    min_segments_for_f0_filter: int = Field(
        default=5,
        ge=3,
        le=100,
        description="Minimum valid F0 measurements required before outlier filtering is applied.",
    )
    f0_outlier_mad_threshold: float = Field(
        default=3.5,
        ge=1.5,
        le=10.0,
        description=(
            "Robust z-score threshold based on MAD. Larger values are less aggressive; "
            "3.5 is a conservative production default."
        ),
    )
    f0_min_hz: float = Field(
        default=50.0,
        ge=30.0,
        le=200.0,
        description="Lower bound passed to pitch estimation.",
    )
    f0_max_hz: float = Field(
        default=500.0,
        ge=200.0,
        le=800.0,
        description="Upper bound passed to pitch estimation.",
    )
    top_n_segments: int = Field(
        default=5,
        ge=1,
        le=100,
        description=(
            "Keep only the N best-quality segments (by dynamic range + duration). "
            "Default 5: a handful of excellent references beats many mediocre ones "
            "for XTTS-v2 conditioning."
        ),
    )
    min_snr_db: float = Field(
        default=MIN_SNR_DB,
        ge=0.0,
        le=60.0,
        description="Reject segments below this SNR.",
    )
    # ── Fase 2 precision levers (all default to current behavior; opt-in) ──
    # These are precision knobs: leave them off until you have a benchmark
    # baseline to A/B against. Enabling any of them changes which segments are
    # selected, which is a precision change the golden rule says must be measured.
    denoise_only_if_noisy: bool = Field(
        default=False,
        description=(
            "If True, denoise is applied only to segments whose estimated "
            "dynamic range is below denoise_snr_threshold_db. Denoising already "
            "clean speech adds artifacts; this avoids it. Default False keeps the "
            "current unconditional behavior."
        ),
    )
    denoise_snr_threshold_db: float = Field(
        default=25.0,
        ge=0.0,
        le=60.0,
        description="Below this estimated dynamic range a segment is denoised "
        "(only consulted when denoise_only_if_noisy is True).",
    )
    enable_overlap_filter: bool = Field(
        default=False,
        description=(
            "If True, drop target-speaker segments that overlap in time with "
            "other speakers beyond max_overlap_ratio. Crosstalk contaminates the "
            "reference. Mirrors enable_f0_outlier_filter; default off."
        ),
    )
    max_overlap_ratio: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Max fraction of a target segment allowed to overlap other "
        "speakers before it is rejected (only when enable_overlap_filter).",
    )
    min_spectral_rolloff_hz: float = Field(
        default=0.0,
        ge=0.0,
        le=24000.0,
        description=(
            "Reject segments whose measured 85%% spectral rolloff is below this "
            "(detects telephone-band audio that was upsampled, which the header "
            "sample-rate gate cannot catch). 0 disables this gate (default)."
        ),
    )

    @model_validator(mode="after")
    def _check_duration_range(self) -> ReferenceConfig:
        if self.max_segment_duration_s <= self.min_segment_duration_s:
            raise ValueError(
                f"max_segment_duration_s ({self.max_segment_duration_s}) must be "
                f"> min_segment_duration_s ({self.min_segment_duration_s})"
            )
        return self


# ─── SynthesisConfig ───────────────────────────────────────────────
class SynthesisConfig(BaseModel):
    """Configuration for the XTTS-v2 inference step."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    model_name: str = Field(
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        description="Coqui TTS model identifier. Don't change unless you know what you're doing.",
    )
    language: SupportedLanguage = Field(
        default="es",
        description="ISO code for target language. 'es' = Spanish.",
    )
    device: DeviceType = Field(
        default="auto",
        description="'auto' detects CUDA, falls back to CPU.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=0.9,
        description=(
            "XTTS-v2 sampling temperature. Lower = more stable / mechanical, "
            "higher = more expressive but risk of voice drift. The official "
            "Coqui docs recommend 0.65-0.85 (https://docs.coqui.ai/en/latest/"
            "models/xtts.html). Values >0.9 produce erratic, drifted output."
        ),
    )
    length_penalty: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Penalty on output length. >1 = shorter outputs.",
    )
    repetition_penalty: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Discourages repeating tokens. 5.0 is XTTS-v2 default.",
    )
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech rate multiplier. 1.0 = natural pace.",
    )
    enable_text_splitting: bool = Field(
        default=True,
        description=(
            "Legacy compatibility switch. If False, XTTS sentence splitting is always "
            "disabled. If True, long_text_strategy decides when to split."
        ),
    )
    long_text_strategy: LongTextStrategy = Field(
        default="auto",
        description=(
            "How to handle long text. 'auto' avoids splitting short utterances to "
            "reduce voice drift, but enables XTTS splitting for longer prose. "
            "'single_pass' never splits. 'coqui_split' always delegates splitting "
            "to XTTS/Coqui."
        ),
    )
    max_single_pass_chars: int = Field(
        default=240,
        ge=80,
        le=800,
        description=(
            "For long_text_strategy='auto', texts at or below this length are sent "
            "as a single XTTS pass to reduce sentence-to-sentence drift."
        ),
    )
    long_text_warning_chars: int = Field(
        default=600,
        ge=200,
        le=5000,
        description=(
            "Emit sidecar/report warnings at or above this length because zero-shot "
            "voice drift becomes more likely and manual listening is mandatory."
        ),
    )
    seed: int | None = Field(
        default=42,
        description=(
            "Random seed for reproducible synthesis. With the same (text, "
            "references, config, seed) inputs, the output WAV must be "
            "byte-identical. Set to None to disable seeding and accept "
            "non-deterministic outputs (legacy behavior). XTTS-v2 sampling "
            "is stochastic — without a seed, regenerating 'the same clip' "
            "is impossible. For family-legacy use, a fixed seed means a "
            "lost output WAV can always be recreated from its sidecar metadata."
        ),
    )
    compute_similarity: bool = Field(
        default=True,
        description=(
            "Try to score synthesized outputs against the reference corpus with "
            "Resemblyzer. If the optional dependency is unavailable, synthesis still "
            "succeeds and the sidecar records similarity_status='skipped'."
        ),
    )
    # ── Speaker-conditioning knobs ────────────────────────────────────
    # Defaults deliberately mirror upstream XttsConfig (idiap coqui-ai-TTS):
    # gpt_cond_len=12, gpt_cond_chunk_len=4, max_ref_len=10,
    # sound_norm_refs=False. Before v0.7.1 the cached-latents fast path used
    # the *raw* get_conditioning_latents defaults (gpt_cond_len=6,
    # max_ref_length=30) while the tts_to_file fallback used XttsConfig's —
    # so the two code paths conditioned the voice differently. Now both
    # paths read these fields, so the voice identity no longer depends on
    # which path happened to run.
    gpt_cond_len: int = Field(
        default=12,
        ge=1,
        le=60,
        description=(
            "Seconds of reference audio used to build the GPT (prosody/style) "
            "conditioning latent. Longer captures more of the speaker's "
            "delivery; upstream default is 12 s. Increase toward 30 s when "
            "the reference corpus is clean and you want tighter style match."
        ),
    )
    gpt_cond_chunk_len: int = Field(
        default=4,
        ge=1,
        le=60,
        description=(
            "Chunk length (s) for GPT-latent averaging; must be <= "
            "gpt_cond_len. Chunking improves stability (upstream default 4)."
        ),
    )
    max_ref_len: int = Field(
        default=10,
        ge=3,
        le=60,
        description=(
            "Maximum seconds taken from EACH reference WAV for the speaker "
            "embedding (upstream XttsConfig default 10)."
        ),
    )
    sound_norm_refs: bool = Field(
        default=False,
        description=(
            "Normalize reference audio amplitude before conditioning. The "
            "corpus phase already loudness-normalizes to target LUFS, so the "
            "default stays False (matching upstream)."
        ),
    )

    @model_validator(mode="after")
    def _validate_conditioning_knobs(self) -> SynthesisConfig:
        """Enforce upstream invariant: gpt_cond_chunk_len <= gpt_cond_len."""
        if self.gpt_cond_chunk_len > self.gpt_cond_len:
            raise ValueError(
                f"gpt_cond_chunk_len ({self.gpt_cond_chunk_len}) must be <= "
                f"gpt_cond_len ({self.gpt_cond_len}); upstream XTTS requires it."
            )
        return self

    cache_conditioning_latents: bool = Field(
        default=True,
        description=(
            "When the underlying XTTS model API is available, compute and cache "
            "speaker conditioning latents for a stable reference set. If the high-level "
            "TTS wrapper does not expose the model API, voicelegacy falls back to "
            "tts_to_file without failing."
        ),
    )


# ─── WorkspacePaths ────────────────────────────────────────────────
class WorkspacePaths(BaseModel):
    """Filesystem layout for a voicelegacy workspace.

    Convention (mirrors speakerscribe):
        workspace/
        ├── interviews_raw/      ← raw audio (mp3, wav, m4a) input
        ├── speakerscribe_out/   ← .json outputs from speakerscribe (diarization)
        ├── reference_corpus/    ← curated clean reference segments (built by us)
        ├── synthesis_out/       ← generated audio files
        ├── reports/             ← quality reports, metadata
        └── runs.db              ← idempotency cache (content-hash → output)
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    workspace: Path = Field(description="Root workspace folder (typically on Drive).")

    @field_validator("workspace", mode="before")
    @classmethod
    def _coerce_to_path(cls, v: object) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @property
    def interviews_raw(self) -> Path:
        return self.workspace / "interviews_raw"

    @property
    def speakerscribe_out(self) -> Path:
        return self.workspace / "speakerscribe_out"

    @property
    def reference_corpus(self) -> Path:
        return self.workspace / "reference_corpus"

    @property
    def synthesis_out(self) -> Path:
        return self.workspace / "synthesis_out"

    @property
    def reports(self) -> Path:
        return self.workspace / "reports"

    @property
    def db_path(self) -> Path:
        return self.workspace / "runs.db"

    def mkdirs(self) -> None:
        """Create all expected subdirectories. Idempotent."""
        for p in (
            self.interviews_raw,
            self.speakerscribe_out,
            self.reference_corpus,
            self.synthesis_out,
            self.reports,
        ):
            p.mkdir(parents=True, exist_ok=True)


# ─── PipelineConfig (top-level) ────────────────────────────────────
class PipelineConfig(BaseModel):
    """Top-level pipeline configuration. Bundles all sub-configs."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    force_rebuild_reference: bool = Field(
        default=False,
        description="If True, rebuild reference corpus from scratch (ignores cache).",
    )
    force_resynthesize: bool = Field(
        default=False,
        description="If True, re-run synthesis even if output already exists.",
    )
    accept_coqui_tos: bool = Field(
        default=False,
        description=(
            "Must be explicitly set to True. Setting this means you have read and "
            "accept the Coqui Public Model License (CPML): https://coqui.ai/cpml"
        ),
    )

    @model_validator(mode="after")
    def _enforce_tos(self) -> PipelineConfig:
        if not self.accept_coqui_tos:
            raise ValueError(
                "You must explicitly accept the CPML license by setting "
                "accept_coqui_tos=True. Read it first: https://coqui.ai/cpml"
            )
        return self
