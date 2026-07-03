"""voicelegacy — Long-form text chunking for robust audiobook synthesis.

XTTS-v2 derails past roughly 250-270 characters per generation in Spanish: it
truncates, loops, or hallucinates. To narrate a paragraph or a chapter we must
split the text into chunks XTTS can handle, synthesize each, and join them
seamlessly. This module is the *text* half of that — the segmenter. The audio
assembly, ASR verification, and orchestration live elsewhere in Fase 3.

Design goals of ``segment_text`` (in priority order):

1. **Never exceed the budget.** Every chunk is <= ``max_chunk_chars``, so XTTS
   never gets a generation it will mangle.
2. **Never split mid-word.** The last resort before slicing a single
   pathological token (a URL, a 250-char string) is a whitespace split.
3. **Respect linguistic structure.** Split on sentence boundaries first, then
   clause boundaries, then whitespace — so prosody and pauses fall where a human
   narrator would put them.
4. **Carry boundary context.** Each chunk records the boundary that *follows*
   it (paragraph / sentence / clause / hard / end). The audio assembler maps
   that to a silence gap (real pause) or a short crossfade (forced split with no
   natural pause), so the join sounds intentional rather than chopped.

Spanish-specific care: opening marks (¿ ¡), abbreviations whose trailing period
is not a sentence end (Sr., Dra., núm., etc.), decimal/thousands numbers
(``3,5`` / ``1.000``), and ellipsis (``...`` / ``…``).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from voicelegacy.audio import (
    concatenate_audio,
    equal_power_crossfade,
    silence,
    trim_silence,
)
from voicelegacy.evaluation import (
    DEFAULT_ASR_MODEL,
    SynthesizeFn,
    normalize_text,
    transcribe,
    word_error_rate,
)
from voicelegacy.logging_config import get_logger
from voicelegacy.telemetry import runtime_snapshot

logger = get_logger()

# Boundary that follows a chunk. The assembler turns these into the gap (or
# crossfade) between this chunk and the next one.
BOUNDARY_PARAGRAPH = "paragraph"  # longest pause
BOUNDARY_SENTENCE = "sentence"  # medium pause
BOUNDARY_CLAUSE = "clause"  # short pause (split landed on , ; : —)
BOUNDARY_HARD = "hard"  # forced whitespace split mid-clause → crossfade, no pause
BOUNDARY_END = "end"  # last chunk, nothing follows
VALID_BOUNDARIES = (
    BOUNDARY_PARAGRAPH,
    BOUNDARY_SENTENCE,
    BOUNDARY_CLAUSE,
    BOUNDARY_HARD,
    BOUNDARY_END,
)

# Conservative safety margin under XTTS-v2's effective per-generation limit. The
# real limit is in tokens, not characters; this is a deliberately low character
# proxy with headroom. A future revision can count tokens with the XTTS
# tokenizer for a tighter bound.
DEFAULT_MAX_CHUNK_CHARS = 220

# Sentence terminators and clause delimiters.
_TERMINATORS = ".!?…"
_CLAUSE_DELIMS = ",;:—–"
_CLOSING_PUNCT = "»\"')]”’"

# Tokens whose trailing period is an abbreviation, not a sentence end. Lowercased
# (with accents) for comparison. Deliberately excludes ambiguous ones like "no"
# (también es palabra) to avoid wrongly merging real sentence breaks.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "sr",
        "sra",
        "srta",
        "srs",
        "sres",
        "dr",
        "dra",
        "dres",
        "d",
        "dña",
        "ud",
        "uds",
        "vd",
        "vds",
        "prof",
        "profa",
        "lic",
        "licda",
        "ing",
        "arq",
        "núm",
        "pág",
        "págs",
        "vol",
        "art",
        "arts",
        "cap",
        "av",
        "avda",
        "ej",
        "máx",
        "mín",
        "depto",
        "dpto",
        "dept",
        "tel",
        "ext",
        "izq",
        "dcha",
        "gral",
        "cía",
        "ltda",
        "esq",
        "apdo",
        "admón",
        "atte",
        "fig",
        "ref",
        "op",
        "cit",
        "ss",
        "aprox",
        "pdte",
        "ee",
        "uu",
        "aa",
        "vs",
    }
)
# Note on "etc.": deliberately NOT listed. In narrative Spanish it ends a
# sentence far more often than it sits mid-sentence, so treating its trailing
# period as a real boundary under-splits less than the alternative. The
# honorific/measurement abbreviations above precede a continuation (e.g.
# "Sr. Pérez"), where NOT splitting is the safer choice for TTS prosody.

_WORD_BEFORE_PERIOD_RE = re.compile(r"([^\s.]+)\.$")


@dataclass(frozen=True)
class Chunk:
    """One synthesizable piece of a long text.

    Attributes:
        index: 0-based position in the chunk sequence.
        text: The text to synthesize. Always non-empty and within budget.
        boundary: What follows this chunk — one of ``VALID_BOUNDARIES``. The
            audio assembler uses it to choose a silence gap or a crossfade.
    """

    index: int
    text: str
    boundary: str

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk's text."""
        return len(self.text)


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines into paragraphs. Single newlines are soft wraps."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _ends_with_abbreviation(fragment: str) -> bool:
    """True if ``fragment`` (ending in '.') ends with a known abbreviation."""
    match = _WORD_BEFORE_PERIOD_RE.search(fragment)
    if not match:
        return False
    return match.group(1).lower() in ABBREVIATIONS


def _split_sentences_es(text: str) -> list[str]:
    """Split Spanish text into sentences, keeping terminators attached.

    Avoids false boundaries on decimals/thousands (``3,5`` / ``1.000``), known
    abbreviations (``Sr.``), and treats runs of terminators (``...``, ``?!``) as
    a single boundary. A boundary requires whitespace or end-of-text after the
    terminator (plus any closing quotes/brackets), so URLs and ``www.x.com`` do
    not split.
    """
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in _TERMINATORS:
            i += 1
            continue

        # Consume a run of terminators (e.g. "...", "?!").
        j = i
        while j < n and text[j] in _TERMINATORS:
            j += 1
        run = text[i:j]

        # A lone '.' between digits is a decimal/thousands separator, not an end.
        if run == "." and i > 0 and text[i - 1].isdigit() and j < n and text[j].isdigit():
            i = j
            continue

        # A lone '.' right after a known abbreviation is not a sentence end.
        if run == "." and _ends_with_abbreviation(text[start : i + 1]):
            i = j
            continue

        # Absorb trailing closing punctuation: «...?» ("..." ) etc.
        k = j
        while k < n and text[k] in _CLOSING_PUNCT:
            k += 1

        # A real boundary needs whitespace or end-of-text after it.
        if k >= n or text[k].isspace():
            sentence = text[start:k].strip()
            if sentence:
                sentences.append(sentence)
            start = k
            i = k
        else:
            i = j

    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences


def _split_clauses(text: str) -> list[str]:
    """Split on clause delimiters (``, ; : — –``), keeping each delimiter on the
    clause it terminates.

    A delimiter sitting between two digits is NOT a split point: that is a
    decimal (``3,5``), a time (``14:30``), or a range (``5–7``), and breaking it
    across chunks would mangle the number.
    """
    clauses: list[str] = []
    start = 0
    n = len(text)
    for i, char in enumerate(text):
        if char not in _CLAUSE_DELIMS:
            continue
        prev_is_digit = i > 0 and text[i - 1].isdigit()
        next_is_digit = i + 1 < n and text[i + 1].isdigit()
        if prev_is_digit and next_is_digit:
            continue
        piece = text[start : i + 1].strip()
        if piece:
            clauses.append(piece)
        start = i + 1
    rest = text[start:].strip()
    if rest:
        clauses.append(rest)
    return clauses


def _split_on_whitespace(text: str, budget: int) -> list[str]:
    """Greedily pack words up to ``budget``. A single word longer than the budget
    is hard-sliced (the only place a token is broken mid-word)."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > budget:
            # Flush what we have, then slice the oversized token into budget bits.
            if current:
                pieces.append(current)
                current = ""
            for offset in range(0, len(word), budget):
                pieces.append(word[offset : offset + budget])
            continue
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= budget:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _split_oversized_sentence(sentence: str, budget: int) -> list[tuple[str, str]]:
    """Break a sentence longer than ``budget`` into (text, boundary) sub-chunks.

    Packs clauses greedily; closing a sub-chunk at a clause delimiter tags it
    ``clause``. A single clause longer than the budget is split on whitespace,
    tagging the forced cuts ``hard``. The final sub-chunk's boundary is a
    placeholder the caller overrides with the sentence's own boundary.
    """
    clauses = _split_clauses(sentence)
    subs: list[tuple[str, str]] = []
    current = ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip() if current else clause
        if len(candidate) <= budget:
            current = candidate
            continue
        if current:
            subs.append((current, BOUNDARY_CLAUSE))
            current = ""
        if len(clause) > budget:
            words = _split_on_whitespace(clause, budget)
            for word_piece in words[:-1]:
                subs.append((word_piece, BOUNDARY_HARD))
            current = words[-1] if words else ""
        else:
            current = clause
    if current:
        subs.append((current, BOUNDARY_CLAUSE))
    return subs


def segment_text(text: str, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[Chunk]:
    """Segment a long text into synthesizable chunks within ``max_chunk_chars``.

    Splits paragraphs → sentences → (if needed) clauses → (last resort)
    whitespace, never breaking a word unless a single token exceeds the budget.
    Each returned chunk carries the boundary that follows it so the audio
    assembler can insert the right pause or crossfade.

    Args:
        text: The text to narrate. May contain blank-line-separated paragraphs.
        max_chunk_chars: Hard ceiling on chunk length, in characters.

    Returns:
        Ordered chunks. Empty list for empty/whitespace-only input.

    Raises:
        ValueError: If ``max_chunk_chars`` is not positive.
    """
    if max_chunk_chars <= 0:
        raise ValueError(f"max_chunk_chars must be positive; got {max_chunk_chars}")
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    raw: list[tuple[str, str]] = []  # (text, terminating boundary)

    for p_idx, paragraph in enumerate(paragraphs):
        is_last_paragraph = p_idx == len(paragraphs) - 1
        sentences = _split_sentences_es(paragraph)
        for s_idx, sentence in enumerate(sentences):
            is_last_sentence = s_idx == len(sentences) - 1
            if is_last_sentence:
                sentence_boundary = BOUNDARY_END if is_last_paragraph else BOUNDARY_PARAGRAPH
            else:
                sentence_boundary = BOUNDARY_SENTENCE

            if len(sentence) <= max_chunk_chars:
                raw.append((sentence, sentence_boundary))
                continue

            subs = _split_oversized_sentence(sentence, max_chunk_chars)
            for sub_idx, (sub_text, sub_boundary) in enumerate(subs):
                if sub_idx == len(subs) - 1:
                    raw.append((sub_text, sentence_boundary))  # inherit sentence's boundary
                else:
                    raw.append((sub_text, sub_boundary))

    return [Chunk(index=i, text=t, boundary=b) for i, (t, b) in enumerate(raw)]


# Default pause/crossfade timings (ms). Single source of truth; LongFormConfig
# (Fase 3, later step) will surface these as configurable knobs.
DEFAULT_CROSSFADE_MS = 15.0
DEFAULT_CLAUSE_PAUSE_MS = 150.0
DEFAULT_SENTENCE_PAUSE_MS = 350.0
DEFAULT_PARAGRAPH_PAUSE_MS = 700.0


def assemble_chunks(
    audio_chunks: Sequence[np.ndarray],
    boundaries: Sequence[str],
    sr: int,
    *,
    crossfade_ms: float = DEFAULT_CROSSFADE_MS,
    clause_pause_ms: float = DEFAULT_CLAUSE_PAUSE_MS,
    sentence_pause_ms: float = DEFAULT_SENTENCE_PAUSE_MS,
    paragraph_pause_ms: float = DEFAULT_PARAGRAPH_PAUSE_MS,
    trim: bool = True,
) -> np.ndarray:
    """Join rendered chunk audio into one continuous track.

    The boundary that FOLLOWS each chunk decides how it joins to the next one:

    * ``hard``      → equal-power crossfade (a forced mid-clause split with no
      natural pause; the crossfade hides the seam);
    * ``clause``    → short silence;
    * ``sentence``  → medium silence;
    * ``paragraph`` → long silence;
    * ``end``       → nothing (only valid on the last chunk; unused for joining).

    Each chunk is trimmed of leading/trailing silence first (when ``trim``) so
    the inserted pauses are deterministic, not chunk padding plus a pause.

    Args:
        audio_chunks: One mono waveform per chunk, in order.
        boundaries: The boundary that follows each chunk (same length as
            ``audio_chunks``; the last entry is typically ``end`` and unused).
        sr: Sample rate shared by all chunks.
        crossfade_ms: Crossfade length used for ``hard`` joins.
        clause_pause_ms: Silence inserted at a ``clause`` boundary.
        sentence_pause_ms: Silence inserted at a ``sentence`` boundary.
        paragraph_pause_ms: Silence inserted at a ``paragraph`` boundary.
        trim: Trim leading/trailing silence from each chunk before joining.

    Returns:
        The assembled mono waveform (empty array when there are no chunks).

    Raises:
        ValueError: If ``audio_chunks`` and ``boundaries`` differ in length.
    """
    if len(audio_chunks) != len(boundaries):
        raise ValueError(
            f"audio_chunks ({len(audio_chunks)}) and boundaries ({len(boundaries)}) "
            "must have the same length."
        )
    if not audio_chunks:
        return np.asarray([], dtype=np.float32)

    def _prep(arr: np.ndarray) -> np.ndarray:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        return trim_silence(flat) if trim else flat

    prepared = [_prep(a) for a in audio_chunks]
    pause_for = {
        BOUNDARY_CLAUSE: clause_pause_ms,
        BOUNDARY_SENTENCE: sentence_pause_ms,
        BOUNDARY_PARAGRAPH: paragraph_pause_ms,
    }

    result = prepared[0]
    for i in range(1, len(prepared)):
        boundary = boundaries[i - 1]
        nxt = prepared[i]
        if boundary == BOUNDARY_HARD:
            result = equal_power_crossfade(result, nxt, sr, crossfade_ms)
        else:
            gap_ms = pause_for.get(boundary, sentence_pause_ms)
            result = concatenate_audio([result, silence(gap_ms, sr), nxt])
    return result.astype(np.float32)


# Default WER ceiling above which a rendered chunk is considered failed (likely
# truncated/hallucinated) and worth retrying.
DEFAULT_MAX_WER = 0.15


@dataclass(frozen=True)
class ChunkVerification:
    """Result of ASR-verifying one rendered chunk against its requested text.

    Attributes:
        passed: True if the chunk is acceptable. When ASR is unavailable
            (``status == "skipped"``) this is True — we cannot verify, so we do
            not block; a retry would not add ASR anyway.
        wer: Word error rate vs the requested text, or None when not computed.
        transcript: What the ASR heard, or None.
        status: ``"ok"`` | ``"skipped"`` (no faster-whisper) | ``"error"``.
    """

    passed: bool
    wer: float | None
    transcript: str | None
    status: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for the sidecar/manifest."""
        return {
            "passed": self.passed,
            "wer": round(self.wer, 4) if self.wer is not None else None,
            "transcript": self.transcript,
            "status": self.status,
        }


def verify_chunk(
    audio: np.ndarray,
    sr: int,
    expected_text: str,
    *,
    asr_model: str = DEFAULT_ASR_MODEL,
    language: str = "es",
    max_wer: float = DEFAULT_MAX_WER,
    fold_accents: bool = False,
    expand_numbers: bool = True,
) -> ChunkVerification:
    """Transcribe a rendered chunk and check it matches the requested text.

    This is the robustness gate for long-form synthesis: the dominant XTTS
    failure on long input is silent truncation/hallucination, and a round-trip
    WER is the cheapest detector. Reuses the harness ASR + WER (no duplication).

    Degrades safely: if faster-whisper is not installed the result is
    ``status="skipped"`` with ``passed=True`` (cannot verify → do not block). Any
    ASR error is ``status="error"`` with ``passed=True`` for the same reason —
    an ASR hiccup should not discard otherwise-fine audio.

    Args:
        audio: The rendered chunk waveform (mono float32).
        sr: Sample rate of ``audio``.
        expected_text: The text the chunk was synthesized from.
        asr_model: faster-whisper model id (defaults to the harness default).
        language: ISO language for the ASR.
        max_wer: WER ceiling; at or below this the chunk passes.
        fold_accents: Passed to the WER text normalizer.
        expand_numbers: Passed to the WER text normalizer.

    Returns:
        A :class:`ChunkVerification`.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        sf.write(
            str(tmp_path), np.asarray(audio, dtype=np.float32).reshape(-1), sr, subtype="PCM_16"
        )
        transcript = transcribe(tmp_path, asr_model, language)
    except ImportError:
        return ChunkVerification(passed=True, wer=None, transcript=None, status="skipped")
    except Exception as exc:
        logger.warning("Chunk ASR verification errored ({}); accepting chunk.", exc)
        return ChunkVerification(passed=True, wer=None, transcript=None, status="error")
    finally:
        tmp_path.unlink(missing_ok=True)

    wer = word_error_rate(
        expected_text,
        transcript,
        normalizer=lambda t: normalize_text(
            t, fold_accents=fold_accents, expand_numbers=expand_numbers
        ),
    )
    return ChunkVerification(passed=wer <= max_wer, wer=wer, transcript=transcript, status="ok")


# ═══════════════════════════════════════════════════════════════════
# Checkpoint / resume — per-chunk on-disk cache
# ═══════════════════════════════════════════════════════════════════
def compute_doc_hash(text: str, config: Mapping[str, object]) -> str:
    """Stable short hash of the input text plus the config that affects output.

    Keys the cache to the exact (text, config) pair: change either and the hash
    changes, so a different cache directory is used and no stale chunk audio is
    reused. The config is sorted so key order does not affect the hash.
    """
    payload = json.dumps(
        {"text": text, "config": dict(sorted(config.items()))},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LongFormCache:
    """On-disk per-chunk cache enabling resumable long-form synthesis.

    Layout: ``<cache_root>/<doc_hash>/`` holding ``chunk_NNNN.wav`` files and a
    ``manifest.json``. Because the directory is the doc hash, changing the text
    or the synthesis config produces a new directory and the old chunks are
    simply not found — no stale audio is ever reused.

    The orchestrator (Fase 3, later step) consults ``has_chunk`` per index to
    skip already-rendered chunks on a resumed run, and calls ``save_chunk`` as
    each chunk is produced so an interrupted run loses at most the chunk in
    flight.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        cache_root: Path,
        doc_hash: str,
        *,
        version: str = "",
        config: Mapping[str, object] | None = None,
    ) -> None:
        self.dir = Path(cache_root) / doc_hash
        self.doc_hash = doc_hash
        self._manifest = self._load_manifest()
        if not self._manifest:
            self._manifest = {
                "doc_hash": doc_hash,
                "voicelegacy_version": version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": dict(config or {}),
                "chunks": {},
            }

    @property
    def manifest_path(self) -> Path:
        """Path to this cache's manifest file."""
        return self.dir / self.MANIFEST_NAME

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache manifest at {}; starting fresh.", self.manifest_path)
        return {}

    def _save_manifest(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def chunk_entry(self, index: int) -> dict | None:
        """Manifest entry for a chunk index, or None if absent."""
        return self._manifest.get("chunks", {}).get(str(index))

    def has_chunk(self, index: int) -> bool:
        """True if chunk ``index`` is cached AND its WAV is present on disk."""
        entry = self.chunk_entry(index)
        return bool(entry) and (self.dir / entry["audio_file"]).exists()

    def completed_indices(self) -> set[int]:
        """Indices that are fully cached (manifest entry + WAV present)."""
        return {int(i) for i in self._manifest.get("chunks", {}) if self.has_chunk(int(i))}

    def load_audio(self, index: int) -> tuple[np.ndarray, int]:
        """Load a cached chunk's audio as mono float32 with its sample rate.

        Raises:
            KeyError: If the chunk is not in the cache.
        """
        entry = self.chunk_entry(index)
        if entry is None:
            raise KeyError(f"chunk {index} not in cache")
        audio, sr = sf.read(str(self.dir / entry["audio_file"]), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)

    def save_chunk(
        self,
        index: int,
        text: str,
        boundary: str,
        audio: np.ndarray,
        sr: int,
        *,
        verification: ChunkVerification | None = None,
        retries: int = 0,
    ) -> Path:
        """Persist one rendered chunk's audio and update the manifest.

        Returns the path to the written WAV.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        audio_file = f"chunk_{index:04d}.wav"
        audio_path = self.dir / audio_file
        sf.write(
            str(audio_path),
            np.asarray(audio, dtype=np.float32).reshape(-1),
            sr,
            subtype="PCM_16",
        )
        self._manifest.setdefault("chunks", {})[str(index)] = {
            "index": index,
            "text": text,
            "boundary": boundary,
            "char_count": len(text),
            "audio_file": audio_file,
            "sample_rate": sr,
            "retries": retries,
            "verification": verification.to_dict() if verification is not None else None,
        }
        self._save_manifest()
        return audio_path

    def clear(self) -> None:
        """Delete all cached audio and reset the manifest (for a forced re-run)."""
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self._manifest = {"doc_hash": self.doc_hash, "chunks": {}}


# ═══════════════════════════════════════════════════════════════════
# Orchestrator — LongFormSynthesizer
# ═══════════════════════════════════════════════════════════════════
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class LongFormConfig:
    """Configuration for long-form synthesis.

    Defaults reproduce sensible audiobook behavior. The precision-relevant knobs
    (chunk budget, WER ceiling, retries) are the ones to tune against a benchmark.
    """

    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    crossfade_ms: float = DEFAULT_CROSSFADE_MS
    clause_pause_ms: float = DEFAULT_CLAUSE_PAUSE_MS
    sentence_pause_ms: float = DEFAULT_SENTENCE_PAUSE_MS
    paragraph_pause_ms: float = DEFAULT_PARAGRAPH_PAUSE_MS
    trim_chunks: bool = True
    asr_verify: bool = True
    max_wer: float = DEFAULT_MAX_WER
    max_retries: int = DEFAULT_MAX_RETRIES
    asr_model: str = DEFAULT_ASR_MODEL
    language: str = "es"
    fold_accents: bool = False
    expand_numbers: bool = True

    def __post_init__(self) -> None:
        if self.max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be > 0; got {self.max_chunk_chars}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0; got {self.max_retries}")
        if self.max_wer < 0:
            raise ValueError(f"max_wer must be >= 0; got {self.max_wer}")
        for name in ("crossfade_ms", "clause_pause_ms", "sentence_pause_ms", "paragraph_pause_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)}")
        if not self.asr_model.strip():
            raise ValueError("asr_model must be a non-empty model id.")
        if not self.language.strip():
            raise ValueError("language must be a non-empty ISO code.")

    def to_dict(self) -> dict[str, object]:
        """Serialize all fields to a JSON-friendly dict."""
        return {
            "max_chunk_chars": self.max_chunk_chars,
            "crossfade_ms": self.crossfade_ms,
            "clause_pause_ms": self.clause_pause_ms,
            "sentence_pause_ms": self.sentence_pause_ms,
            "paragraph_pause_ms": self.paragraph_pause_ms,
            "trim_chunks": self.trim_chunks,
            "asr_verify": self.asr_verify,
            "max_wer": self.max_wer,
            "max_retries": self.max_retries,
            "asr_model": self.asr_model,
            "language": self.language,
            "fold_accents": self.fold_accents,
            "expand_numbers": self.expand_numbers,
        }

    def cache_config(self) -> dict[str, object]:
        """Subset of config that changes the *cached chunk audio*.

        Pause/crossfade timings only affect final assembly (which happens after
        caching), so they are excluded: changing a pause re-assembles from the
        same cache rather than re-synthesizing every chunk.
        """
        return {
            "max_chunk_chars": self.max_chunk_chars,
            "trim_chunks": self.trim_chunks,
            "asr_verify": self.asr_verify,
            "max_wer": self.max_wer,
            "max_retries": self.max_retries,
            "asr_model": self.asr_model,
            "language": self.language,
            "fold_accents": self.fold_accents,
            "expand_numbers": self.expand_numbers,
        }


@dataclass(frozen=True)
class RenderedChunk:
    """Metadata for one rendered chunk (the audio lives in a parallel list/disk)."""

    index: int
    text: str
    boundary: str
    char_count: int
    audio_seconds: float
    sample_rate: int
    verification: ChunkVerification
    retries: int
    from_cache: bool

    @property
    def flagged(self) -> bool:
        """True if ASR verification ran and the chunk did not pass."""
        return self.verification.status == "ok" and not self.verification.passed

    def to_dict(self) -> dict[str, object]:
        """Serialize for the sidecar audit trail."""
        return {
            "index": self.index,
            "text": self.text,
            "boundary": self.boundary,
            "char_count": self.char_count,
            "audio_seconds": round(self.audio_seconds, 3),
            "retries": self.retries,
            "from_cache": self.from_cache,
            "flagged": self.flagged,
            "verification": self.verification.to_dict(),
        }


@dataclass(frozen=True)
class LongFormResult:
    """Outcome of a long-form render: the WAV path plus a full audit trail."""

    output_path: Path
    sidecar_path: Path
    sample_rate: int
    duration_s: float
    chunks: tuple[RenderedChunk, ...]
    synth_seconds: float
    runtime: dict[str, object]
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        """Number of chunks in the render."""
        return len(self.chunks)

    @property
    def flagged_indices(self) -> tuple[int, ...]:
        """Indices of chunks that failed ASR verification (kept best attempt)."""
        return tuple(c.index for c in self.chunks if c.flagged)

    @property
    def rtf(self) -> float | None:
        """Real-time factor over the chunks that were actually synthesized."""
        synth_audio = sum(c.audio_seconds for c in self.chunks if not c.from_cache)
        if synth_audio <= 0:
            return None
        return round(self.synth_seconds / synth_audio, 4)

    def summary(self) -> dict[str, object]:
        """Aggregate stats for the sidecar header."""
        verified = [c for c in self.chunks if c.verification.status == "ok"]
        wers = [c.verification.wer for c in verified if c.verification.wer is not None]
        return {
            "output_path": str(self.output_path),
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 3),
            "n_chunks": self.n_chunks,
            "n_from_cache": sum(1 for c in self.chunks if c.from_cache),
            "n_flagged": len(self.flagged_indices),
            "flagged_indices": list(self.flagged_indices),
            "asr_verified_chunks": len(verified),
            "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
            "max_wer": round(max(wers), 4) if wers else None,
            "rtf": self.rtf,
        }

    def to_dict(self) -> dict[str, object]:
        """Full sidecar payload (summary + runtime + per-chunk detail)."""
        return {
            "summary": self.summary(),
            "runtime": self.runtime,
            "notes": self.notes,
            "chunks": [c.to_dict() for c in self.chunks],
        }


def _package_version() -> str:
    try:
        from voicelegacy import __version__

        return __version__
    except ImportError:  # pragma: no cover - defensive
        return "unknown"


def _verification_from_entry(entry: Mapping[str, object]) -> ChunkVerification:
    """Reconstruct a ChunkVerification from a cache manifest entry."""
    raw = entry.get("verification")
    if not isinstance(raw, Mapping):
        return ChunkVerification(passed=True, wer=None, transcript=None, status="skipped")
    wer = raw.get("wer")
    transcript = raw.get("transcript")
    return ChunkVerification(
        passed=bool(raw.get("passed", True)),
        wer=float(wer) if isinstance(wer, int | float) else None,
        transcript=str(transcript) if isinstance(transcript, str) else None,
        status=str(raw.get("status", "skipped")),
    )


class LongFormSynthesizer:
    """Synthesize arbitrarily long text as one continuous, verified track.

    Ties together the chunker, the ASR-verify gate, the resumable cache and the
    boundary-aware assembler. Synthesis is injected as a callable
    ``(text) -> (waveform, sample_rate)`` — in production the CLI wraps the real
    XTTS model (computing the speaker conditioning once so the voice does not
    drift between chunks); in tests a mock stands in, so the whole orchestrator
    runs without a GPU.

    Robustness: each chunk is ASR-verified and, on failure (the dominant XTTS
    long-text mode is silent truncation/hallucination), re-rolled up to
    ``max_retries`` times — XTTS is stochastic, so a re-roll often fixes it. If
    every attempt fails, the lowest-WER attempt is kept and the chunk is flagged
    in the sidecar for human review: failures become visible and rare instead of
    silent.

    Retry entropy: a re-roll only helps if the sampler actually rolls new
    dice. When the injected callable re-seeds the RNG to a *fixed* value on
    every call (as ``synthesize_to_file`` does with ``SynthesisConfig.seed``,
    default 42, for reproducibility), every retry reproduces the identical
    failing audio and the retry budget is wasted. Pass
    ``synthesize_accepts_attempt=True`` and accept an ``attempt`` keyword in
    the callable to derive a per-attempt seed (e.g. ``base_seed + attempt``):
    retries become genuinely different draws while staying reproducible —
    the kept attempt's index is recorded in the sidecar as ``retries``.
    """

    def __init__(
        self,
        synthesize: SynthesizeFn,
        config: LongFormConfig | None = None,
        *,
        cache_root: Path | None = None,
        synthesize_accepts_attempt: bool = False,
    ) -> None:
        self.synthesize = synthesize
        self.config = config or LongFormConfig()
        self.cache_root = Path(cache_root) if cache_root is not None else None
        # Explicit opt-in (User Control > Automation): introspecting the
        # callable's signature would misfire on mocks and functools.partial.
        self.synthesize_accepts_attempt = synthesize_accepts_attempt

    def _call_synthesize(self, text: str, attempt: int) -> tuple[np.ndarray, int]:
        """Invoke the injected synthesizer, forwarding ``attempt`` if opted in."""
        if self.synthesize_accepts_attempt:
            return self.synthesize(text, attempt=attempt)  # type: ignore[call-arg]
        return self.synthesize(text)

    def _verify(self, audio: np.ndarray, sr: int, text: str) -> ChunkVerification:
        if not self.config.asr_verify:
            return ChunkVerification(passed=True, wer=None, transcript=None, status="skipped")
        return verify_chunk(
            audio,
            sr,
            text,
            asr_model=self.config.asr_model,
            language=self.config.language,
            max_wer=self.config.max_wer,
            fold_accents=self.config.fold_accents,
            expand_numbers=self.config.expand_numbers,
        )

    def _render_with_retry(self, chunk: Chunk) -> tuple[np.ndarray, int, ChunkVerification, int]:
        """Synthesize one chunk, retrying on ASR failure; return the best attempt."""
        best: tuple[np.ndarray, int, ChunkVerification] | None = None
        best_wer = float("inf")
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            audio, sr = self._call_synthesize(chunk.text, attempt)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            verification = self._verify(audio, sr, chunk.text)
            if verification.passed:
                return audio, sr, verification, attempt
            wer = verification.wer if verification.wer is not None else float("inf")
            if best is None or wer < best_wer:
                best, best_wer = (audio, sr, verification), wer
            logger.warning(
                "Chunk {} failed verification (WER={}); retry {}/{}.",
                chunk.index,
                None if verification.wer is None else round(verification.wer, 3),
                attempt + 1,
                self.config.max_retries,
            )
        if best is None:  # unreachable: the loop always runs at least once
            raise RuntimeError("no synthesis attempt produced audio")
        audio, sr, verification = best
        logger.warning("Chunk {} kept best attempt after {} tries; flagged.", chunk.index, attempts)
        return audio, sr, verification, attempts - 1

    def render(
        self,
        text: str,
        out_path: Path,
        *,
        resume: bool = True,
        force: bool = False,
        notes: dict[str, object] | None = None,
    ) -> LongFormResult:
        """Render ``text`` to ``out_path`` and write a sidecar audit JSON.

        Args:
            text: The full text to synthesize.
            out_path: Destination WAV path. The sidecar is written alongside it
                as ``<stem>.sidecar.json``.
            resume: Reuse cached chunks from a previous (interrupted) run.
            force: Clear the cache first and synthesize everything fresh.
            notes: Free-form metadata embedded in the sidecar (e.g. engine info).

        Returns:
            A :class:`LongFormResult`.

        Raises:
            ValueError: If ``text`` produces no chunks.
        """
        chunks = segment_text(text, max_chunk_chars=self.config.max_chunk_chars)
        if not chunks:
            raise ValueError("Text produced no chunks (empty or whitespace-only).")

        cache: LongFormCache | None = None
        if self.cache_root is not None:
            doc_hash = compute_doc_hash(text, self.config.cache_config())
            cache = LongFormCache(
                self.cache_root,
                doc_hash,
                version=_package_version(),
                config=self.config.to_dict(),
            )
            if force:
                cache.clear()

        rendered: list[RenderedChunk] = []
        audio_arrays: list[np.ndarray] = []
        synth_seconds = 0.0
        total = len(chunks)
        logger.info("Long-form render: {} chunks, asr_verify={}.", total, self.config.asr_verify)

        for chunk in chunks:
            if cache is not None and resume and cache.has_chunk(chunk.index):
                audio, sr = cache.load_audio(chunk.index)
                entry = cache.chunk_entry(chunk.index) or {}
                rendered.append(
                    RenderedChunk(
                        index=chunk.index,
                        text=chunk.text,
                        boundary=chunk.boundary,
                        char_count=chunk.char_count,
                        audio_seconds=len(audio) / sr if sr else 0.0,
                        sample_rate=sr,
                        verification=_verification_from_entry(entry),
                        retries=int(entry.get("retries", 0)),
                        from_cache=True,
                    )
                )
                audio_arrays.append(audio)
                logger.info("[{}/{}] chunk {} from cache.", chunk.index + 1, total, chunk.index)
                continue

            started = time.perf_counter()
            audio, sr, verification, retries = self._render_with_retry(chunk)
            synth_seconds += time.perf_counter() - started

            if cache is not None:
                cache.save_chunk(
                    chunk.index,
                    chunk.text,
                    chunk.boundary,
                    audio,
                    sr,
                    verification=verification,
                    retries=retries,
                )
            rendered.append(
                RenderedChunk(
                    index=chunk.index,
                    text=chunk.text,
                    boundary=chunk.boundary,
                    char_count=chunk.char_count,
                    audio_seconds=len(audio) / sr if sr else 0.0,
                    sample_rate=sr,
                    verification=verification,
                    retries=retries,
                    from_cache=False,
                )
            )
            audio_arrays.append(audio)
            logger.info("[{}/{}] chunk {} rendered.", chunk.index + 1, total, chunk.index)

        sample_rate = rendered[0].sample_rate
        final = assemble_chunks(
            audio_arrays,
            [r.boundary for r in rendered],
            sample_rate,
            crossfade_ms=self.config.crossfade_ms,
            clause_pause_ms=self.config.clause_pause_ms,
            sentence_pause_ms=self.config.sentence_pause_ms,
            paragraph_pause_ms=self.config.paragraph_pause_ms,
            trim=self.config.trim_chunks,
        )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), final, sample_rate, subtype="PCM_16")
        duration_s = len(final) / sample_rate if sample_rate else 0.0

        sidecar_path = out_path.with_suffix(".sidecar.json")
        result = LongFormResult(
            output_path=out_path,
            sidecar_path=sidecar_path,
            sample_rate=sample_rate,
            duration_s=duration_s,
            chunks=tuple(rendered),
            synth_seconds=synth_seconds,
            runtime=runtime_snapshot().to_dict(),
            notes=notes or {},
        )
        sidecar_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flagged = result.flagged_indices
        if flagged:
            logger.warning("Render flagged {} chunk(s) for review: {}", len(flagged), list(flagged))
        logger.info("Long-form render complete: {} ({:.1f}s).", out_path, duration_s)
        return result


__all__ = [
    "BOUNDARY_CLAUSE",
    "BOUNDARY_END",
    "BOUNDARY_HARD",
    "BOUNDARY_PARAGRAPH",
    "BOUNDARY_SENTENCE",
    "DEFAULT_CLAUSE_PAUSE_MS",
    "DEFAULT_CROSSFADE_MS",
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_WER",
    "DEFAULT_PARAGRAPH_PAUSE_MS",
    "DEFAULT_SENTENCE_PAUSE_MS",
    "VALID_BOUNDARIES",
    "Chunk",
    "ChunkVerification",
    "LongFormCache",
    "LongFormConfig",
    "LongFormResult",
    "LongFormSynthesizer",
    "RenderedChunk",
    "assemble_chunks",
    "compute_doc_hash",
    "segment_text",
    "verify_chunk",
]
