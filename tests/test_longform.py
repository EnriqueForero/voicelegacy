"""Tests for the long-form text chunker (``voicelegacy.longform.segment_text``).

This is the highest-risk piece of Fase 3 — a Spanish sentence/clause segmenter
has many edge cases (abbreviations, decimals, times, opening marks, ellipsis,
oversized sentences). The tests assert both the hard invariants (never exceed
budget, never split a word, preserve every word in order) and the specific
Spanish boundary behaviors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from voicelegacy.evaluation import load_golden_texts
from voicelegacy.longform import (
    BOUNDARY_CLAUSE,
    BOUNDARY_END,
    BOUNDARY_HARD,
    BOUNDARY_PARAGRAPH,
    BOUNDARY_SENTENCE,
    VALID_BOUNDARIES,
    Chunk,
    _split_clauses,
    _split_oversized_sentence,
    _split_sentences_es,
    assemble_chunks,
    segment_text,
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


# ─── Hard invariants (must hold for ANY input) ─────────────────────
class TestInvariants:
    @pytest.mark.parametrize("budget", [40, 80, 120, 220])
    def test_never_exceeds_budget_on_real_chapter(self, budget: int) -> None:
        chapter = next(t for t in load_golden_texts() if t.category == "chapter").text
        chunks = segment_text(chapter, max_chunk_chars=budget)
        assert chunks
        assert all(c.char_count <= budget for c in chunks)

    def test_preserves_every_word_in_order(self) -> None:
        chapter = next(t for t in load_golden_texts() if t.category == "chapter").text
        chunks = segment_text(chapter, max_chunk_chars=120)
        assert _words(chapter) == _words(" ".join(c.text for c in chunks))

    def test_all_chunks_nonempty_and_indexed(self) -> None:
        chunks = segment_text("Una. Dos. Tres.", max_chunk_chars=220)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert all(c.text.strip() for c in chunks)

    def test_all_boundaries_are_valid(self) -> None:
        chapter = next(t for t in load_golden_texts() if t.category == "chapter").text
        chunks = segment_text(chapter, max_chunk_chars=100)
        assert all(c.boundary in VALID_BOUNDARIES for c in chunks)

    def test_never_splits_mid_word_for_normal_text(self) -> None:
        # Every chunk must start and end on a word/punctuation boundary, i.e.
        # splitting never lands inside a run of word characters.
        text = "Palabra " * 200  # 200 repetitions, forces many splits
        chunks = segment_text(text.strip(), max_chunk_chars=50)
        for chunk in chunks:
            assert chunk.text == chunk.text.strip()
            # no chunk should start or end with a partial "Palabra"
            for token in chunk.text.split():
                assert token in {"Palabra"}  # intact tokens only


# ─── Degenerate inputs ─────────────────────────────────────────────
class TestDegenerate:
    @pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t  \n"])
    def test_empty_or_whitespace_returns_empty(self, text: str) -> None:
        assert segment_text(text) == []

    def test_invalid_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            segment_text("hola", max_chunk_chars=0)

    def test_single_short_sentence_is_one_end_chunk(self) -> None:
        chunks = segment_text("Hola abuela.", max_chunk_chars=220)
        assert len(chunks) == 1
        assert chunks[0].boundary == BOUNDARY_END
        assert chunks[0].text == "Hola abuela."


# ─── Sentence-level boundaries ─────────────────────────────────────
class TestSentenceBoundaries:
    def test_two_sentences(self) -> None:
        chunks = segment_text("Vino el lunes. Se fue el martes.", max_chunk_chars=220)
        assert len(chunks) == 2
        assert chunks[0].boundary == BOUNDARY_SENTENCE
        assert chunks[1].boundary == BOUNDARY_END

    def test_paragraph_boundary(self) -> None:
        text = "Primer parrafo aqui.\n\nSegundo parrafo aqui."
        chunks = segment_text(text, max_chunk_chars=220)
        assert len(chunks) == 2
        assert chunks[0].boundary == BOUNDARY_PARAGRAPH
        assert chunks[1].boundary == BOUNDARY_END

    def test_question_and_exclamation_with_opening_marks(self) -> None:
        chunks = segment_text("¿Vienes hoy? ¡Qué alegría! Te espero.", max_chunk_chars=220)
        assert [c.text for c in chunks] == ["¿Vienes hoy?", "¡Qué alegría!", "Te espero."]

    def test_ellipsis_is_a_single_boundary(self) -> None:
        chunks = segment_text("Bueno... ya veremos.", max_chunk_chars=220)
        assert [c.text for c in chunks] == ["Bueno...", "ya veremos."]

    def test_closing_quote_is_absorbed(self) -> None:
        chunks = segment_text("Dijo «hola». Luego se fue.", max_chunk_chars=220)
        assert chunks[0].text == "Dijo «hola»."
        assert chunks[1].text == "Luego se fue."


# ─── Spanish false-boundary traps (the hard part) ──────────────────
class TestFalseBoundaries:
    def test_abbreviation_period_does_not_split(self) -> None:
        # "Sr." and "Dra." must not be treated as sentence ends.
        s = _split_sentences_es("El Sr. Pérez visitó a la Dra. Gómez. Fue cordial.")
        assert s == ["El Sr. Pérez visitó a la Dra. Gómez.", "Fue cordial."]

    def test_etc_ends_sentence_and_splits(self) -> None:
        # "etc." is treated as a real sentence end (it usually is), unlike the
        # honorific abbreviations below. Documented design choice in longform.py.
        s = _split_sentences_es("Trajo pan, queso, etc. Todo fresco.")
        assert s == ["Trajo pan, queso, etc.", "Todo fresco."]

    def test_thousands_separator_period_does_not_split(self) -> None:
        s = _split_sentences_es("Costó 1.000 pesos. Fue barato.")
        assert s == ["Costó 1.000 pesos.", "Fue barato."]

    def test_url_does_not_split(self) -> None:
        s = _split_sentences_es("Visita www.ejemplo.com hoy. Es gratis.")
        assert s == ["Visita www.ejemplo.com hoy.", "Es gratis."]

    def test_decimal_comma_not_broken_in_clauses(self) -> None:
        assert _split_clauses("cuesta 3,5 millones") == ["cuesta 3,5 millones"]

    def test_time_colon_not_broken_in_clauses(self) -> None:
        assert _split_clauses("a las 14:30 salimos") == ["a las 14:30 salimos"]

    def test_normal_clauses_still_split(self) -> None:
        assert _split_clauses("vino, vio y venció") == ["vino,", "vio y venció"]


# ─── Oversized sentences ───────────────────────────────────────────
class TestOversized:
    def test_oversized_sentence_splits_on_clauses_within_budget(self) -> None:
        sentence = (
            "Mi abuelo llegó a la ciudad en el invierno de 1958, "
            "trabajó en una ferretería del centro durante años, "
            "ahorró peso sobre peso cada domingo en la cocina, "
            "y al final abrió su propio local en el barrio Restrepo."
        )
        chunks = segment_text(sentence, max_chunk_chars=80)
        assert all(c.char_count <= 80 for c in chunks)
        assert _words(sentence) == _words(" ".join(c.text for c in chunks))
        # internal joins are clause-level; the last inherits "end"
        assert chunks[-1].boundary == BOUNDARY_END
        assert all(c.boundary in (BOUNDARY_CLAUSE, BOUNDARY_HARD) for c in chunks[:-1])

    def test_oversized_clause_with_no_commas_splits_on_whitespace_as_hard(self) -> None:
        # One long clause, no delimiters → whitespace split tagged "hard".
        clause = "palabra " * 40  # 40 words, no punctuation
        subs = _split_oversized_sentence(clause.strip(), budget=50)
        assert all(len(t) <= 50 for t, _b in subs)
        assert any(b == BOUNDARY_HARD for _t, b in subs)

    def test_single_token_longer_than_budget_is_hard_sliced(self) -> None:
        giant = "x" * 300  # pathological unsplittable token
        chunks = segment_text(giant, max_chunk_chars=100)
        assert all(c.char_count <= 100 for c in chunks)
        assert "".join(c.text for c in chunks) == giant  # reassembles exactly

    def test_chapter_has_no_overlong_chunks_at_tight_budget(self) -> None:
        chapter = next(t for t in load_golden_texts() if t.category == "chapter").text
        chunks = segment_text(chapter, max_chunk_chars=60)
        assert all(c.char_count <= 60 for c in chunks)


# ─── Chunk value object ────────────────────────────────────────────
def test_chunk_char_count() -> None:
    assert Chunk(0, "hola", BOUNDARY_END).char_count == 4


# ─── Audio assembly (boundary-driven joins) ────────────────────────
class TestAssembleChunks:
    SR = 24000

    def _chunk(self, seconds: float, amp: float = 0.3) -> np.ndarray:
        return np.full(int(self.SR * seconds), amp, dtype=np.float32)

    def test_empty_returns_empty(self) -> None:
        assert assemble_chunks([], [], self.SR).size == 0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            assemble_chunks([self._chunk(0.5)], [BOUNDARY_END, BOUNDARY_END], self.SR)

    def test_single_chunk_returned(self) -> None:
        c = self._chunk(0.5)
        out = assemble_chunks([c], [BOUNDARY_END], self.SR, trim=False)
        assert len(out) == len(c)

    def test_sentence_boundary_inserts_gap(self) -> None:
        c1, c2 = self._chunk(0.5), self._chunk(0.5)
        gap = round(350 / 1000 * self.SR)
        out = assemble_chunks(
            [c1, c2], [BOUNDARY_SENTENCE, BOUNDARY_END], self.SR, trim=False, sentence_pause_ms=350
        )
        assert len(out) == len(c1) + gap + len(c2)

    def test_paragraph_gap_longer_than_sentence_gap(self) -> None:
        c1, c2 = self._chunk(0.5), self._chunk(0.5)
        para = assemble_chunks(
            [c1, c2],
            [BOUNDARY_PARAGRAPH, BOUNDARY_END],
            self.SR,
            trim=False,
            sentence_pause_ms=350,
            paragraph_pause_ms=700,
        )
        sent = assemble_chunks(
            [c1, c2],
            [BOUNDARY_SENTENCE, BOUNDARY_END],
            self.SR,
            trim=False,
            sentence_pause_ms=350,
            paragraph_pause_ms=700,
        )
        assert len(para) > len(sent)

    def test_clause_gap_shorter_than_sentence_gap(self) -> None:
        c1, c2 = self._chunk(0.5), self._chunk(0.5)
        clause = assemble_chunks(
            [c1, c2],
            [BOUNDARY_CLAUSE, BOUNDARY_END],
            self.SR,
            trim=False,
            clause_pause_ms=150,
            sentence_pause_ms=350,
        )
        sent = assemble_chunks(
            [c1, c2],
            [BOUNDARY_SENTENCE, BOUNDARY_END],
            self.SR,
            trim=False,
            clause_pause_ms=150,
            sentence_pause_ms=350,
        )
        assert len(clause) < len(sent)

    def test_hard_boundary_uses_crossfade_not_gap(self) -> None:
        c1, c2 = self._chunk(0.5), self._chunk(0.5)
        overlap = round(15 / 1000 * self.SR)
        out = assemble_chunks(
            [c1, c2], [BOUNDARY_HARD, BOUNDARY_END], self.SR, trim=False, crossfade_ms=15
        )
        # crossfade overlaps, so the result is SHORTER than the two chunks summed
        assert len(out) == len(c1) + len(c2) - overlap

    def test_trim_strips_silence_before_joining(self) -> None:
        # A chunk padded with leading/trailing silence is trimmed before joining,
        # so the assembled length is much less than the raw padded sum.
        body = self._chunk(0.5)
        pad = np.zeros(int(self.SR * 0.4), dtype=np.float32)
        padded = np.concatenate([pad, body, pad])
        out_trim = assemble_chunks(
            [padded, padded], [BOUNDARY_SENTENCE, BOUNDARY_END], self.SR, trim=True
        )
        out_raw = assemble_chunks(
            [padded, padded], [BOUNDARY_SENTENCE, BOUNDARY_END], self.SR, trim=False
        )
        assert len(out_trim) < len(out_raw)

    def test_three_chunks_full_chain(self) -> None:
        chunks = [self._chunk(0.3), self._chunk(0.3), self._chunk(0.3)]
        boundaries = [BOUNDARY_CLAUSE, BOUNDARY_SENTENCE, BOUNDARY_END]
        out = assemble_chunks(chunks, boundaries, self.SR, trim=False)
        assert out.dtype == np.float32
        assert len(out) > sum(len(c) for c in chunks)  # gaps added


# ─── ASR verification gate (verify_chunk) ──────────────────────────
class TestVerifyChunk:
    SR = 24000

    def _audio(self) -> np.ndarray:
        return np.full(self.SR, 0.2, dtype=np.float32)

    def test_matching_transcript_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voicelegacy.longform as lf

        # Transcript matches the requested text modulo case/punctuation (what the
        # normalizer folds). Accents are kept consistent on both sides — with
        # fold_accents=False they are significant by design.
        monkeypatch.setattr(lf, "transcribe", lambda *a, **k: "hola abuela vino el lunes")
        v = lf.verify_chunk(self._audio(), self.SR, "Hola, abuela. Vino el lunes.", max_wer=0.15)
        assert v.status == "ok"
        assert v.passed is True
        assert v.wer == pytest.approx(0.0)

    def test_truncated_transcript_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voicelegacy.longform as lf

        # ASR heard only the first two words of a five-word chunk → high WER.
        monkeypatch.setattr(lf, "transcribe", lambda *a, **k: "hola abuela")
        v = lf.verify_chunk(self._audio(), self.SR, "Hola abuela como estas hoy", max_wer=0.15)
        assert v.status == "ok"
        assert v.passed is False
        assert v.wer is not None and v.wer > 0.15

    def test_missing_asr_skips_and_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voicelegacy.longform as lf

        def _raise_import(*a, **k):
            raise ImportError("faster-whisper not installed")

        monkeypatch.setattr(lf, "transcribe", _raise_import)
        v = lf.verify_chunk(self._audio(), self.SR, "cualquier cosa")
        assert v.status == "skipped"
        assert v.passed is True  # cannot verify → do not block
        assert v.wer is None

    def test_asr_error_does_not_discard_audio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voicelegacy.longform as lf

        def _raise_runtime(*a, **k):
            raise RuntimeError("decoder blew up")

        monkeypatch.setattr(lf, "transcribe", _raise_runtime)
        v = lf.verify_chunk(self._audio(), self.SR, "texto")
        assert v.status == "error"
        assert v.passed is True

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import voicelegacy.longform as lf

        monkeypatch.setattr(lf, "transcribe", lambda *a, **k: "uno dos tres")
        d = lf.verify_chunk(self._audio(), self.SR, "uno dos tres").to_dict()
        assert d["passed"] is True and d["status"] == "ok" and d["wer"] == 0.0


# ─── Checkpoint / resume cache ─────────────────────────────────────
class TestCheckpointCache:
    SR = 24000

    def _audio(self, amp: float = 0.3) -> np.ndarray:
        return np.full(self.SR, amp, dtype=np.float32)

    def test_doc_hash_is_order_independent(self) -> None:
        from voicelegacy.longform import compute_doc_hash

        a = compute_doc_hash("texto", {"max_chunk_chars": 220, "language": "es"})
        b = compute_doc_hash("texto", {"language": "es", "max_chunk_chars": 220})
        assert a == b

    def test_doc_hash_changes_with_text_and_config(self) -> None:
        from voicelegacy.longform import compute_doc_hash

        base = compute_doc_hash("texto", {"max_chunk_chars": 220})
        assert base != compute_doc_hash("otro texto", {"max_chunk_chars": 220})
        assert base != compute_doc_hash("texto", {"max_chunk_chars": 200})

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        cache = LongFormCache(tmp_path, "hash1")
        cache.save_chunk(0, "Hola.", BOUNDARY_SENTENCE, self._audio(0.4), self.SR)
        assert cache.has_chunk(0)
        audio, sr = cache.load_audio(0)
        assert sr == self.SR
        assert len(audio) == self.SR
        assert audio[0] == pytest.approx(0.4, abs=1e-3)

    def test_resume_sees_saved_chunks_after_restart(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        first = LongFormCache(tmp_path, "h")
        first.save_chunk(0, "a", BOUNDARY_SENTENCE, self._audio(), self.SR)
        first.save_chunk(1, "b", BOUNDARY_END, self._audio(), self.SR)
        # Simulate a fresh process: a new instance over the same directory.
        resumed = LongFormCache(tmp_path, "h")
        assert resumed.completed_indices() == {0, 1}
        assert resumed.has_chunk(0) and resumed.has_chunk(1)
        assert not resumed.has_chunk(2)  # orchestrator would only synth chunk 2

    def test_different_hash_is_a_fresh_cache(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        LongFormCache(tmp_path, "old").save_chunk(0, "a", BOUNDARY_END, self._audio(), self.SR)
        # A changed text/config → different hash → nothing cached (no stale reuse).
        fresh = LongFormCache(tmp_path, "new")
        assert fresh.completed_indices() == set()

    def test_has_chunk_false_when_wav_deleted(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        cache = LongFormCache(tmp_path, "h")
        path = cache.save_chunk(0, "a", BOUNDARY_END, self._audio(), self.SR)
        path.unlink()  # wav gone but manifest entry remains
        assert not cache.has_chunk(0)

    def test_load_missing_chunk_raises(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        with pytest.raises(KeyError):
            LongFormCache(tmp_path, "h").load_audio(99)

    def test_clear_removes_everything(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        cache = LongFormCache(tmp_path, "h")
        cache.save_chunk(0, "a", BOUNDARY_END, self._audio(), self.SR)
        cache.clear()
        assert not (tmp_path / "h").exists()
        assert cache.completed_indices() == set()

    def test_verification_is_persisted_in_manifest(self, tmp_path) -> None:
        from voicelegacy.longform import ChunkVerification, LongFormCache

        cache = LongFormCache(tmp_path, "h")
        v = ChunkVerification(passed=True, wer=0.05, transcript="hola", status="ok")
        cache.save_chunk(
            0, "Hola.", BOUNDARY_END, self._audio(), self.SR, verification=v, retries=2
        )
        entry = LongFormCache(tmp_path, "h").chunk_entry(0)
        assert entry["retries"] == 2
        assert entry["verification"]["wer"] == 0.05
        assert entry["verification"]["passed"] is True

    def test_corrupt_manifest_starts_fresh(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormCache

        (tmp_path / "h").mkdir()
        (tmp_path / "h" / "manifest.json").write_text("{ not valid json", encoding="utf-8")
        cache = LongFormCache(tmp_path, "h")  # must not raise
        assert cache.completed_indices() == set()


# ─── LongFormConfig ────────────────────────────────────────────────
class TestLongFormConfig:
    def test_defaults_and_to_dict(self) -> None:
        from voicelegacy.longform import LongFormConfig

        cfg = LongFormConfig()
        assert cfg.asr_verify is True
        assert cfg.to_dict()["max_chunk_chars"] == cfg.max_chunk_chars

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_chunk_chars": 0},
            {"max_retries": -1},
            {"max_wer": -0.1},
            {"sentence_pause_ms": -5},
            {"asr_model": "  "},
            {"language": ""},
        ],
    )
    def test_invalid_config_raises(self, kwargs) -> None:
        from voicelegacy.longform import LongFormConfig

        with pytest.raises(ValueError):
            LongFormConfig(**kwargs)

    def test_cache_config_excludes_assembly_only_knobs(self) -> None:
        from voicelegacy.longform import LongFormConfig

        keys = LongFormConfig().cache_config()
        # pause/crossfade only affect assembly → must NOT invalidate the cache
        assert "crossfade_ms" not in keys
        assert "sentence_pause_ms" not in keys
        assert "max_chunk_chars" in keys and "language" in keys


# ─── LongFormSynthesizer (orchestrator) ────────────────────────────
def _synth_factory(seconds: float = 0.4, sr: int = 24000):
    calls: list[str] = []

    def _synth(text: str):
        calls.append(text)
        return np.full(int(sr * seconds), 0.2, dtype=np.float32), sr

    return _synth, calls


def _verify_sequence(seq):
    """Build a verify_chunk stand-in returning the given (passed, wer) pairs."""
    from voicelegacy.longform import ChunkVerification

    it = iter(seq)

    def _verify(audio, sr, text, **kwargs):
        passed, wer = next(it)
        return ChunkVerification(passed=passed, wer=wer, transcript="t", status="ok")

    return _verify


class TestLongFormSynthesizer:
    SR = 24000

    def test_empty_text_raises(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, _ = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False))
        with pytest.raises(ValueError, match="no chunks"):
            s.render("   ", tmp_path / "o.wav")

    def test_basic_render_writes_wav_and_sidecar(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, calls = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False))
        res = s.render("Hola abuela. ¿Cómo estás? Vino el lunes.", tmp_path / "o.wav")
        assert res.output_path.exists()
        assert res.sidecar_path.exists()
        assert res.n_chunks == len(calls) == 3
        assert res.duration_s > 0
        loaded = json.loads(res.sidecar_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["n_chunks"] == 3
        assert len(loaded["chunks"]) == 3

    def test_asr_verify_disabled_marks_skipped(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, _ = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False))
        res = s.render("Hola.", tmp_path / "o.wav")
        assert res.chunks[0].verification.status == "skipped"
        assert res.flagged_indices == ()

    def test_retry_then_pass(self, tmp_path, monkeypatch) -> None:
        import voicelegacy.longform as lf

        synth, calls = _synth_factory()
        # single chunk: fail first attempt, pass second
        monkeypatch.setattr(lf, "verify_chunk", _verify_sequence([(False, 0.5), (True, 0.05)]))
        s = lf.LongFormSynthesizer(synth, lf.LongFormConfig(asr_verify=True, max_retries=2))
        res = s.render("Hola abuela.", tmp_path / "o.wav")
        assert len(calls) == 2  # synthesized twice
        assert res.chunks[0].retries == 1
        assert res.chunks[0].flagged is False

    def test_all_attempts_fail_flags_and_keeps_best(self, tmp_path, monkeypatch) -> None:
        import voicelegacy.longform as lf

        synth, calls = _synth_factory()
        # 3 attempts (max_retries=2), all fail with different WERs; best is 0.3
        monkeypatch.setattr(
            lf, "verify_chunk", _verify_sequence([(False, 0.5), (False, 0.3), (False, 0.6)])
        )
        s = lf.LongFormSynthesizer(synth, lf.LongFormConfig(asr_verify=True, max_retries=2))
        res = s.render("Hola abuela.", tmp_path / "o.wav")
        assert len(calls) == 3
        assert res.chunks[0].flagged is True
        assert res.chunks[0].retries == 2
        assert res.chunks[0].verification.wer == pytest.approx(0.3)  # kept lowest-WER attempt
        assert res.flagged_indices == (0,)

    def test_resume_skips_cached_chunks(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, calls = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False), cache_root=tmp_path / "c")
        text = "Hola abuela. ¿Cómo estás? Vino el lunes."
        s.render(text, tmp_path / "o.wav")
        assert len(calls) == 3
        calls.clear()
        res2 = s.render(text, tmp_path / "o.wav")  # resume
        assert len(calls) == 0  # nothing re-synthesized
        assert all(c.from_cache for c in res2.chunks)

    def test_force_clears_cache_and_resynthesizes(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, calls = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False), cache_root=tmp_path / "c")
        text = "Hola abuela. Vino el lunes."
        s.render(text, tmp_path / "o.wav")
        n_first = len(calls)
        calls.clear()
        s.render(text, tmp_path / "o.wav", force=True)
        assert len(calls) == n_first  # all re-synthesized

    def test_changed_text_does_not_reuse_cache(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, calls = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False), cache_root=tmp_path / "c")
        s.render("Hola abuela.", tmp_path / "o.wav")
        calls.clear()
        s.render("Texto completamente distinto.", tmp_path / "o2.wav")  # different hash
        assert len(calls) > 0  # fresh synthesis, no stale reuse

    def test_works_without_cache_root(self, tmp_path) -> None:
        from voicelegacy.longform import LongFormConfig, LongFormSynthesizer

        synth, _ = _synth_factory()
        s = LongFormSynthesizer(synth, LongFormConfig(asr_verify=False))  # no cache_root
        res = s.render("Hola abuela. Vino el lunes.", tmp_path / "o.wav")
        assert res.output_path.exists()


# ─── LongFormResult ────────────────────────────────────────────────
class TestLongFormResult:
    def test_rtf_and_summary(self) -> None:
        from voicelegacy.longform import ChunkVerification, LongFormResult, RenderedChunk

        v = ChunkVerification(passed=True, wer=0.1, transcript="t", status="ok")
        chunks = (
            RenderedChunk(0, "a", "sentence", 1, 2.0, 24000, v, 0, from_cache=False),
            RenderedChunk(1, "b", "end", 1, 2.0, 24000, v, 0, from_cache=True),
        )
        res = LongFormResult(
            output_path=Path("/tmp/o.wav"),
            sidecar_path=Path("/tmp/o.sidecar.json"),
            sample_rate=24000,
            duration_s=4.5,
            chunks=chunks,
            synth_seconds=1.0,
            runtime={"cuda_available": False},
        )
        # rtf is over synthesized chunks only (chunk 1 was cached): 1.0 / 2.0 = 0.5
        assert res.rtf == pytest.approx(0.5)
        smry = res.summary()
        assert smry["n_chunks"] == 2
        assert smry["n_from_cache"] == 1
        assert smry["mean_wer"] == pytest.approx(0.1)


# ─── Retry entropy: attempt forwarding (v0.7.1) ─────────────────────
class TestAttemptForwarding:
    def test_opt_in_forwards_incrementing_attempt(self, tmp_path, monkeypatch) -> None:
        """With synthesize_accepts_attempt=True the callable receives the
        0-based attempt index, enabling per-attempt seed derivation so
        retries stop reproducing the identical (seeded) failing audio."""
        import voicelegacy.longform as lf

        attempts_seen: list[int] = []

        def synth(text: str, attempt: int = 0):
            attempts_seen.append(attempt)
            sr = 24000
            return (0.1 * np.ones(sr // 2, dtype=np.float32), sr)

        monkeypatch.setattr(
            lf, "verify_chunk", _verify_sequence([(False, 0.5), (False, 0.4), (True, 0.05)])
        )
        s = lf.LongFormSynthesizer(
            synth,
            lf.LongFormConfig(asr_verify=True, max_retries=2),
            synthesize_accepts_attempt=True,
        )
        res = s.render("Hola abuela.", tmp_path / "o.wav")

        assert attempts_seen == [0, 1, 2]
        assert res.chunks[0].retries == 2

    def test_default_never_passes_attempt_kwarg(self, tmp_path, monkeypatch) -> None:
        """Backward compatibility: a plain (text)->audio callable must keep
        working untouched when the flag is left at its default."""
        import voicelegacy.longform as lf

        def synth_strict(text: str):  # would raise TypeError on any kwarg
            sr = 24000
            return (0.1 * np.ones(sr // 2, dtype=np.float32), sr)

        monkeypatch.setattr(lf, "verify_chunk", _verify_sequence([(False, 0.5), (True, 0.05)]))
        s = lf.LongFormSynthesizer(synth_strict, lf.LongFormConfig(asr_verify=True, max_retries=2))
        res = s.render("Hola abuela.", tmp_path / "o.wav")

        assert res.chunks[0].retries == 1
        assert res.chunks[0].flagged is False
