# Long-form synthesis (`voicelegacy.longform`)

## The problem

XTTS-v2 derails past roughly 250–270 characters per generation in Spanish: it
truncates the sentence, repeats, or hallucinates. A chapter of a few thousand
characters is therefore many short generations that must be (a) cut well,
(b) synthesized without the voice drifting between pieces, (c) verified, and
(d) joined without audible seams. `LongFormSynthesizer` does all four — and,
crucially, **makes failures visible and rare instead of silent**.

This is the output-side counterpart to the benchmark: the benchmark *measures*
long-text failure (high WER on the `long`/`chapter` stimuli); this *fixes* it.

## Architecture

Synthesis is **injected** as a callable `(text) -> (waveform, sample_rate)`. The
orchestrator never imports a TTS engine: in production the CLI wraps the real
XTTS model, in tests a mock stands in, so the whole pipeline runs without a GPU.
The module reuses existing code rather than duplicating it: the WER/ASR come from
`voicelegacy.evaluation` (the harness), the crossfade/trim/loudness from
`voicelegacy.audio`.

Pipeline, per render:

1. **Chunk** the text at linguistic boundaries under the character budget.
2. **Compute speaker conditioning once** (handled inside `synthesis.py`, which
   caches the XTTS conditioning latents keyed by the reference set) so the timbre
   stays consistent across chunks.
3. For each chunk, in order:
   - if cached and `resume`, load it (this is the checkpoint);
   - else synthesize, **ASR-verify**, and **retry** on failure;
   - persist the chunk audio + verification to the cache.
4. **Assemble** the chunks: equal-power crossfade at forced mid-clause splits,
   silence gaps at clause/sentence/paragraph boundaries.
5. Write the WAV and a **sidecar** audit JSON.

## The chunker (`segment_text`)

Hierarchy of split points: paragraph → sentence → clause → whitespace, never
mid-word. Spanish-specific traps it handles: opening marks `¿` `¡`, abbreviations
with internal periods (`Sr.`, `Dra.`, `EE. UU.`) that are not sentence ends,
decimals (`3,5`) and thousands separators, ellipsis (`…`/`...`), times (`14:30`).
Each chunk carries the boundary that follows it, which the assembler uses to
decide crossfade vs pause. Hard invariants (tested against the real golden
chapter at several budgets): never exceed the budget, never split a word,
preserve every word in order.

## Robustness: ASR-verify + retry

The dominant XTTS long-text failure is *silent* truncation/hallucination. After
synthesizing a chunk, `verify_chunk` transcribes it with faster-whisper and
computes WER against the requested text (reusing the harness WER + normalizer).
If WER exceeds `max_wer`, the chunk is **re-rolled** up to `max_retries` times —
XTTS is stochastic, so a re-roll often fixes it. If every attempt fails, the
lowest-WER attempt is kept and the chunk is **flagged** in the sidecar.

Degrades safely: without faster-whisper, verification is `skipped` and chunks are
accepted (a retry would not add ASR); an ASR error never discards otherwise-fine
audio.

This does **not** guarantee perfect long audio. It guarantees that failures are
rare (most are fixed by a re-roll) and *visible* (the rest are flagged for review)
rather than buried in a long file.

## Seamless assembly

Each chunk is trimmed of XTTS's variable leading/trailing silence first, so
pauses are deterministic. Then:

- **forced mid-clause splits** (`hard` boundary) → equal-power crossfade (~15 ms).
  Equal-power (cos/sin) keeps perceived loudness constant; a linear crossfade
  dips ~3 dB at the midpoint, audible as a little dip at every seam.
- **clause / sentence / paragraph** boundaries → increasing silence gaps. These
  joins are the majority, and a real pause sounds better than a crossfade there.

Pause durations are a prosody choice the benchmark does **not** validate (it
measures intelligibility and similarity, not pacing) — tune them by ear.

## Checkpoint / resume

`LongFormCache` stores each rendered chunk as `<cache_root>/<doc_hash>/chunk_NNNN.wav`
plus a `manifest.json`. The directory **is** the hash of (text + the config that
affects chunk audio), so changing the text or a synthesis knob yields a new
directory and no stale audio is reused. On a resumed run the orchestrator skips
chunks already present, so an interrupted Colab session (12 h limit, or a random
disconnect) loses at most the chunk in flight. Pause/crossfade knobs are excluded
from the hash: changing a pause re-assembles from the same cache instead of
re-synthesizing everything.

## Usage

```python
from voicelegacy import LongFormSynthesizer, LongFormConfig

synth = LongFormSynthesizer(synthesize_fn, LongFormConfig(asr_verify=True),
                            cache_root="workspace/synthesis_out/longform_cache")
result = synth.render(text, "chapter_01.wav", resume=True)
print(result.summary())          # duration, n_chunks, flagged, RTF, mean WER
print(result.flagged_indices)    # chunks to review
```

CLI:

```bash
voicelegacy synthesize-long --workspace WS --out chapter_01.wav \
  --text-file chapter_01.txt --resume --accept-tos
```

Outputs `chapter_01.wav` and `chapter_01.sidecar.json` (per-chunk WER, retries,
flagged chunks, RTF, VRAM, config, version).

## Validation (run this on a T4)

Code correctness is covered by unit tests with a mock synthesizer (no GPU). The
*product* claim — that long audio is now continuous and intelligible — is proven
with the benchmark, not asserted:

1. Synthesize the `long` and `chapter` golden stimuli via `synthesize-long`.
2. Re-run `voicelegacy benchmark` and compare WER against the frozen baseline.
3. The expected result is a large WER drop on those categories versus the
   zero-shot baseline. That number is yours to produce — it needs the XTTS
   weights, a GPU, and your reference corpus.

## Known limitation (deliberate, not an oversight)

Persistent-failure **re-chunking** (splitting a chunk that fails every retry into
smaller pieces) is not implemented in this version. Re-chunking mid-render breaks
the cache's index model, and the retry loop already handles the common stochastic
failures. Shipping it half-done would be technical debt; it is left as a future
refinement. The current behavior — retry, then flag the irrecoverable — is
complete and honest.
