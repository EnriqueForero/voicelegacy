"""voicelegacy — Command-line interface.

Provides a minimal CLI for running the pipeline outside of a notebook:

    voicelegacy build-corpus  --workspace /path/to/ws --speaker SPEAKER_00
    voicelegacy synthesize    --workspace /path/to/ws --text "Hola mi nieto."
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from voicelegacy.audio import _ffmpeg_available, convert_directory_to_wav
from voicelegacy.config import (
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.denoise_eval import evaluate_denoise_methods
from voicelegacy.diagnose import diagnose_workspace
from voicelegacy.logging_config import configure_logging
from voicelegacy.pipeline import (
    run_batch_synthesis,
    run_reference_phase,
)
from voicelegacy.quality import select_reference_wavs
from voicelegacy.text_inputs import resolve_text_inputs

app = typer.Typer(help="Voice cloning pipeline for family legacy — XTTS-v2 + speakerscribe.")
console = Console()


def _config_for_attempt(config: SynthesisConfig, attempt: int) -> SynthesisConfig:
    """Derive the per-attempt SynthesisConfig for long-form retry re-rolls.

    ``synthesize_to_file`` re-seeds the RNG to ``config.seed`` before every
    inference, so with a fixed seed every retry reproduces the exact same
    failing audio and the retry budget is wasted. Deriving ``seed + attempt``
    makes each re-roll a genuinely different draw while remaining fully
    reproducible: the sidecar records which attempt was kept (``retries``),
    so ``base_seed + retries`` recreates the published audio byte-for-byte.

    Args:
        config: Base synthesis config.
        attempt: 0-based attempt index (0 = first try).

    Returns:
        The same object for attempt 0 or when seeding is disabled; otherwise
        a copy with ``seed = base_seed + attempt``.
    """
    if attempt == 0 or config.seed is None:
        return config
    return config.model_copy(update={"seed": config.seed + attempt})


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging.")) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")


@app.command("build-corpus")
def build_corpus(
    workspace: Path = typer.Option(..., help="Workspace root containing speakerscribe_out/."),
    speaker: str = typer.Option("SPEAKER_00", help="Target speaker label."),
    top_n: int = typer.Option(10, help="Keep top N best-scoring segments."),
    min_dur: float = typer.Option(4.0, help="Min segment duration (s)."),
    max_dur: float = typer.Option(15.0, help="Max segment duration (s)."),
    min_snr: float = typer.Option(15.0, help="Min SNR in dB."),
    no_denoise: bool = typer.Option(False, "--no-denoise", help="Skip noise reduction."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if outputs exist."),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
) -> None:
    """Build the reference corpus from speakerscribe outputs."""
    paths = WorkspacePaths(workspace=workspace)
    config = PipelineConfig(
        reference=ReferenceConfig(
            target_speaker_label=speaker,
            top_n_segments=top_n,
            min_segment_duration_s=min_dur,
            max_segment_duration_s=max_dur,
            min_snr_db=min_snr,
            apply_denoise=not no_denoise,
        ),
        force_rebuild_reference=force,
        accept_coqui_tos=accept_tos,
    )

    result = run_reference_phase(paths, config)

    table = Table(title="Reference corpus summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total candidates", str(len(result.all_wavs)))
    table.add_row("Passing", str(sum(1 for r in result.reports if r.passed)))
    table.add_row("Top selected", str(len(result.top_wavs)))
    console.print(table)


@app.command("synthesize")
def synthesize(
    workspace: Path = typer.Option(...),
    text: str | None = typer.Option(None, help="Text to vocalize."),
    text_file: Path | None = typer.Option(
        None, "--text-file", help=".txt one utterance per line, or .csv with a text column."
    ),
    language: str = typer.Option("es", help="Language ISO code."),
    top_n: int | None = typer.Option(
        None,
        "--top-n",
        help=(
            "Use only the N best-quality reference WAVs for voice conditioning "
            "(default: ReferenceConfig.top_n_segments = 5). 0 = legacy: all "
            "WAVs, alphabetical."
        ),
    ),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
    force: bool = typer.Option(False, "--force", help="Re-run even if cached."),
) -> None:
    """Synthesize one or more texts using the existing reference corpus."""
    paths = WorkspacePaths(workspace=workspace)
    top_wavs = select_reference_wavs(paths.reference_corpus, top_n=top_n)
    if not top_wavs:
        console.print("[red]No reference WAVs found. Run 'build-corpus' first.[/red]")
        raise typer.Exit(code=2)

    try:
        texts = resolve_text_inputs(text, text_file)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    config = PipelineConfig(
        synthesis=SynthesisConfig(language=language),  # type: ignore[arg-type]
        force_resynthesize=force,
        accept_coqui_tos=accept_tos,
    )

    results = run_batch_synthesis(texts, top_wavs, paths, config)
    for r in results:
        marker = "♻️ " if r.cached else "✨"
        sidecar = f"  metadata={r.metadata_path}" if r.metadata_path else ""
        # soft_wrap=True keeps file paths on a single logical line so output
        # snapshots and CLI test assertions don't break when the rendering
        # width is narrow (e.g. typer.testing.CliRunner uses 80 cols).
        console.print(f"{marker} {r.output_path}{sidecar}", soft_wrap=True)
        if r.similarity_score is not None:
            console.print(
                f"   speaker_similarity_score={r.similarity_score:.3f}",
                soft_wrap=True,
            )


@app.command("diagnose")
def diagnose(
    workspace: Path = typer.Option(..., help="Workspace root to inspect."),
    require_gpu: bool = typer.Option(
        False, "--require-gpu", help="Fail the CUDA check if no GPU is available."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Diagnose local installation and workspace readiness without guessing."""
    import json

    report = diagnose_workspace(workspace, require_gpu=require_gpu)
    if json_output:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        table = Table(title=f"voicelegacy diagnose — {workspace}")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail", overflow="fold")
        table.add_column("Remediation", overflow="fold")
        for check in report.checks:
            style = {"ok": "green", "warn": "yellow", "fail": "red"}[check.status]
            table.add_row(
                check.name,
                f"[{style}]{check.status.upper()}[/{style}]",
                check.detail,
                check.remediation or "",
            )
        console.print(table)
        console.print(
            f"ready={report.ready} | failures={report.failed} | warnings={report.warnings}"
        )
    if not report.ready:
        raise typer.Exit(code=2)


@app.command("convert-audio")
def convert_audio(
    workspace: Path = typer.Option(
        ..., help="Workspace root. Containers in workspace/interviews_raw/ get converted."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-encode even if .wav exists."),
    sample_rate: int = typer.Option(22050, help="Target sample rate for the output WAV."),
) -> None:
    """Convert mp4/m4a/mkv/webm/aac/mov/mp3/ogg/flac in interviews_raw/ to WAV.

    Reemplaza la Cell 17 hand-edited del notebook. Idempotent: skips files
    whose .wav twin already exists unless --overwrite is set.
    """
    paths = WorkspacePaths(workspace=workspace)
    if not paths.interviews_raw.is_dir():
        console.print(f"[red]interviews_raw/ not found at {paths.interviews_raw}[/red]")
        raise typer.Exit(code=2)

    if not _ffmpeg_available():
        console.print(
            "[red]ffmpeg is not on PATH. Install it before running this command "
            "(apt/brew/choco install ffmpeg).[/red]"
        )
        raise typer.Exit(code=2)

    produced = convert_directory_to_wav(
        paths.interviews_raw, target_sr=sample_rate, overwrite=overwrite
    )
    table = Table(title="convert-audio summary")
    table.add_column("File", overflow="fold")
    table.add_column("Size (MB)", justify="right")
    for wav in produced:
        size_mb = wav.stat().st_size / 1e6
        table.add_row(wav.name, f"{size_mb:.1f}")
    table.add_row("─" * 40, "─" * 8)
    table.add_row(f"{len(produced)} WAV(s) ready", "")
    console.print(table)


@app.command("list-speakers")
def list_speakers(
    workspace: Path = typer.Option(..., help="Workspace root containing speakerscribe_out/*.json."),
    show_files: bool = typer.Option(
        True, "--show-files/--no-show-files", help="Also list audio files in interviews_raw/."
    ),
) -> None:
    """List speakers detected by speakerscribe + segment counts and durations.

    Reemplaza la Cell 19 hand-edited del notebook. Útil para verificar el
    label que debes pasar a build-corpus --speaker antes de extraer.
    """
    import json
    from collections import defaultdict

    paths = WorkspacePaths(workspace=workspace)
    json_files = sorted(paths.speakerscribe_out.glob("*.json"))
    if not json_files:
        console.print(f"[red]No speakerscribe JSONs found in {paths.speakerscribe_out}[/red]")
        raise typer.Exit(code=2)

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(f"[red]{jf.name}: malformed JSON ({exc})[/red]")
            continue

        source = data.get("source_audio") or data.get("audio_file") or "(unknown)"
        segments = data.get("segments") or []

        counts: dict[str, int] = defaultdict(int)
        duration: dict[str, float] = defaultdict(float)
        for s in segments:
            lbl = str(s.get("speaker", "?"))
            counts[lbl] += 1
            with suppress(KeyError, ValueError, TypeError):
                duration[lbl] += float(s["end"]) - float(s["start"])

        table = Table(title=f"{jf.name}  (source: {source})")
        table.add_column("Speaker label")
        table.add_column("Segments", justify="right")
        table.add_column("Total duration (s)", justify="right")
        for lbl in sorted(counts):
            table.add_row(lbl, str(counts[lbl]), f"{duration[lbl]:.1f}")
        console.print(table)

    if show_files and paths.interviews_raw.is_dir():
        files = sorted(paths.interviews_raw.iterdir())
        table = Table(title="interviews_raw/")
        table.add_column("File", overflow="fold")
        for f in files:
            table.add_row(f.name)
        console.print(table)


@app.command("evaluate-denoise")
def evaluate_denoise(
    workspace: Path = typer.Option(..., help="Workspace root with interviews_raw/ and reports/."),
    audio: list[Path] | None = typer.Option(
        None,
        "--audio",
        help="Specific audio file(s) to evaluate. Repeat option for multiple files.",
    ),
    include_deepfilter: bool = typer.Option(
        False,
        "--deepfilter",
        help="Also evaluate DeepFilterNet via the optional deepFilter CLI if installed.",
    ),
) -> None:
    """Compare noisereduce against optional DeepFilterNet on real samples.

    Use this with 3-5 representative files before changing production denoise
    defaults. DeepFilterNet is intentionally not enabled blindly.
    """
    paths = WorkspacePaths(workspace=workspace)
    if audio:
        files = [Path(a) for a in audio]
    else:
        files = sorted(
            p
            for p in paths.interviews_raw.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        )[:5]
    if not files:
        console.print("[red]No audio files found. Pass --audio or populate interviews_raw/.[/red]")
        raise typer.Exit(code=2)

    report = evaluate_denoise_methods(
        files,
        paths.reports / "denoise_eval",
        include_deepfilter=include_deepfilter,
    )
    table = Table(title="Denoise evaluation")
    table.add_column("Source", overflow="fold")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Dynamic range", justify="right")
    table.add_column("Output", overflow="fold")
    for row in report["candidates"]:
        stats = row.get("stats") or {}
        table.add_row(
            Path(str(row["source_path"])).name,
            str(row["method"]),
            str(row["status"]),
            str(stats.get("snr_db", "")),
            str(row.get("output_path") or row.get("reason") or ""),
        )
    console.print(table)
    console.print(f"report={report['report_path']}")


@app.command("benchmark")
def benchmark(
    workspace: Path = typer.Option(..., help="Workspace root with reference_corpus/ and reports/."),
    texts: Path | None = typer.Option(
        None, "--texts", help="golden_texts_es.txt-formatted file. Defaults to the packaged set."
    ),
    language: str = typer.Option("es", help="Language ISO code passed to synthesis + ASR."),
    limit: int | None = typer.Option(
        None, "--limit", help="Only run the first N stimuli (quick smoke). Default: all."
    ),
    no_ecapa: bool = typer.Option(False, "--no-ecapa", help="Skip ECAPA speaker similarity."),
    no_resemblyzer: bool = typer.Option(False, "--no-resemblyzer", help="Skip Resemblyzer SECS."),
    no_wer: bool = typer.Option(False, "--no-wer", help="Skip the WER round-trip (ASR)."),
    no_mos: bool = typer.Option(False, "--no-mos", help="Skip the SQUIM MOS proxy."),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
) -> None:
    """Benchmark the current voice against the golden stimulus set.

    Synthesizes each stimulus with the real XTTS model and reference corpus,
    then scores fidelity (SECS, WER/CER, MOS proxy) and speed (RTF, peak VRAM).
    Writes reports/benchmark_<run>.json and appends reports/benchmarks.parquet.
    Run it once to freeze a baseline, then re-run after every precision change.
    """
    import tempfile

    import soundfile as sf

    from voicelegacy.evaluation import (
        BenchmarkConfig,
        append_to_cumulative,
        load_golden_texts,
        run_benchmark,
    )
    from voicelegacy.synthesis import load_xtts_model, release_model, synthesize_to_file

    paths = WorkspacePaths(workspace=workspace)
    top_wavs = sorted(paths.reference_corpus.glob("*.wav"))
    if not top_wavs:
        console.print("[red]No reference WAVs found. Run 'build-corpus' first.[/red]")
        raise typer.Exit(code=2)

    try:
        stimuli = load_golden_texts(texts)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if limit is not None:
        stimuli = stimuli[:limit]

    synth_cfg = SynthesisConfig(language=language)  # type: ignore[arg-type]
    config = BenchmarkConfig(
        compute_secs_ecapa=not no_ecapa,
        compute_secs_resemblyzer=not no_resemblyzer,
        compute_wer=not no_wer,
        compute_mos=not no_mos,
        asr_language=language,
    )

    tts = load_xtts_model(synth_cfg, accept_tos=accept_tos)

    def _synthesize(text: str) -> tuple[Any, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            synthesize_to_file(tts, text, top_wavs, tmp_path, synth_cfg)
            wav, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        finally:
            tmp_path.unlink(missing_ok=True)
        return wav, sr

    try:
        report = run_benchmark(
            _synthesize,
            stimuli,
            top_wavs,
            audio_dir=paths.reports / "benchmark_audio",
            config=config,
            notes={"language": language, "n_reference_wavs": len(top_wavs)},
        )
    finally:
        release_model()

    json_path = report.write_json(paths.reports / f"benchmark_{report.run_id}.json")
    cumulative = append_to_cumulative(report, paths.reports / "benchmarks.parquet")

    summary = report.summary()
    table = Table(title=f"Benchmark {report.run_id} — {summary['n_samples']} stimuli")
    table.add_column("Metric")
    table.add_column("ok", justify="right")
    table.add_column("skipped", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("median", justify="right")
    for name, info in summary["metrics"].items():
        stats = info["stats"] or {}
        table.add_row(
            name,
            str(info["n_ok"]),
            str(info["n_skipped"]),
            f"{stats.get('mean', '—')}",
            f"{stats.get('median', '—')}",
        )
    rtf = summary["rtf"] or {}
    table.add_row(
        "rtf", str(rtf.get("n", 0)), "0", f"{rtf.get('mean', '—')}", f"{rtf.get('median', '—')}"
    )
    console.print(table)
    console.print(f"report={json_path}")
    console.print(f"cumulative={cumulative}")
    console.print(
        "[yellow]Baseline note:[/yellow] freeze this run as the 'before' number. "
        "No precision change should ship without re-running this and comparing."
    )


@app.command("synthesize-long")
def synthesize_long(
    workspace: Path = typer.Option(..., help="Workspace root with reference_corpus/."),
    out: Path = typer.Option(..., "--out", help="Output WAV path. Sidecar written alongside."),
    text: str | None = typer.Option(None, "--text", help="Inline text to synthesize."),
    text_file: Path | None = typer.Option(
        None, "--text-file", help="UTF-8 text file to synthesize (alternative to --text)."
    ),
    language: str = typer.Option("es", help="Language ISO code."),
    max_chunk_chars: int = typer.Option(
        220, "--max-chunk-chars", help="Max characters per synthesis chunk (XTTS limit margin)."
    ),
    max_wer: float = typer.Option(
        0.15, "--max-wer", help="WER ceiling above which a chunk is retried."
    ),
    max_retries: int = typer.Option(2, "--max-retries", help="Re-rolls per failing chunk."),
    no_asr_verify: bool = typer.Option(
        False, "--no-asr-verify", help="Skip ASR verification (faster, less robust)."
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Reuse cached chunks if present."
    ),
    force: bool = typer.Option(
        False, "--force", help="Clear the chunk cache and re-synthesize all."
    ),
    top_n: int | None = typer.Option(
        None,
        "--top-n",
        help=(
            "Use only the N best-quality reference WAVs for voice conditioning "
            "(default: ReferenceConfig.top_n_segments = 5). 0 = legacy: all "
            "WAVs, alphabetical."
        ),
    ),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
) -> None:
    """Synthesize long text (paragraphs, chapters) as one continuous, verified WAV.

    Chunks the text at sentence/clause boundaries under the XTTS character limit,
    synthesizes each chunk reusing the same speaker conditioning (so the voice
    does not drift), ASR-verifies and retries failures, then joins the chunks
    with equal-power crossfades and punctuation-based pauses. Resumable: an
    interrupted run continues from its per-chunk cache. Writes <out>.sidecar.json
    with per-chunk WER, retries and any flagged chunks.
    """
    import tempfile

    import soundfile as sf

    from voicelegacy.longform import LongFormConfig, LongFormSynthesizer
    from voicelegacy.synthesis import load_xtts_model, release_model, synthesize_to_file

    if (text is None) == (text_file is None):
        console.print("[red]Provide exactly one of --text or --text-file.[/red]")
        raise typer.Exit(code=2)
    if text_file is not None:
        if not text_file.exists():
            console.print(f"[red]Text file not found: {text_file}[/red]")
            raise typer.Exit(code=2)
        text = text_file.read_text(encoding="utf-8")
    assert text is not None  # narrowed by the checks above
    if not text.strip():
        console.print("[red]Input text is empty.[/red]")
        raise typer.Exit(code=2)

    paths = WorkspacePaths(workspace=workspace)
    top_wavs = select_reference_wavs(paths.reference_corpus, top_n=top_n)
    if not top_wavs:
        console.print("[red]No reference WAVs found. Run 'build-corpus' first.[/red]")
        raise typer.Exit(code=2)

    synth_cfg = SynthesisConfig(language=language)  # type: ignore[arg-type]
    config = LongFormConfig(
        max_chunk_chars=max_chunk_chars,
        max_wer=max_wer,
        max_retries=max_retries,
        asr_verify=not no_asr_verify,
        language=language,
    )

    tts = load_xtts_model(synth_cfg, accept_tos=accept_tos)

    def _synthesize(chunk_text: str, attempt: int = 0) -> tuple[Any, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            # synthesize_to_file reuses cached conditioning latents across calls,
            # so the speaker timbre stays consistent chunk to chunk. The seed is
            # derived per attempt (base + attempt) so retry re-rolls actually
            # sample new audio instead of reproducing the identical failure.
            synthesize_to_file(
                tts, chunk_text, top_wavs, tmp_path, _config_for_attempt(synth_cfg, attempt)
            )
            wav, sr = sf.read(str(tmp_path), dtype="float32", always_2d=False)
        finally:
            tmp_path.unlink(missing_ok=True)
        return wav, sr

    synthesizer = LongFormSynthesizer(
        _synthesize,
        config,
        cache_root=paths.synthesis_out / "longform_cache",
        synthesize_accepts_attempt=True,
    )
    try:
        result = synthesizer.render(
            text,
            out,
            resume=resume,
            force=force,
            notes={"language": language, "n_reference_wavs": len(top_wavs)},
        )
    finally:
        release_model()

    summary = result.summary()
    table = Table(title=f"Long-form synthesis — {summary['n_chunks']} chunks")
    table.add_column("Field")
    table.add_column("Value", justify="right")
    table.add_row("duration", f"{summary['duration_s']}s")
    table.add_row("chunks", str(summary["n_chunks"]))
    table.add_row("from cache", str(summary["n_from_cache"]))
    table.add_row("ASR-verified", str(summary["asr_verified_chunks"]))
    table.add_row("mean WER", f"{summary['mean_wer']}")
    table.add_row("flagged", str(summary["n_flagged"]))
    table.add_row("RTF", f"{summary['rtf']}")
    console.print(table)
    console.print(f"output={result.output_path}")
    console.print(f"sidecar={result.sidecar_path}")
    if summary["n_flagged"]:
        console.print(
            f"[yellow]{summary['n_flagged']} chunk(s) failed verification and were flagged "
            f"for review (indices {summary['flagged_indices']}). See the sidecar.[/yellow]"
        )


if __name__ == "__main__":
    app()
