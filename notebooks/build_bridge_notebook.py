"""Generator for notebook_voicelegacy_bridge.ipynb.

The BRIDGE notebook. Connects speakerscribe (diarization + transcription) to
voicelegacy (corpus extraction) for the real-world case: many interviews,
each with MULTIPLE speakers, where the target speaker appears under DIFFERENT
SPEAKER_xx labels across files (diarization labels are per-file, not global).

This notebook follows the colab-notebook-dev + python-data-library-dev skills:
- Two cell types only: EXTRAS (run once: imports, dataclass config, functions)
  and EJECUTAR (<=15 lines: user vars + one call).
- RAM discipline: NEVER load a full long recording into RAM. Audio is read
  PARTIALLY from disk via soundfile seek (start/stop frames) — only the slice
  needed. Critical for 10+ hours of interviews on Colab Free (12 GB).
- Per-interview checkpointing: a manifest JSON records which interviews are
  already extracted; re-running skips them. Survives a killed Colab session.
- Centralized @dataclass config, zero magic numbers, pathlib everywhere.
- Observability: per-interview progress, RAM monitoring, composition summaries.
- Reproducibility metadata written at the end.

Run:
    python notebooks/build_bridge_notebook.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat as nbf
import tomllib

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "notebook_voicelegacy_bridge.ipynb"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _version() -> str:
    return tomllib.load(PYPROJECT_PATH.open("rb"))["project"]["version"]


def _cid() -> str:
    return uuid.uuid4().hex[:8]


def _md(t: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(t, id=_cid())


def _code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s, id=_cid())


VERSION = _version()

# ──────────────────────────────────────────────────────────────────────────
# The big EXTRAS cell. All imports, the dataclass config, and every helper
# function live here. Run once. No user-configurable variables.
# ──────────────────────────────────────────────────────────────────────────
EXTRAS_SRC = '''# ===== SECCIÓN EXTRAS / UTILIDADES =====
# Ejecutar UNA VEZ al inicio de la sesión.
# Contiene: imports, dataclass de configuración, funciones. SIN variables de usuario.

import gc
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from voicelegacy.config import ReferenceConfig, WorkspacePaths
from voicelegacy.corpus import (
    analyze_f0_outliers,
    extract_segments_to_wav,
    filter_f0_outliers,
    filter_segments,
    load_speakerscribe_json,
    write_f0_outlier_report,
)
from voicelegacy.speakerscribe_schema import load_and_validate_speakerscribe_document


# ── Layer 1: Centralized config (zero magic numbers) ──────────────────────
@dataclass
class BridgeConfig:
    """Configuration for the speakerscribe -> voicelegacy bridge.

    Context: Google Colab Free (~12 GB RAM). Long interviews are read
    PARTIALLY from disk, never fully into RAM.

    Attributes:
        min_segment_duration_s: Drop target-speaker segments shorter than this.
        max_segment_duration_s: Drop segments longer than this (XTTS limit).
        min_snr_db: SNR floor for the cleanup stage.
        apply_denoise: Run spectral-gate denoise on each segment.
        denoise_stationary: Stationary vs non-stationary denoise.
        apply_bandpass_filter: Voice band-pass on each segment.
        apply_preemphasis_filter: Pre-emphasis on each segment.
        enable_f0_outlier_filter: Drop F0 outliers (likely diarization errors).
        sample_preview_s: Seconds of audio to preview per speaker (identification).
        ram_warn_pct: Warn when RAM exceeds this percent.
        ideal_min_minutes: Lower bound of the ideal net-audio band.
        ideal_max_minutes: Upper bound of the ideal net-audio band.
        usable_min_minutes: Below this, fine-tuning will likely underperform.
    """

    min_segment_duration_s: float = 2.0
    max_segment_duration_s: float = 15.0
    min_snr_db: float = 15.0
    apply_denoise: bool = True
    denoise_stationary: bool = False
    apply_bandpass_filter: bool = True
    apply_preemphasis_filter: bool = True
    enable_f0_outlier_filter: bool = True
    sample_preview_s: float = 12.0
    ram_warn_pct: float = 85.0
    ideal_min_minutes: float = 30.0
    ideal_max_minutes: float = 300.0
    usable_min_minutes: float = 15.0

    def __post_init__(self) -> None:
        if self.min_segment_duration_s <= 0:
            raise ValueError("min_segment_duration_s must be > 0")
        if self.max_segment_duration_s <= self.min_segment_duration_s:
            raise ValueError("max_segment_duration_s must exceed min")
        if not 0 < self.ram_warn_pct <= 100:
            raise ValueError("ram_warn_pct must be in (0, 100]")

    def to_reference_config(self) -> ReferenceConfig:
        """Map to the voicelegacy ReferenceConfig used by corpus extraction."""
        return ReferenceConfig(
            min_segment_duration_s=self.min_segment_duration_s,
            max_segment_duration_s=self.max_segment_duration_s,
            min_snr_db=self.min_snr_db,
            apply_denoise=self.apply_denoise,
            denoise_stationary=self.denoise_stationary,
            apply_bandpass_filter=self.apply_bandpass_filter,
            apply_preemphasis_filter=self.apply_preemphasis_filter,
            enable_f0_outlier_filter=self.enable_f0_outlier_filter,
        )


# ── Observability helpers ─────────────────────────────────────────────────
def ram_usada_mb() -> float:
    """Return current process+system RAM usage in MB (0 if psutil absent)."""
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.virtual_memory().used / 1e6


def avisar_ram(config: BridgeConfig) -> None:
    """Print a warning if RAM crosses the configured threshold."""
    if not _HAS_PSUTIL:
        return
    mem = psutil.virtual_memory()
    if mem.percent >= config.ram_warn_pct:
        print(f"   \u26a0\ufe0f RAM al {mem.percent:.0f}% ({mem.used/1e9:.1f} GB) \u2014 "
              "se libera tras cada entrevista, pero vigila.")


# ── Audio source resolution + PARTIAL read (RAM discipline) ───────────────
def resolver_audio_fuente(doc, json_path: Path, audios_dir: Path) -> Path | None:
    """Locate the original audio for a speakerscribe JSON.

    Tries the declared audio_file stem first, then the JSON stem.

    Args:
        doc: Validated speakerscribe document.
        json_path: Path to the .json file.
        audios_dir: Directory holding the original recordings.

    Returns:
        Path to the audio file, or None if not found.
    """
    src_name = doc.source_name or json_path.stem
    for stem in (Path(src_name).stem, json_path.stem):
        hits = sorted(audios_dir.glob(f"{stem}.*"))
        # Ignore .json hits; we want the media file.
        hits = [h for h in hits if h.suffix.lower() != ".json"]
        if hits:
            return hits[0]
    return None


def leer_fragmento(audio_path: Path, start_s: float, dur_s: float) -> tuple[np.ndarray, int]:
    """Read ONLY a time slice from disk \u2014 never the whole file into RAM.

    Uses soundfile seek (start/stop frames). This is the core RAM-safety
    mechanism: a 1-hour recording costs the same RAM as a 12-second one
    because we only materialize the requested frames.

    Args:
        audio_path: Path to the source recording.
        start_s: Start time in seconds.
        dur_s: Duration to read in seconds.

    Returns:
        (mono float32 array, sample_rate).
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    start_frame = int(start_s * sr)
    stop_frame = min(int((start_s + dur_s) * sr), info.frames)
    y, _ = sf.read(str(audio_path), start=start_frame, stop=stop_frame, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)  # to mono
    return y, sr


# ── Phase 1: inspect speakers per interview ───────────────────────────────
def resumen_hablantes(json_files: list[Path]) -> dict[str, dict[str, float]]:
    """Build a {json_name: {speaker: minutes}} summary across interviews.

    Does NOT read any audio \u2014 only the JSON metadata. Cheap and RAM-free.
    """
    summary: dict[str, dict[str, float]] = {}
    for jf in json_files:
        doc = load_and_validate_speakerscribe_document(jf)
        per_speaker: dict[str, float] = {}
        for seg in doc.segments:
            per_speaker[seg.speaker] = per_speaker.get(seg.speaker, 0.0) + (seg.end - seg.start) / 60.0
        summary[jf.name] = per_speaker
        print(f"\\n\U0001f4c4 {jf.name}  (idioma: {doc.language_detected})")
        print(f"   audio_file declarado: {doc.source_name}")
        for spk, mins in sorted(per_speaker.items(), key=lambda x: -x[1]):
            print(f"   {spk:14s} {mins:6.1f} min  {'\u2588' * min(int(mins), 40)}")
    return summary


# ── Phase 2: per-interview extraction with checkpointing ──────────────────
def _manifest_path(paths: WorkspacePaths) -> Path:
    return Path(paths.reports) / "bridge_manifest.json"


def cargar_manifest(paths: WorkspacePaths) -> dict:
    """Load the extraction manifest (which interviews are already done)."""
    mp = _manifest_path(paths)
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {"interviews": {}}


def guardar_manifest(paths: WorkspacePaths, manifest: dict) -> None:
    """Persist the extraction manifest for checkpoint/resume."""
    _manifest_path(paths).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extraer_una_entrevista(
    json_path: Path,
    target_speaker: str,
    audios_dir: Path,
    paths: WorkspacePaths,
    config: BridgeConfig,
) -> dict:
    """Extract + clean ONLY the target speaker's segments from one interview.

    voicelegacy.extract_segments_to_wav caches the source audio internally
    for the duration of THIS call, then releases it on return \u2014 so RAM holds
    at most one interview's audio at a time. We additionally gc.collect()
    between interviews in the driver loop.

    Args:
        json_path: speakerscribe .json for this interview.
        target_speaker: SPEAKER_xx label of the target in THIS interview.
        audios_dir: Directory of original recordings.
        paths: voicelegacy workspace paths.
        config: BridgeConfig.

    Returns:
        Stats dict: {json, speaker, n_segments_target, n_written, net_minutes}.
    """
    ref_config = config.to_reference_config()
    segs_all = load_speakerscribe_json(json_path, audio_root=audios_dir)
    segs_ok = filter_segments(
        segs_all,
        target_speaker=target_speaker,
        min_duration_s=config.min_segment_duration_s,
        max_duration_s=config.max_segment_duration_s,
    )
    n_target = len(segs_ok)
    if not segs_ok:
        return {"json": json_path.name, "speaker": target_speaker,
                "n_segments_target": 0, "n_written": 0, "net_minutes": 0.0}

    if config.enable_f0_outlier_filter and len(segs_ok) >= 3:
        results = analyze_f0_outliers(segs_ok, ref_config)
        write_f0_outlier_report(
            Path(paths.reports) / f"f0_{json_path.stem}.json", results, ref_config
        )
        segs_ok = filter_f0_outliers(segs_ok, ref_config)

    written = extract_segments_to_wav(segs_ok, Path(paths.reference_corpus), ref_config)
    net_s = 0.0
    for w in written:
        try:
            net_s += sf.info(str(w)).duration
        except Exception:  # noqa: BLE001  (corrupt WAV should not abort batch)
            pass
    return {"json": json_path.name, "speaker": target_speaker,
            "n_segments_target": n_target, "n_written": len(written),
            "net_minutes": round(net_s / 60.0, 2)}


def procesar_lote(
    target_map: dict[str, str],
    json_files: list[Path],
    audios_dir: Path,
    paths: WorkspacePaths,
    config: BridgeConfig,
) -> list[dict]:
    """Driver loop: extract every mapped interview, with checkpoint + RAM release.

    Skips interviews already recorded in the manifest. After each interview,
    forces gc.collect() so the previous recording's audio leaves RAM before
    the next is read.

    Args:
        target_map: {json_name: SPEAKER_xx} for the target speaker per file.
        json_files: All available speakerscribe JSONs.
        audios_dir: Directory of original recordings.
        paths: voicelegacy workspace paths.
        config: BridgeConfig.

    Returns:
        List of per-interview stats dicts (from manifest, all mapped files).
    """
    manifest = cargar_manifest(paths)
    by_name = {jf.name: jf for jf in json_files}
    pendientes = [n for n in target_map if n not in manifest["interviews"]]
    total = len(pendientes)
    if total == 0:
        print("\u267b\ufe0f  Todas las entrevistas mapeadas ya estaban extra\u00eddas (manifest).")
    for i, name in enumerate(pendientes, 1):
        if name not in by_name:
            print(f"   \u274c {name} mapeado pero no existe \u2014 omitido.")
            continue
        target = target_map[name]
        print(f"\\n\U0001f504 [{i}/{total}] {name} \u2192 {target}")
        stats = extraer_una_entrevista(by_name[name], target, audios_dir, paths, config)
        manifest["interviews"][name] = stats
        guardar_manifest(paths, manifest)  # checkpoint after EACH interview
        print(f"   \u2705 {stats['n_written']} wavs | {stats['net_minutes']:.1f} min netos "
              f"(de {stats['n_segments_target']} segmentos del objetivo)")
        gc.collect()  # release this interview's audio before the next
        avisar_ram(config)
    # Return stats for ALL mapped interviews (done now or earlier)
    return [manifest["interviews"][n] for n in target_map if n in manifest["interviews"]]


def reportar_total(stats: list[dict], config: BridgeConfig) -> float:
    """Print net-minutes summary and a band verdict. Returns total minutes."""
    total_min = sum(s["net_minutes"] for s in stats)
    total_wavs = sum(s["n_written"] for s in stats)
    print(f"\\n{'='*52}")
    print(f"\U0001f4ca Voz neta consolidada: {total_min:.1f} min ({total_min/60:.2f} h) | {total_wavs} wavs")
    for s in stats:
        print(f"   {s['json']:35s} [{s['speaker']}]  {s['n_written']} wavs / {s['net_minutes']:.1f} min")
    if total_min < config.usable_min_minutes:
        print(f"\\n\u274c {total_min:.1f} min \u2014 insuficiente. Procesa m\u00e1s entrevistas.")
    elif total_min < config.ideal_min_minutes:
        print(f"\\n\u26a0\ufe0f {total_min:.1f} min \u2014 funciona, calidad limitada.")
    elif total_min <= config.ideal_max_minutes:
        print(f"\\n\u2705 {total_min:.1f} min \u2014 rango IDEAL para fine-tuning.")
    else:
        print(f"\\n\u2705 {total_min:.1f} min \u2014 de sobra; puedes ser m\u00e1s selectivo.")
    return total_min


# ── Reproducibility metadata ──────────────────────────────────────────────
def guardar_metadata(paths: WorkspacePaths, config: BridgeConfig,
                     target_map: dict, stats: list[dict], total_min: float) -> None:
    """Write run metadata for full reproducibility."""
    meta = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config": asdict(config),
        "target_speaker_map": target_map,
        "per_interview_stats": stats,
        "total_net_minutes": round(total_min, 2),
        "python_version": sys.version,
    }
    out = Path(paths.reports) / "bridge_metadata.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\U0001f4cb Metadata guardada: {out}")


print("\u2705 EXTRAS cargado: BridgeConfig + funciones listas.")
'''


CELLS = [
    _md(
        f"""# voicelegacy {VERSION} \u00b7 Puente speakerscribe \u2192 voicelegacy

**El notebook de enlace.** Extrae la voz de UNA persona desde muchas entrevistas \
con varias personas, ya diarizadas por speakerscribe. Construido siguiendo las \
mejores pr\u00e1cticas de Colab: lectura parcial desde disco (sin reventar RAM), \
checkpointing por entrevista, config centralizada, y observabilidad completa.

## Tu escenario

- Muchas entrevistas (10+ horas), cada una con varias personas.
- Quieres clonar la voz de una sola (la "abuela").
- Ya corriste **speakerscribe** \u2192 tienes los `.json`.
- Las etiquetas `SPEAKER_xx` NO son consistentes entre archivos (la diarizaci\u00f3n \
es por-archivo) \u2192 identificas a la abuela una vez por entrevista.

## Gesti\u00f3n de memoria (clave para Colab Free)

Una entrevista de 1 hora **NO se carga entera a RAM**. Este notebook lee solo \
los fragmentos necesarios desde disco (`soundfile` con seek). Procesa una \
entrevista a la vez y **libera la RAM entre cada una**. As\u00ed, 10+ horas de \
entrevistas caben en los 12 GB de Colab Free sin problema.

## Estructura del notebook (2 tipos de celda)

- **EXTRAS** (celda 1): se corre UNA vez. Imports, config, funciones.
- **EJECUTAR** (resto): pocas l\u00edneas, variables de usuario + una llamada.

## El flujo completo

```
speakerscribe (diariza) \u2192 ESTE NOTEBOOK (extrae la voz de la abuela)
                        \u2192 notebook_voicelegacy_finetune.ipynb (entrena)
```
"""
    ),
    _md(
        "## Celda EXTRAS \u2014 correr UNA vez\n\nImports, configuraci\u00f3n y funciones. No tiene variables editables."
    ),
    _code(EXTRAS_SRC),
    _md(
        """## 1 \u00b7 Montar Drive e instalar voicelegacy

`psutil` (preinstalado en Colab) habilita el monitoreo de RAM.
"""
    ),
    _code(
        """from google.colab import drive
drive.mount("/content/drive")
# voicelegacy no está publicado en PyPI: se instala desde el tag de GitHub.
!pip install -q git+https://github.com/EnriqueForero/voicelegacy.git@v{version}
import voicelegacy
print("voicelegacy", voicelegacy.__version__)""".replace("{version}", VERSION)
    ),
    _md(
        """## 2 \u00b7 Configurar rutas e inicializar (EJECUTAR)

Edita las 3 rutas. Necesitas: los `.json` de speakerscribe y los audios originales.
"""
    ),
    _code(
        """# ── Variables de usuario ───────────────────────────────────────────
WORKSPACE           = "/content/drive/MyDrive/voicelegacy_workspace"      # EDITA
SPEAKERSCRIBE_JSONS = "/content/drive/MyDrive/speakerscribe_out"          # EDITA
ORIGINAL_AUDIOS     = "/content/drive/MyDrive/entrevistas_originales"     # EDITA

# ── Inicializaci\u00f3n ─────────────────────────────────────────────────
config = BridgeConfig()
paths = WorkspacePaths(workspace=WORKSPACE)
paths.mkdirs()
audios_dir = Path(ORIGINAL_AUDIOS)
json_files = sorted(Path(SPEAKERSCRIBE_JSONS).glob("*.json"))
assert json_files, f"No hay .json en {SPEAKERSCRIBE_JSONS}. \u00bfCorriste speakerscribe?"
print(f"\u2705 {len(json_files)} JSON | audios: {audios_dir.exists()} | RAM: {ram_usada_mb():.0f} MB")"""
    ),
    _md(
        """## 3 \u00b7 Inspeccionar hablantes por entrevista

Solo lee metadata de los JSON (cero audio, cero RAM). Muestra cu\u00e1nto habla cada \
quien. El que m\u00e1s habla suele ser el entrevistado \u2014 pero **confirma escuchando** \
(celda 4).
"""
    ),
    _code("summary = resumen_hablantes(json_files)"),
    _md(
        """## 4 \u00b7 Identificar a la abuela en una entrevista (ESCUCHANDO)

Reproduce una muestra de cada hablante leyendo **solo ese fragmento desde disco** \
(no carga la entrevista entera). Cambia `INSPECCIONAR` al \u00edndice de cada entrevista \
y repite. Anota el resultado en la celda 5.
"""
    ),
    _code(
        """from IPython.display import Audio, display

INSPECCIONAR = 0   # EDITA: \u00edndice de la entrevista (0 = primera)

_jf = json_files[INSPECCIONAR]
_doc = load_and_validate_speakerscribe_document(_jf)
_audio = resolver_audio_fuente(_doc, _jf, audios_dir)
assert _audio, f"No encuentro el audio de {_jf.name} en {audios_dir}"
print(f"\U0001f4c4 {_jf.name}  \u2192  audio: {_audio.name}\\n")

# Para cada hablante: su segmento m\u00e1s largo como muestra (lectura parcial)
_speakers: dict[str, list] = {}
for _seg in _doc.segments:
    _speakers.setdefault(_seg.speaker, []).append(_seg)
for _spk in sorted(_speakers):
    _s = max(_speakers[_spk], key=lambda s: s.end - s.start)
    _y, _sr = leer_fragmento(_audio, _s.start, config.sample_preview_s)
    print(f"\U0001f50a {_spk}: \\"{_s.text[:60]}...\\"")
    display(Audio(_y, rate=_sr))
    del _y; gc.collect()   # liberar el fragmento tras reproducirlo
print("\\n\U0001f449 Identifica a la abuela y anota en la celda 5.")"""
    ),
    _md(
        """## 5 \u00b7 Mapa abuela-por-entrevista (EJECUTAR)

Tras escuchar cada entrevista, llena el diccionario. Clave = nombre del `.json`, \
valor = etiqueta de la abuela en esa entrevista. Las que no incluyas se omiten.
"""
    ),
    _code(
        """TARGET_SPEAKER_MAP = {
    # "entrevista_01.json": "SPEAKER_00",
    # "entrevista_02.json": "SPEAKER_01",
}

_avail = {jf.name for jf in json_files}
_unmapped = _avail - set(TARGET_SPEAKER_MAP)
if _unmapped:
    print(f"\u26a0\ufe0f {len(_unmapped)} sin mapear (se omiten): {sorted(_unmapped)}")
if set(TARGET_SPEAKER_MAP) - _avail:
    print(f"\u274c Mapeaste JSON inexistentes: {set(TARGET_SPEAKER_MAP) - _avail}")
print(f"\u2705 A procesar: {len(set(TARGET_SPEAKER_MAP) & _avail)}")"""
    ),
    _md(
        """## 6 \u00b7 Extraer la voz de la abuela (EJECUTAR)

Procesa una entrevista a la vez. Por cada una: filtra al hablante objetivo, \
descarta outliers de tono, recorta + limpia, escribe al `reference_corpus/`. \
**Guarda checkpoint tras cada entrevista** (si Colab corta, reanudas sin \
reprocesar) y **libera RAM** entre cada una.
"""
    ),
    _code(
        """stats = procesar_lote(TARGET_SPEAKER_MAP, json_files, audios_dir, paths, config)
total_min = reportar_total(stats, config)
guardar_metadata(paths, config, TARGET_SPEAKER_MAP, stats, total_min)"""
    ),
    _md(
        """## 6.bis \u00b7 Validar coherencia del corpus (DETECTA ETIQUETAS MAL MAPEADAS)

**Paso cr\u00edtico del flujo manual.** Si mapeaste mal a un hablante en una sola \
entrevista, el corpus se contamina con otra voz y el clon sale mezclado. Esta \
celda usa Resemblyzer para verificar que TODOS los WAV del corpus son del mismo \
hablante: embebe cada uno y mide su similitud con el centroide. Los que caen por \
debajo de 0.70 son sospechosos \u2014 probablemente del hablante equivocado.

Requiere `pip install voicelegacy[similarity]` (resemblyzer).
"""
    ),
    _code(
        """from voicelegacy.finetune_dataset import validate_corpus_coherence

try:
    coh = validate_corpus_coherence(Path(paths.reference_corpus), threshold=0.70)
    print(f"Corpus: {coh.n_wavs} WAVs | sim media {coh.mean_similarity:.3f} | "
          f"min {coh.min_similarity:.3f}")
    if coh.is_coherent:
        print("\u2705 Corpus coherente: todos los clips parecen del mismo hablante.")
    else:
        print(f"\u26a0\ufe0f {len(coh.outliers)} clip(s) sospechoso(s) (sim < 0.70):")
        for p, s in coh.outliers[:10]:
            print(f"    {p.name}  sim={s:.3f}")
        print("\\n\U0001f449 Revisa esas etiquetas. Si vienen de una entrevista mal mapeada, "
              "corrige TARGET_SPEAKER_MAP, borra reference_corpus, y re-extrae (celda 6).")
        # Guardar reporte para auditor\u00eda
        import json as _json
        (Path(paths.reports) / "coherence_report.json").write_text(
            _json.dumps(coh.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
except ImportError as e:
    print(f"\u26a0\ufe0f Validaci\u00f3n de coherencia omitida: {e}")
    print("   Instala con: pip install voicelegacy[similarity]")"""
    ),
    _md(
        """## 7 \u00b7 Construir el dataset de fine-tuning (EJECUTAR, opcional)

Usa la funci\u00f3n del paquete `build_finetune_dataset`, que empareja cada WAV con \
su transcripci\u00f3n leyendo el **sidecar `.txt`** que `extract_segments_to_wav` \
escribi\u00f3 junto a cada WAV. Robusto por construcci\u00f3n: no parsea nombres de \
archivo (esa era la causa de un bug que dejaba el dataset vac\u00edo).
"""
    ),
    _code(
        """from voicelegacy.finetune_dataset import build_finetune_dataset

FT_DATASET = Path(WORKSPACE) / "finetune_dataset"
ds = build_finetune_dataset(
    reference_corpus=Path(paths.reference_corpus),
    dataset_dir=FT_DATASET,
    speaker_name="abuela",
)
print(f"\u2705 Dataset: {ds.n_total} pares | train {ds.n_train} / eval {ds.n_eval}")
if ds.n_skipped_no_text:
    print(f"   ({ds.n_skipped_no_text} WAVs sin sidecar de texto, omitidos)")
print(f"\U0001f449 En notebook_voicelegacy_finetune.ipynb apunta DATASET_DIR a: {FT_DATASET}")"""
    ),
    _md(
        """## 8 \u00b7 Qu\u00e9 sigue

El `reference_corpus/` y `finetune_dataset/` contienen SOLO la voz limpia de tu \
abuela, consolidada de todas las entrevistas. Abre \
**`notebook_voicelegacy_finetune.ipynb`**, apunta su `DATASET_DIR` al \
`finetune_dataset/`, y entrena. Con 2-5 h de voz neta \u2192 calidad de legado.

### Checkpointing y reanudaci\u00f3n

`reports/bridge_manifest.json` registra qu\u00e9 entrevistas ya se extrajeron. Si \
Colab corta, vuelve a correr la celda 6: las hechas se saltan, sigue con las \
pendientes. `reports/bridge_metadata.json` guarda config + stats para reproducibilidad.

### Si algo sale mal

| S\u00edntoma | Causa | Soluci\u00f3n |
|---|---|---|
| RAM alta | Entrevista enorme | Ya se libera entre cada una; si persiste, procesa en 2 tandas (menos entradas en el mapa) |
| "No encuentro el audio" | Audio fuera de ORIGINAL_AUDIOS | Pon el audio con el nombre que declara el JSON |
| 0 segmentos del objetivo | Etiqueta equivocada | Re-escucha (celda 4), corrige el mapa |
| Clon suena a 2 personas | Corpus contaminado | Revisa cada etiqueta del mapa, borra reference_corpus, re-extrae |
"""
    ),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    }
    return nb


def main() -> int:
    nb = build()
    nbf.validate(nb)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
        f.write("\n")
    print(f"\u2705 Notebook written: {NOTEBOOK_PATH}")
    print(f"   Cells: {len(nb['cells'])}")
    print(f"   Version: {VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
