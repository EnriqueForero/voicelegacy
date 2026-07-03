"""Generator for notebook_voicelegacy_finetune_standalone.ipynb.

A SELF-CONTAINED fine-tuning notebook that starts from ONE raw recording
(e.g. a 30-minute interview) and does EVERYTHING:

    raw recording (.mp3/.m4a/.wav/.mp4)
        │
        ▼  1. convert to 22.05 kHz mono WAV
        ▼  2. transcribe with faster-whisper (word timestamps, Spanish)
        ▼  3. segment into 2-11s clips at sentence/pause boundaries
        ▼  4. clean each clip (denoise + loudness norm via voicelegacy.audio)
        ▼  5. build LJSpeech-like dataset (metadata_train/eval CSV)
        ▼  6. fine-tune XTTS-v2 GPT encoder (T4 GPU)
        ▼  7. materialize reusable checkpoint
        ▼  8. validate + A/B vs base with speaker_similarity_score

Unlike notebook_voicelegacy_finetune.ipynb (which assumes a pre-built
reference_corpus/ + speakerscribe transcriptions), this one needs NOTHING
but the raw file. It is the right tool when you have a single long recording
and no diarization JSON.

Hyperparameters here are tuned for SMALL datasets (15-25 min net speech),
which is what you get from a 30-minute recording after silence/noise removal:
- Fewer epochs (overfit risk)
- Higher weight decay (regularization)
- Lower learning rate (avoid catastrophic forgetting)

Run:
    python notebooks/build_finetune_standalone_notebook.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat as nbf
import tomllib

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "notebook_voicelegacy_finetune_standalone.ipynb"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _version() -> str:
    return tomllib.load(PYPROJECT_PATH.open("rb"))["project"]["version"]


def _cid() -> str:
    return uuid.uuid4().hex[:8]


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text, id=_cid())


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src, id=_cid())


VERSION = _version()


CELLS = [
    _md(
        f"""# voicelegacy {VERSION} · Fine-tuning desde UNA grabación larga

**Autónomo. Parte de una grabación cruda de ~30 min y hace TODO.**

A diferencia del notebook `notebook_voicelegacy_finetune.ipynb` (que necesita \
un `reference_corpus/` ya construido + transcripciones de speakerscribe), este \
notebook **solo necesita tu archivo de audio**. Transcribe, segmenta, limpia, \
arma el dataset y entrena, todo en una corrida.

## Úsalo si

- Tienes UNA grabación larga (15-60 min) de la persona.
- NO tienes transcripciones ni diarización previa.
- El zero-shot te salió mal (clips de 5-15s dan poco contexto al encoder).

## La verdad sobre 30 minutos

30 min de grabación NO son 30 min de voz útil. Tras quitar silencios, \
respiraciones y ruido, te quedan **~15-22 min de voz neta**. Eso alcanza para \
fine-tuning pero estás en el **límite inferior**. Espera "reconocible y \
aceptable para legado", no "indistinguible del original".

Si tras el A/B (celda final) el fine-tuned NO supera al base por ≥ 0.05 en \
similarity, tu material es insuficiente y no hay notebook que lo arregle: \
necesitas más audio.

## Requisitos

1. **Runtime → Change runtime type → T4 GPU.** Sin GPU no funciona.
2. Aceptar la [Coqui Public Model License (CPML)](https://coqui.ai/cpml).
3. Tu grabación en Drive (mp3/m4a/wav/mp4/ogg/flac).

## Tiempo total estimado (T4 Free)

| Fase | Tiempo |
|---|---|
| Transcripción faster-whisper (30 min audio) | 5-10 min |
| Segmentación + limpieza | 5 min |
| Descarga base XTTS-v2 (una vez) | 5-10 min |
| **Entrenamiento (15-22 min audio, 10 epochs)** | **2-4 h** |
| Validación + A/B | 10 min |

**Total: ~3-5 h.** Cabe en una sesión Colab Free si no te desconectas.
"""
    ),
    # ─── 1. CPML + Drive + GPU ─────────────────────────────────────────
    _md("## 1 · Aceptar CPML, montar Drive, verificar GPU"),
    _code(
        """# ✋ Lee y acepta la CPML: https://coqui.ai/cpml
ACCEPT_CPML = False   # CAMBIA A True

if not ACCEPT_CPML:
    raise RuntimeError("Debes aceptar la CPML cambiando ACCEPT_CPML = True.")

import os
os.environ["COQUI_TOS_AGREED"] = "1"
print("✅ CPML aceptada.")"""
    ),
    _code(
        """from google.colab import drive
drive.mount('/content/drive')

# Verificar T4
gpu = !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
print("GPU:", gpu[0] if gpu else "NINGUNA — cambia el runtime a T4 GPU")
assert gpu and "T4" in gpu[0] or "GPU" in str(gpu), \\
    "Sin GPU. Runtime → Change runtime type → T4 GPU, luego re-corre."
"""
    ),
    # ─── 2. Configuración ──────────────────────────────────────────────
    _md(
        """## 2 · Configuración

Edita SOLO las líneas marcadas `# EDITA`. Los hyperparams ya están calibrados \
para un dataset pequeño (15-25 min). No los toques salvo que entiendas el efecto.
"""
    ),
    _code(
        """from pathlib import Path

# ─── EDITA: ruta a tu grabación cruda en Drive ──────────────────────
RAW_RECORDING = Path("/content/drive/MyDrive/grabacion_abuela_30min.m4a")  # EDITA

# ─── EDITA: workspace de fine-tuning (se crea si no existe) ─────────
WORKSPACE_FT = Path("/content/drive/MyDrive/voicelegacy_finetune_standalone")  # EDITA

# ─── EDITA: identificadores ─────────────────────────────────────────
RUN_NAME  = "abuela_es_v1"   # cambia entre experimentos
LANGUAGE  = "es"             # idioma de la grabación

# ─── Si la grabación tiene MÁS de una persona hablando ──────────────
# faster-whisper NO separa hablantes. Si hay entrevistador + entrevistado,
# pon SOLO_PRIMER_HABLANTE = False y revisa manualmente el CSV en la celda 5
# para borrar las filas del que NO quieres clonar. Si es monólogo, déjalo True.
ES_MONOLOGO = True   # EDITA si hay varias voces

# ─── Hyperparams calibrados para dataset PEQUEÑO (no editar a la ligera) ──
WHISPER_MODEL      = "large-v3"   # mejor calidad de transcripción en español
MIN_SEG_S          = 2.0          # XTTS rechaza < 2s
MAX_SEG_S          = 11.0         # filtro anti-OOM en T4
TARGET_SR          = 22050        # sample rate de entrada XTTS

NUM_EPOCHS         = 10           # MÁS que el notebook de 2-5h (dataset chico necesita más pasadas)
BATCH_SIZE         = 3
GRAD_ACUMM_STEPS   = 84
LEARNING_RATE      = 3e-6         # MÁS conservador que 5e-6 (anti-overfit)
WEIGHT_DECAY       = 5e-2         # MÁS alto que 1e-2 (regularización fuerte)
SAVE_EVERY_N_STEPS = 500          # checkpoints frecuentes (dataset chico = corridas cortas)

# ─── Derivado: no editar ────────────────────────────────────────────
WORK_WAV       = WORKSPACE_FT / "source_22k_mono.wav"
DATASET_DIR    = WORKSPACE_FT / "dataset"
WAVS_DIR       = DATASET_DIR / "wavs"
BASE_XTTS_DIR  = WORKSPACE_FT / "base_xtts"
RUNS_DIR       = WORKSPACE_FT / "runs"
FINETUNED_DIR  = WORKSPACE_FT / "finetuned"
TRANSCRIPT_JSON= WORKSPACE_FT / "transcript_words.json"

for p in (WORKSPACE_FT, DATASET_DIR, WAVS_DIR, BASE_XTTS_DIR, RUNS_DIR, FINETUNED_DIR):
    p.mkdir(parents=True, exist_ok=True)

assert RAW_RECORDING.exists(), f"No existe la grabación: {RAW_RECORDING}"
print(f"Grabación : {RAW_RECORDING}  ({RAW_RECORDING.stat().st_size/(1024*1024):.1f} MB)")
print(f"Workspace : {WORKSPACE_FT}")
print(f"Idioma    : {LANGUAGE}  |  Monólogo: {ES_MONOLOGO}")"""
    ),
    # ─── 3. Instalar dependencias ──────────────────────────────────────
    _md(
        """## 3 · Instalar dependencias

`coqui-tts` (fine-tuning), `faster-whisper` (transcripción), `voicelegacy` \
(limpieza de audio + inferencia final). Ignora los errores rojos de \
`pip dependency resolver` — son informativos.
"""
    ),
    _code(
        """# voicelegacy no está en PyPI: se instala desde el tag de GitHub.
!pip install -q coqui-tts==0.27.5 faster-whisper==1.1.0 git+https://github.com/EnriqueForero/voicelegacy.git@v{version}

import importlib
for mod in ("TTS.tts.models.xtts", "faster_whisper",
            "TTS.tts.layers.xtts.trainer.gpt_trainer",
            "voicelegacy.audio", "voicelegacy.finetuned_inference"):
    try:
        importlib.import_module(mod)
        print(f"  ✅ {{mod}}")
    except Exception as e:
        print(f"  ❌ {{mod}}: {{e!r}}")""".replace("{version}", VERSION)
    ),
    # ─── 4. Convertir a 22.05k mono ────────────────────────────────────
    _md(
        """## 4 · Convertir la grabación a 22.05 kHz mono

XTTS espera 22.05 kHz mono para training. Esta celda convierte tu archivo \
(cualquier formato) usando ffmpeg, que Colab trae preinstalado. Si ya existe \
el WAV convertido, se salta.
"""
    ),
    _code(
        """if WORK_WAV.exists() and WORK_WAV.stat().st_size > 1024:
    print(f"✓ Ya convertido: {WORK_WAV}")
else:
    print("Convirtiendo a 22.05 kHz mono...")
    !ffmpeg -y -i "{RAW_RECORDING}" -ar 22050 -ac 1 -c:a pcm_s16le "{WORK_WAV}" -loglevel error
    print("✅ Convertido.")

import soundfile as sf
info = sf.info(str(WORK_WAV))
print(f"  Duración: {info.duration/60:.1f} min  |  SR: {info.samplerate} Hz  |  canales: {info.channels}")"""
    ),
    # ─── 5. Transcribir + segmentar ────────────────────────────────────
    _md(
        """## 5 · Transcribir con faster-whisper y segmentar

faster-whisper transcribe con timestamps por palabra. Agrupamos palabras en \
segmentos de 2-11s cortando en pausas naturales (silencios > 0.4s) y límites \
de oración. Cada segmento se vuelve un par (wav, texto) del dataset.

**Si la grabación tiene varias voces** (`ES_MONOLOGO = False`): faster-whisper \
NO separa hablantes. Al terminar esta celda, abre `dataset/metadata_train.csv` \
en el explorador de archivos de Colab y borra las filas del hablante que NO \
quieres clonar. Luego re-corre la celda 6 en adelante.
"""
    ),
    _code(
        """import json
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# Cargar modelo Whisper (large-v3 en T4: usa float16)
print(f"Cargando faster-whisper {WHISPER_MODEL}...")
wmodel = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

print("Transcribiendo (5-10 min para 30 min de audio)...")
segments_gen, info = wmodel.transcribe(
    str(WORK_WAV),
    language=LANGUAGE,
    word_timestamps=True,
    vad_filter=True,                     # quita silencios — reduce alucinaciones
    vad_parameters={"min_silence_duration_ms": 400},
    beam_size=5,
    condition_on_previous_text=False,    # reduce alucinación en audio largo
)

# Materializar todas las palabras con timestamps
all_words = []
for seg in segments_gen:
    if seg.words:
        for w in seg.words:
            all_words.append({"start": w.start, "end": w.end, "word": w.word})

print(f"Palabras transcritas: {len(all_words)}")
TRANSCRIPT_JSON.write_text(json.dumps(all_words, ensure_ascii=False), encoding="utf-8")

# Liberar VRAM de Whisper ANTES de cargar XTTS
import gc, torch
del wmodel
gc.collect(); torch.cuda.empty_cache()
print("Whisper liberado de VRAM.")"""
    ),
    _code(
        """# Agrupar palabras en segmentos de MIN_SEG_S..MAX_SEG_S cortando en pausas.
def group_words_into_segments(words, min_s, max_s, pause_gap=0.4):
    segs, cur, cur_start = [], [], None
    for i, w in enumerate(words):
        if cur_start is None:
            cur_start = w["start"]
        cur.append(w)
        dur = w["end"] - cur_start
        # gap al siguiente word
        next_gap = (words[i+1]["start"] - w["end"]) if i + 1 < len(words) else 999
        ends_sentence = w["word"].strip().endswith((".", "?", "!", "…"))
        # Cerrar segmento si: dur en rango y (pausa o fin de oración), o si excede max
        if dur >= min_s and (next_gap >= pause_gap or ends_sentence or dur >= max_s):
            text = "".join(x["word"] for x in cur).strip()
            if len(text) >= 10:
                segs.append({"start": cur_start, "end": w["end"], "text": text})
            cur, cur_start = [], None
        elif dur >= max_s:
            # forzar corte aunque no haya pausa
            text = "".join(x["word"] for x in cur).strip()
            if len(text) >= 10:
                segs.append({"start": cur_start, "end": w["end"], "text": text})
            cur, cur_start = [], None
    return segs

raw_segments = group_words_into_segments(all_words, MIN_SEG_S, MAX_SEG_S)
print(f"Segmentos candidatos: {len(raw_segments)}")

# Cargar el audio una vez. Para UN archivo de 30 min a 22 kHz mono (~80 MB) y
# segmentación secuencial densa, cargar una vez es más eficiente que miles de
# seeks. (El notebook puente, que lee dispersamente de archivos largos, sí usa
# lectura parcial desde disco.) Liberamos el array completo al terminar.
import gc
audio_full, sr = sf.read(str(WORK_WAV), dtype="float32")
assert sr == TARGET_SR

# Recortar, limpiar (voicelegacy.audio) y guardar cada segmento
from voicelegacy.audio import apply_bandpass, denoise, loudness_normalize

rows = []
kept = 0
total_segs = len(raw_segments)
for idx, s in enumerate(raw_segments):
    if idx % 25 == 0:
        print(f"  🔄 segmento {idx}/{total_segs} ({idx/max(total_segs,1):.0%})...")
    a = int(s["start"] * sr)
    b = int(s["end"] * sr)
    clip = audio_full[a:b].astype(np.float32)
    if clip.size < int(MIN_SEG_S * sr):
        continue
    # Limpieza ligera: denoise no-estacionario + bandpass voz + normalización
    try:
        clip = denoise(clip, sr, stationary=False)
        clip = apply_bandpass(clip, sr)
        clip = loudness_normalize(clip, sr, target_lufs=-23.0, peak_ceiling_dbfs=-3.0)
    except Exception as e:
        print(f"  ⚠ clip {idx} limpieza falló ({e!r}); usando crudo")
    name = f"seg_{idx:04d}.wav"
    sf.write(str(WAVS_DIR / name), clip, sr, subtype="PCM_16")
    rows.append((f"wavs/{name}", s["text"], "target_speaker"))
    kept += 1

# Liberar el audio completo de RAM antes de la siguiente fase (training)
del audio_full
gc.collect()
print("  🧹 Audio completo liberado de RAM.")

total_dur_min = sum((sf.info(str(WAVS_DIR / r[0].split('/')[-1])).duration) for r in rows) / 60
print(f"\\nSegmentos guardados: {kept}")
print(f"Voz neta total     : {total_dur_min:.1f} min")

if kept < 50:
    print(f"\\n⚠ Solo {kept} segmentos. XTTS fine-tuning idealmente quiere ≥150. "
          "El resultado puede ser pobre. Considera grabar más audio.")
if total_dur_min < 12:
    print(f"\\n❌ Solo {total_dur_min:.1f} min de voz neta. Es muy poco. "
          "Fine-tuning probablemente NO mejorará. Necesitas más material.")"""
    ),
    _code(
        """# Escribir CSVs LJSpeech-like con split 90/10
import csv, random
random.seed(42)
random.shuffle(rows)
cut = int(len(rows) * 0.9)
train_rows, eval_rows = rows[:cut], rows[cut:]

def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_file", "text", "speaker_name"])
        w.writerows(rows)

write_csv(DATASET_DIR / "metadata_train.csv", train_rows)
write_csv(DATASET_DIR / "metadata_eval.csv",  eval_rows)
print(f"Train: {len(train_rows)}  |  Eval: {len(eval_rows)}")
print(f"\\nCSV en: {DATASET_DIR}")
print("\\n⚠ Si ES_MONOLOGO=False: abre metadata_train.csv y metadata_eval.csv,")
print("   borra las filas del hablante que NO quieres clonar, guarda, y sigue.")"""
    ),
    # ─── 6. Descargar base XTTS ────────────────────────────────────────
    _md("## 6 · Descargar modelo base XTTS-v2 (una vez, ~2 GB)"),
    _code(
        """import urllib.request

BASE_URLS = {
    "model.pth"        : "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config.json"      : "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
    "vocab.json"       : "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "dvae.pth"         : "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "mel_stats.pth"    : "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
    "speakers_xtts.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth",
}
for name, url in BASE_URLS.items():
    out = BASE_XTTS_DIR / name
    if out.exists() and out.stat().st_size > 1024:
        print(f"  ✓ {name} ({out.stat().st_size/(1024*1024):.1f} MB)")
        continue
    print(f"  ↓ {name} …", flush=True)
    urllib.request.urlretrieve(url, out)
    print(f"    ✅ {out.stat().st_size/(1024*1024):.1f} MB")
print("\\nBase XTTS-v2 listo.")"""
    ),
    # ─── 7. Configurar trainer ─────────────────────────────────────────
    _md(
        """## 7 · Configurar el GPTTrainer (calibrado para dataset pequeño)

Solo se entrena el GPT encoder; el vocoder HiFi-GAN queda congelado. Para tu \
dataset pequeño usamos `NUM_EPOCHS=10`, `LEARNING_RATE=3e-6` y \
`WEIGHT_DECAY=5e-2` — más regularización para no sobre-ajustar a 15-22 min.
"""
    ),
    _code(
        """from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
)
from datetime import datetime

GPT_ARGS = GPTArgs(
    max_conditioning_length      = 132300,
    min_conditioning_length      = 66150,
    debug_loading_failures       = False,
    max_wav_length               = int(MAX_SEG_S * 22050),
    max_text_length              = 200,
    mel_norm_file                = str(BASE_XTTS_DIR / "mel_stats.pth"),
    dvae_checkpoint              = str(BASE_XTTS_DIR / "dvae.pth"),
    xtts_checkpoint              = str(BASE_XTTS_DIR / "model.pth"),
    tokenizer_file               = str(BASE_XTTS_DIR / "vocab.json"),
    gpt_num_audio_tokens         = 1026,
    gpt_start_audio_token        = 1024,
    gpt_stop_audio_token         = 1025,
    gpt_use_masking_gt_prompt_approach = True,
    gpt_use_perceiver_resampler  = True,
)
AUDIO_CONFIG = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
DATASET_CONFIG = BaseDatasetConfig(
    formatter="coqui", dataset_name=RUN_NAME, path=str(DATASET_DIR),
    meta_file_train="metadata_train.csv", meta_file_val="metadata_eval.csv",
    language=LANGUAGE,
)
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out_path  = str(RUNS_DIR / f"GPT_XTTS_FT-{RUN_NAME}-{timestamp}")

TRAINER_CONFIG = GPTTrainerConfig(
    output_path=out_path, model_args=GPT_ARGS, run_name=RUN_NAME,
    project_name="voicelegacy_finetune_standalone",
    run_description=f"Standalone FT XTTS-v2 {LANGUAGE} {RUN_NAME}",
    dashboard_logger="tensorboard", logger_uri=None, audio=AUDIO_CONFIG,
    batch_size=BATCH_SIZE, batch_group_size=48, eval_batch_size=BATCH_SIZE,
    num_loader_workers=8, eval_split_max_size=256,
    print_step=50, plot_step=100, log_model_step=500, save_step=SAVE_EVERY_N_STEPS,
    save_n_checkpoints=1, save_checkpoints=True, print_eval=False,
    optimizer="AdamW", optimizer_wd_only_on_weights=True,
    optimizer_params={"betas":[0.9, 0.96], "eps":1e-8, "weight_decay":WEIGHT_DECAY},
    lr=LEARNING_RATE, lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones":[50000*18, 150000*18, 300000*18], "gamma":0.5, "last_epoch":-1},
    test_sentences=[],
)
print(f"Run → {out_path}")
print(f"{NUM_EPOCHS} epochs × batch {BATCH_SIZE} × grad_accum {GRAD_ACUMM_STEPS}, "
      f"lr={LEARNING_RATE}, wd={WEIGHT_DECAY}")"""
    ),
    # ─── 8. Entrenar ───────────────────────────────────────────────────
    _md(
        """## 8 · Entrenar

Esta celda toma 2-4 h. Si Colab te corta, ve a la celda 8.bis para reanudar.

**Si ves `CUDA out of memory`:** baja `BATCH_SIZE` a 2 y sube `GRAD_ACUMM_STEPS` \
a 126 en la celda 2, reinicia runtime, vuelve a correr desde la celda 1.
"""
    ),
    _code(
        """from TTS.tts.datasets import load_tts_samples

train_samples, eval_samples = load_tts_samples(
    DATASET_CONFIG, eval_split=True,
    eval_split_max_size=TRAINER_CONFIG.eval_split_max_size,
    eval_split_size=TRAINER_CONFIG.eval_split_size,
)
model = GPTTrainer.init_from_config(TRAINER_CONFIG)
trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False,
                start_with_eval=True, grad_accum_steps=GRAD_ACUMM_STEPS),
    TRAINER_CONFIG, output_path=out_path, model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
trainer.fit()
print("✅ Entrenamiento terminado.")"""
    ),
    _md("### 8.bis · Reanudar si Colab cortó la sesión"),
    _code(
        """import glob
runs = sorted(glob.glob(str(RUNS_DIR / f"GPT_XTTS_FT-{RUN_NAME}-*")))
assert runs, "No hay runs para reanudar."
last_run = runs[-1]
ckpts = sorted(glob.glob(f"{last_run}/checkpoint_*.pth"))
assert ckpts, f"No hay checkpoints en {last_run}."
restore = ckpts[-1]
print(f"Reanudando desde {restore}")

from TTS.tts.datasets import load_tts_samples
train_samples, eval_samples = load_tts_samples(
    DATASET_CONFIG, eval_split=True,
    eval_split_max_size=TRAINER_CONFIG.eval_split_max_size,
    eval_split_size=TRAINER_CONFIG.eval_split_size,
)
model = GPTTrainer.init_from_config(TRAINER_CONFIG)
trainer = Trainer(
    TrainerArgs(restore_path=restore, skip_train_epoch=False,
                start_with_eval=False, grad_accum_steps=GRAD_ACUMM_STEPS),
    TRAINER_CONFIG, output_path=out_path, model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
trainer.fit()"""
    ),
    # ─── 9. Materializar checkpoint ────────────────────────────────────
    _md("## 9 · Materializar checkpoint reutilizable"),
    _code(
        """import shutil, glob

best = sorted(glob.glob(f"{out_path}/best_model.pth")) or \\
       sorted(glob.glob(f"{out_path}/checkpoint_*.pth"))
assert best, f"No se encontró checkpoint en {out_path}"
ft_model = best[-1]

for old in FINETUNED_DIR.glob("*"):
    old.unlink()
shutil.copy(ft_model, FINETUNED_DIR / "model.pth")
shutil.copy(f"{out_path}/config.json", FINETUNED_DIR / "config.json")
for f in ("vocab.json", "dvae.pth", "mel_stats.pth", "speakers_xtts.pth"):
    shutil.copy(BASE_XTTS_DIR / f, FINETUNED_DIR / f)

print("Checkpoint final:")
for p in sorted(FINETUNED_DIR.iterdir()):
    print(f"  {p.name:24s} {p.stat().st_size/(1024*1024):8.1f} MB")"""
    ),
    # ─── 10. Validar + A/B ─────────────────────────────────────────────
    _md(
        """## 10 · Validar y comparar A/B contra el modelo base

Sintetiza la misma frase con BASE y con FINE-TUNED, mide \
`speaker_similarity_score` contra los segmentos del dataset. **Si \
ΔSim < +0.05, tu material de 30 min fue insuficiente.**
"""
    ),
    _code(
        """from voicelegacy.config import SynthesisConfig
from voicelegacy.finetuned_inference import (
    FineTunedCheckpoint, load_finetuned_model, synthesize_with_finetuned, release_finetuned_model
)
from voicelegacy.synthesis import load_xtts_model, synthesize_to_file, release_model
from voicelegacy.similarity import compute_similarity
from IPython.display import Audio, display
import glob as _glob

# Referencias = primeros 6 segmentos del dataset (limpios)
refs = sorted(WAVS_DIR.glob("*.wav"))[:6]
PHRASE = "Quiero contarte una historia importante de mi vida."

# ── BASE ──
base_tts = load_xtts_model(SynthesisConfig(language=LANGUAGE, device="cuda"), accept_tos=True)
out_base = WORKSPACE_FT / "ab_base.wav"
synthesize_to_file(base_tts, PHRASE, refs, out_base, SynthesisConfig(language=LANGUAGE, seed=42))
release_model()

# ── FINE-TUNED ──
ckpt = FineTunedCheckpoint.from_dir(FINETUNED_DIR)
ft_model = load_finetuned_model(ckpt, device="cuda")
out_ft = WORKSPACE_FT / "ab_finetuned.wav"
synthesize_with_finetuned(ft_model, ckpt, PHRASE, refs, out_ft,
                          SynthesisConfig(language=LANGUAGE, seed=42))

# ── Similarity ──
ref_all = sorted(WAVS_DIR.glob("*.wav"))[:10]
sim_base = compute_similarity(out_base, ref_all).score
sim_ft   = compute_similarity(out_ft,   ref_all).score
print(f"\\n  BASE       similarity: {sim_base:.4f}")
print(f"  FINE-TUNED similarity: {sim_ft:.4f}")
print(f"  ΔSim = {sim_ft - sim_base:+.4f}\\n")

if sim_ft - sim_base >= 0.05:
    print("✅ Fine-tuning APORTA. Usa este checkpoint.")
elif sim_ft - sim_base >= 0.0:
    print("⚠ Marginal. Escucha A/B y decide. Probablemente necesitas más audio.")
else:
    print("❌ Fine-tuning EMPEORA. 30 min fue insuficiente o transcripciones malas. "
          "No uses este checkpoint; consigue más material.")

display(Audio(str(out_base))); print("↑ BASE")
display(Audio(str(out_ft)));   print("↑ FINE-TUNED")"""
    ),
    _md(
        """## 11 · Usar el checkpoint en producción

Cada vez que quieras generar audio nuevo (en esta o futura sesión):

```python
from voicelegacy.config import SynthesisConfig
from voicelegacy.finetuned_inference import (
    FineTunedCheckpoint, load_finetuned_model, synthesize_with_finetuned
)

ckpt  = FineTunedCheckpoint.from_dir(FINETUNED_DIR)
model = load_finetuned_model(ckpt, device="cuda")
refs  = sorted(WAVS_DIR.glob("*.wav"))[:6]

synthesize_with_finetuned(
    model, ckpt,
    text="El texto que quieras que diga.",
    speaker_wav=refs,
    output_path=WORKSPACE_FT / "nuevo_clip.wav",
    config=SynthesisConfig(language=LANGUAGE, seed=42),
)
```

### Qué se guarda (sobrevive a sesión muerta)

| Cache | Vive en | Reutilizable |
|---|---|---|
| WAV convertido 22k | `source_22k_mono.wav` | ✅ |
| Transcripción palabras | `transcript_words.json` | ✅ |
| Dataset (wavs + CSV) | `dataset/` | ✅ |
| Base XTTS-v2 | `base_xtts/` | ✅ entre experimentos |
| Checkpoints intermedios | `runs/<run>/checkpoint_*.pth` | ✅ reanudar |
| Checkpoint final | `finetuned/` | ✅ inferencia |

**Si la sesión muere durante el entrenamiento:** vuelve, monta Drive, salta a \
8.bis. La transcripción y el dataset NO se reprocesan (ya están en Drive).
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
    print(f"✅ Notebook written: {NOTEBOOK_PATH}")
    print(f"   Cells: {len(nb['cells'])}")
    print(f"   Version targeted: {VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
