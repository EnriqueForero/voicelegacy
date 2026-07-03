"""Generator for notebook_voicelegacy_finetune.ipynb.

This produces a Colab notebook that fine-tunes the XTTS-v2 GPT encoder on
a user-supplied corpus, designed for Google Colab Free Tier (T4 GPU).

Why this exists as a generator instead of a hand-edited .ipynb:
1. The .ipynb format is JSON-with-uuids; hand-editing breaks diffs.
2. The companion notebook (notebook_voicelegacy.ipynb) follows the same
   pattern. Keeping both regenerable is a deliberate consistency choice
   that survived audit (see CHANGELOG 0.1.6 — the pre-commit hook
   `check_no_runtime_unassign.py` enforces this for the inference notebook
   and could be extended to this one if it ever grows live bombs).
3. Pre-commit can validate generator output without re-running training.

Run:
    python notebooks/build_finetune_notebook.py

Reference for the fine-tuning recipe:
    https://github.com/idiap/coqui-ai-TTS/blob/main/recipes/ljspeech/xtts_v2/train_gpt_xtts.py
    https://docs.coqui.ai/en/latest/models/xtts.html#training
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import nbformat as nbf
import tomllib

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebooks" / "notebook_voicelegacy_finetune.ipynb"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _version_from_pyproject() -> str:
    return tomllib.load(PYPROJECT_PATH.open("rb"))["project"]["version"]


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text, id=_cell_id())


def _code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src, id=_cell_id())


# ─── Notebook content ─────────────────────────────────────────────────────
VERSION = _version_from_pyproject()


CELLS_SPEC = [
    _md(
        f"""# voicelegacy {VERSION} · XTTS-v2 fine-tuning

**Para Google Colab Free Tier (T4 GPU).** Este notebook fine-tunea el GPT encoder \
de XTTS-v2 con TU corpus, sin tocar el vocoder. El resultado es un checkpoint \
local que el módulo `voicelegacy.finetuned_inference` carga para producir clips \
de mejor calidad que el zero-shot.

## Pre-requisitos

1. **Runtime → Change runtime type → T4 GPU**. Sin GPU, este notebook NO funciona.
2. Aceptar la [Coqui Public Model License (CPML)](https://coqui.ai/cpml).
3. Tener ya el corpus de referencia construido (`reference_corpus/` en Drive) \
con el notebook principal `notebook_voicelegacy.ipynb`. Necesitas ≥ 30 segmentos \
limpios; ideal 2-5 horas de material.

## Decisiones técnicas ancladas en este notebook

| Decisión | Valor | Por qué |
|---|---|---|
| `LANGUAGE` | `es` | Material colombiano. Para otros idiomas cambiar acá. |
| `BATCH_SIZE` | 3 | T4 con 15 GB VRAM aguanta hasta ~4, con margen para fragmentación |
| `GRAD_ACUMM_STEPS` | 84 | Equivale a batch efectivo 252 (recomendación oficial Coqui) |
| `NUM_EPOCHS` | 6 | Sweet spot para 2-5 h de audio. < 4 = under-train. > 10 = overfit |
| `MAX_AUDIO_LENGTH` | 11 s | Filtro de Coqui para evitar OOM en T4 |
| `OPTIMIZER` | AdamW (default) | Mejor para XTTS según [docs Coqui](https://docs.coqui.ai/en/latest/models/xtts.html#training) |
| `LEARNING_RATE` | 5e-6 | Conservador, evita catastrophic forgetting del GPT base |

## Lo que se guarda y NO se reprocesa

Todo va a Drive bajo `WORKSPACE_FT/`:

```
<WORKSPACE_FT>/
├── dataset/
│   ├── wavs/                 ← segmentos preparados para training
│   ├── metadata_train.csv    ← splits 90/10
│   └── metadata_eval.csv
├── base_xtts/                ← modelo base descargado UNA vez
│   ├── model.pth
│   ├── config.json
│   ├── vocab.json
│   ├── dvae.pth
│   ├── mel_stats.pth
│   └── speakers_xtts.pth
├── runs/
│   └── GPT_XTTS_FT-{{timestamp}}/  ← logs + checkpoints intermedios
│       ├── checkpoint_*.pth
│       └── trainer_0_log.txt
└── finetuned/                ← checkpoint final reutilizable
    ├── model.pth             ← copia de best_model.pth
    ├── config.json           ← copia del config usado
    ├── vocab.json            ← copia del base
    ├── dvae.pth              ← copia del base
    ├── mel_stats.pth         ← copia del base
    └── speakers_xtts.pth     ← copia del base
```

**Si Colab Free te corta la sesión a mitad de entrenamiento**, no pierdes nada: \
el Trainer de coqui-tts hace checkpoints intermedios. Al reabrir, retomas con \
`continue_path=` apuntando al run.
"""
    ),
    # ─── Sección 1: Conectar Drive y aceptar CPML ──────────────────────
    _md(
        """## 1 · Conectar Drive y aceptar CPML

Drive es la **única** forma de persistir checkpoints entre sesiones de Colab Free. \
RAM se borra al desconectar. Si no aceptas la CPML aquí, todas las celdas \
siguientes fallarán con `RuntimeError`.
"""
    ),
    _code(
        """# ✋ Lee y acepta la Coqui Public Model License antes de continuar.
# https://coqui.ai/cpml
ACCEPT_CPML = False  # CAMBIA A True si aceptas la licencia.

if not ACCEPT_CPML:
    raise RuntimeError(
        "Debes aceptar la CPML cambiando ACCEPT_CPML = True. "
        "El modelo XTTS-v2 (base y fine-tuneado) está sujeto a esa licencia."
    )

import os
os.environ["COQUI_TOS_AGREED"] = "1"
print("✅ CPML aceptada explícitamente.")"""
    ),
    _code(
        """from google.colab import drive
drive.mount('/content/drive')

# Verificar GPU T4
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"""
    ),
    # ─── Sección 2: Configuración ─────────────────────────────────────
    _md(
        """## 2 · Configuración del fine-tuning

Edita los valores marcados con `# EDITA` y NADA más en esta celda. El resto \
de hyperparams están razonadas en la tabla del bloque de introducción.
"""
    ),
    _code(
        """from pathlib import Path

# ─── EDITA estas 3 rutas ─────────────────────────────────────────────
WORKSPACE       = Path("/content/drive/MyDrive/voicelegacy_workspace")  # mismo del notebook de inferencia
WORKSPACE_FT    = Path("/content/drive/MyDrive/voicelegacy_finetune")   # nuevo, para fine-tuning
RUN_NAME        = "speaker00_es_v1"   # cambia entre experimentos para no sobreescribir

# ─── Idioma del corpus ──────────────────────────────────────────────
LANGUAGE = "es"      # uno de los 16 soportados por XTTS-v2

# ─── Hyperparams de training (razonadas, no cambies sin entender) ───
NUM_EPOCHS         = 6
BATCH_SIZE         = 3
GRAD_ACUMM_STEPS   = 84      # batch efectivo = BATCH_SIZE * GRAD_ACUMM = 252
MAX_AUDIO_LENGTH_S = 11      # filtro upstream Coqui contra OOM en T4
LEARNING_RATE      = 5e-6    # conservador
SAVE_EVERY_N_STEPS = 1000    # checkpoint intermedio (sesión Colab corta)

# ─── Derivado: no editar ────────────────────────────────────────────
REFERENCE_CORPUS = WORKSPACE / "reference_corpus"
DATASET_DIR      = WORKSPACE_FT / "dataset"
BASE_XTTS_DIR    = WORKSPACE_FT / "base_xtts"
RUNS_DIR         = WORKSPACE_FT / "runs"
FINETUNED_DIR    = WORKSPACE_FT / "finetuned"
SPEAKERSCRIBE_OUT = WORKSPACE / "speakerscribe_out"

for p in (WORKSPACE_FT, DATASET_DIR, DATASET_DIR / "wavs", BASE_XTTS_DIR, RUNS_DIR, FINETUNED_DIR):
    p.mkdir(parents=True, exist_ok=True)

print(f"WORKSPACE        : {WORKSPACE}")
print(f"WORKSPACE_FT     : {WORKSPACE_FT}")
print(f"REFERENCE_CORPUS : {REFERENCE_CORPUS} (exists={REFERENCE_CORPUS.exists()})")
print(f"LANGUAGE         : {LANGUAGE}")"""
    ),
    # ─── Sección 3: Instalar dependencias ─────────────────────────────
    _md(
        """## 3 · Instalar dependencias

Instalamos `coqui-tts` (fork mantenido de Idiap) + `voicelegacy` desde PyPI (o \
desde GitHub si aún no hay release público). El paquete `voicelegacy` solo se \
necesita para la inferencia con el checkpoint al final.

**Importante:** la celda muestra `ERROR: pip's dependency resolver...` con \
frecuencia en Colab. Es un mensaje informativo, no un fallo. Si la siguiente \
celda corre sin error, todo está bien.
"""
    ),
    _code(
        """# Coqui TTS (versión validada con voicelegacy 0.3.0)
# voicelegacy no está en PyPI: se instala desde el tag de GitHub.
!pip install -q coqui-tts==0.27.5 git+https://github.com/EnriqueForero/voicelegacy.git@v{version}

# Verificar imports críticos
import importlib
for mod in ("TTS", "TTS.tts.configs.xtts_config", "TTS.tts.models.xtts", "TTS.tts.layers.xtts.trainer.gpt_trainer", "voicelegacy.finetuned_inference"):
    try:
        importlib.import_module(mod)
        print(f"  ✅ {{mod}}")
    except Exception as e:
        print(f"  ❌ {{mod}}: {{e!r}}")""".replace("{version}", VERSION)
    ),
    # ─── Sección 4: Descargar modelo base ─────────────────────────────
    _md(
        """## 4 · Descargar el modelo base XTTS-v2 (una vez)

Si ya existe `BASE_XTTS_DIR/model.pth`, este paso se salta. La descarga pesa \
**~2 GB** y tarda 5-10 min según la red de Colab.
"""
    ),
    _code(
        """import urllib.request, hashlib

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
        size_mb = out.stat().st_size / (1024*1024)
        print(f"  ✓ {name} ya existe ({size_mb:.1f} MB)")
        continue
    print(f"  ↓ {name} …", flush=True)
    urllib.request.urlretrieve(url, out)
    size_mb = out.stat().st_size / (1024*1024)
    print(f"    ✅ {size_mb:.1f} MB")

print("\\nBase XTTS-v2 listo en", BASE_XTTS_DIR)"""
    ),
    # ─── Sección 5: Preparar dataset ──────────────────────────────────
    _md(
        """## 5 · Preparar el dataset desde el reference_corpus

XTTS espera datos en formato **LJSpeech-like**:

```
DATASET_DIR/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── metadata_train.csv      ← formato:  audio_name|transcripción|speaker
└── metadata_eval.csv       ← 10% del total
```

Esta celda hace dos cosas:

1. Copia los WAVs de `reference_corpus/` (ya limpios por `voicelegacy build-corpus`) a `DATASET_DIR/wavs/`, descartando los que excedan `MAX_AUDIO_LENGTH_S`.
2. Lee las **transcripciones** del JSON de speakerscribe (campo `text` por segmento). Las pareja con cada WAV usando el filename.

> **Aviso:** voicelegacy NO transcribe. Las transcripciones vienen de speakerscribe. \
> Si tu WAV de referencia no tiene transcripción correspondiente en el JSON, ese \
> archivo NO se incluye en el dataset y se logea.
"""
    ),
    _code(
        """# El emparejamiento WAV<->texto lo hace el paquete leyendo los sidecars .txt
# que voicelegacy escribe junto a cada WAV en reference_corpus/. NO se parsean
# nombres de archivo (eso causaba un dataset vacío en versiones previas).
from voicelegacy.finetune_dataset import build_finetune_dataset

ds = build_finetune_dataset(
    reference_corpus=REFERENCE_CORPUS,
    dataset_dir=DATASET_DIR,
    speaker_name="target_speaker",
)
print(f"WAVs aceptados : {ds.n_total}")
print(f"Train / Eval   : {ds.n_train} / {ds.n_eval}")
if ds.n_skipped_no_text:
    print(f"Sin sidecar .txt (omitidos): {ds.n_skipped_no_text}")

if ds.n_total < 30:
    raise RuntimeError(
        f"Solo {ds.n_total} WAVs con transcripción. Fine-tuning quiere >=30 "
        f"(idealmente >=200). Verifica que el reference_corpus tenga sidecars "
        f".txt (voicelegacy >= 0.3.4) y que el corpus se construyó del hablante "
        f"correcto."
    )
print(f"Dataset listo en {DATASET_DIR}")"""
    ),
    # ─── Sección 6: Configurar trainer ────────────────────────────────
    _md(
        """## 6 · Configurar el GPTTrainer

El **GPT encoder** es la única parte que se entrena. El **HiFi-GAN vocoder** se \
mantiene congelado (cero gradientes). Esta separación es lo que hace que el \
fine-tuning entre en T4: si tuviéramos que entrenar el vocoder también, ni \
con batch 1 cabría.

Los parámetros más sensibles ya están razonadas en el bloque de configuración \
arriba. Lo único que esta celda agrega es el `GPTArgs` con los token IDs que \
XTTS-v2 espera (no son hyperparams: son constantes del modelo).
"""
    ),
    _code(
        """from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
)
from TTS.utils.manage import ModelManager
from datetime import datetime

# Token IDs internos de XTTS-v2 — constantes, no editar
GPT_ARGS = GPTArgs(
    max_conditioning_length      = 132300,
    min_conditioning_length      = 66150,
    debug_loading_failures       = False,
    max_wav_length               = MAX_AUDIO_LENGTH_S * 22050,  # 22050 Hz inputs
    max_text_length              = 200,
    mel_norm_file                = str(BASE_XTTS_DIR / "mel_stats.pth"),
    dvae_checkpoint              = str(BASE_XTTS_DIR / "dvae.pth"),
    xtts_checkpoint              = str(BASE_XTTS_DIR / "model.pth"),
    tokenizer_file               = str(BASE_XTTS_DIR / "vocab.json"),
    gpt_num_audio_tokens         = 1026,
    gpt_start_audio_token        = 1024,
    gpt_stop_audio_token         = 1025,
    gpt_use_masking_gt_prompt_approach  = True,
    gpt_use_perceiver_resampler  = True,
)

AUDIO_CONFIG = XttsAudioConfig(
    sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
)

DATASET_CONFIG = BaseDatasetConfig(
    formatter   = "coqui",
    dataset_name= RUN_NAME,
    path        = str(DATASET_DIR),
    meta_file_train = "metadata_train.csv",
    meta_file_val   = "metadata_eval.csv",
    language    = LANGUAGE,
)

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out_path  = str(RUNS_DIR / f"GPT_XTTS_FT-{RUN_NAME}-{timestamp}")

TRAINER_CONFIG = GPTTrainerConfig(
    output_path                 = out_path,
    model_args                  = GPT_ARGS,
    run_name                    = RUN_NAME,
    project_name                = "voicelegacy_finetune",
    run_description             = f"Fine-tune XTTS-v2 GPT on {LANGUAGE} for {RUN_NAME}",
    dashboard_logger            = "tensorboard",
    logger_uri                  = None,
    audio                       = AUDIO_CONFIG,
    batch_size                  = BATCH_SIZE,
    batch_group_size            = 48,
    eval_batch_size             = BATCH_SIZE,
    num_loader_workers          = 8,
    eval_split_max_size         = 256,
    print_step                  = 50,
    plot_step                   = 100,
    log_model_step              = 1000,
    save_step                   = SAVE_EVERY_N_STEPS,
    save_n_checkpoints          = 1,   # solo el más reciente (Drive es escaso)
    save_checkpoints            = True,
    print_eval                  = False,
    optimizer                   = "AdamW",
    optimizer_wd_only_on_weights= True,
    optimizer_params            = {"betas":[0.9, 0.96], "eps":1e-8, "weight_decay":1e-2},
    lr                          = LEARNING_RATE,
    lr_scheduler                = "MultiStepLR",
    lr_scheduler_params         = {"milestones":[50_000*18, 150_000*18, 300_000*18], "gamma":0.5, "last_epoch":-1},
    test_sentences              = [],   # sin test sentences = sin reverse pass costoso
)

print(f"Run output → {out_path}")
print(f"Resumen: {NUM_EPOCHS} epochs × batch {BATCH_SIZE} × grad_accum {GRAD_ACUMM_STEPS}")"""
    ),
    # ─── Sección 7: Lanzar training ───────────────────────────────────
    _md(
        """## 7 · Lanzar el entrenamiento

**Esta es la celda que toma tiempo.** Para 2-5 h de audio y 6 epochs en T4: \
~3-5 h en total. Colab Free permite sesiones de hasta ~12 h con inactividad \
moderada — debería caber. Si te corta, ver la siguiente celda para reanudar.

Si ves errores de OOM (`CUDA out of memory`):
- Baja `BATCH_SIZE` a 2.
- Aumenta `GRAD_ACUMM_STEPS` a 126 para mantener batch efectivo 252.
- Reinicia el runtime (Runtime → Restart) y vuelve a correr desde la celda 2.

Puedes ver el progreso en tiempo real en el dashboard de Tensorboard \
(siguiente celda) — opcional pero recomendado.
"""
    ),
    _code(
        """# Cargar muestras
train_samples, eval_samples = load_tts_samples(
    DATASET_CONFIG,
    eval_split=True,
    eval_split_max_size=TRAINER_CONFIG.eval_split_max_size,
    eval_split_size=TRAINER_CONFIG.eval_split_size,
)

# Inicializar el GPTTrainer (carga pesos base, freeza vocoder)
model = GPTTrainer.init_from_config(TRAINER_CONFIG)

# Trainer
trainer = Trainer(
    TrainerArgs(
        restore_path=None,           # arranca desde el base. Usa continue_path para resumir.
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    TRAINER_CONFIG,
    output_path=out_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# ▶ Entrenamiento
trainer.fit()
print("✅ Entrenamiento terminado.")"""
    ),
    _md(
        """### 7.bis · Reanudar entrenamiento si la sesión se cortó

Si Colab Free te desconectó en mitad del entrenamiento, vuelve aquí, **NO** \
corras la celda anterior. Esta celda hace lo mismo pero apunta a un checkpoint \
intermedio (el último que se guardó en Drive).
"""
    ),
    _code(
        """# Encontrar el checkpoint más reciente
import glob
candidate_runs = sorted(glob.glob(str(RUNS_DIR / f"GPT_XTTS_FT-{RUN_NAME}-*")))
if not candidate_runs:
    raise RuntimeError("No hay runs previos para reanudar.")
last_run = candidate_runs[-1]
last_ckpt = sorted(glob.glob(f"{last_run}/checkpoint_*.pth"))
if not last_ckpt:
    raise RuntimeError(f"No hay checkpoints en {last_run}.")
restore = last_ckpt[-1]
print(f"Reanudando desde {restore}")

# Lo mismo que arriba pero con restore_path
train_samples, eval_samples = load_tts_samples(
    DATASET_CONFIG, eval_split=True,
    eval_split_max_size=TRAINER_CONFIG.eval_split_max_size,
    eval_split_size=TRAINER_CONFIG.eval_split_size,
)
model = GPTTrainer.init_from_config(TRAINER_CONFIG)
trainer = Trainer(
    TrainerArgs(
        restore_path=restore,
        skip_train_epoch=False,
        start_with_eval=False,       # ya hicimos eval inicial antes
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    TRAINER_CONFIG, output_path=out_path, model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
trainer.fit()"""
    ),
    # ─── Sección 8: Materializar checkpoint reutilizable ──────────────
    _md(
        """## 8 · Materializar el checkpoint final reutilizable

El Trainer deja `best_model.pth` dentro de `out_path`. Para que \
`voicelegacy.finetuned_inference.FineTunedCheckpoint.from_dir()` pueda \
cargarlo, copiamos los 6 archivos al directorio `FINETUNED_DIR` con los \
nombres EXACTOS que el módulo espera.

Es la única operación de "post-process". A partir de aquí, `FINETUNED_DIR` es \
el directorio que pasas como argumento a la inferencia.
"""
    ),
    _code(
        """import shutil

best_model = sorted(glob.glob(f"{out_path}/best_model.pth"))
if not best_model:
    # fallback a checkpoint más reciente
    best_model = sorted(glob.glob(f"{out_path}/checkpoint_*.pth"))
if not best_model:
    raise RuntimeError(f"No se encontró best_model.pth ni checkpoint_*.pth en {out_path}")

ft_model = best_model[-1]
ft_config = f"{out_path}/config.json"

# Limpiar destino y copiar los 6 archivos
target = FINETUNED_DIR
for old in target.glob("*"):
    old.unlink()

shutil.copy(ft_model,  target / "model.pth")
shutil.copy(ft_config, target / "config.json")
for f in ("vocab.json", "dvae.pth", "mel_stats.pth", "speakers_xtts.pth"):
    shutil.copy(BASE_XTTS_DIR / f, target / f)

# Resumen
print("Checkpoint final:")
for p in sorted(target.iterdir()):
    print(f"  {p.name:24s} {p.stat().st_size/(1024*1024):8.1f} MB")"""
    ),
    # ─── Sección 9: Validar con voicelegacy.finetuned_inference ──────
    _md(
        """## 9 · Validar el checkpoint con `voicelegacy.finetuned_inference`

Carga el checkpoint con el módulo del paquete y sintetiza 1 frase. Esto valida \
end-to-end que el archivo es consumible. Si esto pasa, ya tienes producto \
fine-tuneado listo para usar.
"""
    ),
    _code(
        """from voicelegacy.config import SynthesisConfig
from voicelegacy.finetuned_inference import (
    FineTunedCheckpoint, load_finetuned_model, synthesize_with_finetuned, release_finetuned_model
)

# 1. Validar files
ckpt = FineTunedCheckpoint.from_dir(FINETUNED_DIR)
print("Checkpoint validado:", ckpt.fingerprint)

# 2. Cargar
model = load_finetuned_model(ckpt, device="cuda")

# 3. Sintetizar con las primeras 3 referencias del corpus
refs = sorted(REFERENCE_CORPUS.glob("*.wav"))[:3]
out_wav = WORKSPACE_FT / "test_finetuned.wav"

synthesize_with_finetuned(
    model=model,
    checkpoint=ckpt,
    text="Hola, esta es una prueba del modelo fine-tuneado.",
    speaker_wav=refs,
    output_path=out_wav,
    config=SynthesisConfig(language=LANGUAGE, seed=42),
)

print(f"\\n✅ Generado: {out_wav}  ({out_wav.stat().st_size/1024:.1f} KB)")

# Reproducir
from IPython.display import Audio, display
display(Audio(str(out_wav)))"""
    ),
    # ─── Sección 10: Comparación A/B con similarity ───────────────────
    _md(
        """## 10 · Comparación A/B: base vs fine-tuneado

Sintetiza la misma frase con el modelo BASE y con el FINE-TUNEADO. Compara \
`speaker_similarity_score` contra el corpus de referencia. Si el fine-tuned \
no supera al base por al menos +0.05 en banda media (entre 0.65 y 0.85), el \
fine-tuning **no mejoró nada** — material insuficiente o muletillas del \
speaker amplificadas.
"""
    ),
    _code(
        """from voicelegacy.synthesis import load_xtts_model, synthesize_to_file, release_model
from voicelegacy.similarity import compute_similarity

PHRASE = "Quiero contarte una historia importante de mi vida."

# 1. Inferencia con modelo BASE
release_finetuned_model()  # liberar VRAM antes de cargar el base
base_tts = load_xtts_model(SynthesisConfig(language=LANGUAGE, device="cuda"), accept_tos=True)
out_base = WORKSPACE_FT / "ab_base.wav"
synthesize_to_file(
    tts=base_tts,
    text=PHRASE,
    speaker_wav=refs,
    output_path=out_base,
    config=SynthesisConfig(language=LANGUAGE, seed=42),
)
release_model()

# 2. Inferencia con FINE-TUNED
ft_model = load_finetuned_model(ckpt, device="cuda")
out_ft = WORKSPACE_FT / "ab_finetuned.wav"
synthesize_with_finetuned(
    model=ft_model, checkpoint=ckpt,
    text=PHRASE, speaker_wav=refs, output_path=out_ft,
    config=SynthesisConfig(language=LANGUAGE, seed=42),
)

# 3. Similarity vs corpus
ref_list = sorted(REFERENCE_CORPUS.glob("*.wav"))[:10]
sim_base = compute_similarity(out_base, ref_list).score
sim_ft   = compute_similarity(out_ft,   ref_list).score
print(f"\\n  BASE        similarity: {sim_base:.4f}")
print(f"  FINE-TUNED  similarity: {sim_ft:.4f}")
print(f"  ΔSim = {sim_ft - sim_base:+.4f}")

if sim_ft - sim_base >= 0.05:
    print("\\n✅ Fine-tuning APORTA. Usa el checkpoint en producción.")
elif sim_ft - sim_base >= 0.0:
    print("\\n⚠ Fine-tuning marginal. Escucha A/B antes de decidir.")
else:
    print("\\n❌ Fine-tuning EMPEORA. Posibles causas: overfit, material insuficiente, "
          "transcripciones mal alineadas. NO uses este checkpoint.")

display(Audio(str(out_base))); print("↑ BASE")
display(Audio(str(out_ft)));   print("↑ FINE-TUNED")"""
    ),
    # ─── Sección 11: Cleanup ──────────────────────────────────────────
    _md(
        """## 11 · Cleanup (opcional)

Drive Free son 15 GB. Cada checkpoint pesa ~5 GB. Si el fine-tuning fue exitoso \
y vas a hacer otro experimento, considera borrar los `runs/` intermedios \
(conservan TensorBoard logs, no son críticos):
"""
    ),
    _code(
        """# Mostrar el espacio usado
!du -sh {WORKSPACE_FT} {WORKSPACE_FT}/runs/* 2>/dev/null

# Para borrar runs (descomenta cuando estés seguro):
# import shutil
# for r in (WORKSPACE_FT / 'runs').glob('GPT_XTTS_FT-*'):
#     shutil.rmtree(r)
# print('Limpiado.')"""
    ),
    _md(
        """## Resumen operativo

| Cache | Vive en | Reutilizable |
|---|---|---|
| Base XTTS-v2 (6 archivos, ~2 GB) | `WORKSPACE_FT/base_xtts/` | ✅ entre runs y experimentos |
| Dataset preparado (wavs + CSVs) | `WORKSPACE_FT/dataset/` | ✅ si no cambias SPEAKER_LABEL |
| Checkpoints intermedios | `WORKSPACE_FT/runs/<run>/checkpoint_*.pth` | ✅ para reanudar (`restore_path`) |
| Checkpoint final reutilizable | `WORKSPACE_FT/finetuned/` (6 archivos) | ✅ entre sesiones de inferencia |
| Modelo cargado en VRAM | `_FT_MODEL_CACHE` del módulo | ❌ sesión Colab |
| Conditioning latents | `_FT_LATENTS_CACHE` del módulo | ❌ sesión Colab |

**Si la sesión muere:** vuelve, monta Drive, salta a la celda 7.bis. Nada se pierde.

**Si quieres re-fine-tunear con otro speaker:** cambia `SPEAKER_LABEL` y `RUN_NAME`, \
regenera el dataset (celda 5), corre `trainer.fit()` (celda 7).

**Si quieres comparar 2 checkpoints:** instala uno como `finetuned/`, valida (celda 9), \
muévelo a `finetuned_A/`. Renombra el otro como `finetuned/` y repite. \
`FineTunedCheckpoint.from_dir(<path>)` acepta cualquier ruta — el módulo está pensado \
para esto.
"""
    ),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS_SPEC
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
        },
    }
    return nb


def main() -> int:
    nb = build_notebook()
    # Validate against nbformat schema
    nbf.validate(nb)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
        f.write("\n")  # trailing newline so end-of-file-fixer is idempotent
    print(f"✅ Notebook written: {NOTEBOOK_PATH}")
    print(f"   Cells: {len(nb['cells'])}")
    print(f"   Version targeted: {VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
