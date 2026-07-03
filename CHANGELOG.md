# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

## [0.7.1] — 2026-07-02 — fidelidad de clonación: selección top-N real, re-rolls con entropía y condicionamiento unificado

PATCH de corrección: tres defectos que degradaban silenciosamente la calidad
de la voz clonada, hallados en auditoría de código. Ningún cambio de API
rompe compatibilidad; `LongFormSynthesizer` gana un flag opcional.

### Fixed
- **`synthesize` / `synthesize-long` ahora usan las top-N referencias por
  calidad, no el directorio entero en orden alfabético.** `run_reference_phase`
  escribe TODOS los segmentos filtrados a `reference_corpus/` y su ranking
  top-N vivía solo en el reporte JSON; los comandos de síntesis globeaban el
  directorio, de modo que (a) segmentos que FALLARON la compuerta de SNR/clipping
  entraban al condicionamiento y (b) el latente GPT de XTTS — calculado sobre los
  primeros `gpt_cond_len` segundos — quedaba dominado por lo que ordenara primero
  alfabéticamente (= cronológico), no por lo mejor. Nuevo helper
  `quality.select_reference_wavs` re-puntúa el directorio en síntesis (barato:
  solo estadísticas, sin GPU) y devuelve el top-N mejor-primero. Nueva opción
  `--top-n` en ambos comandos (default: `ReferenceConfig.top_n_segments` = 5;
  `0` = comportamiento legado explícito). Archivos ilegibles se excluyen con
  warning en vez de tumbar la corrida; si nada pasa las compuertas, fail-open
  a todos los decodificables con warning fuerte.
- **Los reintentos de `synthesize-long` ahora re-tiran dados de verdad.**
  `synthesize_to_file` re-sembraba el RNG a `SynthesisConfig.seed` (default 42)
  antes de CADA inferencia, así que cada re-roll reproducía el MISMO audio
  fallido: el presupuesto `max_retries` quemaba GPU sin poder corregir nada.
  El closure del CLI deriva ahora `seed = base + attempt` (helper puro
  `cli._config_for_attempt`), reproducible: el sidecar ya registra `retries`,
  así que `base_seed + retries` recrea el audio publicado byte a byte.
  `LongFormSynthesizer` acepta `synthesize_accepts_attempt=True` (opt-in
  explícito; default False mantiene compatibilidad con callables `(text)`).
- **Ambas rutas de inferencia condicionan la voz igual.** La ruta rápida con
  caché de latentes llamaba `get_conditioning_latents` con los defaults crudos
  (`gpt_cond_len=6`, `max_ref_length=30`) mientras el fallback `tts_to_file`
  usaba los de `XttsConfig` (12/10): mismo corpus, dos voces distintas según
  qué ruta corriera. `SynthesisConfig` expone ahora `gpt_cond_len` (12),
  `gpt_cond_chunk_len` (4), `max_ref_len` (10) y `sound_norm_refs` (False) —
  defaults espejo del upstream idiap — y ambas rutas los leen. La clave del
  caché de latentes incluye los knobs (antes, cambiar `gpt_cond_len` habría
  servido latentes rancios). Validador: `gpt_cond_chunk_len <= gpt_cond_len`
  (invariante upstream).

- **El estimador de rango dinámico (“SNR”) medía ~0 dB para cualquier audio
  real.** `librosa.util.frame(y, ..., axis=0)` con entrada 1-D devuelve
  `(n_frames, frame_length)` y el código promediaba sobre `axis=0` — a través
  de los frames — colapsando 300+ frames en 2048 valores casi idénticos:
  top-10%/bottom-10% ≈ 1 → ~0.1 dB medido en señal limpia tipo voz. Efecto en
  cascada: la compuerta `snr >= 15 dB` de `evaluate_file` rechazaba TODOS los
  candidatos reales (ranking vacío, `run_batch_synthesis` abortaba por
  `MIN_USABLE_REFERENCE_SEGMENTS`) y `denoise_only_if_noisy` denoiseaba
  SIEMPRE (0 < umbral), introduciendo artefactos en cada segmento del corpus.
  Reemplazado el framing por `numpy.lib.stride_tricks.sliding_window_view`
  (eje de muestras inequívoco) con RMS por frame en float64. Medición
  post-fix: señal limpia tipo voz ≈ 19.8 dB, ruidosa (noise=0.02) ≈ 16.9 dB —
  monotónico y discriminante. Tests de regresión anti-eje añadidos.

### Notes
- Los hashes de idempotencia de `run_batch_synthesis` incorporan los campos
  nuevos vía `model_dump_json()`: outputs cacheados de 0.7.0 se re-sintetizan
  una vez tras actualizar (esperado y correcto: el condicionamiento cambió).
- 16 tests nuevos cubren selección/ranking/fail-open/archivos corruptos,
  forwarding de `attempt`, derivación de seed y knobs en ambas rutas.

## [0.7.0] — 2026-06-11 — Fase 3 · síntesis de audio largo (LongFormSynthesizer)

Salto MINOR (0.6.0 → 0.7.0): API pública nueva, el objetivo declarado del proyecto.
XTTS-v2 se descarrila pasados ~250–270 caracteres por generación (trunca/alucina),
que es lo que el baseline expone como WER alto en los estímulos `long`/`chapter`.
El `LongFormSynthesizer` orquesta muchas síntesis cortas en un audio continuo **y
detecta/corrige los fallos** en vez de confiar en que no ocurran. Toda la lógica
(chunking, ensamblado, verificación, caché, orquestación) es testeable **sin GPU**
porque la síntesis se inyecta; los números finales de fidelidad los produce el
usuario corriendo `benchmark` antes/después en una T4.

### Fase 3 — pasos del plan

- **Paso 2 (chunker de texto) — completado.** Nuevo módulo `voicelegacy.longform`
  con `segment_text` y `Chunk`. Segmenta texto largo en trozos sintetizables por
  XTTS (que se descarrila pasados ~250–270 caracteres) respetando jerarquía
  oración → cláusula → espacio, **sin exceder el presupuesto** y **sin partir
  palabras**. Maneja las trampas del español: abreviaturas honoríficas/de medida
  cuyo punto no es fin de oración (`Sr.`, `Dra.`, `núm.`), decimales/miles
  (`3,5` / `1.000`), horas y rangos (`14:30`, `5–7`), signos de apertura
  (`¿ ¡`), puntos suspensivos y comillas de cierre. Cada trozo etiqueta la
  frontera que lo sigue (paragraph/sentence/clause/hard/end) para que el
  ensamblador (paso 3) elija pausa o crossfade. 31 tests, módulo al 94% de
  cobertura. Verificado sobre el "capítulo" real del golden set (4.366 chars →
  33 trozos, máx. 215 chars, 805 palabras preservadas en orden exacto).
  - Aún NO se expone en el `__init__` público ni se sube de versión: es una
    pieza interna del LongFormSynthesizer que se completa en los pasos 3–8
    (ensamblador de audio, verificación ASR + reintentos, checkpoint/resume,
    orquestador, comando CLI). El bump a 0.7.0 ocurre al cerrar la fase.

- **Paso 3 (ensamblador de audio) — completado.** Primitivas DSP reutilizables
  en `audio.py`: `equal_power_crossfade` (crossfade de potencia constante —
  cos/sin— para que no haya caída de ~3 dB en cada costura, a diferencia del
  lineal), `silence` y `concatenate_audio`. Y `longform.assemble_chunks`, que
  une los trozos según la frontera de cada uno: `hard` → crossfade (corte
  forzado a mitad de cláusula, sin pausa real); `clause`/`sentence`/`paragraph`
  → silencio creciente; recorta el silencio de cada trozo antes de unir para
  que las pausas sean deterministas. 16 tests (incluida la verificación de que
  el equal-power no hunde la potencia en material no correlacionado).
- **Paso 4 (compuerta ASR `verify_chunk`) — completado.** Transcribe cada trozo
  renderizado con faster-whisper y mide WER contra el texto pedido, **reutilizando
  el ASR y el WER del harness (sin duplicar código)**. Es el detector barato del
  modo de fallo dominante en texto largo (truncamiento/alucinación). Degrada con
  seguridad: si falta faster-whisper, `status="skipped"` con `passed=True` (no se
  puede verificar → no se bloquea); un error de ASR es `status="error"` con
  `passed=True` (un fallo del ASR no debe descartar audio bueno). El bucle de
  reintento que la consume vive en el orquestador (paso 6). 5 tests.
  - `voicelegacy.longform` va al 95% de cobertura; total de tests 361.

- **Paso 5 (checkpoint / resume) — completado.** `LongFormCache` + `compute_doc_hash`
  en `voicelegacy.longform`. Caché por trozo en disco
  (`<cache_root>/<doc_hash>/chunk_NNNN.wav` + `manifest.json`) para que un capítulo
  pueda **reanudarse** si Colab se desconecta (límite de 12 h): el orquestador
  consulta `has_chunk(i)` y solo sintetiza los trozos faltantes. El directorio
  ES el hash de (texto + config), así que **cambiar el texto o la config invalida
  la caché automáticamente** (no se reusa audio rancio). Guarda la verificación
  ASR y el conteo de reintentos por trozo; tolera un manifiesto corrupto
  (empieza de cero sin reventar). 10 tests, incluida la simulación de reinicio
  (nueva instancia sobre el mismo directorio ve los trozos hechos). Total: 371 tests.

- **Paso 6 (orquestador `LongFormSynthesizer`) — completado.** El corazón de la
  fase: ata chunker + verify + caché + ensamblador con la **síntesis inyectada**
  (corre con XTTS real o con mock; testeable sin GPU). Incluye el **bucle de
  reintento**: cada trozo se verifica con ASR y, si falla (el modo dominante de
  XTTS en texto largo es el truncamiento silencioso), se **re-sintetiza** hasta
  `max_retries` veces — XTTS es estocástico, re-tirar el dado suele arreglarlo.
  Si todos los intentos fallan, **conserva el de menor WER y marca el trozo en el
  sidecar** para revisión humana: los fallos se vuelven visibles y raros en vez de
  silenciosos. Usa `LongFormCache` para reanudar y escribe un sidecar de auditoría
  (`<salida>.sidecar.json`) con WER por trozo, reintentos, trozos marcados, RTF y
  VRAM. Nuevas clases: `LongFormConfig` (dataclass con validación; su
  `cache_config()` excluye pausas/crossfade porque solo afectan el ensamblado,
  no el audio cacheado), `RenderedChunk`, `LongFormResult`, `LongFormSynthesizer`.
  Confirmado el paso 1: `synthesis.py` **ya cachea los latentes de condicionamiento**
  (`_get_conditioning_latents_cached`), lo que habilitará reusar la voz una sola
  vez en el comando CLI sin que derive entre trozos. 18 tests (render end-to-end
  con mock, reintento-luego-pasa, todos-fallan-marca-y-conserva-mejor, resume,
  force, invalidación por cambio de texto). `longform.py` al 96% de cobertura;
  total: 389 tests.

  Decisión de alcance honesta: el **re-troceo** de un trozo que falla persistentemente
  (mencionado en el plan) NO se implementó en este v1. Razón: re-trocear a mitad de
  render rompe el modelo de índices de la caché, y el reintento ya cubre la mayoría
  de los casos (la estocasticidad de XTTS). Entregar re-troceo a medias sería deuda;
  el comportamiento actual (reintento + marcar lo irrecuperable) es completo y honesto.
  El re-troceo queda como refinamiento futuro documentado.

- **Paso 7 (comando CLI `synthesize-long`) — completado.** Sintetiza texto largo
  (inline `--text` o archivo `--text-file`) a un WAV continuo. Construye el callable
  real envolviendo `synthesize_to_file`, que **reutiliza los latentes de
  condicionamiento cacheados** entre trozos (la voz no deriva). Flags:
  `--max-chunk-chars`, `--max-wer`, `--max-retries`, `--no-asr-verify`,
  `--resume/--no-resume`, `--force`, `--accept-tos`. Imprime una tabla resumen
  (duración, trozos, desde caché, WER medio, marcados, RTF) y avisa de los trozos
  marcados. 6 tests con `CliRunner` y modelo mockeado (end-to-end → WAV + sidecar).

### Added (API pública de Fase 3)

- `voicelegacy.LongFormSynthesizer`, `LongFormConfig`, `LongFormResult` y
  `segment_text` exportados desde el paquete raíz.
- Comando `voicelegacy synthesize-long`.
- `docs/P3_LONGFORM.md` (especificación de diseño) y sección en el README.

### Verified

- **395 tests verdes**, `voicelegacy.longform` al ~96% de cobertura, total ≥80%.
- `ruff`/`pre-commit` limpios; `build`+`twine` PASSED; `LongFormSynthesizer` y el
  comando importables/funcionales desde el wheel instalado (render → WAV + sidecar).
- Confirmado el paso 1: `synthesis.py` ya cachea los latentes de condicionamiento
  (`_CONDITIONING_LATENTS_CACHE`), reutilizados entre trozos para consistencia de voz.

### Cierre de Fase 3 — lo que el usuario debe correr en T4

El criterio de éxito (WER bajo en `long`/`chapter`) se **demuestra**, no se asume:
sintetice esos estímulos con `synthesize-long` y re-corra `benchmark` para comparar
contra el baseline congelado. La librería entrega el código + tests con mock; el
número final requiere GPU + el corpus real. Ver `docs/P3_LONGFORM.md` §validación.

- P3-27 (tabla SNR fuente → similarity esperada): experimento empírico con audio real pendiente. Bloqueante para v1.0; no bloqueante para v0.3.0 que aporta valor de ingeniería independiente.
- P3-31 (Polars/DuckDB): no aplica al volumen actual (1-3 entrevistas).

## [0.6.0] — 2026-06-11 — Fase 2 · precisión del corpus de referencia (sin cambiar defaults)

Salto MINOR (0.5.0 → 0.6.0): API aditiva. Atiende el tercer bloque del plan —
precisión del corpus. Todas las palancas nuevas son **opt-in con default igual al
comportamiento actual**: cambiar un default de selección es un cambio de precisión
que la regla de oro exige medir con el harness primero. Aquí se entregan las
herramientas (tested), no un cambio de comportamiento silencioso.

### Added

- **`audio.clean_segment`** — cadena de limpieza **canónica única** (DRY). Antes
  había dos cadenas con órdenes distintos: la inline de `corpus.extract_segments_to_wav`
  (denoise→bandpass→preemphasis, la correcta) y `audio.preprocess_full`
  (bandpass→preemphasis→denoise, la vieja). Ahora ambas delegan en `clean_segment`,
  así que no pueden volver a divergir. Orden fijo: denoise (opcionalmente condicional)
  → band-pass → pre-emphasis → trim → drop-si-<min → loudness-normalize.
- **Denoise condicional por SNR** (`ReferenceConfig.denoise_only_if_noisy` +
  `denoise_snr_threshold_db`): cuando se activa, solo denoisa segmentos cuyo rango
  dinámico estimado cae bajo el umbral. Denoisar audio ya limpio mete artefactos.
  Default off → denoise incondicional como antes.
- **Compuerta de banda por rolloff espectral** (`ReferenceConfig.min_spectral_rolloff_hz`,
  default 0 = desactivada): `AudioStats` ahora incluye `spectral_rolloff_hz` (mediana
  del rolloff 85%%), y `quality.score_segment` puede rechazar audio de banda angosta.
  **Corrige un bug real:** la compuerta anti-telefónico solo miraba el sample-rate del
  header, que NO detecta audio de banda telefónica (~3,4 kHz) **re-muestreado** a 22/44 kHz
  — ese audio pasaba con `sample_rate=44100`. El rolloff se mide de la señal y lo caza.
- **Compuerta de solapamiento de hablantes** (crosstalk) en `corpus`
  (`ReferenceConfig.enable_overlap_filter` + `max_overlap_ratio`, default off, mismo
  patrón que el filtro F0): `compute_overlap_seconds`, `analyze_overlap`,
  `filter_overlapping_segments`, `OverlapResult`, `write_overlap_report`. Rechaza
  segmentos del hablante objetivo solapados con otros hablantes (mismo audio fuente)
  por encima del ratio; el solapamiento se calcula como unión de intervalos (no
  doble-cuenta a dos interlocutores simultáneos). Escribe `reports/overlap_<ts>.json`.
- **20 tests nuevos** (`test_precision_gates.py` + seam update en `test_corpus.py`):
  rolloff narrowband vs wideband, el caso teléfono-re-muestreado que pasa el gate de
  sample-rate pero falla el de rolloff, denoise condicional (skip en limpio / corre en
  ruidoso), drop por duración, y la matemática de overlap (parcial, contención, merge
  de múltiples, distinto hablante/fuente no cuenta). Total: 309 tests.

### Changed

- `corpus.extract_segments_to_wav` y `audio.preprocess_full` ahora usan
  `clean_segment` (refactor interno; el orden de `preprocess_full` se corrige al
  canónico, sin efecto con sus defaults porque band-pass/pre-emphasis están off).
- `QualityReport.to_dict` ahora incluye `spectral_rolloff_hz`.

### Deferred (con motivo, no por descuido)

- **Score de selección perceptual** y **VAD Silero**: el score de selección es una
  palanca de tuning pura; re-ponderarlo sin un número del harness viola la regla de oro.
  Silero VAD es una dependencia opcional pesada de menor prioridad. Ambos se abordan
  después de que exista el baseline para poder medir su efecto. Las correcciones de esta
  fase (rolloff, overlap, DRY) son de exactitud, no de tuning, por eso sí entran ahora.

### Verified

- 309 tests verdes, cobertura 85,1%. `ruff`/`pre-commit` limpios. `build`+`twine` PASSED.
- Refactor confirmado no-rompiente: los 289 tests previos siguen verdes (un test que
  parcheaba un detalle de implementación —`corpus.trim_silence`— se actualizó al nuevo
  seam `audio.trim_silence`, mismo comportamiento verificado).



Salto MINOR (0.4.1 → 0.5.0): API pública nueva. Atiende el segundo hallazgo
estructural de la auditoría — "las compuertas de calidad medían la entrada, nada
medía la salida". Convierte el juicio subjetivo ("¿suena como ella?") en números
reproducibles. **Regla de oro establecida:** ninguna mejora de precisión se
mergea sin un número de este harness.

### Added

- **Módulo `voicelegacy.evaluation`** — harness de evaluación. La síntesis se
  **inyecta** como callable ``(texto) -> (waveform, sample_rate)``: corre con el
  XTTS real en T4 o con un mock en los tests, sin que el módulo importe coqui-tts.
  Métricas (todas opcionales, degradan a ``skipped`` si falta su dependencia, igual
  que `similarity`):
  - **SECS (similitud de hablante)** con **ECAPA-TDNN** (SpeechBrain), el estándar
    de los papers de TTS. Resemblyzer se mantiene como segunda opinión barata.
    Aviso documentado: ECAPA y Resemblyzer viven en **escalas distintas**; las
    bandas 0,60/0,75/0,85 de Resemblyzer NO se transfieren a ECAPA.
  - **WER/CER round-trip**: transcribe la salida con **faster-whisper** y la compara
    contra el texto pedido. Único detector barato del modo de fallo dominante en
    texto largo: truncamiento/alucinación/palabras comidas. El cálculo de WER/CER
    (distancia de edición) se implementó aquí, **sin dependencias nuevas**; solo el
    ASR necesita faster-whisper.
  - **MOS proxy** vía **TorchAudio-SQUIM** objetivo (PESQ/STOI/SI-SDR, sin referencia
    limpia). Proxy consistente, no verdad absoluta.
  - **RTF** (segundos de cómputo / segundos de audio) y **VRAM pico**, vía `telemetry`.
- **Comando `voicelegacy benchmark`** — sintetiza cada estímulo con el modelo y el
  corpus reales, puntúa fidelidad y velocidad, escribe `reports/benchmark_<run>.json`
  y acumula `reports/benchmarks.parquet` (con fallback a JSONL si no hay pandas).
  Persiste el audio sintetizado para auditoría. Flags `--no-ecapa/--no-resemblyzer/
  --no-wer/--no-mos`, `--texts`, `--limit`.
- **Golden set versionado** `voicelegacy/data/golden_texts_es.txt` — sustrato de
  medición congelado: 30 frases cortas, 10 medias (~200 chars, al límite de XTTS),
  5 párrafos largos y 1 "capítulo" (~4.4k chars, prueba de estrés de audiolibro).
  Contenido es-CO con números, fechas, siglas (DIAN, DANE, ICA, SENA, EPS) y nombres
  propios colombianos — donde XTTS tropieza. Se empaqueta como package-data para que
  el comando lo encuentre en el wheel instalado.
- **API pública**: `BenchmarkConfig`, `BenchmarkReport`, `run_benchmark`,
  `load_golden_texts` exportados desde `voicelegacy`.
- **Extra `eval`** (`pip install "voicelegacy[eval] @ git+..."`): speechbrain,
  torchaudio, faster-whisper, num2words, pandas, pyarrow. Floors provisionales,
  a pinar tras validar en la imagen T4. Incluido en `all`.
- **80 tests nuevos** (`test_evaluation.py`, `test_packaging.py` ampliado, comando
  benchmark en `test_cli.py`): orquestación end-to-end con síntesis mock, WER/CER,
  normalizador es, escritura de reporte, agregación, acumulación parquet/JSONL, y el
  skip de cada métrica sin su dependencia. Total: 286 tests.

### Honest limitation (lo que este entorno NO puede entregar)

- **El baseline empírico no está incluido.** Congelar el "antes" (XTTS zero-shot +
  corpus real) exige GPU T4 + pesos de XTTS + el corpus del usuario, ausentes aquí.
  El harness y sus tests con mock están completos y verdes; los números reales
  (SECS/WER/MOS del baseline) los produce el usuario corriendo `voicelegacy benchmark`
  en Colab. "Listo-para-producción-como-código" ≠ "baseline-validado". El criterio de
  aceptación de Fase 1 (tabla baseline en el README) se cierra cuando el usuario corre
  ese comando en T4.

### Verified

- 286 tests verdes, cobertura **83,4%** (piso 80%). Lo no cubierto en `evaluation.py`
  son los cuerpos de inferencia de modelos (ECAPA/ASR/SQUIM), no ejecutables sin GPU
  ni pesos — se validan en T4, no en CI.
- `ruff check`/`format` limpios; `pre-commit run --all-files` verde.
- Comando `benchmark` probado end-to-end con el modelo XTTS mockeado (escribe reporte,
  agrega, imprime tabla).



Salto PATCH (0.4.0 → 0.4.1): no cambia la API pública ni el comportamiento de la
librería. Repara que un tercero **no podía instalar ni usar** el paquete y endurece
las compuertas de CI para que las regresiones de empaquetado no vuelvan en silencio.

### Fixed

- **P0-1 — `pyproject.toml` inválido (bloqueante absoluto).** `[tool.setuptools.packages.find]` tenía `where = [".]` (comilla sin cerrar, línea 54) → `pip install -e .` reventaba con `TOMLDecodeError: Illegal character '\n' (line 54)`. Corregido a `where = ["."]`. **Causa raíz:** ningún generador regenera el archivo (se verificó); fue una regresión manual que el hook `scripts/check_pyproject_toml.py` habría atrapado, pero el hook nunca corría porque no estaba cableado al CI (ver más abajo).
- **P0-6 — el CI validaba el `pyproject.toml` *después* de `pip install -e .`**, es decir después del paso que ya había reventado. La validación se movió **antes** del install (fail-fast) en `ci.yml` y `release.yml`.
- **Fallo latente de CI (no estaba en el plan):** el paso "Validate all generated notebooks" ejecuta `scripts/check_notebook_schema.py`, que importa `nbformat`, pero `nbformat` no se instalaba en CI → ese paso habría fallado con `ModuleNotFoundError`. `nbformat` se añadió al extra `dev`.
- **Deriva de versión:** `__init__.py` quedaba en `0.4.0`; ahora sigue al manifiesto.

### Added

- **P0-2 — extras documentados restaurados.** `[project.optional-dependencies]` solo tenía `dev`, pero README y notebooks instruyen `voicelegacy[similarity|finetune|deepfilter|all]`. Restaurados: `similarity = ["resemblyzer"]`, `finetune = ["faster-whisper>=1.0,<2.0"]`, `deepfilter = ["deepfilternet>=0.5"]`, y `all` como meta-extra **auto-referencial** (`voicelegacy[similarity,finetune,deepfilter]`) para no duplicar specs.
- **`pre-commit run --all-files` en CI.** Hace que los hooks locales (TOML válido, schema de notebook, no `runtime.unassign()` ejecutable) se cumplan de verdad y no solo en teoría. Es ahora la única fuente de verdad para lint+format (se eliminaron los pasos `ruff` duplicados en `ci.yml`).
- **`tests/test_packaging.py`** — test de contrato que parsea el manifiesto real y afirma: TOML válido, los 4 extras existen y no están vacíos, `all` es la unión de los extras de features, versión sincronizada con `__version__`, y piso de cobertura presente. Blinda P0-1/P0-2/P0-5 contra regresión silenciosa.
- **`pre-commit>=3.7,<5.0`** añadido al extra `dev`.

### Changed

- **P0-3 — distribución resuelta.** Los notebooks instalaban `voicelegacy==0.4.0` desde PyPI, pero el paquete **no está publicado en PyPI** (404). Las 3 celdas de instalación afectadas (bridge, finetune, finetune-standalone) ahora instalan desde el tag de GitHub: `pip install git+https://github.com/EnriqueForero/voicelegacy.git@v0.4.1`. El README se alineó al mismo método. (El notebook principal ya instalaba desde una carpeta en Drive; no se tocó.)
- **P0-5 — piso de cobertura unificado.** README prometía 75% y `pyproject.toml` imponía `--cov-fail-under=25`. Una sola verdad: piso a **80** (cobertura real medida: 86.7%). README corregido.
- **P0-4 — `ruff` pineado.** `ruff>=0.9.0` (sin tope, CI no reproducible) → `ruff==0.15.16` en el extra `dev` y en `.pre-commit-config.yaml` (migrado al hook id moderno `ruff-check`). Los notebooks `*.ipynb` se excluyen del lint y del whitespace-fixing (`extend-exclude`): son artefactos generados, se validan por schema, no se lintean como fuente. Esto elimina los 73 errores de ruff (todos en `.ipynb`) y los archivos "reformatables".

### Verified

- `pip install -e .` deja de fallar; `import voicelegacy` funciona.
- **238 tests verdes** (los 6 nuevos de contrato incluidos), cobertura **86.7%**.
- `pre-commit run --all-files`: **todos los hooks verdes**, incluido `validate-pyproject-toml` que antes nunca corría.
- `ruff check .` y `ruff format --check .` limpios bajo 0.15.16.
- Los 4 notebooks regeneran y pasan schema; las celdas de instalación apuntan a `@v0.4.1`.



Salto MINOR (0.3.3 → 0.4.0): API pública nueva + cambios de comportamiento en defaults de limpieza. Atiende los hallazgos críticos de la auditoría externa de v0.3.3.

### Fixed

- **Bug crítico criterio 6 (WAV↔texto roto).** El notebook `notebook_voicelegacy_finetune.ipynb` emparejaba WAVs con transcripciones usando el patrón `{source_stem}_seg{idx:04d}`, que NO coincide con el patrón real que escribe `extract_segments_to_wav` (`{stem}_{idx:04d}_{start:08.2f}`). Resultado: `text_index.get(wav.stem)` devolvía `None` siempre → dataset vacío → `RuntimeError`. **Solución de raíz:** `extract_segments_to_wav` ahora escribe un sidecar `.txt` con la transcripción junto a cada WAV, y la nueva función `build_finetune_dataset` empareja leyendo el sidecar adyacente (no parsea nombres). Reproducido y verificado con un test de integración.

### Added

- **Módulo `voicelegacy.finetune_dataset`** (109 LOC, 96.2% cobertura):
  - `build_finetune_dataset(reference_corpus, dataset_dir, ...)` → arma el dataset LJSpeech-like leyendo los sidecars `.txt`. Robusto por construcción. Devuelve `DatasetBuildResult` con conteos y diagnóstico de descartes.
  - `validate_corpus_coherence(reference_corpus, threshold=0.70, ...)` → **criterio 19**: embebe cada WAV con Resemblyzer y marca clips cuya similitud al centroide cae bajo el umbral. Detecta contaminación silenciosa cuando una entrevista se etiquetó al hablante equivocado en el flujo manual del bridge. Devuelve `CoherenceResult` con outliers y veredicto.
- **Sidecar de transcripción** (`.txt`) escrito junto a cada WAV por `extract_segments_to_wav`.
- **Celda 6.bis en el notebook bridge**: valida coherencia del corpus con Resemblyzer ANTES del dataset, con guía de qué hacer si hay outliers. Escribe `reports/coherence_report.json`.
- **20 tests nuevos** (`test_finetune_dataset.py`), incluido el **test de integración corpus→dataset** (criterio 9) que reproduce y previene la regresión del criterio 6: extrae un corpus real con `extract_segments_to_wav`, arma el dataset, y afirma que NO está vacío + que el interlocutor fue excluido. Total: 238 tests (era 220).

### Changed (cambios de comportamiento en defaults — criterios 1 y 3)

- **`bandpass high_hz` 7600 → 10000 Hz**: 7600 cortaba armónicos altos clave para el timbre; XTTS-v2 emite a 24 kHz.
- **Orden de limpieza en `extract_segments_to_wav`**: ahora `denoise → bandpass → preemphasis` (antes `bandpass → preemphasis → denoise`). noisereduce estima el perfil de ruido del espectro completo; filtrar antes lo privaba de información.
- **`target_loudness_lufs` -23 → -20**: -23 (broadcast EBU) está por debajo del rango de entrenamiento de XTTS-v2 (~-23 a -18); -20 queda dentro.
- **`min_segment_duration_s` 4.0 → 6.0**: consistencia con `MIN_REF_DURATION_S=6.0` y el mínimo recomendado de XTTS-v2 para contexto prosódico.
- **`top_n_segments` 10 → 5**: pocas referencias excelentes superan a muchas mediocres.
- **`notebook_voicelegacy_finetune.ipynb` y bridge** ahora usan `build_finetune_dataset` (función del paquete) en vez de emparejamiento frágil por nombre.

### CI (criterio 11)

- El workflow ahora regenera y valida los **4 notebooks** (antes solo el principal): loop sobre todos los `build_*.py` + schema + hook anti-bomba sobre cada `.ipynb`.

### Verified

- Bug criterio 6 reproducido (patrón viejo nunca coincide) y corregido (test de integración pasa, dataset no vacío, interlocutor excluido).
- `validate_corpus_coherence` detecta contaminación: test con 3 clips de un hablante + 1 ortogonal → marca el outlier, `is_coherent=False`.
- 238 tests verdes, cobertura 86.51% (subió desde 85.90%), `finetune_dataset.py` 96.2%.
- ruff check + format limpios; los 4 notebooks: 0 errores de sintaxis (IPython transformer), schema + anti-bomba OK.
- `python -m build` + `twine check` → PASSED para 0.4.0.

### Not in this release (límites honestos que siguen)

- **Criterio 10 (validación empírica con audio real):** sigue pendiente. Los tests usan audio sintético; nadie ha medido aún `speaker_similarity_score` baseline vs cleanup vs finetune con material real. Es trabajo de Colab con GPU + material del usuario, no de CI.
- Criterios 2 (presets de hiperparámetros legacy vs expresivo) y 4 (renombrar `snr_db`→`dynamic_range_db` en API pública) quedan como mejoras futuras documentadas.

## [0.3.3] — 2026-05-23 — Turno 13 · notebooks conformes a las skills de Colab

Salto PATCH (0.3.2 → 0.3.3): reescritura del notebook puente y mejoras de gestión de memoria en el standalone para cumplir las skills `colab-notebook-dev` y `python-data-library-dev`. Sin cambios en la API del paquete.

### Changed

- **`notebook_voicelegacy_bridge.ipynb` reescrito** siguiendo las skills:
  - **Estructura EXTRAS/EJECUTAR** (skill §4): una celda EXTRAS con imports + `@dataclass BridgeConfig` + todas las funciones; celdas EJECUTAR de pocas líneas con solo variables de usuario + una llamada.
  - **Disciplina de RAM (skill §8, lo más crítico):** el notebook anterior cargaba la entrevista COMPLETA a RAM (`sf.read(str(audio_path))`) solo para reproducir 12 s — con 10+ horas de entrevistas esto reventaba los 12 GB de Colab Free. Ahora usa **lectura parcial desde disco** (`leer_fragmento` con `soundfile` seek start/stop): una entrevista de 1 hora cuesta la misma RAM que una de 12 s. Verificado: lee solo el 18% del archivo para una muestra.
  - **Checkpointing por entrevista (skill §6):** `bridge_manifest.json` registra qué entrevistas ya se extrajeron; re-correr salta las hechas. Sobrevive a sesión Colab cortada. Verificado idempotente.
  - **Liberación de RAM entre entrevistas:** `gc.collect()` tras cada una; monitoreo con `psutil` y aviso al 85%.
  - **Config centralizada `@dataclass` (skill §1):** `BridgeConfig` con `__post_init__` validation, cero magic numbers (umbrales de minutos, SNR, preview, RAM warn todos en la config).
  - **Observabilidad (skill §5):** progreso `[i/total]`, resumen de composición por entrevista, monitoreo de RAM.
  - **Metadata de reproducibilidad (skill §12):** `bridge_metadata.json` con config + stats + python_version.
- **`notebook_voicelegacy_finetune_standalone.ipynb`:** añadida liberación explícita del audio completo (`del audio_full; gc.collect()`) tras la segmentación, progreso `🔄 segmento i/total`, y comentario que justifica por qué para UN archivo con segmentación densa cargar una vez es defendible (vs el bridge que lee dispersamente de muchos archivos largos y por eso usa lectura parcial).

### Verified

- 4 tests end-to-end del flujo del bridge ejecutados con datos sintéticos: (1) lectura parcial lee solo el fragmento pedido (18% del archivo, no 100%); (2) extracción filtra al interlocutor (`Kept N/M segments for SPEAKER_00`); (3) checkpointing idempotente (re-correr salta las hechas); (4) metadata de reproducibilidad guardada.
- La celda EXTRAS ejecuta como módulo real; `BridgeConfig.__post_init__` rechaza valores inválidos.
- 220 tests verdes, cobertura 85.90%, ruff check + format limpios, los 4 notebooks pasan schema + anti-bomba.
- `python -m build` + `twine check` → PASSED para 0.3.3.

### Documented

- `docs/PROCESO_COMPLETO_ENTREVISTAS.md` actualizado con la sección de gestión de memoria (por qué no se carga el audio completo, checkpointing, reanudación).

## [0.3.2] — 2026-05-23 — Turno 12 · notebook puente speakerscribe → voicelegacy

Salto PATCH (0.3.1 → 0.3.2): nuevo notebook de enlace para el caso real de muchas entrevistas con varios hablantes. Sin cambios en la API del paquete.

### Added

- **Notebook `notebooks/notebook_voicelegacy_bridge.ipynb`** (19 celdas, regenerable desde `build_bridge_notebook.py`). Conecta la salida de speakerscribe (entrevistas diarizadas con varias personas) con voicelegacy (corpus de un solo hablante). Resuelve el problema de que las etiquetas `SPEAKER_xx` de speakerscribe NO son consistentes entre archivos (la diarización es por-archivo): incluye una celda de identificación asistida por audio (reproduce una muestra de cada hablante por entrevista) y un mapa `TARGET_SPEAKER_MAP` por entrevista. Recorta + limpia solo los segmentos del hablante objetivo de todas las entrevistas, consolida en un `reference_corpus/` único, y opcionalmente construye el dataset LJSpeech-like para fine-tuning emparejando cada WAV con su transcripción.

### Verified

- Compatibilidad speakerscribe → voicelegacy confirmada ejecutando código real: el schema Pydantic de voicelegacy (`extra="allow"`) parsea el JSON de speakerscribe con todas sus claves (`audio_file`, `language_detected`, `segments[].start/end/text/speaker`, más claves extra `id`, `speaker_overlap_s`, `words`).
- Cadena de extracción probada end-to-end con JSON + WAV sintéticos: `filter_segments(target_speaker="SPEAKER_00")` descartó al interlocutor (`SPEAKER_01`) y extrajo solo los segmentos del objetivo. Log: `Kept 3/4 segments for speaker 'SPEAKER_00'`.
- Las 19 celdas de código validadas con el `TransformerManager` de IPython (cero errores de sintaxis).
- 220 tests verdes, cobertura 85.90%, ruff check + format limpios, los 4 notebooks pasan schema + anti-bomba.

### Documented

- `docs/PROCESO_COMPLETO_ENTREVISTAS.md`: explicación desde cero del flujo de 2 librerías (speakerscribe diariza, voicelegacy clona), por qué el audio debe estar aislado, por qué las etiquetas varían entre archivos, y el paso a paso completo.

## [0.3.1] — 2026-05-19 — Turno 11 · fine-tuning standalone desde grabación cruda

Salto PATCH (0.3.0 → 0.3.1): nuevo notebook autónomo + extra opcional `finetune`. Sin cambios en la API del paquete; el código Python es idéntico a 0.3.0.

### Added

- **Notebook `notebooks/notebook_voicelegacy_finetune_standalone.ipynb`** (27 celdas, regenerable desde `build_finetune_standalone_notebook.py`). A diferencia de `notebook_voicelegacy_finetune.ipynb` (que requiere `reference_corpus/` + transcripciones de speakerscribe), este parte de **UNA grabación cruda** (mp3/m4a/wav/mp4) y hace todo: conversión a 22.05 kHz mono, transcripción con faster-whisper (word timestamps + VAD), segmentación inteligente en clips de 2-11 s por pausas/oraciones, limpieza con `voicelegacy.audio` (denoise + bandpass + loudness), construcción del dataset LJSpeech-like, fine-tuning del GPT encoder, materialización del checkpoint y A/B contra el base.
- **Extra opcional `finetune`** en `pyproject.toml`: `pip install voicelegacy[finetune]` añade `faster-whisper>=1.0,<2.0`. Incluido también en el meta-extra `all`.
- **Hyperparams calibrados para dataset pequeño** (15-25 min de voz neta, lo que rinde una grabación de 30 min tras quitar silencios): `NUM_EPOCHS=10` (vs 6), `LEARNING_RATE=3e-6` (vs 5e-6, anti catastrophic-forgetting), `WEIGHT_DECAY=5e-2` (vs 1e-2, regularización fuerte anti-overfit).

### Verified

- 220 tests verdes (sin cambios — el notebook no toca código del paquete)
- Cobertura 85.90%, piso 80%
- ruff check + format limpios sobre voicelegacy/ tests/ scripts/ notebooks/
- Las 27 celdas de código del notebook standalone validadas con el `TransformerManager` de IPython (cero errores de sintaxis incluyendo magics `!` y `=!`)
- Schema nbformat v4.5 + hook anti-bomba pasan en los 3 notebooks
- `python -m build` + `twine check` → PASSED para 0.3.1

### Documented

- `docs/PLAYBOOK_FINETUNING_30MIN.md`: guía paso a paso completa para fine-tunear desde una grabación de 30 min, con la advertencia honesta de que 30 min es el límite inferior (no el ideal de 2-5 h).

## [0.3.0] — 2026-05-19 — Turno 10 · fine-tuning XTTS-v2

Salto MINOR (0.2.0 → 0.3.0) por feature pública nueva backward-compatible: módulo `voicelegacy.finetuned_inference` para cargar y usar checkpoints XTTS-v2 fine-tuneados. El paquete sigue funcionando idénticamente sin él — la nueva ruta es opt-in.

### Added

- **Módulo `voicelegacy.finetuned_inference`** (411 LOC, 88.7% cobertura). Tres entradas públicas:
  - `FineTunedCheckpoint.from_dir(path)` — valida los 6 archivos requeridos del checkpoint (`model.pth`, `config.json`, `vocab.json`, `dvae.pth`, `mel_stats.pth`, `speakers_xtts.pth`), produce un handle inmutable con `fingerprint` (16 hex chars) para audit trail.
  - `load_finetuned_model(checkpoint, device="auto")` — carga el modelo con `Xtts.init_from_config + load_checkpoint`, no usa la API alta `TTS.api.TTS` porque el checkpoint es local (no hub). Cacheado por `(checkpoint_dir, device)`.
  - `synthesize_with_finetuned(model, checkpoint, text, speaker_wav, output_path, config)` — drop-in del `synthesize_to_file` de `synthesis.py`. Conditioning latents cacheados por `(fingerprint, reference_set)` para evitar mezcla cruzada entre checkpoints.
- **Notebook `notebooks/notebook_voicelegacy_finetune.ipynb`** (27 celdas, regenerable desde `build_finetune_notebook.py`) — flujo completo Colab Free T4: instalación, descarga base XTTS-v2 (cacheada en Drive), preparación dataset LJSpeech-like desde `reference_corpus/` + transcripciones de speakerscribe, configuración del GPTTrainer (sólo GPT, vocoder congelado), entrenamiento con checkpoints intermedios (resumible si Colab corta sesión), materialización de checkpoint reutilizable, validación end-to-end, **comparación A/B base vs fine-tuned con `speaker_similarity_score`**.
- **38 tests nuevos** en `tests/test_finetuned_inference.py` (220 total ahora, era 182). Cubren: validación de los 6 archivos requeridos (paramétrico), fingerprint estable y discriminante, cache hit/miss por device, ImportError graceful cuando coqui-tts no está, fallo de `load_checkpoint` envuelto en `RuntimeError` con mensaje accionable, aislamiento de latents cache entre checkpoints distintos sobre las mismas referencias, contrato drop-in con `synthesis.py`.

### Changed

- **`__version__`** sincronizado a `"0.3.0"` en `voicelegacy/__init__.py` y `pyproject.toml`.
- **`voicelegacy/__init__.py`** exporta los 4 símbolos nuevos: `FineTunedCheckpoint`, `load_finetuned_model`, `release_finetuned_model`, `synthesize_with_finetuned`. `__all__` ordenado alfabéticamente como antes.

### Coverage by module

| Módulo | 0.2.0 | 0.3.0 | Cambio |
|---|---|---|---|
| `finetuned_inference.py` (nuevo) | — | **88.7%** | nuevo |
| **Total** | 85.58% | **85.89%** | +0.31 pp |
| **Tests** | 182 | **220** | +38 |

### Verified

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK
- `python -c "import voicelegacy; print(voicelegacy.__version__)"` → `0.3.0`
- `pyproject.toml::version == __init__.py::__version__` → `0.3.0`
- `ruff check --config pyproject.toml voicelegacy/ tests/ scripts/` → All checks passed
- `ruff format --check --config pyproject.toml voicelegacy/ tests/ scripts/` → 38 files already formatted
- `pytest tests/` → **220 passed in ~10s**
- `pytest tests/ --cov=voicelegacy --cov-fail-under=80` → **85.89%**, piso 80% alcanzado
- `python notebooks/build_notebook.py` → 37 celdas (inferencia)
- `python notebooks/build_finetune_notebook.py` → 27 celdas (fine-tuning)
- `python -m build` → produce `voicelegacy-0.3.0.tar.gz` + `voicelegacy-0.3.0-py3-none-any.whl` sin warnings
- `twine check dist/*` → PASSED en ambos artefactos
- Smoke install desde wheel en venv aislado: `voicelegacy --help` muestra 6 subcomandos, `voicelegacy.__version__` retorna `"0.3.0"`, `FineTunedCheckpoint` importable.

### Not in this release (documented limitations)

- **Inferencia fine-tuned no expuesta vía CLI** todavía. Por diseño: el checkpoint vive en Drive, no en `interviews_raw/`, y el flujo natural es notebook. Si la demanda aparece, agregar `voicelegacy synthesize --finetuned-dir PATH` en 0.4.0.
- **Sin pipeline `run_synthesis_finetuned`** equivalente al `run_synthesis` de `pipeline.py`. El notebook llama directo a `synthesize_with_finetuned`. Razón: el sidecar/runs.db actual no contempla el campo `checkpoint_fingerprint`; añadirlo es scope de 0.4.0.
- **P3-27 (audio real benchmark + decisión zero-shot vs fine-tuned)**: sigue pendiente. v0.3.0 da la herramienta para fine-tunear; falta correrla con material real y comparar A/B.
- **Sin test E2E con coqui-tts real**: imposible en CI sin GPU y sin 2 GB de pesos descargados. Mitigación: el smoke del notebook (celda 9) lo verifica en cada uso.

## [0.2.0] — 2026-05-19 — Turno 9 · ingeniería impecable para PyPI

Salto MINOR (0.1.6 → 0.2.0) por acumulación de cambios significativos desde 0.1.0, incluyendo un breaking change documentado en 0.1.x (P2-24: speakerscribe JSON malformado ahora falla en vez de saltarse segmentos silenciosamente). SemVer 0.x.y permite cualquier cambio en patch, pero documentamos explícitamente para usuarios pre-existentes.

### Added

- **Tests de `similarity.py` ampliados de 4 a 32 casos** (incluye `quality_band` en sus cuatro umbrales, encoder cache hit, `release_encoder`, `compute_similarity_batch` con error parcial recuperable, validación de paths inexistentes, `is_available` con ambas ramas).
- **Tests de `corpus.py` ampliados de 9 a 29 casos** (incluye end-to-end de `extract_segments_to_wav` con audio sintético, `estimate_median_f0_hz` con voz/silencio/duración corta, `analyze_f0_outliers` con detección real de outlier, `filter_f0_outliers`, `build_reference_corpus` completo, branch de búsqueda de extensión alternativa en `load_speakerscribe_json`).
- **`pyproject.toml` con metadatos completos para PyPI**: `Development Status :: 4 - Beta` (subido desde Alpha), `Environment :: GPU :: NVIDIA CUDA`, classifiers ampliados (Conversion, Libraries :: Python Modules, Spanish/English natural language), `license-files = ["LICENSE"]` (PEP 639), `maintainers`, `Issues` URL, keyword extras (`text-to-speech`, `speaker-similarity`, `xtts-v2`, `coqui-tts`, `speakerscribe`, `audio-processing`, `denoising`, `diarization`, `spanish`).
- **Meta-extra `all`** en optional-dependencies: `pip install voicelegacy[all]` instala similarity + deepfilter + notebook + nbformat/ipykernel en un solo comando.
- **Extra `notebook`** para usuarios que quieren `nbformat + ipykernel` sin pip-installing herramientas dev.
- **`PUBLISHING_CHECKLIST.md`**: gate ordenado para tag GitHub y publicación PyPI, con paso explícito de `twine check`, TestPyPI primero, validación de paridad `pyproject.toml::version == __init__.py::__version__`.

### Changed

- **`__version__`** sincronizado a `"0.2.0"` en `voicelegacy/__init__.py` (antes desincronizado: `pyproject.toml=0.2.0` vs `__init__.py=0.1.6`; el bug habría hecho `pip show` y `voicelegacy.__version__` reportar versiones diferentes).
- **`--cov-fail-under`** subido de 75 a 80. Margen restante: 5.58 puntos sobre el piso (85.58% actual).
- **PROGRESS.md tracking honesto**:
  - P1-WADA reclasificado de "✅ cerrado por decisión técnica" a "🛑 Deferred — ver P3-5". Cerrar por no-hacer es contabilidad mañosa; deferred es la categoría correcta.
  - P3 con tabla de cross-reference original (P3-27..P3-31) ↔ renombrado (P3-1..P3-4) para no romper trazabilidad con el plan A01.
  - P3-27 marcado como **prioridad alta — bloqueante para v1.0** (no para v0.2.0).
  - Resumen ejecutivo corregido: P1 8/8 (no 9/9), P3 3 de 5 originales + 1 extra (no 4/5).

### Coverage by module

| Módulo | 0.1.6 | 0.2.0 | Cambio |
|---|---|---|---|
| `similarity.py` | 53.4% | **97.5%** | +44.1 pp |
| `corpus.py` | 42.9% | **95.5%** | +52.6 pp |
| **Total** | 76.21% | **85.58%** | +9.37 pp |
| **Tests** | 134 | **182** | +48 |

### Verified

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK
- `python -c "import voicelegacy; print(voicelegacy.__version__)"` → `0.2.0`
- `grep -E '^version = ' pyproject.toml` y `__version__` sincronizados
- `ruff check --config pyproject.toml voicelegacy/ tests/ scripts/` → All checks passed
- `ruff format --check --config pyproject.toml voicelegacy/ tests/ scripts/` → 36 files already formatted
- `pytest tests/` → **182 passed in ~51s**
- `pytest tests/ --cov=voicelegacy --cov-fail-under=80` → **85.58%**, piso 80% alcanzado
- `python notebooks/build_notebook.py` → 37 celdas, regenerable e idempotente
- `python -m build` → genera `voicelegacy-0.2.0.tar.gz` + `voicelegacy-0.2.0-py3-none-any.whl` sin warnings
- `twine check dist/*` → PASSED en ambos artefactos

### Not in this release (documented limitations)

- **P3-27 (audio real benchmark)**: el experimento que valida si el `speaker_similarity_score` con material real sub-óptimo cae en banda `high` (≥0.75) NO está hecho. v0.2.0 es "ingeniería impecable, validación empírica pendiente". v1.0 requiere ese experimento + bench documentado.
- **Tests con coqui-tts real**: todos los tests usan mocks. Un breaking change en `coqui-tts<1.0` upstream se detectaría en producción, no en CI. Mitigación: `release_conditioning_latents()` + ruta de fallback `tts_to_file` ya implementadas.

## [0.1.6] — 2026-05-18 — Turno 8 · pre-commit hardening

### Fixed

- **Critical**: `.pre-commit-config.yaml` did not parse because three local hooks used `entry: python -c "..."` with inline commands containing `:`, which YAML interpreted as mapping-value separators. The file aborted with `InvalidConfigError` before any hook ran, silently disabling all custom protections (TOML validity, notebook schema, no-live-`runtime.unassign()`). Anyone installing the hooks lost guard against the exact bugs P0-1, P0-2, and notebook-generator divergence.
- The same hook block declared `additional_dependencies: [nbformat]` with `language: system`, a combination pre-commit refuses. The schema validator never installed `nbformat`.
- `notebooks/build_notebook.py` wrote the `.ipynb` without a trailing newline, so `end-of-file-fixer` modified the file after every regeneration, creating a loop where every `python build_notebook.py` made the working tree dirty.

### Added

- `scripts/check_pyproject_toml.py`, `scripts/check_notebook_schema.py`, `scripts/check_no_runtime_unassign.py`: the three local hooks extracted as standalone, testable Python files with docstrings. No more YAML-escaped one-liners.

### Changed

- Bumped `pre-commit/pre-commit-hooks` from `v4.5.0` to `v5.0.0` (the previous version emitted deprecation warnings about stage names).
- `language: system` → `language: python` for the notebook-schema hook so `additional_dependencies` actually installs `nbformat`.
- `files: ^notebooks/.*\.ipynb$` → `files: ^notebooks/[^/]+\.ipynb$` for the notebook hooks, so `notebooks/_archive/` (preserved historical bugs) does not block commits.
- `build_notebook.py` now writes with a trailing newline, keeping `end-of-file-fixer` idempotent across regenerations.

### Verified

- `pre-commit run --all-files` on a freshly initialized repo, no cache: 10 hooks passed (ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files, validate-pyproject-toml, validate-notebook-schema, no-live-runtime-unassign).
- `scripts/check_no_runtime_unassign.py notebooks/_archive/notebook_voicelegacy_*_handedited_37cells.ipynb` correctly fails (positive control: the archived broken notebook is still detected, the hook is real).
- 134 tests passing, coverage 76.31%, ruff clean, TOML valid, notebook regenerable.

## [0.1.5] — 2026-05-18 — Turno 7 · cierre P2-20 + notebook sync

### Added

- `evaluate_file(path, config=ReferenceConfig)` accepts a `ReferenceConfig` so the gating thresholds (`min_segment_duration_s`, `max_segment_duration_s`, `min_snr_db`) have one source of truth. Explicit keyword overrides still take precedence to keep ad-hoc scripts working.
- Tests in `tests/test_quality.py` covering the new `config=` signature and the override precedence rule.
- Notebook now surfaces six features that previously lived only in the code:
  - `voicelegacy diagnose` as the first sanity check after install/import.
  - Sidecar JSON inspection (`speaker_similarity_score`, `quality_band`, `degraded_mode`, `text_plan`, `run_hash`).
  - Optional `evaluate-denoise` cell for comparing `noisereduce` vs DeepFilterNet on real samples.
  - Tunable variables in the config cell for `DENOISE_STATIONARY`, `APPLY_BANDPASS_FILTER`, `APPLY_PREEMPHASIS_FILTER`, `ENABLE_F0_OUTLIER_FILTER`.
  - `LONG_TEXT_STRATEGY`, `MAX_SINGLE_PASS_CHARS`, `LONG_TEXT_WARNING_CHARS`.
  - Troubleshooting table expanded with the matching CLI commands.

### Changed

- `quality.score_segment()` default `min_snr_db` now points to the package constant `MIN_SNR_DB` instead of the literal `15.0` (eliminates the three-source-of-truth bug — closes P2-20 from the original audit §3.4).
- `pipeline.py` uses the simplified `evaluate_file(wav, config=config.reference)` signature in both `run_reference_phase` and `_reference_quality_summary`.
- CLI `synthesize` output uses `console.print(..., soft_wrap=True)` to avoid Rich breaking file paths at the 80-column boundary that `typer.testing.CliRunner` defaults to. The previous behavior made `tests/test_cli.py::test_synthesize_uses_text_file_and_prints_sidecar` fail intermittently in CI on narrow terminals.
- CLI `synthesize` now also prints `speaker_similarity_score` per result when available.

### Fixed

- Notebook cell 6 (`PipelineConfig` assembly) previously referenced `DENOISE_STATIONARY`, `APPLY_BANDPASS_FILTER`, `APPLY_PREEMPHASIS_FILTER` without those names being defined in the user-edited config cell. Any actual notebook run crashed with `NameError`. Variables are now defined in the config cell and consistently propagated.

### Verified

- `python -m ruff format --check .` → 33 files already formatted.
- `python -m ruff check .` → All checks passed!
- `python -m pytest -q` → **134 tests**, **76.31%** coverage, 75% floor reached.
- `python notebooks/build_notebook.py` → validates against nbformat v4.5, writes **37 cells**.
- `pyproject.toml` parses with `tomllib`.

## [0.1.4] — 2026-05-18 — GitHub release candidate

### Added

- GitHub Actions CI gate for install, TOML validation, notebook validation, Ruff format, Ruff lint and pytest coverage.
- Additional mocked XTTS-v2 tests covering CPML acceptance, missing `coqui-tts`, model cache, lower-level XTTS API, conditioning-latent cache and fallback paths.
- CLI execution tests for `build-corpus`, `synthesize --text-file`, and `diagnose --json` with mocked pipeline functions.
- Pipeline tests for optional similarity success/failure branches and batch synthesis preload behavior.

### Changed

- Coverage floor raised from 65% to **75%**.
- README rewritten as an operational guide for local use, Colab use, workspace structure, CLI commands, sidecar interpretation, troubleshooting and zero-shot limits.
- CI explicitly avoids real XTTS weight downloads / GPU inference; those paths remain mocked.

### Verified

- `python -m ruff format --check .` passes.
- `python -m ruff check .` passes.
- `python -m pytest -q` passes: **125 tests**, **77.61%** coverage, 75% floor reached.
- `python notebooks/build_notebook.py` validates and writes the notebook.
- `pyproject.toml` parses with `tomllib`.

## [0.1.3] — 2026-05-18 — P2 core robustness

### Added

- `voicelegacy diagnose --workspace ...` operational readiness command with human and JSON output.
- `voicelegacy synthesize --text-file` for `.txt` and `.csv` batch utterances.
- Pydantic speakerscribe schema validation (`SpeakerscribeDocument`, `SpeakerscribeSegment`).
- Runtime telemetry helpers for elapsed time and CUDA/VRAM snapshots.

### Changed

- `compute_run_hash()` now includes the installed `voicelegacy` version so cache entries are invalidated after algorithm/package upgrades.
- Synthesis sidecar metadata now records `voicelegacy_version`.
- Malformed speakerscribe segments now fail the document explicitly instead of being skipped silently.

### Fixed

- `force_rebuild_reference=True` now backs up existing reference WAVs to `reference_corpus_backup_<UTC>/` instead of deleting them.

## [0.1.2] — 2026-05-18 — P1 quality pass

### Added

- Reproducible synthesis via `SynthesisConfig.seed` and deterministic seeding in `synthesis.py`.
- Optional Resemblyzer speaker-similarity scoring for synthetic outputs.
- Synthesis sidecars with run hash, seed, reference set, source-quality summary, degraded-mode flag and optional speaker-similarity score.
- Adaptive cleanup controls: non-stationary denoise, conservative band-pass filtering and optional pre-emphasis.
- F0 outlier guard for target-speaker contamination detection.
- XTTS conditioning-latent cache with fallback to `tts_to_file()`.
- Grouped quality reasons in reference reports.

### Changed

- `ReferenceConfig.target_loudness_lufs` upper bound tightened to `-16` LUFS.
- `SynthesisConfig.temperature` upper bound tightened to `0.9`.
- The internal SNR heuristic is documented as dynamic-range estimation; `_estimate_snr_db` remains as a backward-compatible alias.
- `load_audio_mono` prefers `soundfile` + `scipy.signal.resample_poly` for WAV-like formats.

### Fixed

- `trim_silence` no longer depends on `librosa.effects.trim`, avoiding slow numba initialization.
- `audio.denoise()` now honors the non-stationary denoise configuration instead of forcing `stationary=True`.

## [0.1.0] — 2026-05-16

### Added

- Pydantic configuration models.
- Audio preprocessing utilities.
- Reference-segment scoring and quality gates.
- Corpus builder for speakerscribe JSON outputs.
- XTTS-v2 wrapper with explicit CPML handling.
- SQLite idempotency cache.
- Two-phase pipeline orchestration.
- Typer CLI.
- Generated Colab notebook.
- Initial test suite.

### Notes

- XTTS-v2 model weights are governed by CPML: https://coqui.ai/cpml
- The project consumes speakerscribe outputs; it does not perform diarization itself.

## 0.1.0-rc.P3 — Turno 6

### Added

- P3 denoise evaluation harness: `voicelegacy evaluate-denoise` compares the current noisereduce path against optional DeepFilterNet on real user-selected samples.
- Optional extras: `deepfilter` and `publish`.
- Explicit long-text policy via `SynthesisConfig.long_text_strategy`, `max_single_pass_chars`, and `long_text_warning_chars`.
- `text_plan` metadata in synthesis sidecars.
- Documentation: `docs/P3_EVALUATION.md`, `docs/ETHICS.md`, and `docs/RELEASE.md`.

### Changed

- XTTS sentence splitting is no longer a blind boolean for all text. The default `auto` policy avoids splitting short utterances and enables splitting for longer prose.
- Release workflow now builds artifacts and draft GitHub releases on tags, but PyPI publishing requires manual workflow dispatch and Trusted Publisher setup.

### Not changed deliberately

- DeepFilterNet is not a production default. It must win on real audio and downstream similarity before replacing noisereduce.
