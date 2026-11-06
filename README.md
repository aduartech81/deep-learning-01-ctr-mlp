# Predicción de clics con MLP: scikit-learn vs PySpark

![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![PySpark](https://img.shields.io/badge/PySpark-4.3-E25A1C) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E) ![Jupyter Book](https://img.shields.io/badge/Jupyter%20Book-1.0-F37726)

**Proyecto 01 de la serie de Deep Learning.** Proyecto Integrador de la asignatura Deep Learning · Maestría en Ingeniería Biomédica, Universidad del Norte
(Barranquilla, Colombia). Autor: **Andrés Duarte Chaves**.

Jupyter Book ejecutable que predice si un usuario hace clic en un anuncio móvil (dataset Avazu,
~40 M de impresiones). Compara MLP en scikit-learn y en PySpark en calidad predictiva y en costo
computacional, y analiza la interpretabilidad con LIME contrastada con SHAP.

**Lo que demuestra este repositorio**
- Protocolo **libre de fuga de información**: partición temporal, ajustes solo con entrenamiento,
  target encoding out-of-fold, umbral y parada temprana con validación.
- **Inferencia estadística** de verdad: 5 semillas con IC 95 %, bootstrap, DeLong, McNemar y Holm.
- **Ingeniería de datos a escala**: DuckDB/Polars fuera de memoria, Parquet particionado y Spark
  con medición de memoria pico (incluida la JVM), escalamiento fuerte/débil y ley de Amdahl.
- **Interpretabilidad crítica**: fidelidad y estabilidad de LIME, sensibilidad al kernel y
  coherencia con SHAP e importancia por permutación.
- **Reproducibilidad**: entorno fijado, semillas, manifiesto SHA-256 de los datos, notebooks
  reanudables y prueba de humo con datos sintéticos.

> *English summary.* Click-through-rate prediction on Avazu with MLPs in scikit-learn vs PySpark:
> leakage-free temporal protocol, rigorous statistical comparison (seeds, bootstrap, DeLong,
> McNemar), compute benchmarking (crossover point, strong/weak scaling, Amdahl) and a critical
> assessment of LIME against SHAP. Fully reproducible Jupyter Book.

**Estado:** código completo y verificado de punta a punta con datos sintéticos; los resultados con
los datos reales de Avazu se publicarán al terminar la ejecución completa.

## Estructura

```
config.yaml              # TODO el diseño experimental (tamaños, semillas, grids, presupuestos)
src/ctr/                 # código reutilizable
  data.py                #   ingesta DuckDB -> Parquet por día, muestras anidadas por hash
  features.py            #   raras, hashing, frecuencia, target encoding OOF (scikit-learn)
  spark_ctr.py           #   el mismo preprocesamiento y modelos en PySpark + métricas Spark UI
  models_sk.py           #   MLP con parada temprana temporal, SGD, HGB, Optuna
  metrics.py             #   métricas, ECE, lift, bootstrap, DeLong, McNemar
  bench.py               #   tiempo, memoria pico (incluye la JVM), Amdahl
  explain.py             #   LIME/SHAP en espacio interpretable, estabilidad, permutación
notebooks/01..10         # capítulos del libro (se ejecutan en orden)
nb_src/                  # mismos notebooks en formato texto (jupytext), útiles para git diff
intro.md, _config.yml, _toc.yml   # Jupyter Book
scripts/run_all.sh       # ejecuta todo y construye el libro
scripts/make_synthetic.py# datos sintéticos SOLO para la prueba de humo
environment.yml          # entorno con versiones fijadas (incluye Java 17 para Spark)
```

## Pasos en el Mac (una sola vez)

1. **Instalar Miniforge** (si no tiene conda): https://github.com/conda-forge/miniforge
2. **Crear el entorno**, en la carpeta del proyecto:
   ```bash
   conda env create -f environment.yml
   conda activate dl-ctr
   python -m ipykernel install --user --name python3
   ```
3. **Descargar Avazu desde Kaggle.**
   - Entre a https://www.kaggle.com/c/avazu-ctr-prediction, pestaña *Rules*, y acepte las reglas
     (sin esto la descarga falla con 403).
   - En Kaggle: *Settings → API → Create New Token*. Mueva el `kaggle.json` descargado:
     ```bash
     mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     ```
   - Descargue solo el archivo de entrenamiento:
     ```bash
     mkdir -p data/raw
     kaggle competitions download -c avazu-ctr-prediction -f train.gz -p data/raw
     # si llega como train.gz.zip:  cd data/raw && unzip train.gz.zip && cd ../..
     ```
     El resultado debe ser `data/raw/train.gz` (~1,1 GB comprimido).
4. **Ajustar `config.yaml` a su Mac** (perfil `full`):
   - `spark.driver_memory`: ~60–70 % de la RAM (12g con 16 GB; 20g con 32 GB).
   - Con 8 GB de RAM: quite `10000000` de `sk_sizes` y `null` de `sizes_freq` y `crossover_sizes`.
   - `bench.cores`: hasta el número de núcleos de su equipo.
   - `bench.cloud_price_usd_per_hour`: precio por hora de una instancia de referencia (opcional).

## Ejecución

Prueba rápida de que todo corre (datos sintéticos, ~30 min, resultados SIN valor empírico):
```bash
CTR_PROFILE=smoke bash scripts/run_all.sh
```
Ejecución real (larga: deje el Mac conectado y sin suspensión, p. ej. `caffeinate -i bash scripts/run_all.sh`):
```bash
bash scripts/run_all.sh            # todos los capítulos
bash scripts/run_all.sh 04         # solo un capítulo (los demás reutilizan resultados guardados)
```
- Los capítulos 4, 5 y 7 son **reanudables**: si se interrumpen, al volver a correrlos saltan lo que
  ya está en `results/*.jsonl`. Para rehacer algo, borre la línea o el archivo correspondiente.
- Orden obligatorio la primera vez: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
  (06–09 leen los resultados de 04–05; 07 usa los mejores hiperparámetros de 04 y 05).
- Tiempos orientativos en un Mac con 16 GB: 01–03 < 30 min; 04 varias horas (10 M de filas);
  05 varias horas (entrenamiento completo); 07 1–3 h. Ajuste `config.yaml` si necesita acortar.
- Durante el capítulo 5 con el entrenamiento completo, abra http://localhost:4040 y guarde capturas
  de *Jobs*, *Stages* y *Storage* en `results/figures/spark_ui_*.png` (evidencia pedida).

El libro queda en `_build/html/index.html`.

## Antes de entregar

- [ ] Ficha de datos (cap. 1): pegar la cláusula de uso de datos de las *Rules* de Kaggle.
- [ ] Cap. 9: revisar y completar el borrador de respuestas P1–P4 y la reflexión con sus palabras.
- [ ] Revisar que no queden salidas del perfil `smoke` (los notebooks muestran el perfil arriba).
- [ ] Capturas de la Spark UI.
- [ ] Subir el repositorio y el libro según indique el profesor.

## Decisiones metodológicas clave

- Partición temporal: entrenamiento días 21–28, validación día 29, prueba día 30.
- Parada temprana del MLP con el día 29 (no con la partición aleatoria de `early_stopping=True`;
  esa variante queda como ablación).
- Tuning de Spark con `ParamGridBuilder` y validación temporal; no `CrossValidator` /
  `TrainValidationSplit` (parten al azar).
- El MLP de Spark densifica la entrada: con hashing se limita a muestras; en el entrenamiento
  completo se usa frecuencia. Lo que excede el presupuesto se omite y se registra.
- Mismas filas en ambos entornos (muestras materializadas una vez); equivalencia del
  preprocesamiento verificada con una aserción en el capítulo 3.
