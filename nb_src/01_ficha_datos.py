# %% [markdown]
# # 1. Ficha de datos e ingesta
#
# **Objetivo del capítulo.** Convertir el CSV de Avazu a Parquet particionado por día (lectura
# perezosa, fuera de memoria) y documentar la ficha obligatoria del enunciado: origen, licencia,
# período, tasa base de clics, tamaño en disco y en memoria, y limitaciones.
#
# **Decisión sobre el dataset.** El enunciado recomienda Criteo 1TB como principal, pero advierte
# que el MLP de Spark no es viable con one-hot sobre millones de categorías y ofrece una
# alternativa sin clúster: usar **Avazu completo** y tratar P1 como un experimento de punto de cruce
# sin presuponer que Spark gana. Este trabajo sigue esa alternativa porque se ejecuta en un único
# equipo (ver anexo de reproducibilidad). Criteo queda como extensión para un entorno con clúster.

# %%
import sys, os, hashlib
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
import numpy as np, pandas as pd, polars as pl
import matplotlib.pyplot as plt
from ctr.nbutils import setup, savefig, show, SERIES, MUTED
from ctr import data
from ctr.bench import measure, hardware_info

cfg, RES, FIG = setup()
hardware_info()

# %% [markdown]
# ## 1.1 Ingesta: CSV → Parquet particionado por día (DuckDB)
#
# DuckDB lee el CSV en streaming y escribe un directorio `day=21 … day=30`. De la columna `hour`
# (formato `YYMMDDHH`) se derivan `day`, `hour_of_day`, `dow` (0 = lunes) y `franja`
# (madrugada 0–5 h, mañana 6–11 h, tarde 12–17 h, noche 18–23 h). La conversión equivalente con
# `SparkSession` se hace y se cronometra en el capítulo 5.

# %%
raw = cfg.path("raw_csv")
assert raw.exists(), f"No se encuentra {raw}. Siga el paso 3 del README (descarga desde Kaggle)."
rec = {}
with measure(rec, "ingesta"):
    info = data.csv_to_parquet(raw, cfg.path("parquet_dir"))
print(info, {k: round(v, 1) for k, v in rec.items()})

# %% [markdown]
# ## 1.2 Tamaño y conteos por día

# %%
def dir_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())

sz = data.split_sizes(cfg)
by_day = sz["by_day"].to_pandas()
by_day["particion"] = np.where(by_day["day"].isin(cfg["split"]["train_days"]), "entrenamiento",
                      np.where(by_day["day"] == cfg["split"]["valid_day"], "validación", "prueba"))
show(by_day)

# memoria en RAM: se estima con una muestra de 200 mil filas y se escala al total
lf = data.scan(cfg.path("parquet_dir"))
n_total = int(by_day["n"].sum())
muestra = lf.head(200_000).collect()
mem_total_gb = muestra.estimated_size("gb") * n_total / muestra.height

ficha = pd.DataFrame([
    ("Nombre", "Avazu Click-Through Rate Prediction (Kaggle, 2014-2015)"),
    ("Origen", "Competencia de Kaggle organizada por Avazu (plataforma de publicidad móvil); archivo train.gz"),
    ("Licencia / uso", "Regido por las reglas de la competencia en Kaggle (pestaña Rules), que deben aceptarse antes de descargar. COMPLETAR: copiar aquí la cláusula de uso de datos tal como aparece en Kaggle."),
    ("Período", f"10 días: 21 al 30 de octubre de 2014 (días presentes: {by_day['day'].min()}–{by_day['day'].max()})"),
    ("Filas totales", f"{n_total:,}"),
    ("Entrenamiento / validación / prueba", f"{sz['train']:,} / {sz['valid']:,} / {sz['test']:,}"),
    ("Tasa base de clics (global)", f"{lf.select(pl.col('click').mean()).collect().item():.4f}"),
    ("Tamaño CSV comprimido en disco", f"{raw.stat().st_size / 2**30:.2f} GB"),
    ("Tamaño Parquet (zstd) en disco", f"{dir_size(cfg.path('parquet_dir')) / 2**30:.2f} GB"),
    ("Tamaño estimado en memoria (Polars, todas las columnas)", f"{mem_total_gb:.1f} GB"),
    ("Variables", "id, click, hour, C1, banner_pos, site_*, app_*, device_*, C14–C21 (anonimizadas)"),
    ("Limitaciones", "Variables C1 y C14–C21 anonimizadas (sin semántica); device_ip/device_id con millones de "
                     "categorías; el test oficial de Kaggle no trae etiqueta, por eso la prueba es el día 30 del "
                     "train; solo 10 días: un único corte temporal (no hay estacionalidad semanal completa); "
                     "datos de 2014: el mercado actual es distinto (validez externa limitada)."),
], columns=["Campo", "Valor"])
ficha.to_csv(RES / "ficha_datos.csv", index=False)
with pd.option_context("display.max_colwidth", 200):
    display(ficha)

# %% [markdown]
# ## 1.3 Manifiesto de datos (reproducibilidad)
#
# Huella SHA-256 del archivo crudo y conteo por día: permite verificar que otra persona trabaja con
# exactamente los mismos datos.

# %%
h = hashlib.sha256()
with open(raw, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        h.update(chunk)
manifest = {"archivo": raw.name, "bytes": raw.stat().st_size, "sha256": h.hexdigest(),
            "filas_por_dia": dict(zip(by_day["day"].astype(int).astype(str), by_day["n"].astype(int))),
            "ingesta_s": rec.get("ingesta_s")}
import json
(RES / "manifiesto_datos.json").write_text(json.dumps(manifest, indent=2))
manifest

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
colors = {"entrenamiento": SERIES[0], "validación": SERIES[1], "prueba": SERIES[2]}
ax.bar(by_day["day"].astype(str), by_day["n"] / 1e6, color=[colors[x] for x in by_day["particion"]], width=0.7)
for k, c in colors.items():
    ax.bar([], [], color=c, label=k)
ax.set_xlabel("Día de octubre de 2014"); ax.set_ylabel("Millones de impresiones")
ax.set_title("Impresiones por día y partición temporal"); ax.legend(ncols=3, loc="upper left")
savefig(fig, FIG, "01_filas_por_dia")
