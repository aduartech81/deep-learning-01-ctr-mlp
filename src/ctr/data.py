"""Ingesta (CSV -> Parquet particionado por día) y carga de particiones temporales.

Toda la lectura es perezosa (Polars / DuckDB) para no cargar los ~40 M de filas en memoria.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

AVAZU_COLUMNS = [
    "id", "click", "hour", "C1", "banner_pos", "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category", "device_id", "device_ip", "device_model",
    "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
DERIVED = ["hour_of_day", "dow", "franja"]


def csv_to_parquet(csv_path: Path, out_dir: Path, memory_limit: str = "8GB", threads: int | None = None) -> dict:
    """Convierte el CSV (o .gz) de Avazu en Parquet particionado por día con DuckDB (fuera de memoria).

    Deriva de `hour` (formato YYMMDDHH): day, hour_of_day, dow (0 = lunes) y franja horaria.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("day=*")):
        return {"status": "ya existía", "path": str(out_dir)}
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    types = {c: "VARCHAR" for c in AVAZU_COLUMNS}
    types.update({"click": "TINYINT", "hour": "INTEGER"})
    cols_sql = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    t0 = time.perf_counter()
    con.execute(f"""
        COPY (
          SELECT *,
                 CAST(substr(CAST(hour AS VARCHAR), 5, 2) AS INTEGER) AS day,
                 CAST(substr(CAST(hour AS VARCHAR), 7, 2) AS VARCHAR) AS hour_of_day,
                 CAST(isodow(make_date(2000 + CAST(substr(CAST(hour AS VARCHAR), 1, 2) AS INTEGER),
                                       CAST(substr(CAST(hour AS VARCHAR), 3, 2) AS INTEGER),
                                       CAST(substr(CAST(hour AS VARCHAR), 5, 2) AS INTEGER))) - 1 AS VARCHAR) AS dow,
                 CASE WHEN CAST(substr(CAST(hour AS VARCHAR), 7, 2) AS INTEGER) < 6 THEN 'madrugada'
                      WHEN CAST(substr(CAST(hour AS VARCHAR), 7, 2) AS INTEGER) < 12 THEN 'manana'
                      WHEN CAST(substr(CAST(hour AS VARCHAR), 7, 2) AS INTEGER) < 18 THEN 'tarde'
                      ELSE 'noche' END AS franja
          FROM read_csv('{Path(csv_path).as_posix()}', header=true, columns={{{cols_sql}}})
        ) TO '{out_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (day), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)
    secs = time.perf_counter() - t0
    con.close()
    return {"status": "creado", "path": str(out_dir), "seconds": secs}


def scan(parquet_dir: Path) -> pl.LazyFrame:
    """Lectura perezosa de todo el Parquet; `day` se recupera de la partición hive."""
    return pl.scan_parquet(str(Path(parquet_dir) / "day=*" / "*.parquet"), hive_partitioning=True)


def _days_filter(lf: pl.LazyFrame, days) -> pl.LazyFrame:
    return lf.filter(pl.col("day").is_in(list(days)))


def load_split(cfg, split: str, n: int | None = None, seed: int = 0, columns=None) -> pl.DataFrame:
    """Devuelve la partición temporal pedida ('train' | 'valid' | 'test').

    Si `n` está definido, toma una muestra determinista por hash del `id`. Las muestras están
    anidadas (la de 100 mil está contenida en la de 1 millón), lo que hace comparables las
    curvas de aprendizaje y de costo. El resultado se ordena por `id` para alinear predicciones
    entre scikit-learn y PySpark.
    """
    sp = cfg["split"]
    days = {"train": sp["train_days"], "valid": [sp["valid_day"]], "test": [sp["test_day"]]}[split]
    lf = _days_filter(scan(cfg.path("parquet_dir")), days)
    cols = columns or (["id", "click", "day"] + cfg.feature_cols)
    lf = lf.select(cols)
    if n is not None:
        # muestreo por hash: estable, perezoso y anidado; el seed es fijo para que las muestras
        # sean las mismas en ambos entornos (la variación entre semillas va en los modelos)
        total = _days_filter(scan(cfg.path("parquet_dir")), days).select(pl.len()).collect().item()
        if n < total:
            frac = min(1.0, max(n * 1.05, n + 2000) / total)
            thr = int(frac * (2**64 - 1))
            lf = (lf.with_columns(_h=pl.col("id").hash(seed=1234))
                    .filter(pl.col("_h") <= pl.lit(thr, dtype=pl.UInt64))
                    .sort("_h").head(n).drop("_h"))
    return lf.sort("id").collect(engine="streaming")


def sample_path(cfg, split: str, n: int | None) -> Path:
    """Materializa (una sola vez) la muestra en Parquet para que scikit-learn y PySpark usen
    exactamente las mismas filas. Devuelve la ruta."""
    d = cfg.path("results_dir") / "samples"
    d.mkdir(parents=True, exist_ok=True)
    tag = "all" if n is None else str(int(n))
    p = d / f"{split}_{tag}.parquet"
    if not p.exists():
        load_split(cfg, split, n=n).write_parquet(p)
    return p


def read_sample(cfg, split: str, n: int | None) -> pl.DataFrame:
    return pl.read_parquet(sample_path(cfg, split, n))


def eval_sets(cfg):
    """Validación (submuestra opcional del día 29) y prueba (día 30, completo por defecto)."""
    va = read_sample(cfg, "valid", cfg["valid_max_rows"])
    te = read_sample(cfg, "test", cfg["test_max_rows"])
    return va, te


def split_sizes(cfg) -> dict:
    sp = cfg["split"]
    lf = scan(cfg.path("parquet_dir"))
    by_day = lf.group_by("day").agg(pl.len().alias("n"), pl.col("click").mean().alias("ctr")).sort("day").collect()
    d = dict(zip(by_day["day"].to_list(), by_day["n"].to_list()))
    return {
        "train": int(sum(d.get(x, 0) for x in sp["train_days"])),
        "valid": int(d.get(sp["valid_day"], 0)),
        "test": int(d.get(sp["test_day"], 0)),
        "by_day": by_day,
    }


def add_crosses(df: pl.DataFrame, crosses) -> tuple[pl.DataFrame, list[str]]:
    names = []
    for a, b in crosses:
        name = f"{a}_x_{b}"
        df = df.with_columns((pl.col(a).cast(pl.Utf8) + "|" + pl.col(b).cast(pl.Utf8)).alias(name))
        names.append(name)
    return df, names


def to_numpy_labels(df: pl.DataFrame) -> np.ndarray:
    return df["click"].to_numpy().astype(np.int8)
