"""Genera un CSV SINTÉTICO con el mismo esquema de Avazu, solo para la prueba de humo.

Los resultados obtenidos con estos datos NO tienen ningún valor empírico: sirven únicamente para
verificar que todos los notebooks corren de punta a punta antes de usar los datos reales.

Uso:  python scripts/make_synthetic.py --rows-per-day 30000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def zipf_choice(rng, n_cat, size, a=1.3, prefix=""):
    ranks = np.arange(1, n_cat + 1)
    w = 1 / ranks**a
    w /= w.sum()
    idx = rng.choice(n_cat, size=size, p=w)
    return idx, np.array([f"{prefix}{i:08x}" for i in range(n_cat)])[idx]


def main(rows_per_day=30000, seed=7, out="data/smoke/train_synthetic.csv.gz"):
    rng = np.random.default_rng(seed)
    days = list(range(21, 31))
    n = rows_per_day * len(days)
    day = np.repeat(days, rows_per_day)
    hod = rng.integers(0, 24, n)
    hour = (14100000 + day * 100 + hod).astype(np.int64)
    cols = {}
    site_idx, cols["site_id"] = zipf_choice(rng, 3000, n, 1.2, "s")
    cols["site_domain"] = np.array([f"d{(i * 7) % 2500:07x}" for i in range(3000)])[site_idx]
    cols["site_category"] = np.array([f"c{i % 20:07x}" for i in range(3000)])[site_idx]
    app_idx, cols["app_id"] = zipf_choice(rng, 4000, n, 1.4, "a")
    cols["app_domain"] = np.array([f"ad{(i * 3) % 300:06x}" for i in range(4000)])[app_idx]
    cols["app_category"] = np.array([f"ac{i % 25:06x}" for i in range(4000)])[app_idx]
    is_default = rng.random(n) < 0.8
    _, dev = zipf_choice(rng, 60000, n, 1.05, "e")
    cols["device_id"] = np.where(is_default, "a99f214a", dev)
    _, cols["device_ip"] = zipf_choice(rng, 150000, n, 1.02, "i")
    model_idx, cols["device_model"] = zipf_choice(rng, 5000, n, 1.1, "m")
    cols["device_type"] = rng.choice(["0", "1", "4", "5"], n, p=[0.05, 0.85, 0.08, 0.02])
    cols["device_conn_type"] = rng.choice(["0", "2", "3", "5"], n, p=[0.85, 0.08, 0.05, 0.02])
    cols["C1"] = rng.choice(["1005", "1002", "1010", "1012"], n, p=[0.9, 0.05, 0.03, 0.02])
    cols["banner_pos"] = rng.choice(["0", "1", "7"], n, p=[0.72, 0.27, 0.01])
    c14_idx, cols["C14"] = zipf_choice(rng, 2500, n, 1.1, "")
    cols["C14"] = (15000 + c14_idx).astype(str)
    cols["C15"] = rng.choice(["320", "300", "728"], n, p=[0.93, 0.05, 0.02])
    cols["C16"] = rng.choice(["50", "250", "90"], n, p=[0.94, 0.05, 0.01])
    cols["C17"] = (1700 + c14_idx % 430).astype(str)
    cols["C18"] = rng.choice(["0", "1", "2", "3"], n, p=[0.42, 0.07, 0.17, 0.34])
    cols["C19"] = (35 + c14_idx % 68).astype(str)
    cols["C20"] = np.where(rng.random(n) < 0.47, "-1", (100000 + c14_idx % 170).astype(str))
    cols["C21"] = (20 + c14_idx % 60).astype(str)

    # señal: efectos por categoría + hora + deriva temporal
    eff_site = rng.normal(0, 0.8, 3000)
    eff_app = rng.normal(0, 0.6, 4000)
    eff_model = rng.normal(0, 0.4, 5000)
    eff_c14 = rng.normal(0, 0.7, 2500)
    logit = (-1.75 + eff_site[site_idx] * 0.7 + eff_app[app_idx] * 0.5 + eff_model[model_idx] * 0.5
             + eff_c14[c14_idx] * 0.8 + 0.25 * np.sin(2 * np.pi * hod / 24)
             + np.where(cols["banner_pos"] == "1", 0.25, 0.0) + np.where(cols["C18"] == "2", -0.3, 0.0)
             + 0.03 * (day - 25))
    p = 1 / (1 + np.exp(-logit))
    click = (rng.random(n) < p).astype(np.int8)
    ids = rng.integers(1, 2**63 - 1, n, dtype=np.int64).astype(np.uint64) * 2 + rng.integers(0, 2, n).astype(np.uint64)

    df = pd.DataFrame({"id": ids.astype(str), "click": click, "hour": hour})
    for c in ["C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain",
              "app_category", "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
              "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21"]:
        df[c] = cols[c]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"{len(df):,} filas sintéticas -> {out}  (tasa de clic {click.mean():.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-day", type=int, default=30000)
    ap.add_argument("--out", default="data/smoke/train_synthetic.csv.gz")
    a = ap.parse_args()
    main(a.rows_per_day, out=a.out)
