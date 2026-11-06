"""Utilidades comunes de los notebooks: configuración, estilo de gráficos y registro de corridas."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config

# Paleta categórica validada (orden fijo; el color sigue a la entidad, no al rango)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIV_CMAP = mpl.colors.LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19d9c", "#b42d2c"])

FRAMEWORK_COLOR = {"sklearn": SERIES[0], "spark": SERIES[1]}
MODEL_COLOR = {"mlp": SERIES[0], "sgd": SERIES[1], "lr": SERIES[1], "hgb": SERIES[2], "gbt": SERIES[2]}
ENC_COLOR = {"hash": SERIES[0], "freq": SERIES[1], "target": SERIES[2]}


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 6, "figure.dpi": 110, "savefig.dpi": 150,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "font.family": "sans-serif",
    })


def setup(profile=None):
    """Carga configuración, aplica estilo y devuelve (cfg, rutas)."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="pyspark")
    style()
    cfg = load_config(profile)
    res = cfg.path("results_dir")
    fig = cfg.path("figures_dir")
    (res / "preds").mkdir(parents=True, exist_ok=True)
    print(f"Perfil: {cfg.profile}  |  resultados en: {res}")
    if cfg.profile == "smoke":
        print("ATENCIÓN: perfil 'smoke' con datos SINTÉTICOS. Los números no tienen valor empírico.")
    return cfg, res, fig


def savefig(fig, figdir: Path, name: str):
    fig.tight_layout()
    fig.savefig(Path(figdir) / f"{name}.png", bbox_inches="tight")


def run_key(**kw) -> str:
    return "|".join(f"{k}={kw[k]}" for k in sorted(kw))


def done_keys(path: Path) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def save_preds(res: Path, key: str, p: np.ndarray) -> str:
    name = "".join(ch if ch.isalnum() else "_" for ch in key)[:180] + ".npy"
    np.save(Path(res) / "preds" / name, np.asarray(p, dtype=np.float32))
    return name


def load_preds(res: Path, name: str) -> np.ndarray:
    return np.load(Path(res) / "preds" / name)


def show(df: pd.DataFrame, n=30, floatfmt="{:.4f}"):
    from IPython.display import display
    with pd.option_context("display.float_format", floatfmt.format, "display.max_columns", 50,
                           "display.width", 200):
        display(df.head(n))
