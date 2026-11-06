"""Carga de la configuración y rutas del proyecto."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    profile: str
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    def path(self, key: str) -> Path:
        p = ROOT / self.raw["paths"][key]
        if key.endswith("_dir"):
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def high_card(self):
        return list(self.raw["features"]["high_card"])

    @property
    def low_card(self):
        return list(self.raw["features"]["low_card"])

    @property
    def feature_cols(self):
        return self.high_card + self.low_card

    @property
    def seeds(self):
        return [self.raw["seed_base"] + i for i in range(self.raw["n_seeds"])]


def load_config(profile: str | None = None) -> Config:
    profile = profile or os.environ.get("CTR_PROFILE", "full")
    with open(ROOT / "config.yaml", encoding="utf8") as f:
        y = yaml.safe_load(f)
    if profile not in y["profiles"]:
        raise ValueError(f"Perfil desconocido: {profile}")
    raw = _deep_merge(y["common"], y["profiles"][profile])
    if profile == "smoke":  # la prueba de humo nunca toca los datos reales
        raw["paths"] = dict(raw["paths"], raw_csv="data/smoke/train_synthetic.csv.gz",
                            parquet_dir="data/smoke/parquet", results_dir="results_smoke",
                            figures_dir="results_smoke/figures")
    return Config(profile, raw)
