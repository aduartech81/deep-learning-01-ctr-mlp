"""Preprocesamiento para scikit-learn con la misma lógica que el pipeline de PySpark.

1. Agrupación de categorías raras en `__RARE__` (umbral ajustado SOLO con entrenamiento).
2. Codificación de cada categoría a un código entero (0 = `__RARE__` o no vista).
3. Tres estrategias de codificación (factor experimental):
   - 'hash'  : hashing trick (MurmurHash3 de "columna=valor", sin signo alterno, igual que el
               FeatureHasher de Spark) para alta cardinalidad + one-hot para baja cardinalidad.
   - 'freq'  : log1p(frecuencia en entrenamiento) + StandardScaler.
   - 'target': target encoding out-of-fold (TargetEncoder de scikit-learn, con cross-fitting
               interno de 5 particiones sobre el entrenamiento) + StandardScaler.

Los códigos enteros son además el espacio interpretable que usa LIME.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.utils import murmurhash3_32

from .data import add_crosses

RARE = "__RARE__"


class CTRFeaturizer:
    def __init__(self, high_card, low_card, min_count=20, crosses=None):
        self.high_card = list(high_card)
        self.low_card = list(low_card)
        self.min_count = int(min_count)
        self.crosses = crosses or []
        self._luts = {}
        self._scalers = {}
        self._te = None

    # ------------------------------------------------------------------ ajuste
    def fit(self, df: pl.DataFrame):
        if self.crosses:
            df, names = add_crosses(df, self.crosses)
            self.high_card = self.high_card + [n for n in names if n not in self.high_card]
        self.cols = self.high_card + self.low_card
        self.vocab, self.names, self.counts = {}, {}, {}
        n = df.height
        for c in self.cols:
            vc = (df.select(pl.col(c).cast(pl.Utf8).fill_null("__NA__").alias("v"))
                    .group_by("v").len().sort(["len", "v"], descending=[True, False]))
            kept = vc.filter(pl.col("len") >= self.min_count)
            values = kept["v"].to_list()
            cnt = kept["len"].to_numpy().astype(np.int64)
            rare_cnt = int(n - cnt.sum())
            self.vocab[c] = dict(zip(values, range(1, len(values) + 1)))
            self.names[c] = [RARE] + values
            self.counts[c] = np.concatenate([[max(rare_cnt, 1)], cnt])
        self.n_train_ = n
        return self

    def fit_lazy(self, lf: pl.LazyFrame):
        """Igual que fit, pero cuenta en streaming sin cargar el conjunto (entrenamiento completo)."""
        assert not self.crosses, "fit_lazy no soporta cruces"
        self.cols = self.high_card + self.low_card
        self.vocab, self.names, self.counts = {}, {}, {}
        n = lf.select(pl.len()).collect(engine="streaming").item()
        for c in self.cols:
            vc = (lf.select(pl.col(c).cast(pl.Utf8).fill_null("__NA__").alias("v")).group_by("v").len()
                    .filter(pl.col("len") >= self.min_count).sort(["len", "v"], descending=[True, False])
                    .collect(engine="streaming"))
            values, cnt = vc["v"].to_list(), vc["len"].to_numpy().astype(np.int64)
            self.vocab[c] = dict(zip(values, range(1, len(values) + 1)))
            self.names[c] = [RARE] + values
            self.counts[c] = np.concatenate([[max(int(n - cnt.sum()), 1)], cnt])
        self.n_train_ = n
        return self

    def encode_lazy(self, lf: pl.LazyFrame, chunk_col="day"):
        """Codifica por bloques (un día a la vez) para no cargar todas las cadenas en memoria."""
        keys = sorted(lf.select(pl.col(chunk_col).unique()).collect()[chunk_col].to_list())
        codes, ys = [], []
        for k in keys:
            df = lf.filter(pl.col(chunk_col) == k).select(["click"] + self.cols).collect(engine="streaming")
            codes.append(self.encode(df))
            ys.append(df["click"].to_numpy().astype(np.int8))
            del df
        return np.concatenate(codes), np.concatenate(ys)

    # ------------------------------------------------------------- codificación
    def encode(self, df: pl.DataFrame) -> np.ndarray:
        """Matriz (n, n_cols) de códigos int32; 0 = __RARE__ / no visto en entrenamiento."""
        if self.crosses:
            df, _ = add_crosses(df, self.crosses)
        out = np.empty((df.height, len(self.cols)), dtype=np.int32)
        for j, c in enumerate(self.cols):
            s = df.get_column(c).cast(pl.Utf8).fill_null("__NA__")
            out[:, j] = s.replace_strict(self.vocab[c], default=0, return_dtype=pl.Int32).to_numpy()
        return out

    def _hash_lut(self, dim: int):
        if dim in self._luts:
            return self._luts[dim]
        luts = {}
        for c in self.high_card:
            names = self.names[c]
            luts[c] = np.fromiter((murmurhash3_32(f"{c}={v}", seed=0, positive=True) % dim for v in names),
                                  dtype=np.int64, count=len(names))
        offsets, off = {}, dim
        for c in self.low_card:
            offsets[c] = off
            off += len(self.names[c])
        self._luts[dim] = (luts, offsets, off)
        return self._luts[dim]

    def n_features(self, encoding: str, hash_dim: int | None = None) -> int:
        if encoding == "hash":
            return self._hash_lut(hash_dim)[2]
        return len(self.cols)

    def transform_codes(self, codes: np.ndarray, encoding: str, hash_dim: int | None = None,
                        y: np.ndarray | None = None, fit: bool = False):
        """Convierte códigos en la matriz del modelo. `fit=True` solo con entrenamiento."""
        n = codes.shape[0]
        if encoding == "hash":
            luts, offsets, dim_total = self._hash_lut(hash_dim)
            cols_idx = []
            for j, c in enumerate(self.cols):
                if c in luts:
                    cols_idx.append(luts[c][codes[:, j]])
                else:
                    cols_idx.append(offsets[c] + codes[:, j].astype(np.int64))
            indices = np.stack(cols_idx, axis=1).ravel()
            indptr = np.arange(0, n * len(self.cols) + 1, len(self.cols), dtype=np.int64)
            X = sp.csr_matrix((np.ones(indices.shape[0], dtype=np.float32), indices, indptr),
                              shape=(n, dim_total))
            X.sum_duplicates()  # colisiones de hash dentro de una fila se suman
            return X
        if encoding == "freq":
            X = np.empty(codes.shape, dtype=np.float32)
            for j, c in enumerate(self.cols):
                X[:, j] = np.log1p(self.counts[c][codes[:, j]])
            key = "freq"
        elif encoding == "target":
            if fit:
                self._te = TargetEncoder(categories="auto", target_type="binary", cv=5)
                X = self._te.fit_transform(codes, y).astype(np.float32)  # out-of-fold en entrenamiento
            else:
                X = self._te.transform(codes).astype(np.float32)
            key = "target"
        else:
            raise ValueError(encoding)
        if fit:
            self._scalers[key] = StandardScaler().fit(X)
        return self._scalers[key].transform(X).astype(np.float32)

    def fit_transform(self, df, y, encoding, hash_dim=None):
        self.fit(df)
        codes = self.encode(df)
        return codes, self.transform_codes(codes, encoding, hash_dim, y=y, fit=True)

    def feature_names_interpretable(self):
        return list(self.cols)

    def cardinality(self) -> dict:
        return {c: len(self.names[c]) for c in self.cols}
