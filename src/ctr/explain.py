"""Interpretabilidad local (LIME) y contraste (SHAP, importancia por permutación).

Problema: con hashing las columnas del modelo no son legibles. Solución: LIME y SHAP perturban
el ESPACIO INTERPRETABLE (un código entero por variable original, ya con raras agrupadas) y la
función de predicción aplica por dentro la misma codificación (hashing/frecuencia) antes de
llamar al modelo. Así cada explicación queda en términos de site_id, device_model, hora, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


class RawModel:
    """Envuelve featurizer + modelo para predecir desde códigos interpretables."""

    def __init__(self, featurizer, clf, encoding, hash_dim=None):
        self.f, self.clf, self.encoding, self.hash_dim = featurizer, clf, encoding, hash_dim

    def predict_proba(self, codes):
        codes = np.asarray(np.rint(codes), dtype=np.int32)
        X = self.f.transform_codes(codes, self.encoding, self.hash_dim)
        return self.clf.predict_proba(X)

    def p1(self, codes):
        return self.predict_proba(codes)[:, 1]


def select_cases(y, p, thr, n) -> pd.DataFrame:
    """Falsos positivos y falsos negativos con mayor confianza errónea (selección sistemática)."""
    y = np.asarray(y)
    p = np.asarray(p)
    fp_idx = np.flatnonzero((y == 0) & (p >= thr))
    fn_idx = np.flatnonzero((y == 1) & (p < thr))
    fp_top = fp_idx[np.argsort(-p[fp_idx])][:n]
    fn_top = fn_idx[np.argsort(p[fn_idx])][:n]
    return pd.DataFrame({"fila": np.r_[fp_top, fn_top],
                         "tipo": ["FP"] * len(fp_top) + ["FN"] * len(fn_top),
                         "y": y[np.r_[fp_top, fn_top]], "p": p[np.r_[fp_top, fn_top]]})


def make_lime(train_codes, featurizer, kernel_width=None, seed=0):
    from lime.lime_tabular import LimeTabularExplainer
    names = featurizer.feature_names_interpretable()
    cat_names = {j: featurizer.names[c] for j, c in enumerate(names)}
    return LimeTabularExplainer(training_data=np.asarray(train_codes, dtype=float),
                                feature_names=names, categorical_features=list(range(len(names))),
                                categorical_names=cat_names, class_names=["no_clic", "clic"],
                                mode="classification", kernel_width=kernel_width,
                                discretize_continuous=False, random_state=seed)


def lime_explain(explainer, row_codes, predict_fn, num_samples, num_features, seed=None):
    if seed is not None:
        explainer.random_state = np.random.RandomState(seed)
    exp = explainer.explain_instance(np.asarray(row_codes, dtype=float), predict_fn,
                                     num_features=num_features, num_samples=num_samples,
                                     labels=(1,))
    weights = dict(exp.as_map()[1])
    return {"weights": weights, "r2": float(exp.score), "local_pred": float(np.ravel(exp.local_pred)[0]),
            "intercept": float(exp.intercept[1]), "exp": exp}


def topk(weights: dict, k: int) -> set:
    return set(sorted(weights, key=lambda j: -abs(weights[j]))[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def mean_pairwise_jaccard(sets) -> float:
    vals = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals)) if vals else np.nan


def shap_values(predict_p1, background_codes, cases_codes, seed=0, max_evals=None):
    """SHAP agnóstico al modelo (algoritmo de permutación) sobre el espacio interpretable."""
    import shap
    masker = shap.maskers.Independent(np.asarray(background_codes, dtype=float), max_samples=len(background_codes))
    n_feat = cases_codes.shape[1]
    explainer = shap.explainers.Permutation(lambda X: predict_p1(X), masker, seed=seed)
    sv = explainer(np.asarray(cases_codes, dtype=float), max_evals=max_evals or (2 * n_feat + 1) * 5)
    return sv.values


def permutation_importance_codes(predict_p1, codes, y, names, n_repeats=3, seed=0) -> pd.DataFrame:
    """Importancia global: caída de ROC AUC al permutar cada variable original."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict_p1(codes))
    rows = []
    for j, name in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            c2 = codes.copy()
            c2[:, j] = rng.permutation(c2[:, j])
            drops.append(base - roc_auc_score(y, predict_p1(c2)))
        rows.append({"variable": name, "caida_auc_media": np.mean(drops), "caida_auc_sd": np.std(drops, ddof=1) if n_repeats > 1 else np.nan})
    return pd.DataFrame(rows).sort_values("caida_auc_media", ascending=False).reset_index(drop=True)


def rank_agreement(w_a: dict, w_b: np.ndarray, n_feat: int, k: int) -> dict:
    """Coherencia LIME vs SHAP en un caso: Jaccard del top-k y Spearman de |atribución|."""
    a = np.array([abs(w_a.get(j, 0.0)) for j in range(n_feat)])
    b = np.abs(np.asarray(w_b))
    top_a = set(np.argsort(-a)[:k])
    top_b = set(np.argsort(-b)[:k])
    rho = stats.spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan
    return {"jaccard_topk": jaccard(top_a, top_b), "spearman_abs": float(rho)}
