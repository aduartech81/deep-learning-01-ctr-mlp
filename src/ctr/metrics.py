"""Métricas de discriminación, probabilísticas, de umbral y de negocio, e inferencia estadística.

Incluye: bootstrap de Poisson vectorizado (rápido para millones de filas), prueba de DeLong
(implementación rápida de Sun y Xu, 2014) y prueba de McNemar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

EPS = 1e-7


# ------------------------------------------------------------------ calibración
def ece(y, p, n_bins=15, strategy="uniform") -> float:
    """Error de calibración esperado. strategy='uniform' (clásico) o 'quantile'."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tot = np.bincount(idx, minlength=len(edges) - 1)
    sp_ = np.bincount(idx, weights=p, minlength=len(edges) - 1)
    sy = np.bincount(idx, weights=y, minlength=len(edges) - 1)
    m = tot > 0
    return float(np.sum(np.abs(sy[m] - sp_[m])) / len(p))


def best_threshold(y, p, criterion="f1") -> float:
    """Umbral que maximiza F1 en VALIDACIÓN (nunca en prueba)."""
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    ps = np.asarray(p)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = ys.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, EPS)
    # solo posiciones donde cambia el puntaje
    last = np.r_[ps[1:] != ps[:-1], True]
    k = np.argmax(np.where(last, f1, -1))
    return float(ps[k])


def all_metrics(y, p, thr) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p, 15, "uniform"),
        "ece_q": ece(y, p, 15, "quantile"),
        "threshold": thr,
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "mean_p": float(p.mean()), "base_rate": float(y.mean()),
    }


def lift_table(y, p, n_bins=10) -> pd.DataFrame:
    """Curva de lift y de ganancia acumulada por deciles de probabilidad."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["decil"] = np.arange(len(df)) * n_bins // len(df) + 1
    g = df.groupby("decil").agg(n=("y", "size"), clics=("y", "sum"), p_media=("p", "mean"))
    base = df["y"].mean()
    g["ctr"] = g["clics"] / g["n"]
    g["lift"] = g["ctr"] / base
    g["pct_poblacion"] = g["n"].cumsum() / g["n"].sum()
    g["ganancia_acum"] = g["clics"].cumsum() / g["clics"].sum()
    g["lift_acum"] = g["ganancia_acum"] / g["pct_poblacion"]
    return g.reset_index()


# ------------------------------------------------------------------ bootstrap
def _grouped(y, p):
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    return order, ps, ys, starts


def bootstrap_ci(y, p, n_boot=500, seed=0, alpha=0.05) -> pd.DataFrame:
    """IC por bootstrap de Poisson (pesos ~ Poisson(1)) para ROC AUC, PR AUC, log loss y Brier.

    Ordena una sola vez y calcula cada réplica en O(n) vectorizado: viable con millones de filas.
    """
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    order, ps, ys, starts = _grouped(y, p)
    ll_i = -(ys * np.log(ps) + (1 - ys) * np.log(1 - ps))
    br_i = (ps - ys) ** 2
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        w = rng.poisson(1.0, size=len(ys)).astype(float)
        wp = np.add.reduceat(w * ys, starts)
        wn = np.add.reduceat(w * (1 - ys), starts)
        Wp, Wn = wp.sum(), wn.sum()
        neg_below = np.cumsum(wn) - wn
        auc = np.sum(wp * (neg_below + 0.5 * wn)) / (Wp * Wn)
        # PR AUC (average precision) recorriendo umbrales de mayor a menor
        tp = np.cumsum(wp[::-1])
        fp = np.cumsum(wn[::-1])
        prec = tp / np.maximum(tp + fp, EPS)
        ap = np.sum(wp[::-1] / Wp * prec)
        W = w.sum()
        out.append((auc, ap, np.sum(w * ll_i) / W, np.sum(w * br_i) / W))
    arr = np.array(out)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({"metrica": ["roc_auc", "pr_auc", "log_loss", "brier"],
                         "ic_inf": lo, "ic_sup": hi, "sd_boot": arr.std(axis=0)})


# ------------------------------------------------------------------ DeLong
def _midrank(x):
    return stats.rankdata(x, method="average")


def delong_test(y, p1, p2) -> dict:
    """Prueba de DeLong para dos AUC correlacionadas (mismas filas de prueba)."""
    y = np.asarray(y).astype(bool)
    m, n = y.sum(), (~y).sum()
    res = {}
    v01, v10 = [], []
    aucs = []
    for p in (p1, p2):
        p = np.asarray(p, float)
        pos, neg = p[y], p[~y]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.r_[pos, neg])
        auc = (tz[:m].sum() / m - (m + 1) / 2) / n
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
        aucs.append(auc)
    sx = np.cov(np.vstack(v01))
    sy = np.cov(np.vstack(v10))
    S = sx / m + sy / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-300))
    res.update(auc1=aucs[0], auc2=aucs[1], diff=aucs[0] - aucs[1], se=float(np.sqrt(var)),
               z=float(z), p_value=float(2 * stats.norm.sf(abs(z))))
    return res


def mcnemar_test(y, yhat1, yhat2) -> dict:
    """McNemar sobre aciertos/errores de dos clasificadores en las mismas filas."""
    y = np.asarray(y)
    c1 = np.asarray(yhat1) == y
    c2 = np.asarray(yhat2) == y
    b = int(np.sum(c1 & ~c2))
    c = int(np.sum(~c1 & c2))
    if b + c == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_value": 1.0, "metodo": "sin discordancias"}
    if b + c < 25:
        pv = stats.binomtest(b, b + c, 0.5).pvalue
        return {"b": b, "c": c, "stat": float(min(b, c)), "p_value": float(pv), "metodo": "exacta"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "stat": float(stat), "p_value": float(stats.chi2.sf(stat, 1)),
            "metodo": "chi2 con corrección de continuidad"}


def mean_sd_ci(values, alpha=0.05) -> dict:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    k = len(v)
    mu = v.mean() if k else np.nan
    sd = v.std(ddof=1) if k > 1 else np.nan
    if k > 1:
        h = stats.t.ppf(1 - alpha / 2, k - 1) * sd / np.sqrt(k)
    else:
        h = np.nan
    return {"media": mu, "sd": sd, "ic_inf": mu - h, "ic_sup": mu + h, "n": k}


def summarize_runs(df: pd.DataFrame, by, metrics) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for mname in metrics:
            s = mean_sd_ci(g[mname].values)
            row[f"{mname}_media"] = s["media"]
            row[f"{mname}_sd"] = s["sd"]
            row[f"{mname}_ic"] = (f"[{s['ic_inf']:.4f}, {s['ic_sup']:.4f}]" if s["n"] > 1 else "n=1")
        row["n_semillas"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)
