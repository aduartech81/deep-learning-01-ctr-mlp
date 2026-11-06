"""Ejecución estandarizada de una corrida (modelo, codificación, tamaño, semilla) con medición."""
from __future__ import annotations

import numpy as np

from . import metrics as M
from . import models_sk as MS
from .bench import measure


def run_sk_model(model, Xtr, ytr, Xva, yva, Xte, yte, seed, params=None, class_weight=None,
                 max_epochs=30, patience=3):
    """Entrena, elige el umbral en validación y evalúa en prueba. Devuelve (registro, p_test)."""
    rec = {"framework": "sklearn", "model": model, "seed": seed,
           "class_weight": class_weight or "none", "n_train": Xtr.shape[0]}
    with measure(rec, "train", Xtr.shape[0]):
        if model == "mlp":
            clf = MS.fit_mlp_temporal(Xtr, ytr, Xva, yva, params, seed, max_epochs, patience, class_weight)
            rec["best_epoch"] = clf.best_epoch_
            rec["epochs_run"] = len(clf.history_)
        elif model == "mlp_es_interna":
            clf = MS.fit_mlp_internal_es(Xtr, ytr, params, seed)
            rec["epochs_run"] = clf.n_iter_
        elif model == "sgd":
            clf = MS.fit_sgd(Xtr, ytr, Xva, yva, seed, class_weight=class_weight)
            rec["alpha"] = clf.chosen_alpha_
        elif model == "hgb":
            clf = MS.fit_hgb(Xtr, ytr, Xva, yva, seed, class_weight=class_weight)
            rec["best_iter"] = clf.n_iter
        else:
            raise ValueError(model)
    p_va = clf.predict_proba(Xva)[:, 1]
    thr = M.best_threshold(yva, p_va)
    with measure(rec, "predict", Xte.shape[0]):
        p_te = clf.predict_proba(Xte)[:, 1]
    rec.update({f"val_{k}": v for k, v in M.all_metrics(yva, p_va, thr).items()
                if k in ("roc_auc", "pr_auc", "log_loss")})
    rec.update(M.all_metrics(yte, p_te, thr))
    return rec, p_te, clf
