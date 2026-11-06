"""Modelos de scikit-learn: MLP con parada temprana TEMPORAL, líneas base y búsqueda con Optuna.

Por qué no se usa `early_stopping=True` como modelo principal: MLPClassifier separa al azar un
10 % del entrenamiento para decidir cuándo parar. Eso mezcla días y contradice el protocolo
temporal. Aquí se entrena época a época con `partial_fit` y se para según la log loss del día 29
(validación). La variante con `early_stopping=True` se conserva como ablación.
"""
from __future__ import annotations

import copy
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

EPS = 1e-7


def parse_hidden(h) -> tuple:
    if isinstance(h, (tuple, list)):
        return tuple(int(x) for x in h)
    return tuple(int(x) for x in str(h).split("-"))


def balanced_weights(y):
    y = np.asarray(y)
    n1 = y.sum()
    n0 = len(y) - n1
    w = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
    return w.astype(np.float64)


def fit_mlp_temporal(Xtr, ytr, Xva, yva, params: dict, seed: int, max_epochs=30, patience=3,
                     class_weight=None, tol=1e-5):
    """MLP (ReLU, Adam) entrenado por épocas; conserva los pesos de la mejor época en validación."""
    clf = MLPClassifier(hidden_layer_sizes=parse_hidden(params["hidden_layer_sizes"]),
                        alpha=params["alpha"], learning_rate_init=params["learning_rate_init"],
                        batch_size=int(params["batch_size"]), solver="adam", shuffle=True,
                        random_state=seed)
    sw = balanced_weights(ytr) if class_weight == "balanced" else None
    best, wait, best_state = np.inf, 0, None
    hist = []
    for ep in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.partial_fit(Xtr, ytr, classes=np.array([0, 1]), sample_weight=sw)
        t_ep = time.perf_counter() - t0
        pva = np.clip(clf.predict_proba(Xva)[:, 1], EPS, 1 - EPS)
        ll = log_loss(yva, pva, labels=[0, 1])
        hist.append({"epoch": ep, "train_loss": float(clf.loss_), "val_log_loss": ll,
                     "val_auc": roc_auc_score(yva, pva), "epoch_s": t_ep})
        if ll < best - tol:
            best, wait = ll, 0
            best_state = (copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_), ep)
        else:
            wait += 1
            if wait >= patience:
                break
    clf.coefs_, clf.intercepts_, best_ep = best_state
    clf.best_epoch_ = best_ep
    clf.history_ = hist
    return clf


def fit_mlp_internal_es(Xtr, ytr, params, seed, max_iter=200):
    """Variante del enunciado (early_stopping=True, validación interna ALEATORIA). Solo ablación."""
    clf = MLPClassifier(hidden_layer_sizes=parse_hidden(params["hidden_layer_sizes"]),
                        alpha=params["alpha"], learning_rate_init=params["learning_rate_init"],
                        batch_size=int(params["batch_size"]), early_stopping=True,
                        n_iter_no_change=3, max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xtr, ytr)
    return clf


def fit_sgd(Xtr, ytr, Xva, yva, seed, alphas=(1e-6, 1e-5, 1e-4, 1e-3), class_weight=None):
    """Regresión logística por SGD; alpha elegido por log loss en validación."""
    best = None
    for a in alphas:
        clf = SGDClassifier(loss="log_loss", alpha=a, max_iter=20, tol=1e-4, random_state=seed,
                            class_weight=class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(Xtr, ytr)
        ll = log_loss(yva, np.clip(clf.predict_proba(Xva)[:, 1], EPS, 1 - EPS), labels=[0, 1])
        if best is None or ll < best[0]:
            best = (ll, a, clf)
    best[2].chosen_alpha_ = best[1]
    return best[2]


class HGBAtIteration:
    """HistGradientBoosting con número de iteraciones elegido en validación (staged)."""

    def __init__(self, clf, n_iter):
        self.clf, self.n_iter = clf, n_iter

    def predict_proba(self, X):
        for i, p in enumerate(self.clf.staged_predict_proba(X), start=1):
            if i == self.n_iter:
                return p
        return p


def fit_hgb(Xtr, ytr, Xva, yva, seed, max_iter=300, class_weight=None):
    """Árboles potenciados; requiere matriz densa (se usa con 'freq' o 'target')."""
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=False, random_state=seed,
                                         class_weight=class_weight)
    clf.fit(Xtr, ytr)
    lls = [log_loss(yva, np.clip(p[:, 1], EPS, 1 - EPS), labels=[0, 1])
           for p in clf.staged_predict_proba(Xva)]
    k = int(np.argmin(lls)) + 1
    m = HGBAtIteration(clf, k)
    m.val_curve_ = lls
    return m


def optuna_search(Xtr, ytr, Xva, yva, space: dict, n_trials: int, seed: int, max_epochs, patience):
    """Búsqueda bayesiana (TPE) con validación temporal (día 29)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", space["hidden_layer_sizes"]),
            "alpha": trial.suggest_float("alpha", *space["alpha"], log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", *space["learning_rate_init"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        }
        t0 = time.perf_counter()
        clf = fit_mlp_temporal(Xtr, ytr, Xva, yva, params, seed, max_epochs, patience)
        h = clf.history_[clf.best_epoch_ - 1]
        trial.set_user_attr("val_auc", h["val_auc"])
        trial.set_user_attr("best_epoch", clf.best_epoch_)
        trial.set_user_attr("fit_s", time.perf_counter() - t0)
        return h["val_log_loss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study
