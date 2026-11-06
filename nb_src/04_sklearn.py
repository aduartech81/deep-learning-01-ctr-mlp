# %% [markdown]
# # 4. Modelado con scikit-learn (muestras escalonadas)
#
# **Diseño.**
# 1. Búsqueda bayesiana (Optuna, TPE) del MLP sobre una muestra de entrenamiento, con validación
#    temporal (día 29) y la codificación principal (hashing $2^{14}$).
# 2. Corridas finales con los mejores hiperparámetros en muestras de 100 mil, 1 millón y
#    10 millones de filas (anidadas), para cada codificación y modelo:
#    - `sgd`: regresión logística por SGD (línea base lineal).
#    - `hgb`: `HistGradientBoostingClassifier` (línea base no lineal; requiere matriz densa, por eso
#      se usa con frecuencia y target encoding).
#    - `mlp`: `MLPClassifier` (ReLU, Adam) con **parada temprana temporal**.
# 3. Semillas: 5 repeticiones en los tamaños `sk_seed_sizes` (media, desviación, IC 95 %); una en
#    el resto por costo.
# 4. Ablaciones: dimensión del hashing, cruces, profundidad, ponderación de clases y la parada
#    temprana interna del enunciado.
#
# **Sobre `early_stopping=True`.** El enunciado lo propone, pero scikit-learn separa al azar un
# 10 % del entrenamiento para decidir cuándo parar, lo que mezcla días y contradice la división
# temporal. Se reemplaza por entrenamiento época a época (`partial_fit`) con parada según la log
# loss del día 29, y la variante del enunciado se conserva como ablación para medir la diferencia.
#
# Las corridas se registran en `results/runs_sklearn.jsonl` y el notebook se puede reanudar: lo que
# ya está registrado no se recalcula.

# %%
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ctr.nbutils import setup, savefig, show, run_key, done_keys, save_preds, SERIES, MODEL_COLOR, ENC_COLOR, MUTED
from ctr import data, models_sk as MS
from ctr.features import CTRFeaturizer
from ctr.experiments import run_sk_model
from ctr.bench import append_jsonl, read_jsonl, measure

cfg, RES, FIG = setup()
RUNS = RES / "runs_sklearn.jsonl"
ABL = RES / "ablaciones_sklearn.jsonl"
va, te = data.eval_sets(cfg)
yva, yte = va["click"].to_numpy(), te["click"].to_numpy()
te.select("id").write_parquet(RES / "test_ids.parquet")   # orden de referencia de la prueba
DIM = cfg["hash_dim_main"]
print(f"validación: {len(yva):,} filas (CTR {yva.mean():.4f}) | prueba: {len(yte):,} filas (CTR {yte.mean():.4f})")

# %% [markdown]
# ## 4.1 Búsqueda de hiperparámetros con Optuna (validación temporal)

# %%
best_path = RES / "mlp_best_params.json"
n_opt = cfg["optuna_train_size"]
tr = data.read_sample(cfg, "train", n_opt)
feat = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"])
c_tr, Xtr = feat.fit_transform(tr, tr["click"].to_numpy(), "hash", DIM)
Xva = feat.transform_codes(feat.encode(va), "hash", DIM)
ytr = tr["click"].to_numpy()

if best_path.exists():
    best = json.loads(best_path.read_text())
    trials = pd.read_csv(RES / "optuna_trials.csv")
else:
    t0 = time.perf_counter()
    study = MS.optuna_search(Xtr, ytr, Xva, yva, cfg["mlp_search_space"], cfg["optuna_trials"],
                             cfg["seed_base"], cfg["max_epochs"], cfg["patience"])
    trials = study.trials_dataframe()
    trials.to_csv(RES / "optuna_trials.csv", index=False)
    best = dict(study.best_params)
    best["val_log_loss"] = study.best_value
    best["busqueda_s"] = time.perf_counter() - t0
    best_path.write_text(json.dumps(best, indent=2))
print(json.dumps(best, indent=2))
show(trials.sort_values("value").head(10))

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
axs[0].plot(trials["number"], trials["value"], "o", color=SERIES[0], markersize=5, label="ensayo")
axs[0].plot(trials["number"], trials["value"].cummin(), color=SERIES[1], label="mejor hasta ahora")
axs[0].set_title("Historia de la optimización"); axs[0].set_xlabel("Ensayo"); axs[0].set_ylabel("log loss validación"); axs[0].legend()
axs[1].scatter(trials["params_alpha"], trials["value"], color=SERIES[0], s=24)
axs[1].set_xscale("log"); axs[1].set_title("alpha vs log loss"); axs[1].set_xlabel("alpha")
g = trials.groupby("params_hidden_layer_sizes")["value"].min()
axs[2].bar(g.index.astype(str), g.values, color=SERIES[0], width=0.6)
axs[2].set_ylim(g.min() * 0.995, g.max() * 1.003); axs[2].set_title("Mejor log loss por arquitectura")
savefig(fig, FIG, "04_optuna")

PARAMS = {k: best[k] for k in ["hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size"]}

# %% [markdown]
# ## 4.2 Corridas finales: tamaños × codificaciones × modelos × semillas

# %%
ENCODINGS = ["hash", "freq", "target"]
MODELS = {"hash": ["sgd", "mlp"], "freq": ["sgd", "hgb", "mlp"], "target": ["sgd", "hgb", "mlp"]}
done = done_keys(RUNS)
for n in cfg["sk_sizes"]:
    n_avail = data.split_sizes(cfg)["train"]
    if n > n_avail:
        print(f"Se omite n={n:,}: el entrenamiento solo tiene {n_avail:,} filas"); continue
    seeds = cfg.seeds if n in cfg["sk_seed_sizes"] else [cfg["seed_base"]]
    pending = [(e, m, s) for e in ENCODINGS for m in MODELS[e] for s in seeds
               if run_key(fw="sklearn", model=m, enc=e, n=n, seed=s, cw="none") not in done]
    if not pending:
        continue
    prep = {}
    with measure(prep, "lectura", n):
        tr = data.read_sample(cfg, "train", n)
    ytr = tr["click"].to_numpy()
    feat = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"])
    with measure(prep, "ajuste_vocab", n):
        feat.fit(tr)
        c_tr = feat.encode(tr)
    c_va, c_te = feat.encode(va), feat.encode(te)
    del tr
    for enc in ENCODINGS:
        if not any(p[0] == enc for p in pending):
            continue
        with measure(prep, f"transform_{enc}", n):
            Xtr = feat.transform_codes(c_tr, enc, DIM, y=ytr, fit=True)
        Xva_, Xte_ = feat.transform_codes(c_va, enc, DIM), feat.transform_codes(c_te, enc, DIM)
        for (e, m, s) in pending:
            if e != enc:
                continue
            key = run_key(fw="sklearn", model=m, enc=e, n=n, seed=s, cw="none")
            t0 = time.perf_counter()
            rec, p_te, clf = run_sk_model(m, Xtr, ytr, Xva_, yva, Xte_, yte, s, PARAMS,
                                          max_epochs=cfg["max_epochs"], patience=cfg["patience"])
            rec.update({"key": key, "encoding": e, "hash_dim": DIM if e == "hash" else None, "n": n,
                        "lectura_s": prep["lectura_s"], "prep_s": prep["ajuste_vocab_s"] + prep[f"transform_{enc}_s"],
                        "prep_peak_mb": max(prep["ajuste_vocab_peak_mb"], prep[f"transform_{enc}_peak_mb"])})
            if s == cfg["seed_base"]:
                rec["pred_file"] = save_preds(RES, key, p_te)
                if m == "mlp":
                    rec["history"] = clf.history_
            append_jsonl(RUNS, rec)
            print(f"n={n:>10,} {e:6s} {m:4s} seed={s} AUC={rec['roc_auc']:.4f} logloss={rec['log_loss']:.4f} "
                  f"train={rec['train_s']:.1f}s  ({time.perf_counter() - t0:.0f}s)")

# %%
runs = read_jsonl(RUNS)
from ctr.metrics import summarize_runs
tabla = summarize_runs(runs, ["n", "encoding", "model"], ["roc_auc", "pr_auc", "log_loss", "brier", "ece", "f1", "train_s", "train_peak_mb"])
tabla.to_csv(RES / "tabla_sklearn.csv", index=False)
show(tabla[["n", "encoding", "model", "n_semillas", "roc_auc_media", "roc_auc_sd", "roc_auc_ic",
            "pr_auc_media", "log_loss_media", "ece_media", "f1_media", "train_s_media", "train_peak_mb_media"]], 60)

# %% [markdown]
# ## 4.3 Curvas de aprendizaje y de costo

# %%
agg = runs.groupby(["model", "encoding", "n"]).agg(auc=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
                                                   t=("train_s", "mean"), mem=("train_peak_mb", "mean")).reset_index()
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
styles = {"hash": "-", "freq": "--", "target": ":"}
for (m, e), g in agg.groupby(["model", "encoding"]):
    lab = f"{m} · {e}"
    axs[0].errorbar(g["n"], g["auc"], yerr=g["auc_sd"].fillna(0), color=MODEL_COLOR[m], linestyle=styles[e], marker="o", label=lab, capsize=3)
    axs[1].plot(g["n"], g["t"], color=MODEL_COLOR[m], linestyle=styles[e], marker="o", label=lab)
    axs[2].plot(g["n"], g["mem"] / 1024, color=MODEL_COLOR[m], linestyle=styles[e], marker="o", label=lab)
for ax, t, yl in zip(axs, ["Curva de aprendizaje (prueba)", "Curva de costo: entrenamiento", "Memoria pico en entrenamiento"],
                     ["ROC AUC", "segundos", "GB"]):
    ax.set_xscale("log"); ax.set_xlabel("Filas de entrenamiento"); ax.set_title(t); ax.set_ylabel(yl)
axs[1].set_yscale("log"); axs[0].legend(fontsize=7, ncols=2)
savefig(fig, FIG, "04_curvas_aprendizaje_costo")

# %%
# curva de entrenamiento del MLP (época a época) en la corrida más grande con hashing
h = runs[(runs.model == "mlp") & (runs.encoding == "hash") & runs["history"].notna()].sort_values("n").iloc[-1]
hist = pd.DataFrame(h["history"])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(hist["epoch"], hist["train_loss"], marker="o", color=SERIES[0], label="pérdida de entrenamiento")
ax.plot(hist["epoch"], hist["val_log_loss"], marker="o", color=SERIES[1], label="log loss validación (día 29)")
ax.axvline(h["best_epoch"], color=MUTED, linestyle="--", linewidth=1)
ax.text(h["best_epoch"], ax.get_ylim()[1], " mejor época", color=MUTED, va="top", fontsize=8)
ax.set_xlabel("Época"); ax.set_title(f"MLP · hashing · n = {int(h['n']):,}"); ax.legend()
savefig(fig, FIG, "04_mlp_epocas")

# %% [markdown]
# ## 4.4 Ablaciones (n = `ablation_size`, varias semillas)

# %%
NA = cfg["ablation_size"]
seeds_abl = cfg.seeds[: cfg["ablation_seeds"]]
done_abl = done_keys(ABL)
tr = data.read_sample(cfg, "train", NA); ytr = tr["click"].to_numpy()
feat = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"]).fit(tr)
c_tr, c_va, c_te = feat.encode(tr), feat.encode(va), feat.encode(te)

def abl(factor, nivel, model, enc, dim, params, s, cw=None, f=feat, ctr=c_tr, cva=c_va, cte=c_te):
    key = run_key(factor=factor, nivel=nivel, model=model, seed=s)
    if key in done_abl:
        return
    Xtr = f.transform_codes(ctr, enc, dim, y=ytr, fit=True)
    rec, _, _ = run_sk_model(model, Xtr, ytr, f.transform_codes(cva, enc, dim), yva,
                             f.transform_codes(cte, enc, dim), yte, s, params, cw,
                             cfg["max_epochs"], cfg["patience"])
    rec.update({"key": key, "factor": factor, "nivel": str(nivel), "encoding": enc, "n": NA,
                "dim_entrada": Xtr.shape[1]})
    append_jsonl(ABL, rec)

for s in seeds_abl:
    for dim in cfg["hash_dims"]:                                   # dimensión del hashing
        abl("dimension_hashing", f"2^{int(np.log2(dim))}", "mlp", "hash", dim, PARAMS, s)
    for hl in cfg["mlp_search_space"]["hidden_layer_sizes"]:       # profundidad / ancho
        abl("arquitectura", hl, "mlp", "hash", DIM, dict(PARAMS, hidden_layer_sizes=hl), s)
    for m in ["mlp", "sgd", "hgb"]:                                 # ponderación de clases
        enc = "hash" if m != "hgb" else "freq"
        for cw in [None, "balanced"]:
            abl("ponderacion_clases", f"{m}:{cw or 'ninguna'}", m, enc, DIM, PARAMS, s, cw)
    abl("parada_temprana", "temporal (día 29)", "mlp", "hash", DIM, PARAMS, s)
    abl("parada_temprana", "interna aleatoria (enunciado)", "mlp_es_interna", "hash", DIM, PARAMS, s)

feat_x = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"], crosses=cfg["features"]["crosses"]).fit(tr)
cx_tr, cx_va, cx_te = feat_x.encode(tr), feat_x.encode(va), feat_x.encode(te)
for s in seeds_abl:                                                # cruces de variables
    abl("cruces", "sin cruces", "mlp", "hash", DIM, PARAMS, s)
    abl("cruces", "con cruces", "mlp", "hash", DIM, PARAMS, s, None, feat_x, cx_tr, cx_va, cx_te)

ab = read_jsonl(ABL)
tab_abl = summarize_runs(ab, ["factor", "nivel"], ["roc_auc", "pr_auc", "log_loss", "ece", "f1", "recall", "train_s"])
tab_abl.to_csv(RES / "tabla_ablaciones_sklearn.csv", index=False)
show(tab_abl[["factor", "nivel", "n_semillas", "roc_auc_media", "roc_auc_sd", "pr_auc_media", "log_loss_media",
              "ece_media", "f1_media", "recall_media", "train_s_media"]], 40)

# %%
facs = list(tab_abl["factor"].unique())
fig, axs = plt.subplots(1, len(facs), figsize=(4 * len(facs), 3.8), squeeze=False)
for ax, fac in zip(axs[0], facs):
    t = tab_abl[tab_abl.factor == fac]
    ax.bar(range(len(t)), t["roc_auc_media"], yerr=t["roc_auc_sd"].fillna(0), color=SERIES[0], width=0.6, capsize=3)
    ax.set_xticks(range(len(t)), t["nivel"], rotation=35, ha="right", fontsize=8)
    lo = (t["roc_auc_media"] - t["roc_auc_sd"].fillna(0)).min()
    ax.set_ylim(lo - 0.01, (t["roc_auc_media"] + t["roc_auc_sd"].fillna(0)).max() + 0.005)
    ax.set_title(fac.replace("_", " "), fontsize=10)
axs[0][0].set_ylabel("ROC AUC (prueba)")
savefig(fig, FIG, "04_ablaciones")
