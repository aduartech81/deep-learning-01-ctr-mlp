# %% [markdown]
# # 8. Interpretabilidad: LIME, contrastado con SHAP e importancia por permutación
#
# **Modelo explicado:** el MLP de scikit-learn con los mejores hiperparámetros, hashing $2^{14}$,
# entrenado con `ablation_size` filas.
#
# **Espacio interpretable.** Con hashing las columnas del modelo no son legibles. LIME y SHAP
# perturban un vector de 24 códigos, uno por variable original (con raras ya agrupadas), y la
# función de predicción aplica por dentro el mismo hashing antes de llamar al MLP. Así cada
# explicación habla de `site_id`, `device_model`, `hour_of_day`, etc. La correspondencia
# código → valor original queda en `featurizer.names`.
#
# **Cómo perturba LIME aquí.** Con variables categóricas, LIME reemplaza el valor de la instancia
# por valores muestreados de la distribución de entrenamiento, y su representación interpretable es
# binaria: "la variable conserva su valor" o no. El peso de LIME se lee como el aporte de **tener ese
# valor concreto** frente a un valor típico.
#
# **Selección sistemática de casos:** los falsos positivos y falsos negativos con mayor confianza
# errónea en prueba, con el umbral elegido en validación.
#
# **Evaluación de LIME:**
# - Fidelidad: $R^2$ del modelo sustituto local (`exp.score`).
# - Estabilidad: repetir con distintas semillas y medir el Jaccard medio entre los top-k.
# - Sensibilidad: ancho del kernel y número de muestras perturbadas.
# - Coherencia: top-k de LIME vs SHAP (Jaccard, Spearman) e importancia global por permutación.

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ctr.nbutils import setup, savefig, show, SERIES, MUTED, INK2
from ctr import data, metrics as M, explain as X
from ctr.features import CTRFeaturizer
from ctr.models_sk import fit_mlp_temporal

cfg, RES, FIG = setup()
E = cfg["explain"]
DIM = cfg["hash_dim_main"]
SEED = cfg["seed_base"]
PARAMS = {k: v for k, v in json.loads((RES / "mlp_best_params.json").read_text()).items()
          if k in ("hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size")}
tr = data.read_sample(cfg, "train", cfg["ablation_size"])
va, te = data.eval_sets(cfg)
ytr, yva, yte = tr["click"].to_numpy(), va["click"].to_numpy(), te["click"].to_numpy()
feat = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"]).fit(tr)
c_tr, c_va, c_te = feat.encode(tr), feat.encode(va), feat.encode(te)
clf = fit_mlp_temporal(feat.transform_codes(c_tr, "hash", DIM), ytr, feat.transform_codes(c_va, "hash", DIM), yva,
                       PARAMS, SEED, cfg["max_epochs"], cfg["patience"])
model = X.RawModel(feat, clf, "hash", DIM)
p_va, p_te = model.p1(c_va), model.p1(c_te)
thr = M.best_threshold(yva, p_va)
NAMES = feat.feature_names_interpretable()
NF = len(NAMES)
print(f"AUC prueba = {M.all_metrics(yte, p_te, thr)['roc_auc']:.4f} | umbral (validación) = {thr:.4f}")

# %% [markdown]
# ## 8.1 Casos: falsos positivos y negativos de mayor confianza errónea

# %%
cases = X.select_cases(yte, p_te, thr, E["n_cases"])
def label(j, code):
    v = feat.names[NAMES[j]][int(code)]
    return f"{NAMES[j]}={v[:14]}"
show(cases)
rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(c_tr), size=min(20000, len(c_tr)), replace=False)

# %% [markdown]
# ## 8.2 LIME: explicación de referencia, fidelidad y estabilidad entre semillas

# %%
NS_REF = max(E["lime_num_samples"])
lime_ref = X.make_lime(c_tr[bg_idx], feat, seed=SEED)
default_kw = np.sqrt(NF) * 0.75
rows, ref_w = [], {}
for _, cs in cases.iterrows():
    row = c_te[int(cs.fila)]
    exps = [X.lime_explain(lime_ref, row, model.predict_proba, NS_REF, NF, seed=s) for s in range(E["lime_seeds"])]
    ref_w[int(cs.fila)] = exps[0]["weights"]
    tops = [X.topk(e["weights"], E["top_k"]) for e in exps]
    rows.append({"fila": int(cs.fila), "tipo": cs.tipo, "p_modelo": cs.p, "pred_local_LIME": exps[0]["local_pred"],
                 "R2_medio": np.mean([e["r2"] for e in exps]), "R2_min": np.min([e["r2"] for e in exps]),
                 "jaccard_topk_entre_semillas": X.mean_pairwise_jaccard(tops),
                 "top_k": ", ".join(label(j, row[j]) for j in sorted(exps[0]["weights"], key=lambda j: -abs(exps[0]["weights"][j]))[:E["top_k"]])})
lime_tab = pd.DataFrame(rows)
lime_tab.to_csv(RES / "lime_casos.csv", index=False)
with pd.option_context("display.max_colwidth", 120):
    show(lime_tab, 40)
print(f"Fidelidad media R² = {lime_tab.R2_medio.mean():.3f} | estabilidad media (Jaccard top-{E['top_k']}) = {lime_tab.jaccard_topk_entre_semillas.mean():.3f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, tipo in zip(axs, ["FP", "FN"]):
    sub = cases[cases.tipo == tipo]
    if sub.empty:
        ax.set_visible(False); continue
    f = int(sub.iloc[0].fila); w = ref_w[f]; row = c_te[f]
    top = sorted(w, key=lambda j: -abs(w[j]))[:10][::-1]
    ax.barh([label(j, row[j]) for j in top], [w[j] for j in top],
            color=[SERIES[1] if w[j] > 0 else SERIES[0] for j in top], height=0.6)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title(f"{tipo} · p = {sub.iloc[0].p:.3f} (real = {int(sub.iloc[0].y)})", fontsize=10)
    ax.set_xlabel("Peso LIME (naranja empuja hacia clic, azul hacia no clic)")
savefig(fig, FIG, "08_lime_casos")

# %% [markdown]
# ## 8.3 Sensibilidad al ancho del kernel y al número de muestras

# %%
sens = []
for fk in E["kernel_width_factors"]:
    expl = X.make_lime(c_tr[bg_idx], feat, kernel_width=default_kw * fk, seed=SEED)
    for ns in E["lime_num_samples"]:
        for _, cs in cases.iterrows():
            row = c_te[int(cs.fila)]
            e = X.lime_explain(expl, row, model.predict_proba, ns, NF, seed=0)
            sens.append({"factor_kernel": fk, "num_samples": ns, "fila": int(cs.fila), "R2": e["r2"],
                         "jaccard_vs_referencia": X.jaccard(X.topk(e["weights"], E["top_k"]), X.topk(ref_w[int(cs.fila)], E["top_k"]))})
sens = pd.DataFrame(sens)
sens_tab = sens.groupby(["factor_kernel", "num_samples"])[["R2", "jaccard_vs_referencia"]].agg(["mean", "std"]).round(3)
sens_tab.to_csv(RES / "lime_sensibilidad.csv")
show(sens_tab)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for i, ns in enumerate(sorted(sens.num_samples.unique())):
    g = sens[sens.num_samples == ns].groupby("factor_kernel")[["R2", "jaccard_vs_referencia"]].mean()
    axs[0].plot(g.index, g["R2"], marker="o", color=SERIES[i], label=f"{ns} muestras")
    axs[1].plot(g.index, g["jaccard_vs_referencia"], marker="o", color=SERIES[i], label=f"{ns} muestras")
axs[0].set(title="Fidelidad (R² local)", xlabel="Factor sobre el ancho por defecto")
axs[1].set(title=f"Acuerdo con la referencia (Jaccard top-{E['top_k']})", xlabel="Factor sobre el ancho por defecto", ylim=(0, 1.05))
axs[0].legend(); axs[1].legend()
savefig(fig, FIG, "08_lime_sensibilidad")

# %% [markdown]
# ## 8.4 Contraste con SHAP (permutación, agnóstico al modelo) e importancia por permutación

# %%
sc = cases.head(E["shap_cases"]) if len(cases) > E["shap_cases"] else cases
bg = c_tr[rng.choice(len(c_tr), size=E["background"], replace=False)]
phi = X.shap_values(model.p1, bg, c_te[sc.fila.astype(int).values], seed=SEED)
agree = []
for i, (_, cs) in enumerate(sc.iterrows()):
    agree.append({"fila": int(cs.fila), "tipo": cs.tipo, **X.rank_agreement(ref_w[int(cs.fila)], phi[i], NF, E["top_k"])})
agree = pd.DataFrame(agree)
agree.to_csv(RES / "lime_vs_shap.csv", index=False)
show(agree)
print(f"Acuerdo LIME–SHAP: Jaccard top-{E['top_k']} medio = {agree.jaccard_topk.mean():.3f}; Spearman medio = {agree.spearman_abs.mean():.3f}")

idx = rng.choice(len(c_te), size=min(E["perm_rows"], len(c_te)), replace=False)
perm = X.permutation_importance_codes(model.p1, c_te[idx], yte[idx], NAMES, n_repeats=3, seed=SEED)
perm.to_csv(RES / "importancia_permutacion.csv", index=False)

glob = pd.DataFrame({"variable": NAMES,
                     "LIME_|w|_medio": [np.mean([abs(ref_w[f].get(j, 0)) for f in ref_w]) for j in range(NF)],
                     "SHAP_|phi|_medio": np.abs(phi).mean(axis=0)}).merge(perm[["variable", "caida_auc_media"]], on="variable")
for c in ["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"]:
    glob[f"rango_{c.split('_')[0]}"] = glob[c].rank(ascending=False)
glob = glob.sort_values("caida_auc_media", ascending=False)
glob.to_csv(RES / "coherencia_global.csv", index=False)
show(glob, 30)
print("Spearman entre rankings globales:")
print(glob[["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"]].corr(method="spearman").round(3))

# %%
top = glob.head(12)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, c, t in zip(axs, ["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"],
                    ["LIME: |peso| medio (casos)", "SHAP: |φ| medio (casos)", "Permutación: caída de AUC (global)"]):
    ax.barh(top["variable"][::-1], top[c][::-1], color=SERIES[0], height=0.6); ax.set_title(t, fontsize=10)
savefig(fig, FIG, "08_coherencia_metodos")

# %% [markdown]
# ## 8.5 Qué aporta LIME y qué no puede garantizar
#
# - **Aporta:** una explicación legible y local de cada error, útil para depurar (por ejemplo,
#   descubrir que un falso positivo se debe a un `site_id` con CTR histórico alto).
# - **No garantiza:** (1) fidelidad global: el sustituto es lineal y local, y un $R^2$ bajo indica
#   que la explicación describe mal al modelo; (2) estabilidad: depende del muestreo aleatorio
#   (Jaccard < 1 entre semillas); (3) independencia del ancho del kernel, que no tiene un valor
#   "correcto"; (4) causalidad: describe al **modelo**, no al usuario que hace clic; (5) con
#   variables correlacionadas, perturbar una a la vez genera combinaciones irreales.
# - **SHAP** reparte la predicción con propiedades axiomáticas (eficiencia, simetría), pero con un
#   fondo de referencia que también hay que elegir, y es más costoso.
#
# Interprete los números de 8.2–8.4 con estos criterios en el capítulo 9.
