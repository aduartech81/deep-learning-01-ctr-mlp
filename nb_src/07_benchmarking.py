# %% [markdown]
# # 7. Benchmarking computacional
#
# **Qué se mide.** Tiempo de pared, memoria pico (proceso Python + hijos, que incluye la JVM de
# Spark) y filas por segundo, por separado para lectura, preprocesamiento, entrenamiento y
# predicción.
#
# **Carga de trabajo comparable.** Para el punto de cruce y el escalamiento se fija la carga:
# codificación por frecuencia ($d$ = 24), la mejor arquitectura de cada entorno y un número **fijo**
# de épocas/iteraciones (el de la mejor configuración encontrada), sin parada temprana. Así el
# tiempo depende del volumen y de los núcleos, no de cuándo se detuvo cada corrida.
#
# **Experimentos.**
# 1. Desglose por etapa de las corridas de los capítulos 4 y 5.
# 2. Punto de cruce: volumen creciente, ambos entornos, todos los núcleos.
# 3. Escalamiento fuerte: tamaño fijo con 1, 2, 4 y 8 núcleos → $S(p) = T_1/T_p$, $E(p) = S(p)/p$ y
#    ajuste de Amdahl $S(p) = 1/(s + (1-s)/p)$.
# 4. Escalamiento débil: tamaño proporcional a los núcleos ($E_w(p) = T_1/T_p$; ideal = 1).
# 5. Evidencia de la Spark UI y costo estimado en la nube.

# %%
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
import numpy as np, pandas as pd, polars as pl, psutil
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from ctr.nbutils import setup, savefig, show, run_key, done_keys, SERIES, FRAMEWORK_COLOR, MUTED
from ctr import data, spark_ctr as S
from ctr.features import CTRFeaturizer
from ctr.models_sk import parse_hidden
from ctr.bench import measure, append_jsonl, read_jsonl, fit_amdahl, amdahl, hardware_info

cfg, RES, FIG = setup()
B = cfg["bench"]
BENCH = RES / "bench.jsonl"
HW = hardware_info(); print(HW)
CORES = [p for p in B["cores"] if p <= psutil.cpu_count()]
sk_best = json.loads((RES / "mlp_best_params.json").read_text())
sp_best = json.loads((RES / "spark_mlp_best_params.json").read_text())["freq"]
sk_trials = pd.read_csv(RES / "optuna_trials.csv").sort_values("value")
SK_EPOCHS = int(sk_trials.iloc[0].get("user_attrs_best_epoch", 10) or 10)
print("scikit-learn:", sk_best, "| épocas fijas:", SK_EPOCHS)
print("Spark:", sp_best)
va_pl, te_pl = data.eval_sets(cfg)
yte = te_pl["click"].to_numpy()
N_TRAIN = data.split_sizes(cfg)["train"]

# %% [markdown]
# ## 7.1 Desglose por etapa (corridas de los capítulos 4 y 5)

# %%
runs = pd.concat([read_jsonl(RES / "runs_sklearn.jsonl"), read_jsonl(RES / "runs_spark.jsonl")], ignore_index=True)
if "omitida" in runs:
    runs = runs[runs["omitida"].isna()]
st = (runs.groupby(["framework", "model", "encoding", "n"])
          [["lectura_s", "prep_s", "train_s", "predict_s", "prep_peak_mb", "train_peak_mb", "train_rows_per_s", "predict_rows_per_s"]]
          .median().reset_index())
st.to_csv(RES / "bench_etapas.csv", index=False)
show(st, 60, "{:.2f}")

# %% [markdown]
# ## 7.2 Funciones de carga fija

# %%
def sk_job(n, cores, rec):
    """Lectura + preprocesamiento + entrenamiento (épocas fijas) + predicción en scikit-learn."""
    need_gb = (n or N_TRAIN) * 24 * 4 * 3 / 2**30
    if need_gb > 0.7 * psutil.virtual_memory().available / 2**30:
        raise MemoryError(f"se estiman {need_gb:.1f} GB; supera el 70 % de la RAM disponible")
    feat = CTRFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"])
    if n is None:   # entrenamiento completo: conteo en streaming y codificación por días
        lf = data.scan(cfg.path("parquet_dir")).filter(pl.col("day").is_in(cfg["split"]["train_days"]))
        with measure(rec, "lectura"):
            pass
        with measure(rec, "prep"):
            feat.fit_lazy(lf); codes, y = feat.encode_lazy(lf)
            X = feat.transform_codes(codes, "freq", y=y, fit=True); del codes
    else:
        with measure(rec, "lectura", n):
            tr = data.read_sample(cfg, "train", n)
        with measure(rec, "prep", n):
            y = tr["click"].to_numpy(); feat.fit(tr)
            X = feat.transform_codes(feat.encode(tr), "freq", y=y, fit=True); del tr
    Xte = feat.transform_codes(feat.encode(te_pl), "freq")
    with threadpool_limits(limits=cores):
        clf = MLPClassifier(hidden_layer_sizes=parse_hidden(sk_best["hidden_layer_sizes"]), alpha=sk_best["alpha"],
                            learning_rate_init=sk_best["learning_rate_init"], batch_size=int(sk_best["batch_size"]),
                            random_state=cfg["seed_base"])
        with measure(rec, "train", X.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                for _ in range(SK_EPOCHS):
                    clf.partial_fit(X, y, classes=np.array([0, 1]))
        with measure(rec, "predict", Xte.shape[0]):
            p = clf.predict_proba(Xte)[:, 1]
    rec["n"] = X.shape[0]; rec["roc_auc"] = roc_auc_score(yte, p)


def sp_job(n, cores, rec, spark):
    """Lo mismo en Spark con local[cores]."""
    teS = spark.read.parquet(str(data.sample_path(cfg, "test", cfg["test_max_rows"])))
    with measure(rec, "lectura"):
        trS = S.read_train(spark, cfg, n, data.sample_path).cache(); rec["n"] = trS.count()
    with measure(rec, "prep", rec["n"]):
        feat = S.SparkFeaturizer(cfg.high_card, cfg.low_card, cfg["rare_min_count"]).fit(trS)
        trP, _, teP, d = S.prepare(feat, trS, teS, teS, "freq")
        trP = trP.cache(); trP.count(); teP = teP.cache(); teP.count()
    trS.unpersist()
    with measure(rec, "train", rec["n"]):
        model = S.make_model("mlp", d, sp_best, cfg["seed_base"]).fit(trP)
    with measure(rec, "predict", len(yte)):
        pr = S.predict_proba(model, teP)
    rec["roc_auc"] = roc_auc_score(yte, pr["p"]); trP.unpersist(); teP.unpersist()


def run_bench(exp, fw, n, cores, rep, spark=None):
    key = run_key(exp=exp, fw=fw, n=n if n is not None else "all", cores=cores, rep=rep)
    if key in done_b:
        return
    rec = {"key": key, "exp": exp, "framework": fw, "cores": cores, "rep": rep, "n_pedido": n}
    try:
        sk_job(n, cores, rec) if fw == "sklearn" else sp_job(n, cores, rec, spark)
        rec["total_s"] = sum(rec.get(f"{s}_s", 0) for s in ["lectura", "prep", "train", "predict"])
        rec["peak_mb"] = max(rec.get(f"{s}_peak_mb", 0) for s in ["lectura", "prep", "train", "predict"])
    except MemoryError as e:
        rec["fallo"] = f"memoria: {e}"
    except Exception as e:  # p. ej., OutOfMemoryError de la JVM
        rec["fallo"] = f"{type(e).__name__}: {str(e)[:200]}"
    append_jsonl(BENCH, rec)
    print(key, {k: round(v, 2) for k, v in rec.items() if k.endswith("_s") or k == "roc_auc"}, rec.get("fallo", ""))

done_b = done_keys(BENCH)

# %% [markdown]
# ## 7.3 Punto de cruce (todos los núcleos)

# %%
allc = max(CORES)
for rep in range(B["reps"]):
    for n in B["crossover_sizes"]:
        run_bench("cruce", "sklearn", n, allc, rep)
spark = S.get_spark(cfg, allc)
for rep in range(B["reps"]):
    for n in B["crossover_sizes"]:
        run_bench("cruce", "spark", n, allc, rep, spark)
ui = S.ui_stage_metrics(spark); ui.to_csv(RES / "spark_ui_cruce.csv", index=False)
spark.stop()

bench = read_jsonl(BENCH)
ok = bench[bench["fallo"].isna()] if "fallo" in bench else bench
cr = (ok[ok.exp == "cruce"].groupby(["framework", "n"])
        [["lectura_s", "prep_s", "train_s", "predict_s", "total_s", "peak_mb", "roc_auc"]].median().reset_index())
show(cr, 40, "{:.3f}")
if "fallo" in bench:
    show(bench[bench["fallo"].notna()][["exp", "framework", "n_pedido", "cores", "fallo"]])

def crossover(cr, col):
    a = cr[cr.framework == "sklearn"].set_index("n")[col]; b = cr[cr.framework == "spark"].set_index("n")[col]
    ns = sorted(set(a.index) & set(b.index))
    diff = [(n, np.log(b[n]) - np.log(a[n])) for n in ns]
    for (n0, d0), (n1, d1) in zip(diff, diff[1:]):
        if d0 > 0 >= d1:   # Spark pasa de más lento a más rápido: interpolación log-log
            return float(np.exp(np.log(n0) + (np.log(n1) - np.log(n0)) * d0 / (d0 - d1)))
    return None

cruce = {c: crossover(cr, c) for c in ["train_s", "total_s"]}
print("Punto de cruce estimado (filas):", {k: (f"{v:,.0f}" if v else "no se observa en el rango medido") for k, v in cruce.items()})
(RES / "punto_cruce.json").write_text(json.dumps(cruce))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
for fw, g in cr.groupby("framework"):
    axs[0].plot(g["n"], g["train_s"], marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
    axs[1].plot(g["n"], g["total_s"], marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
    axs[2].plot(g["n"], g["peak_mb"] / 1024, marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
for ax, t, yl, key in zip(axs, ["Entrenamiento", "Extremo a extremo (lectura → predicción)", "Memoria pico"],
                          ["segundos", "segundos", "GB"], ["train_s", "total_s", None]):
    ax.set_xscale("log"); ax.set_xlabel("Filas de entrenamiento"); ax.set_title(t); ax.set_ylabel(yl); ax.legend()
    if key:
        ax.set_yscale("log")
        if cruce.get(key):
            ax.axvline(cruce[key], color=MUTED, linestyle="--", linewidth=1)
            ax.text(cruce[key], ax.get_ylim()[1], " cruce", color=MUTED, va="top", fontsize=8)
savefig(fig, FIG, "07_punto_cruce")

# barras comparativas por etapa en el mayor tamaño común
nmax = cr.groupby("n")["framework"].nunique().loc[lambda s: s == 2].index.max()
g = cr[cr.n == nmax].set_index("framework")[["lectura_s", "prep_s", "train_s", "predict_s"]]
fig, ax = plt.subplots(figsize=(8, 3.4))
left = np.zeros(len(g))
for i, c in enumerate(g.columns):
    ax.barh(g.index, g[c], left=left, color=SERIES[i], label=c.replace("_s", ""), height=0.5, edgecolor="#fcfcfb", linewidth=2)
    left += g[c].values
ax.set_xlabel("segundos"); ax.set_title(f"Tiempo por etapa (n = {int(nmax):,})"); ax.legend(ncols=4, loc="lower right")
savefig(fig, FIG, "07_etapas")

# %% [markdown]
# ## 7.4 Escalamiento fuerte (tamaño fijo) y ley de Amdahl

# %%
NS = B["strong_size"]
for rep in range(B["reps"]):
    for p in CORES:
        run_bench("fuerte", "sklearn", NS, p, rep)
for p in CORES:
    spark = S.get_spark(cfg, p)
    for rep in range(B["reps"]):
        run_bench("fuerte", "spark", NS, p, rep, spark)
    spark.stop()

bench = read_jsonl(BENCH); ok = bench[bench["fallo"].isna()] if "fallo" in bench else bench
fu = ok[ok.exp == "fuerte"].groupby(["framework", "cores"])[["train_s", "total_s"]].median().reset_index()
am_rows = []
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
pp = np.linspace(1, max(CORES), 50)
for fw, g in fu.groupby("framework"):
    g = g.sort_values("cores")
    if len(g) < 2:
        continue
    fa = fit_amdahl(g["cores"], g["train_s"])
    am_rows.append({"framework": fw, **{k: v for k, v in fa.items() if not isinstance(v, list)}})
    axs[0].plot(g["cores"], fa["speedup"], "o", color=FRAMEWORK_COLOR[fw], label=f"{fw} medido")
    axs[0].plot(pp, amdahl(pp, fa["fraccion_secuencial"]), color=FRAMEWORK_COLOR[fw], linewidth=1.4, linestyle="--",
                label=f"{fw} Amdahl s={fa['fraccion_secuencial']:.2f}")
    axs[1].plot(g["cores"], fa["eficiencia"], marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
axs[0].plot(pp, pp, color=MUTED, linewidth=1, linestyle=":", label="ideal")
axs[0].set(title=f"Escalamiento fuerte (n = {NS:,}): aceleración", xlabel="Núcleos", ylabel="S(p) = T1/Tp")
axs[1].axhline(1, color=MUTED, linewidth=1, linestyle=":"); axs[1].set(title="Eficiencia E(p) = S(p)/p", xlabel="Núcleos", ylim=(0, 1.2))
axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
savefig(fig, FIG, "07_escalamiento_fuerte")
amd = pd.DataFrame(am_rows); amd.to_csv(RES / "amdahl.csv", index=False); show(amd)

# %% [markdown]
# ## 7.5 Escalamiento débil (tamaño ∝ núcleos)

# %%
for rep in range(B["reps"]):
    for p in CORES:
        run_bench("debil", "sklearn", B["weak_base"] * p, p, rep)
for p in CORES:
    spark = S.get_spark(cfg, p)
    for rep in range(B["reps"]):
        run_bench("debil", "spark", B["weak_base"] * p, p, rep, spark)
    spark.stop()
bench = read_jsonl(BENCH); ok = bench[bench["fallo"].isna()] if "fallo" in bench else bench
de = ok[ok.exp == "debil"].groupby(["framework", "cores"])[["train_s", "n"]].median().reset_index()
fig, ax = plt.subplots(figsize=(6.5, 4))
for fw, g in de.groupby("framework"):
    g = g.sort_values("cores")
    ax.plot(g["cores"], g["train_s"].iloc[0] / g["train_s"], marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
ax.axhline(1, color=MUTED, linewidth=1, linestyle=":")
ax.set(title=f"Escalamiento débil ({B['weak_base']:,} filas por núcleo)", xlabel="Núcleos", ylabel="Eficiencia débil T1/Tp", ylim=(0, 1.3))
ax.legend()
savefig(fig, FIG, "07_escalamiento_debil")
show(de)

# %% [markdown]
# ## 7.6 Spark UI y costo en la nube
#
# El costo se calcula como (tiempo total en horas) × (precio por hora de la instancia), con el
# precio que se defina en `config.yaml` (`cloud_price_usd_per_hour`). Verifique el precio vigente en
# la página del proveedor el día de la entrega y regístrelo con fecha.

# %%
ui = pd.read_csv(RES / "spark_ui_cruce.csv")
if "shuffleWriteBytes" in ui:
    print({"etapas": len(ui), "shuffle_GB": (ui["shuffleReadBytes"].sum() + ui["shuffleWriteBytes"].sum()) / 2**30,
           "spill_disco_GB": ui["diskBytesSpilled"].sum() / 2**30,
           "sesgo_p95": float(ui["sesgo_max_sobre_mediana"].quantile(0.95))})
price = B["cloud_price_usd_per_hour"]
if price:
    cr["costo_usd"] = cr["total_s"] / 3600 * price
    show(cr[["framework", "n", "total_s", "costo_usd"]], 40, "{:.4f}")
else:
    print("Defina cloud_price_usd_per_hour en config.yaml para estimar el costo.")
