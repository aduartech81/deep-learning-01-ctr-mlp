# %% [markdown]
# # 9. Comparación de resultados y reflexión crítica
#
# Este capítulo consolida los resultados y responde las preguntas de investigación **con los
# números producidos por los capítulos anteriores**. Las respuestas de 9.3 se redactan
# automáticamente a partir de esos números, con reglas explícitas (umbrales de significancia y de
# magnitud). Son un borrador basado en evidencia: revíselas y complételas con su interpretación
# antes de entregar.

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from ctr.nbutils import setup, savefig, show, SERIES, FRAMEWORK_COLOR, MODEL_COLOR, MUTED
from ctr.bench import read_jsonl
from ctr import metrics as M

cfg, RES, FIG = setup()
runs = pd.concat([read_jsonl(RES / "runs_sklearn.jsonl"), read_jsonl(RES / "runs_spark.jsonl")], ignore_index=True)
omit = runs[runs["omitida"].notna()] if "omitida" in runs else runs.iloc[:0]
runs = runs[runs["omitida"].isna()] if "omitida" in runs else runs
runs["n"] = runs["n"].astype(int)
tests = pd.read_csv(RES / "pruebas_hipotesis.csv")
cruce = json.loads((RES / "punto_cruce.json").read_text())
amd = pd.read_csv(RES / "amdahl.csv") if (RES / "amdahl.csv").exists() else pd.DataFrame()
abl = pd.read_csv(RES / "tabla_ablaciones_sklearn.csv")
lime = pd.read_csv(RES / "lime_casos.csv"); lvs = pd.read_csv(RES / "lime_vs_shap.csv")
glob = pd.read_csv(RES / "coherencia_global.csv")
bench = read_jsonl(RES / "bench.jsonl")
bench_ok = bench[bench["fallo"].isna()] if "fallo" in bench else bench

# %% [markdown]
# ## 9.1 Tabla comparativa final (por entorno, modelo y codificación)

# %%
MET = ["roc_auc", "pr_auc", "log_loss", "brier", "ece", "f1", "train_s", "predict_s", "train_peak_mb"]
final = M.summarize_runs(runs, ["framework", "model", "encoding", "n"], MET)
cols = ["framework", "model", "encoding", "n", "n_semillas"] + [f"{m}_media" for m in MET] + ["roc_auc_sd", "roc_auc_ic"]
final = final[cols].sort_values(["n", "framework", "encoding", "model"])
final.to_csv(RES / "tabla_final.csv", index=False)
show(final, 80)
if len(omit):
    print("Corridas omitidas por presupuesto de tiempo (también son evidencia de costo):")
    show(omit[["framework", "model", "encoding", "n", "motivo"]])

# %%
# tiempos de entrenamiento y predicción del MLP por entorno y tamaño (codificación por frecuencia)
g = runs[(runs.model == "mlp") & (runs.encoding == "freq")].groupby(["framework", "n"])[["train_s", "predict_s", "roc_auc"]].median().reset_index()
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
for fw, gg in g.groupby("framework"):
    for ax, c in zip(axs, ["train_s", "predict_s", "roc_auc"]):
        ax.plot(gg["n"], gg[c], marker="o", color=FRAMEWORK_COLOR[fw], label=fw)
for ax, t in zip(axs, ["Entrenamiento (s)", "Predicción del día 30 (s)", "ROC AUC en prueba"]):
    ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("Filas de entrenamiento"); ax.legend()
axs[0].set_yscale("log"); axs[1].set_yscale("log")
savefig(fig, FIG, "09_mlp_entornos")

# %% [markdown]
# ## 9.2 Ablaciones (scikit-learn)

# %%
show(abl[["factor", "nivel", "n_semillas", "roc_auc_media", "roc_auc_sd", "log_loss_media", "ece_media", "f1_media", "train_s_media"]], 40)

# %% [markdown]
# ## 9.3 Respuestas a las preguntas de investigación (borrador generado desde los resultados)

# %%
def fmt(x, d=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"

out = []
# ---------- P1
cr = (bench_ok[bench_ok.exp == "cruce"].groupby(["framework", "n"])[["train_s", "total_s", "roc_auc"]].median().reset_index()
      if len(bench_ok) else pd.DataFrame())
if cruce.get("train_s"):
    p1 = (f"Se observó un punto de cruce en entrenamiento alrededor de **{cruce['train_s']:,.0f} filas**"
          + (f" y extremo a extremo alrededor de **{cruce['total_s']:,.0f} filas**." if cruce.get("total_s") else
             "; extremo a extremo, sin embargo, Spark no llegó a superar a scikit-learn en el rango medido (la lectura y el preprocesamiento pesan)."))
    p1 += " **H1 se sostiene** en el rango medido."
else:
    p1 = ("En el rango medido **no se observó punto de cruce**: scikit-learn fue más rápido en todos los volúmenes "
          "en una sola máquina. **H1 se sostiene solo en su segunda parte** (por debajo del cruce scikit-learn es más eficiente); "
          "el cruce, si existe, está por encima del volumen probado o requiere un clúster real.")
if len(cr):
    q = cr.pivot(index="n", columns="framework", values="roc_auc")
    if {"sklearn", "spark"} <= set(q.columns):
        p1 += f" La AUC de la carga fija fue {fmt(q['sklearn'].mean())} (scikit-learn) vs {fmt(q['spark'].mean())} (Spark) en promedio sobre los tamaños."
if len(amd):
    p1 += " Fracción secuencial de Amdahl: " + "; ".join(f"{r.framework} s = {r.fraccion_secuencial:.2f} (aceleración máxima teórica ≈ {r.aceleracion_max_teorica:.1f}×)" for r in amd.itertuples()) + "."
out.append("**P1. ¿A partir de qué volumen PySpark supera a scikit-learn en tiempo sin degradar la calidad?**\n\n" + p1)

# ---------- P2
sk = runs[(runs.framework == "sklearn") & (runs.model == "mlp")]
nn = sk.groupby("n")["encoding"].nunique()
n2 = int(nn[nn == nn.max()].index.max())
e2 = sk[sk.n == n2].groupby("encoding")[["roc_auc", "log_loss", "train_s", "train_peak_mb"]].mean()
prep = pd.read_csv(RES / "prep_codificaciones.csv")
p2 = f"Con n = {n2:,}, MLP: " + "; ".join(f"{e}: AUC {fmt(r.roc_auc)}, log loss {fmt(r.log_loss)}, entrenamiento {fmt(r.train_s,1)} s" for e, r in e2.iterrows()) + ". "
p2 += "Memoria de la matriz de entrada: " + "; ".join(f"{r.codificacion}: {r.MB_matriz:,.0f} MB" for r in prep.itertuples()) + ". "
if {"hash", "freq"} <= set(e2.index):
    d = e2.loc["hash", "roc_auc"] - e2.loc["freq", "roc_auc"]
    mem_h = prep[prep.codificacion.str.startswith("hash")]["MB_matriz"].median(); mem_f = prep[prep.codificacion == "freq"]["MB_matriz"].iloc[0]
    if abs(d) < 0.005:
        p2 += f"Hashing y frecuencia quedan a menos de 0,005 de AUC (Δ = {d:+.4f}), con memorias de {mem_h:,.0f} vs {mem_f:,.0f} MB: **H2 se sostiene**."
    else:
        p2 += (f"La diferencia de AUC hashing − frecuencia es {d:+.4f}: **H2 no se sostiene tal como está formulada**; "
               + ("el hashing es claramente mejor, pero ocupa más memoria que la frecuencia (la hipótesis suponía lo contrario)." if d > 0 and mem_h > mem_f
                  else "conviene reformularla según el sentido de la diferencia y el costo en memoria medido."))
hd = abl[abl.factor == "dimension_hashing"]
if len(hd):
    p2 += " Dimensión del hashing (ablación): " + "; ".join(f"{r.nivel}: AUC {fmt(r.roc_auc_media)}" for r in hd.itertuples()) + "."
out.append("**P2. ¿Cómo afecta la codificación de alta cardinalidad al desempeño y al costo?**\n\n" + p2)

# ---------- P3
p3 = []
for r in tests.itertuples():
    sig = r.p_DeLong_Holm < 0.05
    p3.append(f"{r.A} vs {r.B}: ΔAUC = {r.dif_AUC:+.4f} (p Holm = {r.p_DeLong_Holm:.2g}) → "
              + ("diferencia significativa" if sig else "sin diferencia significativa")
              + (", pero de magnitud pequeña (< 0,005)" if sig and abs(r.dif_AUC) < 0.005 else ""))
out.append("**P3. ¿Aporta el MLP una mejora significativa sobre líneas base más simples?**\n\n- " + "\n- ".join(p3))

# ---------- P4
p4 = (f"Fidelidad media de LIME: R² = {lime.R2_medio.mean():.3f} (mínimo {lime.R2_min.min():.3f}). "
      f"Estabilidad entre semillas: Jaccard medio del top-{cfg['explain']['top_k']} = {lime.jaccard_topk_entre_semillas.mean():.3f}. "
      f"Acuerdo con SHAP: Jaccard {lvs.jaccard_topk.mean():.3f}, Spearman {lvs.spearman_abs.mean():.3f}. ")
rho = glob[["LIME_|w|_medio", "SHAP_|phi|_medio", "caida_auc_media"]].corr(method="spearman")
p4 += f"Coherencia global (Spearman de rankings): LIME–permutación {rho.iloc[0, 2]:.2f}, SHAP–permutación {rho.iloc[1, 2]:.2f}. "
p4 += ("Las explicaciones son **estables y razonablemente fieles**." if lime.R2_medio.mean() > 0.5 and lime.jaccard_topk_entre_semillas.mean() > 0.7
       else "Las explicaciones **no son suficientemente estables o fieles** para tomarlas como evidencia por sí solas: deben contrastarse con SHAP y con la importancia global.")
out.append("**P4. ¿Son las explicaciones de LIME estables y fieles al modelo?**\n\n" + p4)
display(Markdown("\n\n".join(out)))
(RES / "respuestas_borrador.md").write_text("\n\n".join(out), encoding="utf8")

# %% [markdown]
# ## 9.4 Reflexión crítica
#
# Responda cada punto citando los números de las tablas anteriores (los borradores de 9.3 son el
# punto de partida).
#
# 1. **¿Qué entorno fue más rápido, en qué rango de volumen y por qué?** Considere: costo fijo de
#    arrancar la JVM y planificar trabajos; serialización Python ↔ JVM; que un `.gz` no es divisible
#    (una sola tarea de lectura); que en una sola máquina Spark compite por los mismos núcleos y
#    memoria; la densificación del MLP de MLlib frente al producto disperso de scikit-learn.
# 2. **¿Cuál fue más preciso y mejor calibrado, y es significativa la diferencia?** Use la tabla
#    6.5 (DeLong/McNemar con Holm), los IC por bootstrap y las curvas de calibración. Recuerde que con
#    millones de filas casi todo es "significativo": discuta la magnitud.
# 3. **¿Cuándo se justifica PySpark y cuándo es sobrecosto?** Datos que no caben en una máquina,
#    tolerancia a fallos, integración con un lago de datos y pipelines programados, frente a la
#    simplicidad y velocidad de scikit-learn mientras los datos caben en RAM.
# 4. **Limitaciones arquitectónicas del MLP nativo de Spark** frente a una red con embeddings
#    (capítulo 5.6): sin embeddings, sin dropout ni L2, sigmoide fija, entrada densa, solo l-bfgs/gd.
# 5. **¿Qué aporta LIME, qué no garantiza y cómo se compara con SHAP?** (capítulo 8.5 y P4).
#
# ### Amenazas a la validez
#
# - **Deriva temporal:** el PSI del capítulo 2 mide cuánto cambia la distribución entre los días
#   21–28 y el 30; una sola ventana de prueba (un día) no permite generalizar a otras semanas.
# - **Fuga de información:** controlada con división temporal, ajuste de vocabularios, escaladores y
#   codificadores solo con entrenamiento, target encoding out-of-fold y umbral elegido en
#   validación. Riesgo residual: el cross-fitting del `TargetEncoder` es aleatorio **dentro** de
#   los días de entrenamiento.
# - **Anonimización:** C1 y C14–C21 no tienen semántica conocida; las explicaciones sobre ellas
#   no se pueden interpretar en términos de negocio.
# - **Sesgo de muestreo:** Avazu es una muestra de 2014 de una sola plataforma; el submuestreo por
#   hash es uniforme, pero las muestras de 100 mil no representan bien las categorías raras.
# - **Validez del benchmark:** una sola máquina (Spark en modo local), carga de otros procesos,
#   caché del sistema de archivos y número de repeticiones limitado.
# - **Diferencias de implementación:** los hashers de scikit-learn y Spark usan semillas distintas
#   (mismos principios, índices diferentes) y los MLP difieren en activación y optimizador, de modo
#   que "scikit-learn vs Spark" compara implementaciones completas, no solo motores de cómputo.
