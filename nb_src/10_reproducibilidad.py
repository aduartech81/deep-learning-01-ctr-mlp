# %% [markdown]
# # Anexo A. Reproducibilidad
#
# Este anexo registra todo lo necesario para repetir los resultados: versiones de las librerías,
# hardware, configuración de Spark, semillas, configuración del experimento y manifiesto de datos.
#
# **Para reproducir de punta a punta** (ver `README.md`):
#
# ```bash
# conda env create -f environment.yml && conda activate dl-ctr
# # descargar train.gz de Kaggle en data/raw/
# bash scripts/run_all.sh          # ejecuta los notebooks en orden y construye el Jupyter Book
# ```

# %%
import sys, json, platform, subprocess
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src") if Path.cwd().name == "notebooks" else str(Path.cwd() / "src"))
from importlib.metadata import version, PackageNotFoundError
import pandas as pd, yaml
from ctr.nbutils import setup, show
from ctr.bench import hardware_info

cfg, RES, FIG = setup()

# %%
pkgs = ["numpy", "pandas", "polars", "duckdb", "pyarrow", "scikit-learn", "scipy", "pyspark", "optuna", "lime",
        "shap", "matplotlib", "statsmodels", "threadpoolctl", "psutil", "jupyter-book"]
def v(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return "no instalado"
try:
    java = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.strip().splitlines()[0]
except Exception as e:
    java = f"no disponible ({e})"
env = pd.DataFrame({"paquete": ["python", "java"] + pkgs, "version": [platform.python_version(), java] + [v(p) for p in pkgs]})
show(env, 30)
env.to_csv(RES / "versiones.csv", index=False)

# %%
hw = hardware_info()
hw

# %%
print("Configuración del experimento (perfil:", cfg.profile, ")")
print(yaml.safe_dump(cfg.raw, allow_unicode=True, sort_keys=False))
print("Semillas de modelos:", cfg.seeds)
print("Muestreo: hash del id con semilla fija 1234 (muestras anidadas e idénticas en ambos entornos)")

# %%
from ctr import spark_ctr as S
spark = S.get_spark(cfg)
conf = pd.DataFrame(sorted(spark.sparkContext.getConf().getAll()), columns=["clave", "valor"])
conf = conf[conf.clave.str.match(r"spark\.(driver\.memory|master|sql\.shuffle|sql\.execution\.arrow|executor|app\.name|driver\.maxResultSize)")]
show(conf, 30)
spark.stop()

# %%
man = RES / "manifiesto_datos.json"
json.loads(man.read_text()) if man.exists() else "Ejecute el capítulo 1 para generar el manifiesto."
