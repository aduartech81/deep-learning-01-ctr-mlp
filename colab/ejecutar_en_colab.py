# %% [markdown]
# # Ejecutar el proyecto en Google Colab
#
# Este cuaderno corre los 10 capítulos con los datos reales de Avazu y **guarda todo en Google
# Drive después de cada capítulo**. Si Colab se desconecta, vuelva a ejecutar todas las celdas:
# recupera lo guardado y continúa donde iba.
#
# **Antes de empezar (una sola vez)**
# 1. *Entorno de ejecución → Cambiar tipo de entorno de ejecución*: **CPU** y activar **RAM alta**
#    (High-RAM). No use GPU: los modelos de este proyecto solo usan CPU y la GPU gasta unidades.
# 2. Panel de la llave (**Secretos**, a la izquierda): cree `KAGGLE_API_TOKEN` con el token de
#    kaggle.com/settings → API → *Generate New Token* (o, con credenciales antiguas, `KAGGLE_USERNAME`
#    y `KAGGLE_KEY` del archivo `kaggle.json`) y active el acceso de este cuaderno. Opcional:
#    `GITHUB_TOKEN` (token *fine-grained* con permiso *Contents: read and write* solo para este
#    repositorio) para subir los resultados al final.
# 3. Acepte las reglas de la competencia en kaggle.com/c/avazu-ctr-prediction → Rules.
#
# Mantenga la pestaña abierta mientras corre.

# %%
# 1. Hardware y elección del perfil
import os, subprocess, psutil
ram_gb = psutil.virtual_memory().total / 2**30
cores = os.cpu_count()
print(f"RAM: {ram_gb:.1f} GB | núcleos: {cores}")
if ram_gb >= 24:
    PROFILE = "colab"
elif ram_gb >= 11:
    PROFILE = "mac8"
else:
    raise SystemExit("Menos de 11 GB de RAM: active 'RAM alta' en el tipo de entorno de ejecución.")
DRIVER_MEM = f"{max(4, int(ram_gb * 0.55))}g"
print(f"Perfil elegido: {PROFILE} | memoria del driver de Spark: {DRIVER_MEM}")
if cores < 8:
    print(f"Nota: el escalamiento del capítulo 7 llegará hasta {cores} núcleos (hay que declararlo en el informe).")

# %%
# 2. Google Drive como respaldo persistente
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/deep-learning-01-ctr-mlp"
os.makedirs(f"{DRIVE}/done", exist_ok=True)
REPO = "/content/deep-learning-01-ctr-mlp"

# %%
# 3. Código del repositorio y restauración de lo ya guardado en Drive
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/aduartech81/deep-learning-01-ctr-mlp.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
os.chdir(REPO)
for sub in ["results", "data/parquet", "notebooks"]:
    src = f"{DRIVE}/{sub}"
    if os.path.exists(src):
        os.makedirs(sub, exist_ok=True)
        subprocess.run(f'cp -r "{src}/." "{sub}/"', shell=True, check=True)
        print("restaurado desde Drive:", sub)
print(sorted(os.listdir(f"{DRIVE}/done")) or "ningún capítulo terminado todavía")

# %%
# 4. Dependencias (Java 17 + versiones fijadas del proyecto)
subprocess.run("apt-get -qq update && apt-get -qq install -y openjdk-17-jdk-headless > /dev/null", shell=True, check=True)
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
subprocess.run("pip -q install -r requirements.txt", shell=True, check=True)
print(subprocess.run("java -version", shell=True, capture_output=True, text=True).stderr.splitlines()[0])

# %%
# 5. Datos de Avazu desde Kaggle (solo si aún no están convertidos a Parquet)
from google.colab import userdata
if not os.path.exists("data/parquet") or not any(n.startswith("day=") for n in os.listdir("data/parquet")):
    def _secret(name):
        try:
            return userdata.get(name)
        except Exception:
            return None
    if _secret("KAGGLE_API_TOKEN"):          # token nuevo de Kaggle ("Generate New Token")
        os.environ["KAGGLE_API_TOKEN"] = _secret("KAGGLE_API_TOKEN")
    elif _secret("KAGGLE_USERNAME") and _secret("KAGGLE_KEY"):   # credenciales antiguas (kaggle.json)
        os.environ["KAGGLE_USERNAME"] = _secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = _secret("KAGGLE_KEY")
    else:
        raise SystemExit("Falta el secreto KAGGLE_API_TOKEN (o KAGGLE_USERNAME y KAGGLE_KEY) con acceso activado.")
    os.makedirs("data/raw", exist_ok=True)
    subprocess.run("kaggle competitions download -c avazu-ctr-prediction -f train.gz -p data/raw", shell=True, check=True)
    if os.path.exists("data/raw/train.gz.zip"):
        subprocess.run("cd data/raw && unzip -o train.gz.zip && rm train.gz.zip", shell=True, check=True)
    print(subprocess.run("ls -lh data/raw", shell=True, capture_output=True, text=True).stdout)
else:
    print("Parquet ya disponible (restaurado desde Drive): no hace falta descargar.")

# %%
# 6. Ejecutar capítulo por capítulo, respaldando en Drive después de cada uno
import time
env = dict(os.environ, CTR_PROFILE=PROFILE, CTR_SPARK_DRIVER_MEMORY=DRIVER_MEM, SKIP_BOOK="1",
           PYARROW_IGNORE_TIMEZONE="1")
CHAPTERS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

def backup(include_parquet=False):
    subs = ["results", "notebooks"] + (["data/parquet"] if include_parquet else [])
    for sub in subs:
        os.makedirs(f"{DRIVE}/{sub}", exist_ok=True)
        subprocess.run(f'cp -r "{sub}/." "{DRIVE}/{sub}/"', shell=True, check=True)

for ch in CHAPTERS:
    if os.path.exists(f"{DRIVE}/done/{ch}"):
        print(f"{ch}: ya terminado"); continue
    t0 = time.time()
    print(f"=== {ch} ({time.strftime('%H:%M')}) perfil {PROFILE}", flush=True)
    r = subprocess.run(["bash", "scripts/run_all.sh", ch], env=env)
    backup(include_parquet=(ch == "01"))
    if r.returncode != 0:
        raise SystemExit(f"El capítulo {ch} falló: revise logs/{ch}_*.log (también quedó respaldo en Drive).")
    open(f"{DRIVE}/done/{ch}", "w").write(f"{time.time() - t0:.0f} s\n")
    print(f"{ch}: listo en {(time.time() - t0) / 60:.0f} min", flush=True)
print("Todos los capítulos terminados.")

# %% [markdown]
# ## 7. Subir los resultados a GitHub
#
# **Opción A (desde aquí):** requiere el secreto `GITHUB_TOKEN`. Ejecute la celda siguiente.
#
# **Opción B (desde su Mac):** copie `notebooks/` y `results/` de la carpeta de Drive
# `deep-learning-01-ctr-mlp` a su copia local del repositorio y ejecute
# `bash scripts/subir_resultados.sh`.

# %%
token = userdata.get("GITHUB_TOKEN")
check = subprocess.run("grep -l 'Perfil: smoke' notebooks/*.ipynb", shell=True, capture_output=True, text=True)
assert not check.stdout.strip(), f"Hay notebooks con salidas sintéticas: {check.stdout}"
cmds = f"""
git config user.name "Andrés Duarte Chaves"
git config user.email "200296813+aduartech81@users.noreply.github.com"
git add notebooks results
git commit -m "Resultados con Avazu (Colab, perfil {PROFILE}): notebooks ejecutados, tablas y figuras"
git push https://x-access-token:{token}@github.com/aduartech81/deep-learning-01-ctr-mlp.git HEAD:main
"""
r = subprocess.run(cmds, shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:].replace(token, "***"))
