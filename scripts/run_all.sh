#!/usr/bin/env bash
# Ejecuta todos los notebooks en orden (guardando salidas) y construye el Jupyter Book.
# Uso:
#   bash scripts/run_all.sh            # perfil full (datos reales de Avazu)
#   CTR_PROFILE=smoke bash scripts/run_all.sh   # prueba de humo con datos sintéticos
#   bash scripts/run_all.sh 04 05      # solo los notebooks cuyo nombre empieza por 04 y 05
set -euo pipefail
cd "$(dirname "$0")/.."
export CTR_PROFILE="${CTR_PROFILE:-full}"
export PYARROW_IGNORE_TIMEZONE=1
mkdir -p logs

if [[ "$CTR_PROFILE" == "smoke" && ! -f data/smoke/train_synthetic.csv.gz ]]; then
  python scripts/make_synthetic.py --rows-per-day 30000
fi

sel=("$@")
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  name=$(basename "$nb")
  if [[ ${#sel[@]} -gt 0 ]]; then
    ok=0; for s in "${sel[@]}"; do [[ "$name" == "$s"* ]] && ok=1; done
    [[ $ok -eq 0 ]] && continue
  fi
  echo "=== $(date '+%H:%M:%S') ejecutando $name (perfil $CTR_PROFILE)"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    "$nb" 2>&1 | tee "logs/${name%.ipynb}.log"
done

echo "=== construyendo el Jupyter Book"
jupyter-book build . 2>&1 | tail -5
echo "Listo: abra _build/html/index.html"
