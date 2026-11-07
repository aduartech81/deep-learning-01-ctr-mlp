#!/usr/bin/env bash
# Sube a GitHub los notebooks EJECUTADOS con Avazu real y los resultados livianos (tablas, figuras, json).
# El Jupyter Book se vuelve a publicar solo en GitHub Pages.
# Uso:  bash scripts/subir_resultados.sh "mensaje opcional"
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) seguridad: nada del perfil de prueba con datos sintéticos
if grep -l "Perfil: smoke" notebooks/*.ipynb >/dev/null 2>&1; then
  echo "ERROR: hay notebooks ejecutados con el perfil 'smoke' (datos sintéticos):"
  grep -l "Perfil: smoke" notebooks/*.ipynb
  echo "Vuelva a ejecutarlos con el perfil full antes de subir."; exit 1
fi
# 2) los notebooks deben tener salidas
sin=$(for f in notebooks/[0-9][0-9]_*.ipynb; do grep -q '"output_type"' "$f" || echo "$f"; done)
if [ -n "$sin" ]; then echo "Aviso: estos notebooks no tienen salidas (no ejecutados):"; echo "$sin"; fi
# 3) tamaño de lo que se va a subir
echo "Tamaño de results/ (sin preds ni samples):"
du -sh --exclude=preds --exclude=samples results 2>/dev/null || du -sh results
read -r -p "¿Subir notebooks ejecutados y resultados? (s/n) " r; [ "$r" = "s" ] || { echo "Cancelado."; exit 1; }

git pull --ff-only
git add notebooks results
git commit -m "${1:-Resultados con Avazu: notebooks ejecutados, tablas y figuras}"
git push
echo "Listo. El libro se actualiza en unos minutos en https://aduartech81.github.io/deep-learning-01-ctr-mlp/"
