#!/usr/bin/env bash
set -euo pipefail

# v04 accuracy-preserving truncated run.
# This mirrors v03/run_macro_intersection_h60_mech100_XLA.sh. The simulation
# change is only the printed layer limit; output/profiling paths are separate so
# the first-5-layer run does not overwrite the v03 full-layer run.

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
cd "${WORK_ROOT}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/thermal_macro1mm_intersection_first5_h60_mech100_xla_v04_${RUN_ID}}"
PROFILE_JSON="${PROFILE_JSON:-${OUT_ROOT}/profile.json}"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "v04 mech100 XLA first-5-layer run with v03 h60 parameters"
echo "WORK_ROOT    = ${WORK_ROOT}"
echo "REPO_ROOT    = ${REPO_ROOT}"
echo "OUT_ROOT     = ${OUT_ROOT}"
echo "PROFILE_JSON = ${PROFILE_JSON}"
echo "Extra args   = $*"
echo "============================================================"

PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}" \
python "${REPO_ROOT}/159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py" \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver jax \
  --xla-fallback-to-spsolve \
  --xla-show-devices \
  --profile-json "${PROFILE_JSON}" \
  --profile-label "v04-v03-h60-mech100-first5" \
  --config materials/Ti-6Al-4V/ti64_material_config_initial.json \
  --inp "${WORK_ROOT}/schema/0119_c3d4_only.inp" \
  --max-cells 0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --max-print-layers 5 \
  --path-file "${WORK_ROOT}/output/geometry_path_macro1mm_first10_h60/path_macro1mm_first10_h60.csv" \
  --path-output "" \
  --layer-activation-mode layer_on_scan \
  --layer-activation-geometry intersection \
  --future-layer-mode void \
  --active-window-below-layers 5 \
  --old-layer-thermal-factor 1.0e-6 \
  --old-layer-cooling-h 1.0e4 \
  --laser-power 3000 \
  --absorptivity 0.5 \
  --emissivity 0.53 \
  --beam-radius 1.0e-3 \
  --source-depth 5.0e-4 \
  --dt 1.0e-4 \
  --powder-mode powder \
  --cooling-steps 20 \
  --mechanics-model linear_elastic \
  --mechanics-every 100 \
  --thermal-output-every 1000 \
  --mechanics-output-every 1000 \
  --summary-every 100 \
  --output-dir "${OUT_ROOT}" \
  "$@"

echo "============================================================"
echo "Done. Key outputs:"
echo "  output dir: ${OUT_ROOT}"
echo "  profile:    ${PROFILE_JSON}"
echo "============================================================"
