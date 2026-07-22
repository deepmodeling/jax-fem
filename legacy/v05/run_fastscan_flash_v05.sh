#!/usr/bin/env bash
set -euo pipefail

# v05 fast-scan: identical parameters to v04 run_macro_fastscan_flash.sh but
# entry point is the v05 wrapper (incremental J2 with stored plastic strain).
# KEEP THE FLAG BLOCK IN SYNC with 159_local/v04/run_macro_fastscan_flash.sh.

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
cd "${WORK_ROOT}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/fastscan_flash_v05_full91_${RUN_ID}}"
PROFILE_JSON="${PROFILE_JSON:-${OUT_ROOT}/profile.json}"
mkdir -p "${OUT_ROOT}"

echo "=== v05 fast-scan (incremental plastic history), OUT_ROOT=${OUT_ROOT} ==="

PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}" \
python "${REPO_ROOT}/159_local/v05/am_thermal_stress_macro_intersection_mech100_v05.py" \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver spsolve \
  --profile-json "${PROFILE_JSON}" \
  --profile-label "fastscan-flash-v05" \
  --config materials/Ti-6Al-4V/ti64_material_config_physfix.json \
  --inp "${WORK_ROOT}/schema/0119_c3d4_only.inp" \
  --max-cells 0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --layers 91 \
  --scan-steps-per-layer 1 \
  --hatch-lines-per-layer 1 \
  --laser-power 0 \
  --dt 1.0e-2 \
  --layer-activation-mode layer_on_scan \
  --layer-activation-geometry intersection \
  --future-layer-mode void \
  --active-window-below-layers 5 \
  --old-layer-thermal-factor 1.0e-6 \
  --old-layer-cooling-h 1.0e4 \
  --inactive-mass-factor 1.0 \
  --powder-mode powder \
  --surface-selection exterior \
  --boundary-tol 1.0e-4 \
  --quadrature-order 2 \
  --solidus-temperature 0 --liquidus-temperature 0 --latent-heat 0 \
  --stress-relaxation-temperature 1100 \
  --reset-activation-temperature \
  --activation-reset-temperature 1100 \
  --recoat-time 10.0 \
  --recoat-steps 10 \
  --cooling-steps 100 --cooling-dt 6.0 \
  --mechanics-model j2_plastic \
  --yield-saturation-stress 1.15e9 \
  --bottom-mechanics-bc elastic \
  --bottom-foundation-stiffness 1.0e12 \
  --mechanics-every 22 \
  --mechanics-rel-tol 1e-5 \
  --mechanics-max-iter 50 \
  --mechanics-line-search \
  --release-after-cooling \
  --thermal-output-every 44 \
  --mechanics-output-every 22 \
  --summary-every 11 \
  --output-dir "${OUT_ROOT}" \
  "$@"

echo "=== done: ${OUT_ROOT} (release: ${OUT_ROOT}/release.vtu) ==="
