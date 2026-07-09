#!/usr/bin/env bash
set -euo pipefail

# v04 first-5-layer run with PHYSICS CORRECTIONS enabled (2026-07-08 review):
#   1. --surface-selection exterior : convection/radiation on every mesh-exterior
#      face above the base plane. The legacy box-plane selectors matched 0 faces
#      on the real 0119 part, leaving the model nearly adiabatic.
#   2. --boundary-tol 1e-4          : base-plane node selection catches the real
#      ~1225-node bottom face instead of the 8 exactly-coplanar nodes, fixing
#      both the fixed-temperature bottom BC and the mechanical clamp.
#   3. --recoat-time 10             : 10 s recoater dwell inserted at every layer
#      transition of the path file (10 implicit steps of 1 s each). The h60 path
#      CSV contains scan vectors only, with zero interlayer cooling.
#   4. --cooling-steps 300 --cooling-dt 2.0 : 10 min final cooldown so the part
#      approaches ambient before stress release.
#   5. --release-after-cooling      : release solve with 3-2-1 anchor BC gives
#      the actual post-build residual stress instead of the still-clamped state.
#   6. --reset-activation-temperature : freshly activated layer nodes start at
#      the preheat/ambient temperature like real spread powder.
#   7. --mechanics-model j2_plastic : yield-capped residual stress (config table).
#      NOTE: the yield table currently ends at 1273 K (flat 405 MPa above); see
#      materials/Ti-6Al-4V/TODO.md for the recommended high-T extension.
#   8. --quadrature-order 2 : 4-point TET4 rule. The legacy single-point rule
#      has a rank-1 mass matrix that produced +-1900 K spurious temperature
#      oscillations in powder and suppressed melt-pool peaks below solidus.
#
# Results are intentionally NOT comparable to the legacy first5 run.

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
cd "${WORK_ROOT}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/thermal_macro1mm_intersection_first5_h60_mech100_xla_v04_physfix_${RUN_ID}}"
PROFILE_JSON="${PROFILE_JSON:-${OUT_ROOT}/profile.json}"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "v04 mech100 XLA first-5-layer run, physics-corrected (physfix)"
echo "WORK_ROOT    = ${WORK_ROOT}"
echo "OUT_ROOT     = ${OUT_ROOT}"
echo "PROFILE_JSON = ${PROFILE_JSON}"
echo "Extra args   = $*"
echo "============================================================"

PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}" \
python "${REPO_ROOT}/159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py" \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver spsolve \
  --xla-show-devices \
  --profile-json "${PROFILE_JSON}" \
  --profile-label "v04-h60-mech100-first5-physfix" \
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
  --inactive-mass-factor 1.0 \
  --laser-power 3000 \
  --absorptivity 0.5 \
  --emissivity 0.53 \
  --beam-radius 1.0e-3 \
  --source-depth 5.0e-4 \
  --dt 1.0e-4 \
  --powder-mode powder \
  --surface-selection exterior \
  --boundary-tol 1.0e-4 \
  --quadrature-order 2 \
  --recoat-time 10.0 \
  --recoat-steps 10 \
  --cooling-steps 300 \
  --cooling-dt 2.0 \
  --reset-activation-temperature \
  --release-after-cooling \
  --mechanics-model j2_plastic \
  --yield-saturation-stress 1.15e9 \
  --mechanics-every 100 \
  --thermal-output-every 1000 \
  --mechanics-output-every 1000 \
  --summary-every 100 \
  --output-dir "${OUT_ROOT}" \
  "$@"

echo "============================================================"
echo "Done. Key outputs:"
echo "  output dir: ${OUT_ROOT}"
echo "  release:    ${OUT_ROOT}/release.vtu"
echo "  profile:    ${PROFILE_JSON}"
echo "============================================================"
