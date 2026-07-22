#!/usr/bin/env bash
set -euo pipefail

# FAST-SCAN whole-part run (2026-07-09): layer-lumped "flash heating" mode.
#
# Rationale: in route-B (consolidation-on-activation) the stress physics is
# "layer enters stress-free at the relaxation temperature -> cools under
# constraint -> residual stress builds". The laser raster only adds a local
# transient on top. Dropping the raster and keeping ONLY per-layer flash
# activation + interlayer recoat cooling reduces 826k thermal steps to ~1k
# (~90x) while preserving layer-scale temperature/stress distributions.
# This mirrors the industrial layer-agglomeration approach (Simufact/Netfabb).
#
# What is sacrificed: track-level thermal history (melt-track gradients and
# scan-direction anisotropy). Do not use for melt-pool or track-scale claims.
#
# Per layer: 1 laser-off step (whole layer flash-activates hot at the
# relaxation temperature) + recoat dwell (10 implicit steps over 10 s).
# Mechanics every 22 steps ~= every 2 layers, forced at the last step.

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
cd "${WORK_ROOT}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/fastscan_flash_full91_${RUN_ID}}"
PROFILE_JSON="${PROFILE_JSON:-${OUT_ROOT}/profile.json}"

mkdir -p "${OUT_ROOT}"

echo "============================================================"
echo "fast-scan flash-heating whole-part run (91 macro layers)"
echo "OUT_ROOT     = ${OUT_ROOT}"
echo "Extra args   = $*"
echo "============================================================"

PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}" \
python "${REPO_ROOT}/159_local/v04/am_thermal_stress_macro_intersection_mech100_XLA.py" \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver spsolve \
  --profile-json "${PROFILE_JSON}" \
  --profile-label "fastscan-flash-full91" \
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

echo "============================================================"
echo "Done. Outputs: ${OUT_ROOT} (release: ${OUT_ROOT}/release.vtu)"
echo "============================================================"
