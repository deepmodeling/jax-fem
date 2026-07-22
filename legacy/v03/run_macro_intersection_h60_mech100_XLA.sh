#!/usr/bin/env bash
set -euo pipefail

cd /home/user/work/159

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  # Use the known local JAX environment when this script is launched from a
  # plain shell where python3 may not provide jax/meshio.
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
python /home/user/work/159/jax-fem/159_local/v03/am_thermal_stress_macro_intersection_mech100_XLA.py \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver jax \
  --xla-fallback-to-spsolve \
  --xla-show-devices \
  --config materials/Ti-6Al-4V/ti64_material_config_initial.json \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --max-print-layers 100 \
  --path-file /home/user/work/159/output/geometry_path_macro1mm_first10_h60/path_macro1mm_first10_h60.csv \
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
  --output-dir /home/user/work/159/output/thermal_macro1mm_intersection_first10_h60_mech100_xla \
  "$@"
