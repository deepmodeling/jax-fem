#!/usr/bin/env bash
set -euo pipefail

cd /home/user/work/159

# # 1) Generate geometry-aware scan path with 60 hatch lines.
# PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
# python3 /home/user/work/159/jax-fem/159_local/v02/geometry_aware_layer_path_planner.py \
#   --inp /home/user/work/159/schema/0119_c3d4_only.inp \
#   --mesh-length-scale 1.0 \
#   --build-axis x \
#   --base-side min \
#   --layer-thickness 1.0e-3 \
#   --max-print-layers 100 \
#   --planning-entity cell \
#   --auto-expand-layer-band \
#   --max-layer-band 5.0e-3 \
#   --min-samples-per-layer 20 \
#   --scan-axis auto \
#   --scan-start-frac 0.0 \
#   --scan-end-frac 1.0 \
#   --hatch-start-frac 0.0 \
#   --hatch-end-frac 1.0 \
#   --hatch-lines-per-layer 60 \
#   --no-auto-scan-steps-from-speed \
#   --scan-steps-per-segment 100 \
#   --beam-radius 1.0e-3 \
#   --source-depth 5.0e-4 \
#   --laser-power 3000 \
#   --dt 1.0e-4 \
#   --path-output /home/user/work/159/output/geometry_path_macro1mm_first10_h60/path_macro1mm_first10_h60.csv \
#   --output-dir /home/user/work/159/output/geometry_path_macro1mm_first10_h60

# 2) Run thermal-mechanical simulation with linear elastic stress solve every 100 steps.
PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
python3 /home/user/work/159/jax-fem/159_local/v03/am_thermal_stress_macro_intersection_mech100.py \
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
  --thermal-output-every 100 \
  --mechanics-output-every 100 \
  --summary-every 100 \
  --output-dir /home/user/work/159/output/thermal_macro1mm_intersection_first10_h60_mech100
