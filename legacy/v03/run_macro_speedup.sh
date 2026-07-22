#!/usr/bin/env bash
set -euo pipefail

cd /home/user/work/159

# ============================================================
# GPU / JAX runtime settings
# ============================================================

# 强制使用 GPU。如果环境没有正确识别 GPU，JAX 会直接报错。
export JAX_PLATFORM_NAME=gpu

# 允许 JAX 预分配显存，通常对长时间 FEM 求解更稳定。
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# ============================================================
# Full-part B-group configuration
# ============================================================

# 全部件共有 91 层
PRINT_LAYERS=91

# 每层 hatch lines 数量，保持原来的 60
HATCH_LINES_PER_LAYER=60

# ------------------------------------------------------------
# B 组热传导 / 扫描时间步控制
# ------------------------------------------------------------
# 原始设置：
#   SCAN_STEPS_PER_SEGMENT=100
#   DT=1.0e-4
#
# B 组设置：
#   SCAN_STEPS_PER_SEGMENT=25
#   DT=4.0e-4
#
# 单条扫描段总时间保持不变：
#   100 * 1.0e-4 = 25 * 4.0e-4 = 1.0e-2 s
#
# 热传导 step 数约减少到原来的 1/4。
#
SCAN_STEPS_PER_SEGMENT=25
DT=4.0e-4

# ------------------------------------------------------------
# Cooling steps
# ------------------------------------------------------------
# 原始冷却时间：
#   20 * 1.0e-4 = 2.0e-3 s
#
# B 组 dt = 4.0e-4，因此使用：
#   5 * 4.0e-4 = 2.0e-3 s
#
# 保持每层扫描后的冷却总时间近似不变。
#
COOLING_STEPS=5

# ------------------------------------------------------------
# VTU 输出频率
# ------------------------------------------------------------
# 原始约：
#   9000 step / layer
#
# B 组 scan steps 从 100 降到 25 后：
#   9000 * 25 / 100 = 2250 step / layer
#
# 目标每层约 10 个 VTU：
#   2250 / 10 = 225
#
OUTPUT_EVERY=225

# ------------------------------------------------------------
# 力学求解频率
# ------------------------------------------------------------
# MECH_EVERY=225 表示每层约 10 次线弹性应力求解。
#
MECH_EVERY=225

# ============================================================
# New output paths
# ============================================================

PATH_TAG="full91_h${HATCH_LINES_PER_LAYER}_scan25_dt4em4"
RUN_TAG="${PATH_TAG}_mech${MECH_EVERY}_out${OUTPUT_EVERY}"

# 新建全件扫描路径 CSV
PATH_DIR="/home/user/work/159/output/geometry_path_macro1mm_${PATH_TAG}"
PATH_FILE="${PATH_DIR}/path_macro1mm_${PATH_TAG}.csv"

# 新建全件 VTU 输出目录
VTU_OUTPUT_DIR="/home/user/work/159/output/thermal_macro1mm_intersection_${RUN_TAG}"

mkdir -p "${PATH_DIR}"
mkdir -p "${VTU_OUTPUT_DIR}"

echo "============================================================"
echo "AM thermal-mechanical full-part B-group scan"
echo "PRINT_LAYERS              = ${PRINT_LAYERS}"
echo "HATCH_LINES_PER_LAYER     = ${HATCH_LINES_PER_LAYER}"
echo "SCAN_STEPS_PER_SEGMENT    = ${SCAN_STEPS_PER_SEGMENT}"
echo "DT                        = ${DT}"
echo "COOLING_STEPS             = ${COOLING_STEPS}"
echo "OUTPUT_EVERY              = ${OUTPUT_EVERY}"
echo "MECH_EVERY                = ${MECH_EVERY}"
echo "PATH_FILE                 = ${PATH_FILE}"
echo "VTU_OUTPUT_DIR            = ${VTU_OUTPUT_DIR}"
echo "============================================================"

# ============================================================
# 0) Confirm JAX device
# ============================================================

PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
python3 - <<'PY'
import jax

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("JAX enable x64:", jax.config.read("jax_enable_x64"))
PY

# ============================================================
# 1) Generate full-part geometry-aware scan path CSV
# ============================================================

/usr/bin/time -v \
env PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
python3 /home/user/work/159/jax-fem/159_local/v02/geometry_aware_layer_path_planner.py \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --mesh-length-scale 1.0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --max-print-layers "${PRINT_LAYERS}" \
  --planning-entity cell \
  --auto-expand-layer-band \
  --max-layer-band 5.0e-3 \
  --min-samples-per-layer 20 \
  --scan-axis auto \
  --scan-start-frac 0.0 \
  --scan-end-frac 1.0 \
  --hatch-start-frac 0.0 \
  --hatch-end-frac 1.0 \
  --hatch-lines-per-layer "${HATCH_LINES_PER_LAYER}" \
  --no-auto-scan-steps-from-speed \
  --scan-steps-per-segment "${SCAN_STEPS_PER_SEGMENT}" \
  --beam-radius 1.0e-3 \
  --source-depth 5.0e-4 \
  --laser-power 3000 \
  --dt "${DT}" \
  --path-output "${PATH_FILE}" \
  --output-dir "${PATH_DIR}" \
  2>&1 | tee "${PATH_DIR}/path_planner.log"

# ============================================================
# 1.5) Check generated path layer count
# ============================================================

python3 - <<PY
import csv
from pathlib import Path

path = Path("${PATH_FILE}")
print("Checking path file:", path)

with path.open("r", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []

    layer_candidates = [
        "layer",
        "layer_id",
        "layer_idx",
        "layer_index",
        "print_layer",
        "print_layer_id",
    ]

    layer_col = None
    for c in layer_candidates:
        if c in fieldnames:
            layer_col = c
            break

    if layer_col is None:
        print("WARNING: Could not find layer column.")
        print("CSV columns:", fieldnames)
    else:
        layers = set()
        rows = 0
        for row in reader:
            rows += 1
            layers.add(row[layer_col])

        print("Layer column:", layer_col)
        print("Number of path rows:", rows)
        print("Number of unique layers:", len(layers))
        print("Expected layers:", ${PRINT_LAYERS})

        if len(layers) != ${PRINT_LAYERS}:
            print("WARNING: generated layer count does not match PRINT_LAYERS.")
PY

# ============================================================
# 2) Run full-part thermal-mechanical simulation
# ============================================================

/usr/bin/time -v \
env PYTHONPATH=/home/user/work/159/jax-fem/159_local/v01:/home/user/work/159/jax-fem \
python3 /home/user/work/159/jax-fem/159_local/v03/am_thermal_stress_macro_intersection_mech100.py \
  --config materials/Ti-6Al-4V/ti64_material_config_initial.json \
  --inp /home/user/work/159/schema/0119_c3d4_only.inp \
  --max-cells 0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --max-print-layers "${PRINT_LAYERS}" \
  --path-file "${PATH_FILE}" \
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
  --dt "${DT}" \
  --powder-mode powder \
  --cooling-steps "${COOLING_STEPS}" \
  --mechanics-model linear_elastic \
  --mechanics-every "${MECH_EVERY}" \
  --thermal-output-every "${OUTPUT_EVERY}" \
  --mechanics-output-every "${OUTPUT_EVERY}" \
  --summary-every "${OUTPUT_EVERY}" \
  --output-dir "${VTU_OUTPUT_DIR}" \
  2>&1 | tee "${VTU_OUTPUT_DIR}/run.log"