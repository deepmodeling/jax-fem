#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 维护入口 2/2：全件精细扫描（full fine scan）—— 91 层逐道路径，约 24 小时
#
# 使用真实 h60 扫描路径（825,876 步）按 STRIDE 抽稀后逐道求解。
#   STRIDE 上限 = 5（光斑重叠判据：0.4mm/行 x STRIDE <= 2mm 光斑直径），
#   STRIDE=5 => 165,176 热步，估计 ~2.5 天；更粗的抽稀会把扫描线退化成
#   孤立能量脉冲，塑性统计成为混叠伪影（实测 flash/s64/s16 屈服单元
#   46/1748/50 不单调）。24h 内的全件请改用 run_fast_scan.sh 或等待
#   sequential flash / 沿道热源涂抹实现。
# 能量守恒：dt 随抽稀自动放大（由路径时间列差分），功率不变。
# 物理内核 = v05（增量 J2 + ε_p）+ 全部 physfix 物理修复 + 弹性地基。
#
# 与 fast scan 的区别：保留道次级热瞬变（更完整的高温屈服采样、扫描热
# 累积），代价是 ~24x 时长。真正全分辨率（STRIDE=1）当前约 264h，需等
# 求解器效率阶段完成。
#
# 可调环境变量：
#   STRIDE=16        路径抽稀步距（越小越精细越慢；1=全分辨率）
#   MECH_EVERY=300   力学求解间隔（热步数）
#   RUN_ID / OUT_ROOT 常规输出控制
#
# 用法:  bash 159_local/run_full_fine_scan_24h.sh [附加参数覆盖]
# 建议:  nohup 或 tmux 中运行；输出 output/finescan_v05_full91_<时间戳>/
# ============================================================================

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
PATH_CSV="${WORK_ROOT}/output/geometry_path_macro1mm_first10_h60/path_macro1mm_first10_h60.csv"
STRIDE="${STRIDE:-5}"
MECH_EVERY="${MECH_EVERY:-300}"

# Beam-overlap criterion (2026-07-13): consecutive deposits must overlap or
# the track degenerates into isolated high-energy pulses whose peak
# temperatures (and therefore yield statistics) are aliasing artifacts —
# measured: released plastic cells flash/s64/s16 = 46/1748/50, non-monotonic.
# Path step is 0.4 mm and beam diameter is 2 mm, so STRIDE <= 5 is required.
# Larger strides need ALLOW_ALIASED_STRIDE=1 and along-track source smearing
# (not implemented yet).
if [ "${STRIDE}" -gt 5 ] && [ "${ALLOW_ALIASED_STRIDE:-0}" != "1" ]; then
  echo "ERROR: STRIDE=${STRIDE} > 5 breaks the beam-overlap criterion" >&2
  echo "  (0.4 mm/row x STRIDE must stay <= 2 mm beam diameter)." >&2
  echo "  Deposits become isolated energy pulses; plasticity becomes an" >&2
  echo "  aliasing artifact. Set ALLOW_ALIASED_STRIDE=1 only for debugging." >&2
  exit 2
fi

cd "${WORK_ROOT}"
if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/finescan_v05_full91_${RUN_ID}}"
PROFILE_JSON="${PROFILE_JSON:-${OUT_ROOT}/profile.json}"
mkdir -p "${OUT_ROOT}"

SUBSAMPLED_CSV="${OUT_ROOT}/path_stride${STRIDE}.csv"
python "${REPO_ROOT}/159_local/v05/subsample_path.py" "${PATH_CSV}" "${SUBSAMPLED_CSV}" "${STRIDE}"

echo "=== v05 fine scan: STRIDE=${STRIDE}, MECH_EVERY=${MECH_EVERY}, OUT_ROOT=${OUT_ROOT} ==="

PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}" \
python "${REPO_ROOT}/159_local/v05/am_thermal_stress_macro_intersection_mech100_v05.py" \
  --xla-platform gpu \
  --xla-preallocate off \
  --xla-linear-solver spsolve \
  --profile-json "${PROFILE_JSON}" \
  --profile-label "finescan-v05-full91-stride${STRIDE}" \
  --config materials/Ti-6Al-4V/ti64_material_config_physfix.json \
  --inp "${WORK_ROOT}/schema/0119_c3d4_only.inp" \
  --max-cells 0 \
  --build-axis x \
  --base-side min \
  --layer-thickness 1.0e-3 \
  --path-file "${SUBSAMPLED_CSV}" \
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
  --mechanics-every "${MECH_EVERY}" \
  --mechanics-rel-tol 1e-5 \
  --mechanics-max-iter 50 \
  --mechanics-line-search \
  --release-after-cooling \
  --thermal-output-every 3000 \
  --mechanics-output-every 3000 \
  --summary-every 600 \
  --output-dir "${OUT_ROOT}" \
  "$@"

echo "=== done: ${OUT_ROOT} (release: ${OUT_ROOT}/release.vtu) ==="
echo "boundary analysis: bash ${REPO_ROOT}/159_local/v05/run_v05_boundary_analysis.sh ${OUT_ROOT}"
