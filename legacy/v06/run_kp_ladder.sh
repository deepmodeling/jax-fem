#!/usr/bin/env bash
# ============================================================================
# v06 实验：粉末侧向约束 k_p 敏感度阶梯（被动法向支撑的部件尺度效应）
#
# 设计（2026-07-15，见 v06/README.md 粉末约束条目与 v07 求解器文档）：
#   - 载具：全域快扫 flash full91（消融用 flash、真值用 stride-1 的既定分工；
#     前 5 层由底板 BC 主导，测不到粉末效应的主作用区）。
#   - 阶梯：k_p ∈ {0, 1e8, 1e9, 1e10, 1e11} Pa/m。0 = 现状自由侧面基线；
#     1e11 接近粘接极限的粗代理（双向弹簧）。
#   - 求解器：--xla-linear-solver pardiso（v07，real-slice 7.1x，VTU 逐位一致）。
#   - 目标量：release 位移场、边缘屈服带（v05 boundary analysis）、塑性单元数。
#   - 哨兵量：建造期 u_max 序列（线性弹簧适用域检查）。
#   - 判据：目标量漂移 < flash 模式自身散布（lane3 消融尺度）→ 判不敏感，
#     默认保持 none；敏感 → 升级 first5 两点对照 + 烧结壳网格化提上日程。
#
# 用法：
#   bash 159_local/v06/run_kp_ladder.sh                    # 完整阶梯（约 5h）
#   KP_POINTS="0 1e9" EXTRA_ARGS="--layers 3 --cooling-steps 5 --recoat-steps 2" \
#     bash 159_local/v06/run_kp_ladder.sh                  # 链路冒烟
#
# 失败语义：单点失败不中断链（记录 manifest 后继续下一点）。
# ============================================================================
set -u

WORK_ROOT="/home/user/work/159"
REPO_ROOT="${WORK_ROOT}/jax-fem"
V05_RUNNER="${REPO_ROOT}/159_local/v05/run_fastscan_flash_v05.sh"
BOUNDARY_RUNNER="${REPO_ROOT}/159_local/v05/run_v05_boundary_analysis.sh"

KP_POINTS="${KP_POINTS:-0 1e8 1e9 1e10 1e11}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LADDER_ID="${LADDER_ID:-$(date +%Y%m%d_%H%M%S)}"
LADDER_ROOT="${WORK_ROOT}/output/kp_ladder_${LADDER_ID}"
MANIFEST="${LADDER_ROOT}/ladder_manifest.tsv"

mkdir -p "${LADDER_ROOT}"
printf "kp\tstatus\texit_code\twall_s\tout_dir\n" > "${MANIFEST}"
echo "=== k_p ladder ${LADDER_ID}: points [${KP_POINTS}] root ${LADDER_ROOT} ==="

for KP in ${KP_POINTS}; do
  OUT="${LADDER_ROOT}/kp_${KP}"
  if [ "${KP}" = "0" ]; then
    POWDER_ARGS="--powder-mechanics-bc none"
  else
    POWDER_ARGS="--powder-mechanics-bc elastic --powder-foundation-stiffness ${KP}"
  fi

  echo "--- kp=${KP} -> ${OUT}"
  T0=$(date +%s)
  # OUT_ROOT/RUN_ID are consumed by the v05 runner; appended args override
  # its defaults (argparse keeps the last occurrence).
  OUT_ROOT="${OUT}" RUN_ID="kp${KP}_${LADDER_ID}" \
    bash "${V05_RUNNER}" \
      --xla-linear-solver pardiso \
      ${POWDER_ARGS} \
      ${EXTRA_ARGS} \
      > "${LADDER_ROOT}/kp_${KP}_console.log" 2>&1
  RC=$?
  WALL=$(( $(date +%s) - T0 ))

  if [ "${RC}" -eq 0 ] && [ -f "${OUT}/release.vtu" ]; then
    STATUS="complete"
    # 边缘屈服带分析（非致命：分析失败不影响阶梯继续）
    if [ -x "${BOUNDARY_RUNNER}" ] || [ -f "${BOUNDARY_RUNNER}" ]; then
      bash "${BOUNDARY_RUNNER}" "${OUT}" \
        >> "${LADDER_ROOT}/kp_${KP}_console.log" 2>&1 \
        || echo "WARNING: boundary analysis failed for kp=${KP} (non-fatal)"
    fi
  else
    STATUS="FAILED"
    echo "WARNING: kp=${KP} failed (exit=${RC}); tail of console:"
    tail -5 "${LADDER_ROOT}/kp_${KP}_console.log" || true
  fi
  printf "%s\t%s\t%d\t%d\t%s\n" "${KP}" "${STATUS}" "${RC}" "${WALL}" "${OUT}" \
    >> "${MANIFEST}"
done

echo "=== ladder finished; manifest: ==="
cat "${MANIFEST}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi
python "${REPO_ROOT}/159_local/v06/analyze_kp_ladder.py" "${LADDER_ROOT}" \
  | tee "${LADDER_ROOT}/ladder_summary.txt"
echo "=== summary: ${LADDER_ROOT}/ladder_summary.txt ==="
