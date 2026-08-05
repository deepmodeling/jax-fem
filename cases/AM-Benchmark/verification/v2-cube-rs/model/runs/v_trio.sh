#!/usr/bin/env bash
# ============================================================================
# 三件套(Fable5 2026-08-05 文献对齐包)在 **graded** 网格上的验证。
#
# 基线:frac=0.01 + 原始 E(T) 在 graded 上确定性失败于 ledger=40(两次)。
# 验收(Fable5 原话):graded 在**不加大正则化剂量**的前提下收敛。
# 所以以下所有臂 min-tangent-frac 一律保持 0.01,不动剂量。
#
#   A  ecol         件1:E(T) 随固相线坍塌 + 有限地板(1 % E_RT,不为零)
#   B  ecol_anneal  件1 + 件2:再加熔化重置(legacy_reset + reset-plastic-on-melt
#                   + 凝固零应力参考 1273.15 K)。件2 靠求解器既有开关,未改共享代码。
#
# 件3(截断温度)不在本轮:与件1**叠加会自相矛盾**——把流动曲线钳在 T_cut
# 而让 E 继续坍塌,固相线处 H1/E 冲到 5.99(塑性支比弹性还硬),等于把
# D-V2-22 的病重新造一遍。件1 与件3 是互斥的两条路线,不是可叠加的两味药。
#
# 判据:越过 ledger=40 即 PASS_STEP。上限 2400 s。两臂并行(graded 单进程 ~9.3 GB)。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=8

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/trio.log
RES=$VT/trio_results.txt
cd /home/user/work/159

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }
newton_failures() { grep -c 'did not converge' "$1/run.log" 2>/dev/null | tr -d '\n'; }

run_case() {
  local OUT=$1 CFG=$2 TMO=$3; shift 3
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
  timeout "$TMO" python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$M/v2_multitrack_c3d8.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "trio-$(basename "$OUT")" \
    --xla-platform cpu --xla-preallocate off --xla-linear-solver pardiso \
    --xla-pardiso-mode phase23 \
    --build-axis z --base-side min --layer-thickness 4.0e-5 --layers 1 \
    --support-thickness 4.0e-4 --path-file "$OUT/path.csv" --path-length-scale 1.0 \
    --source-model legacy --beam-radius 5.0e-5 --source-depth 1.0e-4 \
    --laser-power 50 --dt 1.9e-5 \
    --layer-activation-mode layer_on_scan --layer-activation-geometry intersection \
    --future-layer-mode void --active-window-below-layers 0 --inactive-mass-factor 1.0 \
    --powder-mode powder --surface-selection exterior --boundary-tol 1.0e-6 \
    --quadrature-order 2 --ambient 313.0 --preheat-temperature 353.15 \
    --bottom-thermal-bc fixed --bottom-temperature 353.15 \
    --cooling-steps 40 --cooling-dt 0.02 --final-cooldown-temperature 353.15 \
    --mechanics-model j2_plastic --bottom-mechanics-bc fixed \
    --mechanics-every 20 --mechanics-rel-tol 5e-5 --mechanics-acceptance abaqus \
    --mechanics-temperature-floor 293.15 --thermal-mass-lumping \
    --thermal-output-every 100 --mechanics-output-every 20 --summary-every 2 \
    --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3 \
    "$@" \
    > "$OUT/run.log" 2>&1
  return $?
}

one() {
  local TAG=$1 CFG=$2; shift 2
  local OUT=$OUTROOT/v2_trio_$TAG
  run_case "$OUT" "$CFG" 2400 "$@"
  local RC=$?
  local N; N=$(ledger_rows "$OUT")
  local NF; NF=$(newton_failures "$OUT")
  local V; if [ "$N" -gt 40 ]; then V=PASS_STEP; else V=FAIL_SAME; fi
  printf '%-14s %-10s ledger=%-6s newton_nonconv=%-4s rc=%s\n' \
    "$TAG" "$V" "$N" "$NF" "$RC" >> "$RES"
  say "$TAG 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
}

: > "$RES"
say "======== 三件套验证(graded)启动 pid=$$ ========"
say "基线 frac=0.01 + 原始 E:确定性失败于 ledger=40。剂量保持 0.01 不变。"

# A:件1 单独。相位历史沿用原设定(paper_irreversible),只换 E(T) 与随之重建的流动曲线。
one ecol "$M/v2_material_config_fc_ecol.json" \
  --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
  --stress-relaxation-temperature 0 &

# B:件1 + 件2。legacy_reset 是 reset_plastic_on_melt / 凝固零应力参考生效的前提
# (events.py:114 paper_irreversible 分支会直接跳过这两者)。
one ecol_anneal "$M/v2_material_config_fc_ecol.json" \
  --reset-plastic-on-melt --phase-history-model legacy_reset \
  --stress-relaxation-temperature 1273.15 &

wait
say "结果表:"
cat "$RES" | tee -a "$LOG"
say "TRIO_DONE"
