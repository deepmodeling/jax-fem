#!/usr/bin/env bash
# ============================================================================
# P0(Fable5 排序):--min-tangent-frac 阈值扫描,在 **graded** 网格上。
#
# 已知:frac=0.01 在 graded 上**确定性失败**两次,均停在 ledger=40
#       (force_ratio 5.97e5);同一 frac 在 coarse 上两臂都跑完 240+40 步。
#       => 阈值是网格相关的。本扫描找 graded 能过的最小档。
#
# 判据:越过 ledger=40(graded 在 0.01 下的确定性停点)即 PASS_STEP。
# run_case 与 v_sub6.sh / v_fcfull.sh 逐字相同,除网格用 graded、配置用扫描臂。
# 路径同样截断到 200 行,与 v2_sub6_graded 的失败基线可比。
#
# 两档并行:graded 单进程实测 RSS 9.3 GB,本机 31 GB 且当前无其他任务。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=8

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/p0sweep.log
RES=$VT/p0sweep_results.txt
cd /home/user/work/159

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }
newton_failures() { grep -c 'did not converge' "$1/run.log" 2>/dev/null | tr -d '\n'; }

run_case() {
  local OUT=$1 CFG=$2 TMO=$3
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
  timeout "$TMO" python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$M/v2_multitrack_c3d8.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "p0-$(basename "$OUT")" \
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
    --stress-relaxation-temperature 0 \
    --cooling-steps 40 --cooling-dt 0.02 --final-cooldown-temperature 353.15 \
    --mechanics-model j2_plastic --bottom-mechanics-bc fixed \
    --mechanics-every 20 --mechanics-rel-tol 5e-5 --mechanics-acceptance abaqus \
    --mechanics-temperature-floor 293.15 --thermal-mass-lumping \
    --thermal-output-every 100 --mechanics-output-every 20 --summary-every 2 \
    --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
    --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3 \
    > "$OUT/run.log" 2>&1
  return $?
}

one() {
  local TAG=$1 FRAC=$2
  local OUT=$OUTROOT/v2_p0_graded_$TAG
  run_case "$OUT" "$M/v2_material_config_fc_offset_mt${TAG}.json" 2400
  local RC=$?
  local N; N=$(ledger_rows "$OUT")
  local NF; NF=$(newton_failures "$OUT")
  local V; if [ "$N" -gt 40 ]; then V=PASS_STEP; else V=FAIL_SAME; fi
  printf '%-8s frac=%-6s %-10s ledger=%-6s newton_nonconv=%-4s rc=%s\n' \
    "$TAG" "$FRAC" "$V" "$N" "$NF" "$RC" >> "$RES"
  say "frac=$FRAC 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
}

: > "$RES"
say "======== P0 阈值扫描(graded 网格)启动 pid=$$ ========"
say "基线:frac=0.01 在 graded 上确定性失败于 ledger=40(两次)"
say "判据:越过 ledger=40 即 PASS_STEP;上限 2400 s"
one 002 0.02 &
one 005 0.05 &
wait
say "结果表:"
cat "$RES" | tee -a "$LOG"
say "P0SWEEP_DONE"
