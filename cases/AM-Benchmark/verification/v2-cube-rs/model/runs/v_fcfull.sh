#!/usr/bin/env bash
# ============================================================================
# 阶段 B:采纳臂的**完整**运行(不设 timeout),两臂并行。
#   $1 = 主臂名(offset 侧),$2 = 括号臂名(cap 侧)
#
# 与 v_hotreg.sh / v_bisect.sh / v_fcladder.sh 的 run_case 逐字相同(除输出目录/
# 配置/无 timeout),保证与失败臂可比。
#
# 内存注记(2026-08-03 教训,见 v_sub5.sh):coarse 网格 30000 单元单进程 RSS
# 约 2-3 GB,两个并行安全;fine/graded 网格才是 18.8 GB 的 OOM 来源,那些必须串行。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=8

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/fcfull.log
RES=$VT/fcfull_results.txt
cd /home/user/work/159

ARM_MAIN=${1:?需要主臂名}
ARM_BRACKET=${2:?需要括号臂名}

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }
newton_failures() { grep -c "did not converge" "$1/run.log" 2>/dev/null | tr -d '\n'; }
is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}

run_case() {
  local OUT=$1 CFG=$2; shift 2
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
  python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$M/v2_multitrack_c3d8_coarse.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "fcfull-$(basename "$OUT")" \
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
    "$@" \
    > "$OUT/run.log" 2>&1
  return $?
}

one_arm() {
  local ARM=$1
  local OUT=$OUTROOT/v2_full_$ARM
  run_case "$OUT" "$M/v2_material_config_fc_$ARM.json"
  local RC=$?
  local N; N=$(ledger_rows "$OUT")
  local NF; NF=$(newton_failures "$OUT")
  local V; if is_complete "$OUT"; then V=COMPLETE; else V=INCOMPLETE; fi
  printf '%-10s %-11s ledger=%-6s newton_nonconv=%-4s rc=%s\n' \
    "$ARM" "$V" "$N" "$NF" "$RC" >> "$RES"
  say "$ARM 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
}

: > "$RES"
say "======== 阶段 B 完整运行:$ARM_MAIN(主) + $ARM_BRACKET(括号) ========"
one_arm "$ARM_MAIN" &
one_arm "$ARM_BRACKET" &
wait

say "结果表:"
cat "$RES" | tee -a "$LOG"
say "FCFULL_DONE"
