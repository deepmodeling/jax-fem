#!/usr/bin/env bash
# ============================================================================
# D-V2-22 二分:分离"flow_curve_table 消费路径" / "高温正则化" / "近零切线"
#
# 前情:A 相四臂(asis/offset/cap/asis_iter)全部失败且全部停在同一处
# (ledger=12, global_step=10),近零切线被从 H1/E=2.98 括到 0.10 而收敛行为
# 不变 -> 近零奇异被证伪为病因。
#
#   flat      = T1 占位对(490 MPa + 1e9*eps_p,全温区常数),但经 flow_curve_table
#               交付。T1 本身走 yield_table+hardening_table 且跑通 154 步。
#               失败 => 病因在共享代码的 flow_curve_table 消费路径 -> 停下报告。
#   hotfloor  = offset 近零端 + T1 高温地板。通过 => 病因是 D-V2-19 的高温对。
#
# 判据不是"跑完",而是**能否越过 ledger=12**(四个失败臂的共同停点)。
# 每臂 900 s 上限:失败臂 4.5 min 就到 12 行,健康臂(T1 到过 156 行)远早于此越过。
#
# 串行、flock 互斥、零网络。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=12

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/bisect.log
RES=$VT/bisect_results.txt
cd /home/user/work/159

LOCK=$VT/bisect.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "另一个 bisect 监督器已在运行,本次退出。" >&2
  exit 3
fi
trap 'rm -f "$VT/BISECT_RUNNING"' EXIT
touch "$VT/BISECT_RUNNING"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }
newton_failures() { grep -c "did not converge" "$1/run.log" 2>/dev/null || echo 0; }

# 与 v_fcladder.sh 的 run_case 逐字相同(除输出目录/配置),保证可比。
run_case() {
  local OUT=$1 CFG=$2 TMO=$3; shift 3
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
  timeout "$TMO" python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$M/v2_multitrack_c3d8_coarse.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "bisect-$(basename "$OUT")" \
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
    --thermal-output-every 100 --mechanics-output-every 100 --summary-every 2 \
    --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
    --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3 \
    "$@" \
    > "$OUT/run.log" 2>&1
  return $?
}

: > "$RES"
say "================ bisect supervisor 启动 pid=$$ ================"
say "判据:越过 ledger=12(四失败臂的共同停点)即 PASS_STEP"

for ARM in flat hotfloor; do
  OUT=$OUTROOT/v2_fc_$ARM
  say "$ARM 开始 (上限 900 s)"
  run_case "$OUT" "$M/v2_material_config_fc_$ARM.json" 900
  RC=$?
  N=$(ledger_rows "$OUT")
  NF=$(newton_failures "$OUT")
  if [ "$N" -gt 12 ]; then V=PASS_STEP; else V=FAIL_SAME; fi
  say "$ARM 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
  printf '%-10s %-10s ledger=%-5s newton_nonconv=%-4s rc=%s\n' \
    "$ARM" "$V" "$N" "$NF" "$RC" >> "$RES"
done

say "结果表:"
cat "$RES" | tee -a "$LOG"
say "BISECT_DONE"
