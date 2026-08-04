#!/usr/bin/env bash
# ============================================================================
# D-V2-19 单变量试验梯(D-V2-22 因果被证伪后的下一刀)
#
# 已知:
#   A 相四臂 asis/offset/cap/asis_iter 全部 FAIL,全部停在 ledger=12。
#   flat / hotfloor 两个二分臂 PASS_STEP(ledger 152 / 165,newton_nonconv=0)。
#   -> flow_curve_table 消费路径无罪;病因在高温分支。
#
# 但 hotfloor 一次改了两样(高温屈服量级 1->490 MPa,高温硬化 1e7->1e9 Pa),
# 所以还分不出是哪一样。本梯固定 offset 的近零处理,单变量地扫 D-V2-19 这一对:
#
#   hy10   floor 1e6 -> 1e7 Pa   (屈服轴,弱)     H_reg 不动
#   hy100  floor 1e6 -> 1e8 Pa   (屈服轴,强)     H_reg 不动
#   htan   H_reg 1e7 -> 1e9 Pa   (硬化轴)         floor 不动
#
# 判据与 v_bisect.sh 逐字相同:越过 ledger=12 即 PASS_STEP。
# run_case 与 v_bisect.sh / v_fcladder.sh 逐字相同(除输出目录/配置),保证可比。
# 三臂并行(各 6 线程),上限 1200 s。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=6

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/hotreg.log
RES=$VT/hotreg_results.txt
cd /home/user/work/159

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }
newton_failures() { grep -c "did not converge" "$1/run.log" 2>/dev/null || echo 0; }

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
    --profile-label "hotreg-$(basename "$OUT")" \
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

one_arm() {
  local ARM=$1
  local OUT=$OUTROOT/v2_fc_$ARM
  run_case "$OUT" "$M/v2_material_config_fc_$ARM.json" 1200
  local RC=$?
  local N; N=$(ledger_rows "$OUT")
  local NF; NF=$(newton_failures "$OUT" | tr -d '\n')
  local V; if [ "$N" -gt 12 ]; then V=PASS_STEP; else V=FAIL_SAME; fi
  printf '%-8s %-10s ledger=%-6s newton_nonconv=%-4s rc=%s\n' \
    "$ARM" "$V" "$N" "$NF" "$RC" >> "$RES"
  say "$ARM 结束 rc=$RC ledger=$N newton_nonconv=$NF -> $V"
}

: > "$RES"
say "============ D-V2-19 单变量梯启动 pid=$$ ============"
say "判据:越过 ledger=12 即 PASS_STEP(与 v_bisect.sh 同)"

for ARM in hy10 hy100 htan; do
  say "$ARM 启动(后台,上限 1200 s)"
  one_arm "$ARM" &
done
wait

say "结果表:"
cat "$RES" | tee -a "$LOG"
say "HOTREG_DONE"
