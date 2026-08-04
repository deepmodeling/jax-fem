#!/usr/bin/env bash
# ============================================================================
# D-V2-22 flow-curve 切线正则化阶梯 + D-V2-07 基板研究重跑
#
# 背景:v2_sub5_* 三变体全部在力学 Newton 上失败(force_ratio 1.5e6 而位移修正
# 5e-14),而把 flow_curve_table 换成常数占位表的 T1 变体跑通 154 步。诊断锁定
# flow_curve 本身:J-C 幂律 dsigma/deps_p = B*n*eps_p^(n-1) 在 eps_p->0 发散,
# 首格 0.002 采样出 H = 178.7 GPa > E = 171.0 GPa;近零加密后升到 510 GPa
# (即"表太粗"被排除,必须正则化幂律本身)。
#
# A 相:四臂筛选(coarse 网格 30k + 截断路径 200 行,失败点在 ~step 24,覆盖充分)
#   asis       原样(对照,预期失败)
#   offset     幂律原点移到 0.2% 偏移点(H1/E = 0.23;ε_p=0.2 处 1391 MPa,
#              与工程 UTS ~1000 MPa 的真应力量级相符,而 asis 给 1746 MPa)
#   cap        斜率限制 H <= 0.10 E(括号臂,H1/E = 0.10)
#   asis_iter  原样 + max-iter 200 + max-cuts 6(对照:排除"只是求解器设置")
#
# B 相:A 相优胜臂(优先 offset)跑满三个基板变体(prefix v2_sub6)
# C 相:analyze_substrate_study.py v2_sub6
#
# 断网无关:全部本地计算,不访问任何网络。
# 抗回收:由 Windows 侧 keepalive 客户端保活发行版(见 FCLADDER_RUNNING 哨兵)。
# 幂等:台账 summary complete==true 的运行直接跳过,被杀后重启不重做。
# 串行:峰值内存 = 单 runner(2026-08-03 并行触发 18.8 GB OOM 的教训)。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=12

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUTROOT=/home/user/work/159/output
LOG=$VT/fcladder.log
RES=$VT/fcladder_results.txt
SENTINEL=$VT/FCLADDER_RUNNING
cd /home/user/work/159

# 互斥:两次启动会并发写同一输出目录并互相污染台账(2026-08-03 实锤)。
LOCK=$VT/fcladder.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "另一个 fcladder 监督器已在运行($LOCK 被占),本次退出。" >&2
  exit 3
fi

touch "$SENTINEL"
trap 'rm -f "$SENTINEL"' EXIT

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

avail_gb() { free -g | awk '/^Mem:/ {print $7}'; }

is_complete() {
  python3 - "$1" <<'PY' 2>/dev/null
import json, os, sys
p = os.path.join(sys.argv[1], "thermal_energy_ledger_summary.json")
try:
    ok = json.load(open(p)).get("complete") is True
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

ledger_rows() { wc -l < "$1/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0; }

newton_failures() { grep -c "did not converge" "$1/run.log" 2>/dev/null || echo 0; }

# ---------------------------------------------------------------------------
# 单次运行。$1=输出目录 $2=材料配置 $3=网格后缀 $4=路径行数(0=全长)
# $5=超时秒 $6..=附加 runner 参数
# ---------------------------------------------------------------------------
run_case() {
  local OUT=$1 CFG=$2 SUF=$3 NROWS=$4 TMO=$5; shift 5
  rm -rf "$OUT"; mkdir -p "$OUT"
  python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
    --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
  if [ "$NROWS" -gt 0 ]; then
    python "$VT/head_path.py" "$OUT/path.csv" "$NROWS" >> "$OUT/path.log" 2>&1
  fi
  timeout "$TMO" python -m jax_fem_am.simulation.runner \
    --config "$CFG" \
    --inp "$M/v2_multitrack_c3d8${SUF}.inp" \
    --output-dir "$OUT" --profile-json "$OUT/profile.json" \
    --profile-label "fcladder-$(basename "$OUT")" \
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
    "$@" \
    > "$OUT/run.log" 2>&1
  return $?
}

say "================ fcladder supervisor 启动 pid=$$ ================"
say "A 相:四臂筛选(coarse 30k,路径 200 行,单臂超时 60 min)"

: > "$RES"
declare -A VERDICT

# 臂定义:名称 -> "配置文件|附加参数"
run_arm() {
  local NAME=$1 CFG=$2; shift 2
  local OUT=$OUTROOT/v2_fc_$NAME
  if is_complete "$OUT"; then
    say "A/$NAME 已完成,跳过"
  else
    for ATT in 1 2; do
      say "A/$NAME 尝试 $ATT 开始"
      run_case "$OUT" "$CFG" _coarse 200 3600 "$@"
      local RC=$?
      if is_complete "$OUT"; then
        say "A/$NAME 尝试 $ATT 完成 (rc=$RC, ledger=$(ledger_rows "$OUT"))"
        break
      fi
      say "A/$NAME 尝试 $ATT 未完成 (rc=$RC, ledger=$(ledger_rows "$OUT"), newton_fail=$(newton_failures "$OUT"))"
    done
  fi
  if is_complete "$OUT"; then
    VERDICT[$NAME]=PASS
  else
    VERDICT[$NAME]=FAIL
  fi
  printf '%-12s %-6s ledger=%-5s newton_nonconv=%-4s\n' \
    "$NAME" "${VERDICT[$NAME]}" "$(ledger_rows "$OUT")" "$(newton_failures "$OUT")" >> "$RES"
  say "A/$NAME 判定 ${VERDICT[$NAME]}"
}

run_arm asis      "$M/v2_material_config_fc_asis.json"   --mechanics-max-iter 50  --mechanics-line-search --mechanics-max-cuts 3
run_arm offset    "$M/v2_material_config_fc_offset.json" --mechanics-max-iter 50  --mechanics-line-search --mechanics-max-cuts 3
run_arm cap       "$M/v2_material_config_fc_cap.json"    --mechanics-max-iter 50  --mechanics-line-search --mechanics-max-cuts 3
run_arm asis_iter "$M/v2_material_config_fc_asis.json"   --mechanics-max-iter 200 --mechanics-line-search --mechanics-max-cuts 6

say "A 相结束。结果表:"
cat "$RES" | tee -a "$LOG"

# ---------------------------------------------------------------------------
WIN=""
if [ "${VERDICT[offset]}" = PASS ]; then
  WIN=offset
elif [ "${VERDICT[cap]}" = PASS ]; then
  WIN=cap
fi

if [ -z "$WIN" ]; then
  say "两个正则化臂均未通过 -> B 相不启动。诊断需升级(可能牵涉共享代码,须先报告用户/主线)。"
  say "FCLADDER_DONE (A only)"
  exit 0
fi
say "优胜臂 = $WIN;进入 B 相(三基板变体全长路径,单例超时 6 h)"

for MODE in coarse graded fine; do
  SUF=""; [ "$MODE" != graded ] && SUF="_$MODE"
  OUT=$OUTROOT/v2_sub6_$MODE
  if is_complete "$OUT"; then
    say "B/$MODE 已完成,跳过"
    continue
  fi
  # 内存预检:coarse 30k 单元实测 RSS 5.6 GB;fine 110k 单元约 3.7 倍 = 20 GB。
  # 2026-08-03 的 OOM 是三变体并行(18.8 GB);串行可行但仍需留余量,否则 OOM
  # 会在数小时后才失败,白费机时。
  NEED=8; [ "$MODE" = graded ] && NEED=14; [ "$MODE" = fine ] && NEED=24
  AV=$(avail_gb)
  if [ "${AV:-0}" -lt "$NEED" ]; then
    say "B/$MODE 跳过:可用内存 ${AV} GB < 需要 ${NEED} GB(避免数小时后 OOM)"
    printf 'B/%-8s %-6s reason=insufficient_memory avail=%sGB need=%sGB\n' \
      "$MODE" SKIP "$AV" "$NEED" >> "$RES"
    continue
  fi
  say "B/$MODE 内存预检通过(可用 ${AV} GB >= ${NEED} GB)"
  for ATT in 1 2; do
    say "B/$MODE 尝试 $ATT 开始"
    run_case "$OUT" "$M/v2_material_config_fc_$WIN.json" "$SUF" 0 21600 \
      --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3
    RC=$?
    if is_complete "$OUT"; then
      say "B/$MODE 尝试 $ATT 完成 (rc=$RC, ledger=$(ledger_rows "$OUT"))"
      break
    fi
    say "B/$MODE 尝试 $ATT 未完成 (rc=$RC, ledger=$(ledger_rows "$OUT"), newton_fail=$(newton_failures "$OUT"))"
  done
  printf 'B/%-8s %-6s ledger=%-5s newton_nonconv=%-4s\n' "$MODE" \
    "$(is_complete "$OUT" && echo PASS || echo FAIL)" \
    "$(ledger_rows "$OUT")" "$(newton_failures "$OUT")" >> "$RES"
done

say "B 相结束,进入 C 相分析"
python "$M/analyze_substrate_study.py" v2_sub6 > "$VT/substrate_analysis_sub6.txt" 2>&1
say "C 相结束 (rc=$?)"
say "最终结果表:"
cat "$RES" | tee -a "$LOG"
say "FCLADDER_DONE (A+B+C), winner=$WIN"
