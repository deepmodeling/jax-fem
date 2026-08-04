#!/usr/bin/env bash
# V2 基板离散敏感性,第 5 次(串行 + 幂等 + 自愈)
#   教训 2026-08-03:三变体并行触发内核 OOM(单进程 RSS 18.8 GB / 虚拟机 31 GB,
#   journal 10:23:39 实锤)。改串行:峰值内存 = 单 runner。
#   幂等:台账 summary complete==true 的变体直接跳过 -> 被杀后重启不重做。
#   自愈:每变体最多 3 次尝试。
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=12
M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
cd /home/user/work/159
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$VT/sub5.log"; }
say "sub5 supervisor 启动 pid=$$"

is_complete() {
  python3 - "$1" <<'PY' 2>/dev/null
import json, sys, os
p = os.path.join(sys.argv[1], "thermal_energy_ledger_summary.json")
try:
    ok = json.load(open(p)).get("complete") is True
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

for MODE in coarse graded fine; do
  SUF=""
  [ "$MODE" != graded ] && SUF="_$MODE"
  OUT=/home/user/work/159/output/v2_sub5_${MODE}
  if is_complete "$OUT"; then
    say "$MODE 已完成,跳过"
    continue
  fi
  for ATTEMPT in 1 2 3; do
    say "$MODE 尝试 $ATTEMPT 开始"
    rm -rf "$OUT"; mkdir -p "$OUT"
    python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
      --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
    python -m jax_fem_am.simulation.runner \
      --config "$M/v2_material_config.json" \
      --inp "$M/v2_multitrack_c3d8${SUF}.inp" \
      --output-dir "$OUT" --profile-json "$OUT/profile.json" \
      --profile-label "vtrack-sub5-$MODE" \
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
      --mechanics-max-iter 50 --mechanics-line-search --mechanics-max-cuts 3 \
      --mechanics-temperature-floor 293.15 --thermal-mass-lumping \
      --thermal-output-every 100 --mechanics-output-every 100 --summary-every 2 \
      --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
      > "$OUT/run.log" 2>&1
    RC=$?
    if is_complete "$OUT"; then
      say "$MODE 尝试 $ATTEMPT 完成 (rc=$RC, steps=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null))"
      break
    fi
    say "$MODE 尝试 $ATTEMPT 未完成 (rc=$RC),$([ $ATTEMPT -lt 3 ] && echo 重试 || echo 放弃)"
  done
done

say "全部变体处理完毕,开始分析"
python "$M/analyze_substrate_study.py" v2_sub5 > "$VT/substrate_analysis.txt" 2>&1
say "SUB5_ALL_DONE"
