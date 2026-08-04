#!/usr/bin/env bash
# ============================================================================
# D-V2-07 基板离散敏感性探针,第 6 次 —— 用 D-V2-19-R1 采纳臂重跑。
#
# 第 5 次(2026-08-03)三变体全部失败,病因是 flow_curve 高温端的切线比 H/E
# (D-V2-19-R1),与基板剖分无关。本次唯一的变化是 --config 指向采纳臂
# v2_material_config_fc_offset_mt.json,其余与 v_sub5.sh 逐字相同。
#
# 串行(2026-08-03 教训:三变体并行触发内核 OOM,单进程 RSS 18.8 GB / 虚拟机 31 GB)。
# 幂等:台账 summary complete==true 的变体直接跳过。自愈:每变体最多 2 次尝试。
# ============================================================================
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=12
M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
CFG=$M/v2_material_config_fc_offset_mt.json
cd /home/user/work/159
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$VT/sub6.log"; }
say "sub6 supervisor 启动 pid=$$ config=$(basename "$CFG")"

is_complete() {
  python3 -c "
import json,sys,os
p=os.path.join('$1','thermal_energy_ledger_summary.json')
try: ok=json.load(open(p)).get('complete') is True
except Exception: ok=False
sys.exit(0 if ok else 1)" 2>/dev/null
}

# coarse 变体与阶段 B 主臂完全同构(同网格 v2_multitrack_c3d8_coarse.inp、同配置
# offset_mt、同路径、同求解器参数),因此直接复用那次运行,不重跑一遍。
COARSE_SRC=/home/user/work/159/output/v2_full_offset_mt
COARSE_DST=/home/user/work/159/output/v2_sub6_coarse
if [ -d "$COARSE_SRC" ] && [ ! -e "$COARSE_DST" ]; then
  ln -s "$COARSE_SRC" "$COARSE_DST"
  say "coarse 变体 -> 符号链接到阶段 B 主臂 $(basename "$COARSE_SRC")"
fi

# fine 变体本次不跑,理由是实测的内存包络而不是偷懒:
#   coarse(30k 单元)实测单进程 RSS 5.8-7.1 GB  ->  fine(约 110k 单元)外推 ~22-26 GB
#   本机 31 GB,且另一会话的 GPU 任务常驻约 3.3 GB;2026-08-03 的内核 OOM
#   (单进程 18.8 GB)正是 fine 造成的。留待 GPU 任务空闲时段单独跑。
# 需要跑 fine 时把下面的列表改回 "coarse graded fine" 即可(脚本幂等,已完成的会跳过)。
for MODE in coarse graded; do
  SUF=""
  [ "$MODE" != graded ] && SUF="_$MODE"
  OUT=/home/user/work/159/output/v2_sub6_${MODE}
  if is_complete "$OUT"; then
    say "$MODE 已完成,跳过"
    continue
  fi
  for ATTEMPT in 1 2; do
    say "$MODE 尝试 $ATTEMPT 开始 (mesh v2_multitrack_c3d8${SUF}.inp)"
    rm -rf "$OUT"; mkdir -p "$OUT"
    python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
      --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
    # coarse 变体复用阶段 B 主臂的运行,而那次运行的路径被 head_path.py 截断到
    # 前 200 行。三个变体必须承受**逐字相同的载荷**,否则比的就不是基板剖分了,
    # 所以这里同样截断。(v_sub5.sh 用的是完整路径;sub5 三变体全部失败,
    # 没有需要保持可比的基线。)
    python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
    python -m jax_fem_am.simulation.runner \
      --config "$CFG" \
      --inp "$M/v2_multitrack_c3d8${SUF}.inp" \
      --output-dir "$OUT" --profile-json "$OUT/profile.json" \
      --profile-label "vtrack-sub6-$MODE" \
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
      --thermal-output-every 100 --mechanics-output-every 20 --summary-every 2 \
      --no-reset-plastic-on-melt --phase-history-model paper_irreversible \
      > "$OUT/run.log" 2>&1
    RC=$?
    NC=$(grep -c 'did not converge' "$OUT/run.log" 2>/dev/null | tr -d '\n')
    if is_complete "$OUT"; then
      say "$MODE 尝试 $ATTEMPT 完成 (rc=$RC, ledger=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null), nonconv=$NC)"
      break
    fi
    say "$MODE 尝试 $ATTEMPT 未完成 (rc=$RC, ledger=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null), nonconv=$NC)"
  done
done

say "全部变体处理完毕,开始分析"
python "$M/analyze_substrate_study.py" v2_sub6 > "$VT/substrate_analysis6.txt" 2>&1
say "SUB6_ALL_DONE"
