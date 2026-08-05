#!/usr/bin/env bash
# 三件套验收判据 3(Fable5):coarse 网格上新旧方案的 RS 差,应落在
# offset-vs-cap 括号(2–3 %)量级内;若远超,说明地板/坍塌选择在动答案,要重审。
#
# 与 v2_full_offset_mt(已完成,240+40 步,原始 E)逐项同构,只换 --config
# (即 E_collapse + 按坍塌 E 重建的流动曲线),因此可逐单元相减。
set -u
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
export PYTHONPATH=/home/user/work/159/jax-fem JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MKL_NUM_THREADS=8

M=/home/user/work/159/jax-fem/cases/AM-Benchmark/verification/v2-cube-rs/model
VT=/home/user/work/159/vtmp
OUT=/home/user/work/159/output/v2_full_ecol
cd /home/user/work/159
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$VT/ecolcoarse.log"; }

say "coarse 上的 E 坍塌臂开始(与 v2_full_offset_mt 同构,只换 config)"
rm -rf "$OUT"; mkdir -p "$OUT"
python "$M/make_v2_path_multitrack.py" --tracks 3 --sample-step 12.5e-6 \
  --power 50 --output "$OUT/path.csv" > "$OUT/path.log" 2>&1
python "$VT/head_path.py" "$OUT/path.csv" 200 >> "$OUT/path.log" 2>&1
python -m jax_fem_am.simulation.runner \
  --config "$M/v2_material_config_fc_ecol.json" \
  --inp "$M/v2_multitrack_c3d8_coarse.inp" \
  --output-dir "$OUT" --profile-json "$OUT/profile.json" \
  --profile-label "ecol-coarse" \
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
RC=$?
N=$(wc -l < "$OUT/thermal_energy_ledger.jsonl" 2>/dev/null || echo 0)
NF=$(grep -c 'did not converge' "$OUT/run.log" 2>/dev/null | tr -d '\n')
say "coarse ecol 结束 rc=$RC ledger=$N newton_nonconv=$NF"
say "ECOLCOARSE_DONE"
