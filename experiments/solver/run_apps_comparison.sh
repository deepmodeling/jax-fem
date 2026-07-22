#!/bin/bash
# V07 cross-application comparison: baseline vs pardiso/phase23 on stock
# jax-fem validation apps (unmodified, intercepted at linear_solver).
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate jax-fem-env
cd /home/user/work/159/jax-fem
export MPLBACKEND=Agg
export PYTHONPATH=/home/user/work/159/jax-fem
OUT=output/05_bench/v07_apps
mkdir -p $OUT
H=experiments/solver/bench_apps.py

run_case () {  # name app_path arm timeout snapshot_src
  local name=$1 app=$2 arm=$3 to=$4 snap=$5
  echo "=== $name / $arm ==="
  timeout "$to" python $H "$app" "$arm" "$OUT/${name}_${arm}.json" 2>&1 | tail -2
  if [ -d "$snap" ]; then
    rm -rf "$OUT/${name}_${arm}_out"
    cp -r "$snap" "$OUT/${name}_${arm}_out"
  fi
}

# 1) phase-field fracture: staggered u/d, baseline is spsolve
PFF=applications/phase_field_fracture
run_case pff $PFF/example.py baseline 3600 $PFF/output
run_case pff $PFF/example.py phase23  3600 $PFF/output

# 2) wave: 200 implicit steps, constant matrix, baseline is default jax GPU
WAVE=applications/wave
run_case wave $WAVE/example.py baseline 3600 $WAVE/output
run_case wave $WAVE/example.py phase23  3600 $WAVE/output
run_case wave $WAVE/example.py spsolve  3600 $WAVE/output

# 3) scalability: 3D hyperelastic 50^3 (~397k dofs), baseline is petsc
SCAL=applications/scalability
run_case scal $SCAL/example_forward.py baseline 3600 $SCAL/output
run_case scal $SCAL/example_forward.py phase23  3600 $SCAL/output
run_case scal $SCAL/example_forward.py spsolve  3600 $SCAL/output

echo ALL_DONE
