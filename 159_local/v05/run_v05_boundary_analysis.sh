#!/usr/bin/env bash
# v05 boundary-stress analysis over a fast-scan/physfix output directory.
# Usage: run_v05_boundary_analysis.sh <run_output_dir> [extra postprocess args]
set -euo pipefail

RUN_DIR="${1:?usage: run_v05_boundary_analysis.sh <run_output_dir> [args]}"
shift || true

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

python /home/user/work/159/jax-fem/159_local/v05/postprocess_boundary_stress.py "$RUN_DIR" "$@"
