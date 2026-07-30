#!/usr/bin/env bash
# ============================================================================
# AMB2018-01 L0 pipeline run (N=50, 109k elements, 13 computational layers).
# Flash deposition per D-02: front activation at liquidus, laser power 0 -
# energy enters via deposition enthalpy; energy-input fidelity is L1's topic.
# Layer clock: constant 53 s (schedule mean; per-layer times need a runner
# extension - registered L0 deviation). Every material value traces to
# derived/material/* (PROVISIONAL-L0 labels inside).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/user/miniconda3/envs/jax-fem-env/bin/python}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/home/user/work/output/amb_l0_${RUN_ID}}"
MESH="${CASE_DIR}/derived/meshes/amb_d07_L0.inp"
CONFIG="${CASE_DIR}/derived/material/amb_material_config_L0.json"

for f in "${MESH}" "${CONFIG}"; do
  [[ -f "$f" ]] || { echo "l0: missing input: $f" >&2; exit 2; }
done
mkdir -p "${OUT_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# A.3/D-07 temperatures: substrate underside Dirichlet 73.9 C = 347.05 K
PLATE_K=347.05

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m jax_fem_am.simulation.runner \
  --config "${CONFIG}" \
  --inp "${MESH}" \
  --mesh-length-scale 1.0e-3 \
  --output-dir "${OUT_ROOT}" \
  --profile-json "${OUT_ROOT}/profile.json" \
  --profile-label "amb-l0-${RUN_ID}" \
  --xla-platform cpu \
  --xla-preallocate off \
  --xla-linear-solver pardiso \
  --build-axis z \
  --base-side min \
  --support-thickness 12.7e-3 \
  --layers 13 \
  --layer-thickness 1.0e-3 \
  --layer-activation-mode front \
  --layer-activation-geometry centroid \
  --future-layer-mode void \
  --powder-mode powder \
  --reset-activation-temperature \
  --activation-reset-temperature 1630.0 \
  --laser-power 0.0 \
  --scan-steps-per-layer 1 \
  --dt 1.0 \
  --dwell-steps-between-layers 53 \
  --preheat-temperature "${PLATE_K}" \
  --bottom-temperature "${PLATE_K}" \
  --bottom-thermal-bc fixed \
  --ambient "${PLATE_K}" \
  --surface-selection exterior \
  --boundary-tol 1.0e-6 \
  --quadrature-order 2 \
  --mechanics-every 10 \
  --stress-relaxation-temperature 1273.15 \
  --bottom-mechanics-bc fixed \
  --cooling-steps 60 \
  --cooling-dt 10.0 \
  "$@" 2>&1 | tee "${OUT_ROOT}/run.log"
