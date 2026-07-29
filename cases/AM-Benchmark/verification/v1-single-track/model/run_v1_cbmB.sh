#!/usr/bin/env bash
# ============================================================================
# V1 single-track thermal run - Balbaa2022 parity at NIST AMB2018-02 case B
# (195 W, 800 mm/s, spot D4sigma 100 um -> r = 50 um, OPD 100 um,
#  20 um powder layer on 280 um substrate, A = 0.62).
# Thermal-only (--mechanics-every 0). Zero-calibration: every input traces to
# inputs/balbaa-model.json, inputs/nist-meltpool.json, or a registered entry
# in inputs/deviations.yaml.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v python >/dev/null 2>&1 && [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
POWER="${POWER:-195}"
SPEED="${SPEED:-0.800}"
CASE_TAG="${CASE_TAG:-cbmB}"
DT="${DT:-1.0e-5}"
RHO_SOLID="${RHO_SOLID:-}"          # optional override for D-V1-18 bounding runs
THERMAL_OUT_EVERY="${THERMAL_OUT_EVERY:-2}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/v1_${CASE_TAG}_P${POWER}_${RUN_ID}}"

MESH="${MESH:-${SCRIPT_DIR}/v1_single_track_c3d8.inp}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/v1_material_config.json}"
PATH_FILE="${OUT_ROOT}/v1_path_${CASE_TAG}.csv"

for f in "${MESH}" "${CONFIG}"; do
  [[ -f "$f" ]] || { echo "v1: missing input: $f" >&2; exit 2; }
done
mkdir -p "$(dirname "${OUT_ROOT}")"
if ! mkdir "${OUT_ROOT}" 2>/dev/null; then
  echo "v1: refusing existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/make_v1_path.py" \
  --power "${POWER}" --speed "${SPEED}" --output "${PATH_FILE}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM:-cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PYTHONUNBUFFERED=1

# Balbaa initial/ambient (deviations D-V1-05 baseline):
# preheat 80 C = 353.15 K (Sec 2.1 Eq 2), ambient sink 313 K (Table 1),
# bottom fixed at preheat.
PREHEAT_K=353.15
AMBIENT_K=313.0

cd "${WORK_ROOT}"
SOLVER_CMD=(
  "${PYTHON_BIN}" -m jax_fem_am.simulation.runner
  --config "${CONFIG}"
  --inp "${MESH}"
  --output-dir "${OUT_ROOT}"
  --profile-json "${OUT_ROOT}/profile.json"
  --profile-label "v1-${CASE_TAG}-P${POWER}"
  --xla-platform "${XLA_PLATFORM:-cpu}"
  --xla-preallocate off
  --xla-linear-solver "${LINEAR_SOLVER:-pardiso}"
  --build-axis z
  --base-side min
  --layer-thickness 2.0e-5
  --layers 1
  --support-thickness 2.8e-4
  --path-file "${PATH_FILE}"
  --path-length-scale 1.0
  --source-model legacy
  --beam-radius 5.0e-5
  --source-depth 1.0e-4
  --laser-power "${POWER}"
  --dt "${DT}"
  --layer-activation-mode layer_on_scan
  --layer-activation-geometry intersection
  --future-layer-mode void
  --active-window-below-layers 0
  --inactive-mass-factor 1.0
  --powder-mode powder
  --surface-selection exterior
  --boundary-tol 1.0e-6
  --quadrature-order 2
  --ambient "${AMBIENT_K}"
  --preheat-temperature "${PREHEAT_K}"
  --bottom-thermal-bc fixed
  --bottom-temperature "${PREHEAT_K}"
  --front-surface-loss-radiation
  --cooling-steps 30
  --cooling-dt "${DT}"
  --mechanics-every 0
  # workaround for shared-code bug (stepper.py:966 calls
  # set_flow_curve_active_mask on the _ThermalOnlyMechanicsProblem surrogate,
  # which lacks it): build the full mechanics Problem, never solve it.
  # Reported to main session 2026-07-29; do NOT fix solver code from V1.
  --no-xla-thermal-only-mechanics-surrogate
  --thermal-mass-lumping
  --thermal-output-every "${THERMAL_OUT_EVERY}"
  --summary-every 25
  --phase-history-model paper_irreversible
)
if [[ -n "${RHO_SOLID}" ]]; then
  SOLVER_CMD+=(--rho "${RHO_SOLID}")
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  SOLVER_CMD+=(${EXTRA_ARGS})
fi

printf '%q ' "${SOLVER_CMD[@]}" > "${OUT_ROOT}/solver_command.txt"
printf '\n' >> "${OUT_ROOT}/solver_command.txt"

"${SOLVER_CMD[@]}"

"${PYTHON_BIN}" -m jax_fem_am.verification.run_audit "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v1_run_audit.json" \
  --ambient "${AMBIENT_K}" \
  --quality-threshold 0.05 || echo "v1 WARNING: run_audit failed" >&2

echo "v1 ${CASE_TAG} complete: ${OUT_ROOT}"
