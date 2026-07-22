#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
XLA_PLATFORM="${XLA_PLATFORM:-cpu}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/v06_smoke_${RUN_ID}}"
LASER_POWER_W="${LASER_POWER_W:-0}"
BEAM_RADIUS_M="${BEAM_RADIUS_M:-0}"
SOURCE_DEPTH_M="${SOURCE_DEPTH_M:-0}"
RUN_LABEL="${RUN_LABEL:-v06-smoke-cube}"

MATERIAL_CONFIG="${MATERIAL_CONFIG:-${WORK_ROOT}/materials/Ti-6Al-4V/ti64_material_config_physfix.json}"
MESH_FILE="${REPO_ROOT}/jax_fem_am/verification/fixtures/unit_cube_6tet.inp"
XRD_PROTOCOL="${REPO_ROOT}/jax_fem_am/verification/fixtures/unit_cube_xrd_protocol.json"

if [[ ! -f "${MATERIAL_CONFIG}" ]]; then
  echo "v06 smoke: material config not found: ${MATERIAL_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${MESH_FILE}" ]]; then
  echo "v06 smoke: verification mesh not found: ${MESH_FILE}" >&2
  exit 2
fi
if [[ ! -f "${XRD_PROTOCOL}" ]]; then
  echo "v06 smoke: XRD protocol not found: ${XRD_PROTOCOL}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_ROOT}")"
if ! mkdir "${OUT_ROOT}" 2>/dev/null; then
  echo "v06 smoke: refusing existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

cd "${WORK_ROOT}"
SOLVER_CMD=(
  "${PYTHON_BIN}" -m jax_fem_am.simulation.runner
  --config "${MATERIAL_CONFIG}"
  --inp "${MESH_FILE}"
  --output-dir "${OUT_ROOT}"
  --profile-json "${OUT_ROOT}/profile.json"
  --profile-label "${RUN_LABEL}"
  --xla-platform "${XLA_PLATFORM}"
  --xla-preallocate off
  --build-axis z
  --base-side min
  --layer-thickness 0.001
  --layers 1
  --laser-power "${LASER_POWER_W}"
  --beam-radius "${BEAM_RADIUS_M}"
  --source-depth "${SOURCE_DEPTH_M}"
  --scan-speed 0
  --scan-steps-per-layer 1
  --activation-reset-temperature 1100
  --cooling-steps 2
  --cooling-dt 1
  --bottom-thermal-bc fixed
  --bottom-mechanics-bc fixed
  --mechanics-every 1
  --thermal-output-every 1
  --mechanics-output-every 1
  --summary-every 1
  --release-after-cooling
  --reset-plastic-on-melt
)
if [[ -n "${SOLIDUS_TEMPERATURE_K:-}" || -n "${LIQUIDUS_TEMPERATURE_K:-}" ]]; then
  if [[ -z "${SOLIDUS_TEMPERATURE_K:-}" || -z "${LIQUIDUS_TEMPERATURE_K:-}" ]]; then
    echo "v06 smoke: solidus and liquidus overrides must be provided together" >&2
    exit 2
  fi
  SOLVER_CMD+=(
    --solidus-temperature "${SOLIDUS_TEMPERATURE_K}"
    --liquidus-temperature "${LIQUIDUS_TEMPERATURE_K}"
  )
fi
printf '%q ' "${SOLVER_CMD[@]}" > "${OUT_ROOT}/solver_command.txt"
printf '\n' >> "${OUT_ROOT}/solver_command.txt"

finalize_manifest() {
  local prior_status=$?
  trap - EXIT
  set +e
  "${PYTHON_BIN}" -m jax_fem_am.verification.provenance \
    --repo-root "${REPO_ROOT}" \
    --work-root "${WORK_ROOT}" \
    --run-dir "${OUT_ROOT}" \
    --mesh "${MESH_FILE}" \
    --material-config "${MATERIAL_CONFIG}" \
    --xrd-protocol "${XRD_PROTOCOL}" \
    --label "${RUN_LABEL}" \
    --output "${OUT_ROOT}/v06_manifest.json" \
    --require-complete
  local manifest_status=$?
  if (( prior_status != 0 )); then
    if [[ -f "${OUT_ROOT}/v06_manifest.json" ]]; then
      echo "v06 smoke failed; forensic manifest written to ${OUT_ROOT}" >&2
    else
      echo "v06 smoke failed; forensic manifest could not be written (status ${manifest_status})" >&2
    fi
    exit "${prior_status}"
  fi
  if (( manifest_status != 0 )); then
    echo "v06 smoke artifacts are incomplete; inspect ${OUT_ROOT}/v06_manifest.json" >&2
    exit "${manifest_status}"
  fi
  echo "v06 smoke complete: ${OUT_ROOT}"
  exit 0
}
trap finalize_manifest EXIT

"${SOLVER_CMD[@]}"

"${PYTHON_BIN}" -m jax_fem_am.verification.run_audit "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_run_audit.json" \
  --ambient 300 \
  --quality-threshold 0.05 \
  --source-free-upper-bound 1100

"${PYTHON_BIN}" -m jax_fem_am.verification.xrd_vtu \
  --vtu "${OUT_ROOT}/step_000002_cooling.vtu" \
  --protocol "${XRD_PROTOCOL}" \
  --quality-threshold 0.05 \
  --output "${OUT_ROOT}/xrd_operator_smoke.json"

"${PYTHON_BIN}" -m jax_fem_am.verification.response_gate \
  --run-dir "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_response_gate.json"
