#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

export LASER_POWER_W="${LASER_POWER_W:-0.01}"
export BEAM_RADIUS_M="${BEAM_RADIUS_M:-0.0005}"
export SOURCE_DEPTH_M="${SOURCE_DEPTH_M:-0.001}"
export SOLIDUS_TEMPERATURE_K="${SOLIDUS_TEMPERATURE_K:-0}"
export LIQUIDUS_TEMPERATURE_K="${LIQUIDUS_TEMPERATURE_K:-0}"
export RUN_LABEL="${RUN_LABEL:-v06-nonzero-thermal-mechanical-cube}"
export OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/v06_nonzero_smoke_${RUN_ID}}"

exec bash "${SCRIPT_DIR}/run_smoke.sh"
