#!/usr/bin/env bash
# Reproducible optimized continuation of the two-layer r3 Kaess case.
# Physics inputs stay aligned with r3; results go to a new directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/home/user/work/159}"
RUN_ID="${RUN_ID:-r3opt_$(date -u +%Y%m%dT%H%M%SZ)}"

export WORK_ROOT RUN_ID
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PLATE_TEMP_C="${PLATE_TEMP_C:-150}"
export POWER_TAG="${POWER_TAG:-P250}"
export ELEMENT_TYPE="${ELEMENT_TYPE:-c3d8}"
export POWDER_SOLID="${POWDER_SOLID:-1}"
export POWDER_SOLID_E="${POWDER_SOLID_E:-10e9}"
export POWDER_SOLID_YIELD="${POWDER_SOLID_YIELD:-1e6}"
export POWDER_SOLID_HARDENING="${POWDER_SOLID_HARDENING:-1e7}"
export PATH_ARGS="${PATH_ARGS:---layers 2}"
export OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/kaess_p2_T150C_P250_c3d8_r3opt_phase23_${RUN_ID}}"

# residual-only saves the final unused mechanics tangent assembly; phase23
# reuses PARDISO CSR indices and symbolic analysis while still refactorizing
# changed Newton matrices. EXTRA_ARGS remains available for explicit overrides.
export EXTRA_ARGS="${EXTRA_ARGS:-} --mechanics-residual-only-check --xla-pardiso-mode phase23"

exec bash "${SCRIPT_DIR}/run_kaess_phase2.sh"
