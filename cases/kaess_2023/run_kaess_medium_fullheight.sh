#!/usr/bin/env bash
# Medium-scale Kaess whole-height regression (target: about 2-3 hours).
#
# This is deliberately a reduced-order process test, not a reference-resolution
# result: the full 0.3 mm build height is activated as 3 x 100 um macro layers,
# while the complete C3D8 powder-margin mesh, moving source, J2 mechanics,
# cooldown, saw-cut release, XRD operator, audit, response gate, and provenance
# pipeline are retained.  The three-pass laser exposure is intentionally lower
# than the ten-pass reference exposure, so temperatures/stresses are regression
# signals only and must not be interpreted as reference-energy predictions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/home/user/work/159}"
RUN_ID="${RUN_ID:-medium3h_$(date -u +%Y%m%dT%H%M%SZ)}"

export WORK_ROOT RUN_ID
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
# Lock the scientific configuration so stale environment variables from r3/r4
# cannot silently turn this dedicated regression into a different case.
export PLATE_TEMP_C="150"
export ROOM_TEMP_K="293.15"
export RELAX_TEMP_K="0"
export POWER_TAG="P250"
export ELEMENT_TYPE="c3d8"
export POWDER_SOLID="1"
export POWDER_SOLID_E="10e9"
export POWDER_SOLID_YIELD="1e6"
export POWDER_SOLID_HARDENING="1e7"
export MATERIAL_CONFIG="${WORK_ROOT}/materials/316L/ss316l_material_config_kaess.json"
export MESH_FILE="${SCRIPT_DIR}/kaess_cantilever_c3d8_powder_margin.inp"

# Three macro layers span the same 0.3 mm height as the 10 x 30 um reference.
# A 50 um path sample equals one beam radius and gives 334 path states.
export BUILD_LAYERS="3"
export LAYER_THICKNESS="1.0e-4"
export PATH_SAMPLE_STEP="5.0e-5"
export PATH_ARGS="--power 250.0 --speed 0.850 --hatch 1.0e-4 --jump-speed 5.0 --rotation-deg 67.0 --start-angle-deg 46.0 --sample-step ${PATH_SAMPLE_STEP}"

# Two macro-layer waits at 45 s retain the 90 s aggregate recoat dwell of the
# nine 10 s waits in the reference build.  Temporal resolution is intentionally
# coarser.  Reference-reset events may still force mechanics between cadence
# points, so MECH_EVERY is not advertised as the primary speed control.
export RECOAT_TIME="45.0"
export RECOAT_STEPS="10"
export COOLING_STEPS="30"
export COOLING_DT="1.0"
export MECH_EVERY="20"
export MECH_ACCEPTANCE="abaqus"
export MECH_REL_TOL="5e-5"
export MECH_MAX_CUTS="3"
export MECH_T_FLOOR="293.15"
export THERMAL_LUMPING="1"

export XLA_PLATFORM="cpu"
export LINEAR_SOLVER="pardiso"
export RUN_LABEL="kaess-2023-medium-fullheight-macro${BUILD_LAYERS}-T${PLATE_TEMP_C}C-${POWER_TAG}-${ELEMENT_TYPE}"
export OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/kaess_medium_fullheight_T${PLATE_TEMP_C}C_${POWER_TAG}_${ELEMENT_TYPE}_${RUN_ID}}"

# Phase23 reuses PARDISO symbolic analysis.  The residual-only check skips an
# unused final mechanics tangent.  Summary cadence supports the documented
# step-50 and step-100 runtime gates.
export EXTRA_ARGS="--mechanics-residual-only-check --xla-pardiso-mode phase23 --summary-every 25"

print_plan() {
  printf '%s\n' \
    "RUN_LABEL=${RUN_LABEL}" \
    "ELEMENT_TYPE=${ELEMENT_TYPE}" \
    "MESH_FILE=${MESH_FILE}" \
    "MATERIAL_CONFIG=${MATERIAL_CONFIG}" \
    "BUILD_LAYERS=${BUILD_LAYERS}" \
    "LAYER_THICKNESS=${LAYER_THICKNESS}" \
    "PATH_SAMPLE_STEP=${PATH_SAMPLE_STEP}" \
    "PATH_ARGS=${PATH_ARGS}" \
    "RECOAT_TIME=${RECOAT_TIME}" \
    "RECOAT_STEPS=${RECOAT_STEPS}" \
    "COOLING_STEPS=${COOLING_STEPS}" \
    "COOLING_DT=${COOLING_DT}" \
    "MECH_EVERY=${MECH_EVERY}" \
    "MECH_ACCEPTANCE=${MECH_ACCEPTANCE}" \
    "MECH_REL_TOL=${MECH_REL_TOL}" \
    "XLA_PLATFORM=${XLA_PLATFORM}" \
    "LINEAR_SOLVER=${LINEAR_SOLVER}" \
    "EXTRA_ARGS=${EXTRA_ARGS}" \
    "EXPECTED_STEPS=384" \
    "OUT_ROOT=${OUT_ROOT}"
}

case "${1:-}" in
  --print-plan)
    print_plan
    exit 0
    ;;
  "")
    ;;
  *)
    echo "usage: ${0##*/} [--print-plan]" >&2
    exit 2
    ;;
esac

echo "kaess medium full-height plan: macro_layers=${BUILD_LAYERS} layer_thickness=${LAYER_THICKNESS} path_sample=${PATH_SAMPLE_STEP} expected_steps=384"
echo "kaess medium full-height output: ${OUT_ROOT}"

exec bash "${SCRIPT_DIR}/run_kaess_phase2.sh"
