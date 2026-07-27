#!/usr/bin/env bash
# ============================================================================
# Kaess 2023 code-to-code benchmark - Phase 2: real moving Gaussian heat
# source on the reference scan pattern (67 deg rotation, hatch 100 um).
#
# 定位标注:对比实验(数值对标),非实验验证。
# Reference standard parameters: 250 W / 850 mm/s / 30 um / plate 150 C,
# beam radius 50 um, eta=0.5 (config), hemispherical volumetric source
# approximated by the v03 Gaussian with --source-depth = beam radius.
#
# Expect the run-audit undershoot/source-free gates to be strict here: this
# case doubles as the regression testbed for the G1 activation-undershoot fix.
#
# Usage:
#   bash run_kaess_phase2.sh                    # standard parameters
#   PLATE_TEMP_C=450 bash run_kaess_phase2.sh   # preheat ladder point
#   PATH_ARGS="--power 100" POWER_TAG=P100 bash run_kaess_phase2.sh
# ============================================================================
set -euo pipefail

if ! command -v python >/dev/null 2>&1 && [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
XLA_PLATFORM="${XLA_PLATFORM:-cpu}"
LINEAR_SOLVER="${LINEAR_SOLVER:-pardiso}"

PLATE_TEMP_C="${PLATE_TEMP_C:-150}"
PLATE_TEMP_K="$(${PYTHON_BIN} -c "print(273.15 + ${PLATE_TEMP_C})")"
ROOM_TEMP_K="${ROOM_TEMP_K:-293.15}"      # final plate cooldown target (paper 2.3)
# Reference model has NO stress relaxation/annealing mechanism -> disabled (0).
RELAX_TEMP_K="${RELAX_TEMP_K:-0}"
POWER_TAG="${POWER_TAG:-P250}"
PATH_ARGS="${PATH_ARGS:-}"
# The reference case is 10 x 30 um.  Keep those defaults, but expose both
# values so reduced-order launchers can cover the same 0.3 mm build height
# with a smaller number of explicitly documented macro layers.
BUILD_LAYERS="${BUILD_LAYERS:-10}"
LAYER_THICKNESS="${LAYER_THICKNESS:-3.0e-5}"
RECOAT_TIME="${RECOAT_TIME:-10.0}"
RECOAT_STEPS="${RECOAT_STEPS:-10}"
COOLING_STEPS="${COOLING_STEPS:-30}"
COOLING_DT="${COOLING_DT:-1.0}"
MECH_EVERY="${MECH_EVERY:-20}"
# DEVIATION (documented): G1 activation undershoot drives freshly activated
# quads below 0 K; the floor keeps the mechanics chain (thermal strain +
# material tables) physical so Newton converges. Thermal field stays
# unclamped -> run_audit undershoot gate still reports the artifact.
# Remove once the lumped-mass/substep G1 fix lands. Set MECH_T_FLOOR="" to disable.
MECH_T_FLOOR="${MECH_T_FLOOR:-293.15}"
# Abaqus-parity: automatic increment cutback (2,4,8 substeps) when a
# mechanics Newton solve stalls; substeps are pure continuation, final
# substep solves the exact original problem. 0 disables.
MECH_MAX_CUTS="${MECH_MAX_CUTS:-3}"
# DEVIATION (documented): j2 tangent/residual mismatch leaves a Newton stall
# floor that wanders up to ~2e-5 even at 8 cutback substeps; 5e-5 keeps 2.4x
# margin above it and is still well inside engineering stress accuracy
# (solver help: 1e-6-scale is already plenty). Measured 2026-07-17.
MECH_REL_TOL="${MECH_REL_TOL:-5e-5}"
# P0 (2026-07-22): Abaqus/Standard-style dual acceptance criteria (force
# residual vs increment out-of-balance scale + displacement correction +
# linear-convergence fallback) - the reference-parity criteria under which
# both the j2 stall floor and the B-bar/powder stall states count as
# converged (experiments/solver/ABAQUS_SOLVER_NOTES.md). Set
# MECH_ACCEPTANCE=legacy for the pre-P0 single-residual behavior.
MECH_ACCEPTANCE="${MECH_ACCEPTANCE:-abaqus}"
# G1 fix (validated 2026-07-17, lump3L vs cutback3L): TET4 vertex-quadrature
# capacitance lumping - Abaqus first-order heat-transfer element behavior.
# Kills activation undershoot exactly (T_min == plate temperature bitwise),
# constrained_valid gate passes, ~2.4x faster steps, cutback engagements
# 9 -> 1 on the 3-layer smoke. Set THERMAL_LUMPING="" to disable.
THERMAL_LUMPING="${THERMAL_LUMPING:-1}"

# ELEMENT_TYPE=c3d8 (default): reference-parity hexes with B-bar; c3d4 =
# legacy volumetric-locking comparison arm (mesh selection further below).
ELEMENT_TYPE="${ELEMENT_TYPE:-c3d8}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/kaess_p2_T${PLATE_TEMP_C}C_${POWER_TAG}_${ELEMENT_TYPE}_${RUN_ID}}"
RUN_LABEL="${RUN_LABEL:-kaess-2023-phase2-T${PLATE_TEMP_C}C-${POWER_TAG}-${ELEMENT_TYPE}}"

MATERIAL_CONFIG="${MATERIAL_CONFIG:-${WORK_ROOT}/materials/316L/ss316l_material_config_kaess.json}"
# POWDER_SOLID=1: paper-parity weak-solid powder (E=10 GPa / sigma_y=1 MPa,
# Kaess 2023 sec 2.2) - switches to the powder-filled mesh (inter-wall gaps
# meshed as a POWDER elset; lateral margins still unmeshed, documented
# deviation) and makes powder load-bearing during the build, depowdered at
# release. Default off = phase-2 legacy behavior (powder carries no load).
POWDER_SOLID="${POWDER_SOLID:-}"
# v03 auto-enables B-bar on HEX8 (Abaqus C3D8 selective reduced
# integration), curing the TET4+J2 volumetric locking diagnosed 2026-07-21.
# The c3d8 powder mesh also meshes the 0.1 mm lateral margins (48x28x22 =
# 29,568 elements, cell-for-cell the paper's mesh); the c3d4 powder mesh
# keeps its legacy gap-only fill.
if [[ -n "${POWDER_SOLID}" ]]; then
  if [[ "${ELEMENT_TYPE}" == "c3d8" ]]; then
    MESH_FILE="${MESH_FILE:-${SCRIPT_DIR}/kaess_cantilever_c3d8_powder_margin.inp}"
  else
    MESH_FILE="${MESH_FILE:-${SCRIPT_DIR}/kaess_cantilever_c3d4_powder.inp}"
  fi
  POWDER_ARGS=(--powder-elset POWDER
               --powder-solid-E "${POWDER_SOLID_E:-10e9}"
               --powder-solid-yield "${POWDER_SOLID_YIELD:-1e6}"
               --powder-solid-hardening "${POWDER_SOLID_HARDENING:-1e7}")
else
  MESH_FILE="${MESH_FILE:-${SCRIPT_DIR}/kaess_cantilever_${ELEMENT_TYPE}.inp}"
  POWDER_ARGS=()
fi
RELEASE_CELL_SET="${RELEASE_CELL_SET:-${SCRIPT_DIR}/inputs/release-cellset.json}"
REFERENCE_POWDER_MESH="${SCRIPT_DIR}/kaess_cantilever_c3d8_powder_margin.inp"
if [[ "$(realpath "${MESH_FILE}")" == "$(realpath "${REFERENCE_POWDER_MESH}")" ]]; then
  # Formal 29,568-cell reference mesh: use the exact content-addressed
  # Figure-7 partition. The JSON keeps W3 attached and removes W1/W2 only;
  # the same W3 minimal in-plane anchors persist from build through release.
  RELEASE_ARGS=(--release-anchor-mode paper_minimal_root
                --release-cell-set "${RELEASE_CELL_SET}")
else
  # Non-reference/TET diagnostic meshes have different cell identities and
  # cannot consume the formal artifact. Their box cut is never promotion
  # eligible and remains an explicitly labelled legacy comparison arm.
  RELEASE_ARGS=(--release-anchor-mode box
                --release-anchor-box 7.75e-4 9.75e-4 0 5.0e-4 -1.0e-9 1.0e-9
                --release-cut-box 0 7.0e-4 0 5.0e-4 0 2.999e-4)
  echo "kaess p2: DIAGNOSTIC_ONLY geometric release box; paper release gate is ineligible" >&2
fi
EXTRA_ARGS="${EXTRA_ARGS:-}"

for f in "${MATERIAL_CONFIG}" "${MESH_FILE}" "${RELEASE_CELL_SET}"; do
  [[ -f "$f" ]] || { echo "kaess p2: missing input: $f" >&2; exit 2; }
done

mkdir -p "$(dirname "${OUT_ROOT}")"
if ! mkdir "${OUT_ROOT}" 2>/dev/null; then
  echo "kaess p2: refusing existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

# Scan path generated per-run so PATH_ARGS (power/speed/layers) are recorded.
# Base layer arguments mirror the solver.  Trailing PATH_ARGS may explicitly
# override path-only layers for legacy truncated smoke wrappers.
PATH_FILE="${OUT_ROOT}/kaess_path.csv"
"${PYTHON_BIN}" "${SCRIPT_DIR}/make_kaess_path.py" \
  --layers "${BUILD_LAYERS}" \
  --layer-thickness "${LAYER_THICKNESS}" \
  --output "${PATH_FILE}" ${PATH_ARGS}

XRD_PROTOCOL="${OUT_ROOT}/kaess_xrd_protocol.json"
"${PYTHON_BIN}" - "$XRD_PROTOCOL" "$ROOM_TEMP_K" <<'PYEOF'
import json, sys
path, temp = sys.argv[1], float(sys.argv[2])
eye = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
proto = {
  "gauges": [{
    "center_m": [0.5e-3, 0.25e-3, 0.45e-3],
    "geometry_model": "rectangular_box",
    "direction_specimen": [1.0, 0.0, 0.0],
    "id": "kaess_beam_mid_eps_xx",
    "rotation_gauge_to_specimen": eye,
    "size_m": [0.2e-3, 0.2e-3, 0.1e-3],
  }],
  "mesh_to_specimen": {
    "registration_rms_m": 0.0, "rotation": eye,
    "scale_m_per_mesh_unit": 1.0, "translation_m": [0.0, 0.0, 0.0],
  },
  "maximum_registration_rms_fraction_of_min_gauge": 0.25,
  "measurement_temperature_k": temp,
  "minimum_material_fill_fraction": 0.95,
  "required_state": "attached_to_build_plate_before_EDM",
  "schema_version": "v06.xrd-gauges/1",
  "temperature_tolerance_k": 2.0,
}
json.dump(proto, open(path, "w"), indent=1)
PYEOF

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
  --xla-linear-solver "${LINEAR_SOLVER}"
  --build-axis z
  --base-side min
  --layer-thickness "${LAYER_THICKNESS}"
  --layers "${BUILD_LAYERS}"
  --support-thickness 3.0e-4
  --path-file "${PATH_FILE}"
  --path-length-scale 1.0
  --beam-radius 5.0e-5
  --source-depth 5.0e-5
  --dt 3.0e-5
  --layer-activation-mode layer_on_scan
  --layer-activation-geometry intersection
  --future-layer-mode void
  --active-window-below-layers 10
  --inactive-mass-factor 1.0
  --powder-mode powder
  --surface-selection exterior
  --boundary-tol 1.0e-6
  --quadrature-order 2
  --ambient "${PLATE_TEMP_K}"
  --preheat-temperature "${PLATE_TEMP_K}"
  --bottom-thermal-bc fixed
  --bottom-temperature "${PLATE_TEMP_K}"
  --stress-relaxation-temperature "${RELAX_TEMP_K}"
  --reset-activation-temperature
  --activation-reset-temperature "${PLATE_TEMP_K}"
  --recoat-time "${RECOAT_TIME}"
  --recoat-steps "${RECOAT_STEPS}"
  --cooling-steps "${COOLING_STEPS}" --cooling-dt "${COOLING_DT}"
  --final-cooldown-temperature "${ROOM_TEMP_K}"
  --mechanics-model j2_plastic
  # Paper Section 2.3 minimal-rigid-body bottom restraint. The exact
  # in-plane nodes are unpublished; min_min is the frozen primary variant.
  --bottom-mechanics-bc paper_minimal
  --paper-minimal-anchor-corner min_min
  --mechanics-every "${MECH_EVERY}"
  --mechanics-rel-tol "${MECH_REL_TOL}"
  --mechanics-acceptance "${MECH_ACCEPTANCE}"
  --mechanics-max-iter 50
  --mechanics-line-search
  ${MECH_T_FLOOR:+--mechanics-temperature-floor "${MECH_T_FLOOR}"}
  --mechanics-max-cuts "${MECH_MAX_CUTS}"
  ${THERMAL_LUMPING:+--thermal-mass-lumping}
  ${POWDER_ARGS[@]+"${POWDER_ARGS[@]}"}
  --release-after-cooling
  "${RELEASE_ARGS[@]}"
  --thermal-output-every 200
  --mechanics-output-every 200
  --summary-every 100
  --no-reset-plastic-on-melt
  --phase-history-model paper_irreversible
)
printf '%q ' "${SOLVER_CMD[@]}" ${EXTRA_ARGS} > "${OUT_ROOT}/solver_command.txt"
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
    echo "kaess p2 failed; forensic manifest at ${OUT_ROOT}" >&2
    exit "${prior_status}"
  fi
  if (( manifest_status != 0 )); then
    echo "kaess p2 artifacts incomplete; inspect ${OUT_ROOT}/v06_manifest.json" >&2
    exit "${manifest_status}"
  fi
  echo "kaess p2 complete: ${OUT_ROOT}"
  exit 0
}
trap finalize_manifest EXIT

"${SOLVER_CMD[@]}" ${EXTRA_ARGS}

LAST_COOLING_VTU="$(ls "${OUT_ROOT}"/step_*_cooling.vtu 2>/dev/null | sort | tail -1)"

# NOTE: no --source-free-upper-bound here (laser is ON); undershoot and
# invariant gates still apply inside run_audit.
"${PYTHON_BIN}" -m jax_fem_am.verification.run_audit "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_run_audit.json" \
  --ambient "${ROOM_TEMP_K}" \
  --quality-threshold 0.05

# Output name must be xrd_operator_smoke.json: that is the artifact role
# filename jax_fem_am.verification.provenance requires for a complete claim. Best-effort: a
# measurement-operator failure (e.g. gauge volume in void on truncated
# smoke runs) must not abort the remaining verification artifacts; the
# manifest then simply reports the missing XRD artifact.
if [[ -n "${LAST_COOLING_VTU}" ]]; then
  set +e
  "${PYTHON_BIN}" -m jax_fem_am.verification.xrd_vtu \
    --vtu "${LAST_COOLING_VTU}" \
    --protocol "${XRD_PROTOCOL}" \
    --quality-threshold 0.05 \
    --output "${OUT_ROOT}/xrd_operator_smoke.json"
  XRD_RC=$?
  set -e
  if (( XRD_RC != 0 )); then
    echo "kaess p2 WARNING: xrd_vtu operator failed (rc=${XRD_RC}); continuing" >&2
  fi
fi

"${PYTHON_BIN}" -m jax_fem_am.verification.response_gate \
  --run-dir "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_response_gate.json"
