#!/usr/bin/env bash
# ============================================================================
# Kaess 2023 code-to-code benchmark - Phase 1: pipeline connectivity
# (macro direct-solidify flash mode, NO moving heat source yet).
#
# 定位标注:对比实验(数值对标),非实验验证。
# See 159_local/v06/validation/cases/kaess_2023_benchmark_plan.md.
#
# Geometry: 40x20x22-hex-derived TET4 cantilever (make_kaess_mesh.py),
#   beam 1.0x0.5x0.3 mm on a 0.3 mm support block, units meters, build axis z.
# Release: partial saw cut - support nodes with x <= ROOT_X_M stay clamped
#   (--release-anchor-mode box), the rest of the part springs free.
#   ROOT_X_M is an INFERRED assumption until Kaess Fig 3/7 are digitized.
#
# Usage:
#   bash run_kaess_phase1.sh                          # standard 150 C plate
#   PLATE_TEMP_C=450 bash run_kaess_phase1.sh         # preheat ladder point
#   EXTRA_ARGS="--layers 3" bash run_kaess_phase1.sh  # quick smoke
# ============================================================================
set -euo pipefail

if ! command -v python >/dev/null 2>&1 && [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
XLA_PLATFORM="${XLA_PLATFORM:-cpu}"
LINEAR_SOLVER="${LINEAR_SOLVER:-pardiso}"

PLATE_TEMP_C="${PLATE_TEMP_C:-150}"
PLATE_TEMP_K="$(${PYTHON_BIN} -c "print(273.15 + ${PLATE_TEMP_C})")"
ROOM_TEMP_K="${ROOM_TEMP_K:-293.15}"      # final plate cooldown target (paper 2.3)
# Reference model has NO stress relaxation/annealing mechanism -> disabled (0)
# for code-to-code faithfulness. Set >0 to reactivate the v06 knob.
RELAX_TEMP_K="${RELAX_TEMP_K:-0}"
RESET_TEMP_K="${RESET_TEMP_K:-1643.15}"   # solidus: fresh layer enters stress-free

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${WORK_ROOT}/output/kaess_p1_T${PLATE_TEMP_C}C_${RUN_ID}}"
RUN_LABEL="kaess-2023-phase1-T${PLATE_TEMP_C}C"

MATERIAL_CONFIG="${MATERIAL_CONFIG:-${WORK_ROOT}/materials/316L/ss316l_material_config_kaess.json}"
MESH_FILE="${MESH_FILE:-${SCRIPT_DIR}/kaess_cantilever_c3d4.inp}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

for f in "${MATERIAL_CONFIG}" "${MESH_FILE}"; do
  [[ -f "$f" ]] || { echo "kaess p1: missing input: $f" >&2; exit 2; }
done

mkdir -p "$(dirname "${OUT_ROOT}")"
if ! mkdir "${OUT_ROOT}" 2>/dev/null; then
  echo "kaess p1: refusing existing OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

# XRD gauge protocol: box in the beam mid-span, measured at plate temperature
# (attached state, after final cooling to the fixed-bottom plate temperature).
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

export PYTHONPATH="${REPO_ROOT}/159_local/v01:${REPO_ROOT}/159_local:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"

cd "${WORK_ROOT}"
SOLVER_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/159_local/v06/driver.py"
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
  --layer-thickness 3.0e-5
  --layers 10
  --support-thickness 3.0e-4
  --scan-steps-per-layer 1
  --hatch-lines-per-layer 1
  --laser-power 0
  --dt 5.0e-3
  --layer-activation-mode layer_on_scan
  --layer-activation-geometry intersection
  --future-layer-mode void
  --active-window-below-layers 10
  --inactive-mass-factor 1.0
  --powder-mode powder
  --surface-selection exterior
  --boundary-tol 1.0e-6
  --quadrature-order 2
  --solidus-temperature 0 --liquidus-temperature 0 --latent-heat 0
  --ambient "${PLATE_TEMP_K}"
  --preheat-temperature "${PLATE_TEMP_K}"
  --bottom-thermal-bc fixed
  --bottom-temperature "${PLATE_TEMP_K}"
  --stress-relaxation-temperature "${RELAX_TEMP_K}"
  --reset-activation-temperature
  --activation-reset-temperature "${RESET_TEMP_K}"
  --recoat-time 10.0
  --recoat-steps 10
  --cooling-steps 30 --cooling-dt 1.0
  --final-cooldown-temperature "${ROOM_TEMP_K}"
  --mechanics-model j2_plastic
  # bottom clamp: paper fixes z (+rotation) with x/y partially permitted;
  # elastic (normal-spring) foundation leaves x/y rigid modes exactly
  # singular on this mesh, so full clamp is used - documented deviation.
  --bottom-mechanics-bc fixed
  --mechanics-every 5
  --mechanics-rel-tol 1e-5
  --mechanics-max-iter 50
  --mechanics-line-search
  --release-after-cooling
  --release-anchor-mode box
  # saw cut per paper Fig 7: only the wide ROOT wall (x 0.775-0.975 mm)
  # keeps its build-plate connection...
  --release-anchor-box 7.75e-4 9.75e-4 0 5.0e-4 -1.0e-9 1.0e-9
  # ...and the sawed-off walls W1/W2 are DEACTIVATED in the release solve
  # (paper deletes them; leaving them attached releases their locked
  # stress into the beam tip - observed as a spurious plastic kink).
  --release-cut-box 0 7.0e-4 0 5.0e-4 0 2.999e-4
  --thermal-output-every 11
  --mechanics-output-every 11
  --summary-every 11
  --reset-plastic-on-melt
)
printf '%q ' "${SOLVER_CMD[@]}" ${EXTRA_ARGS} > "${OUT_ROOT}/solver_command.txt"
printf '\n' >> "${OUT_ROOT}/solver_command.txt"

finalize_manifest() {
  local prior_status=$?
  trap - EXIT
  set +e
  "${PYTHON_BIN}" -m v06.provenance \
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
    echo "kaess p1 failed; forensic manifest at ${OUT_ROOT}" >&2
    exit "${prior_status}"
  fi
  if (( manifest_status != 0 )); then
    echo "kaess p1 artifacts incomplete; inspect ${OUT_ROOT}/v06_manifest.json" >&2
    exit "${manifest_status}"
  fi
  echo "kaess p1 complete: ${OUT_ROOT}"
  exit 0
}
trap finalize_manifest EXIT

"${SOLVER_CMD[@]}" ${EXTRA_ARGS}

LAST_COOLING_VTU="$(ls "${OUT_ROOT}"/step_*_cooling.vtu 2>/dev/null | sort | tail -1)"

"${PYTHON_BIN}" -m v06.verification.run_audit "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_run_audit.json" \
  --ambient "${ROOM_TEMP_K}" \
  --quality-threshold 0.05 \
  --source-free-upper-bound "${RESET_TEMP_K}"

if [[ -n "${LAST_COOLING_VTU}" ]]; then
  "${PYTHON_BIN}" -m v06.measurement.xrd_vtu \
    --vtu "${LAST_COOLING_VTU}" \
    --protocol "${XRD_PROTOCOL}" \
    --quality-threshold 0.05 \
    --output "${OUT_ROOT}/xrd_operator_kaess.json"
fi

"${PYTHON_BIN}" -m v06.verification.response_gate \
  --run-dir "${OUT_ROOT}" \
  --output "${OUT_ROOT}/v06_response_gate.json"
