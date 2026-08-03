#!/usr/bin/env bash
# ============================================================================
# AMB2018-01 L0 pipeline run - REAL-CONDITION WIRING (rev 4, 2026-07-31).
#
# rev 4 correction (run-6 post-mortem): without --powder-elset the
# layer-plane activation + macro consolidation solidified ALL 59150
# build cells, including the 43810 POWDER cells (leg gaps + margin,
# 74 % of the build volume) - the bridge fused into a slab. With
# --powder-elset POWDER those cells are excluded from activation every
# step (stepper.py permanent-powder masks; events.py non_fixture derives
# from active), stay STATE_POWDER, and keep powder-conductivity thermal
# participation - the real recoat physics. Weak-solid mechanics
# regularization (E 10 GPa / yield 1 MPa / hardening 10 MPa) is the
# Kaess-golden-validated numerical treatment, not material data.
#
# rev 3 corrections (run-5 post-mortem):
#   * --scan-steps-per-layer is PER HATCH LINE (log: hatch=1/4 scan=1/8 ..
#     8/8, then hatch=2/4 scan=1/8). rev 2's 4 lines x 8 steps x 20 s
#     = 640 s scan deposited 4x the intended layer energy (77.4 kJ vs
#     19.3 kJ). Fix: laser power 195 -> 48.75 W (/4) conserves the D-10
#     per-layer absorbed energy over the 640 s scan phase; instantaneous
#     power is an aggregation construct at dt=20 s lumping anyway (peaks
#     are sub-grid at part scale; consolidation is activation-driven).
#   * dwell 125 -> 101 steps so the layer clock stays 640+2020 = 2660 s.
#   * mechanics Newton diverged at step ~10 under the 4x overfeed with
#     the DEFAULT tolerances (1e-9/1e-11 = the known j2 stagnation-floor
#     trap). Wire the Kaess-golden-validated acceptance quartet:
#     rel-tol 5e-5 / abaqus acceptance / max-iter 50 / line search.
#   * --scan-speed set to the actual effective value 0.075 m / 160 s
#     (the runner steps the beam by line-length/steps, not by speed).
#
# Physics mapping (every number traces to the ledger / D-10 schedule):
#
#   DEPOSITION   macro consolidation-on-activation (the solver's documented
#                part-scale branch: liquidus == solidus disables the melt-
#                detection path that can never trigger at lumped resolution).
#                New material is born at plate temperature (Kaess G1
#                convention) and is heated by the REAL laser energy below.
#   T_ref        anchored ONCE at --stress-relaxation-temperature for each
#                consolidated point (Denlinger stress-free-above-T_cut
#                semantics == D-11 T_cut; 1273.15 K = the 1000 C literature
#                precedent, swept later by D-11). legacy_reset is the branch
#                that implements this anchor; in macro mode nothing remelts,
#                so T_ref is never rewritten and eqp is never erased.
#   ENERGY       real per-layer absorbed energy: 48.75 W x 0.62 over the
#                640 s scan phase = 19.34 kJ per computational layer
#                == 195 W x 0.62 x 160 s (the D-10 aggregate laser-on
#                time for 50 physical legs layers is 152 s). Raster: 4
#                serpentine lines over the part footprint x 0-75 mm,
#                y +-2.5 mm; beam 2 mm (mesh scale), depth 1 mm.
#   CLOCK        2660 s per computational layer (640 scan + 2020 dwell)
#                ~= D-10 N=50 aggregate (legs 2608 s, bridge 2740 s);
#                uniform because per-layer clocks need a runner extension.
#   LATENT       0 in macro mode: every point melts AND solidifies inside
#                one lumped window, so latent heat round-trips to zero net.
#
# Registered L0 deviations (uniform scan time under-feeds the bridge band
# ~-29 %, comp-layer 13 is 24 ridge layers but gets the uniform clock,
# total absorbed 251 kJ vs ~289 kJ real = -13 %): all fixed at L1 via
# per-layer schedule support.
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
# XLA_PLATFORM=gpu PYTHON_BIN=.../jax-fem-gpu/bin/python bash run_l0.sh
# switches the run to the GPU environment; default cpu = validated config.
XLA_PLATFORM="${XLA_PLATFORM:-cpu}"
export JAX_PLATFORM_NAME="${XLA_PLATFORM}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

PLATE_K=347.05   # A.3/D-07: substrate underside Dirichlet 73.9 C

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m jax_fem_am.simulation.runner \
  --config "${CONFIG}" \
  --inp "${MESH}" \
  --mesh-length-scale 1.0e-3 \
  --output-dir "${OUT_ROOT}" \
  --profile-json "${OUT_ROOT}/profile.json" \
  --profile-label "amb-l0-rev2-${RUN_ID}" \
  --xla-platform "${XLA_PLATFORM}" \
  --xla-preallocate off \
  --xla-linear-solver pardiso \
  --build-axis z \
  --base-side min \
  --support-thickness 12.7e-3 \
  --layers 13 \
  --layer-thickness 1.0e-3 \
  --layer-activation-mode layer_on_scan \
  --layer-activation-geometry centroid \
  --future-layer-mode void \
  --powder-mode powder \
  --reset-activation-temperature \
  --activation-reset-temperature "${PLATE_K}" \
  --powder-elset POWDER \
  --powder-solid-E 10e9 \
  --powder-solid-yield 1e6 \
  --powder-solid-hardening 1e7 \
  --solidus-temperature 1552.0 \
  --liquidus-temperature 1552.0 \
  --latent-heat 0.0 \
  --phase-history-model legacy_reset \
  --stress-relaxation-temperature 1273.15 \
  --laser-power 48.75 \
  --source-model legacy \
  --beam-radius 2.0e-3 \
  --source-depth 1.0e-3 \
  --scan-start 0.0 \
  --scan-end 0.075 \
  --hatch-start -0.0025 \
  --hatch-end 0.0025 \
  --hatch-lines-per-layer 4 \
  --serpentine \
  --scan-speed 4.6875e-4 \
  --scan-steps-per-layer 8 \
  --dt 20.0 \
  --dwell-steps-between-layers 101 \
  --preheat-temperature "${PLATE_K}" \
  --bottom-temperature "${PLATE_K}" \
  --bottom-thermal-bc fixed \
  --ambient "${PLATE_K}" \
  --surface-selection exterior \
  --boundary-tol 1.0e-6 \
  --quadrature-order 2 \
  --mechanics-every 16 \
  --mechanics-rel-tol 5e-5 \
  --mechanics-acceptance abaqus \
  --mechanics-max-iter 50 \
  --mechanics-line-search \
  --mechanics-max-cuts 3 \
  --bottom-mechanics-bc fixed \
  --cooling-steps 120 \
  --cooling-dt 30.0 \
  "$@" 2>&1 | tee "${OUT_ROOT}/run.log"
