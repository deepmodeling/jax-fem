#!/usr/bin/env bash
# ============================================================================
# Kaess 2023 benchmark: build-plate preheat ladder (paper Table 3 / Fig 8a,9a)
# 20, 150, 300, 450, 600, 750, 900 C - trend anchor: residual stress and
# deflection decrease with preheat, pronounced from 450 C.
#
# 定位标注:对比实验(数值对标),非实验验证。
# NOTE: with the flash phase-1 driver the ladder shape is directly shaped by
# RELAX_TEMP_K (stress-relaxation knob) - this ladder doubles as its
# calibration against the reference curve once Fig 9a is digitized.
#
# Usage:
#   bash run_kaess_preheat_ladder.sh            # phase 1 (flash) ladder
#   PHASE=2 bash run_kaess_preheat_ladder.sh    # phase 2 (moving source)
#   PLATE_TEMPS_C="150 450 900" bash run_kaess_preheat_ladder.sh
# ============================================================================
set -u

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/home/user/work/159}"
PHASE="${PHASE:-1}"
PLATE_TEMPS_C="${PLATE_TEMPS_C:-20 150 300 450 600 750 900}"
LADDER_ID="${LADDER_ID:-$(date +%Y%m%d_%H%M%S)}"
LADDER_ROOT="${WORK_ROOT}/output/kaess_preheat_p${PHASE}_${LADDER_ID}"
MANIFEST="${LADDER_ROOT}/ladder_manifest.tsv"
RUNNER="${SCRIPT_DIR}/run_kaess_phase${PHASE}.sh"

mkdir -p "${LADDER_ROOT}"
printf "plate_c\tstatus\texit\twall_s\tout_dir\n" > "${MANIFEST}"
echo "=== kaess preheat ladder (phase ${PHASE}) ${LADDER_ID}: [${PLATE_TEMPS_C}] ==="

for TC in ${PLATE_TEMPS_C}; do
  OUT="${LADDER_ROOT}/T${TC}C"
  echo "--- plate ${TC} C -> ${OUT}"
  T0=$(date +%s)
  PLATE_TEMP_C="${TC}" RUN_ID="lad${TC}_${LADDER_ID}" OUT_ROOT="${OUT}" \
    bash "${RUNNER}" > "${LADDER_ROOT}/T${TC}C_console.log" 2>&1
  RC=$?
  WALL=$(( $(date +%s) - T0 ))
  if [ "${RC}" -eq 0 ] && [ -f "${OUT}/release.vtu" ]; then
    STATUS=complete
  elif [ -f "${OUT}/release.vtu" ]; then
    # solver + release finished but a verification gate rejected the run
    # (expected until the G1 undershoot fix: thermal below-ambient gate).
    STATUS=gated
  else
    STATUS=FAILED
    tail -4 "${LADDER_ROOT}/T${TC}C_console.log" || true
  fi
  printf "%s\t%s\t%d\t%d\t%s\n" "${TC}" "${STATUS}" "${RC}" "${WALL}" "${OUT}" >> "${MANIFEST}"
done

echo "=== ladder finished ==="
cat "${MANIFEST}"

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi
python "${SCRIPT_DIR}/analyze_kaess.py" "${LADDER_ROOT}"/T*C \
  --json "${LADDER_ROOT}/ladder_summary.json" \
  | tee "${LADDER_ROOT}/ladder_summary.txt"
