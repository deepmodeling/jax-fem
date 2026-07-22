#!/usr/bin/env bash
# ============================================================================
# Kaess 2023 phase-2 overnight queue: moving-heat-source runs at 4 plate
# temperatures spanning the Fig 9a ladder (150 = standard, then 450/900/20).
# Designed to be launched DETACHED inside WSL (survives session loss):
#
#   nohup setsid bash cases/kaess_2023/run_kaess_p2_overnight.sh \
#     > /home/user/work/159/output/kaess_p2_overnight.log 2>&1 &
#
# Progress:  /home/user/work/159/output/kaess_p2_overnight_<ID>/queue_status.tsv
# Each point ~2.2 h -> full queue ~9 h.
# 定位标注:对比实验(数值对标),非实验验证。
# ============================================================================
set -u

if [ -f /home/user/miniforge3/etc/profile.d/conda.sh ]; then
  source /home/user/miniforge3/etc/profile.d/conda.sh
  conda activate jax-fem-env
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-/home/user/work/159}"
QUEUE_ID="${QUEUE_ID:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${WORK_ROOT}/output/kaess_p2_overnight_${QUEUE_ID}"
STATUS="${QUEUE_ROOT}/queue_status.tsv"
PLATE_TEMPS_C="${PLATE_TEMPS_C:-150 450 900 20}"

mkdir -p "${QUEUE_ROOT}"
printf "plate_c\tstatus\texit\twall_s\tout_dir\tfinished_at\n" > "${STATUS}"
echo "=== kaess phase-2 overnight queue ${QUEUE_ID}: [${PLATE_TEMPS_C}] ==="
echo "host pid $$ started $(date -Is)"

for TC in ${PLATE_TEMPS_C}; do
  OUT="${QUEUE_ROOT}/T${TC}C"
  echo "--- $(date -Is) plate ${TC} C -> ${OUT}"
  T0=$(date +%s)
  PLATE_TEMP_C="${TC}" RUN_ID="ovn${TC}_${QUEUE_ID}" OUT_ROOT="${OUT}" \
    bash "${SCRIPT_DIR}/run_kaess_phase2.sh" \
    > "${QUEUE_ROOT}/T${TC}C_console.log" 2>&1
  RC=$?
  WALL=$(( $(date +%s) - T0 ))
  if [ "${RC}" -eq 0 ] && [ -f "${OUT}/release.vtu" ]; then
    ST=complete
  elif [ -f "${OUT}/release.vtu" ]; then
    ST=gated
  else
    ST=FAILED
    tail -5 "${QUEUE_ROOT}/T${TC}C_console.log" || true
  fi
  printf "%s\t%s\t%d\t%d\t%s\t%s\n" "${TC}" "${ST}" "${RC}" "${WALL}" "${OUT}" "$(date -Is)" >> "${STATUS}"
done

echo "=== queue finished $(date -Is) ==="
cat "${STATUS}"
python "${SCRIPT_DIR}/analyze_kaess.py" "${QUEUE_ROOT}"/T*C \
  --json "${QUEUE_ROOT}/queue_summary.json" \
  | tee "${QUEUE_ROOT}/queue_summary.txt"
touch "${QUEUE_ROOT}/QUEUE_DONE"
