#!/bin/bash
# Inspect a running v05 solver process: args, output dir, progress, ETA.
PID="${1:?usage: inspect_run.sh <pid>}"
tr '\0' '\n' < "/proc/$PID/cmdline" | grep -E "path_stride|output/" | head -4
OUT=$(tr '\0' '\n' < "/proc/$PID/cmdline" | grep -A1 -x -- "--output-dir" | tail -1)
echo "output dir: $OUT"
LOG="$OUT/stdout.log"
if [ ! -f "$LOG" ]; then
  LOG=$(ls "$OUT"/*.log 2>/dev/null | head -1)
fi
echo "log: $LOG"
if [ -f "$LOG" ]; then
  grep "global_step=" "$LOG" | tail -1 | cut -c1-160
else
  echo "(no log file; run may write stdout to the user's terminal)"
fi
ls "$OUT" 2>/dev/null | tail -3
ELAPSED=$(ps -o etimes= -p "$PID" | tr -d ' ')
echo "elapsed: ${ELAPSED}s"
