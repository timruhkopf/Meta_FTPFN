#!/bin/bash
# One-shot: sleep until the next 01:00 local time, wait for the GPU to be
# free (in case step4_pathway or anything else is still running), then
# launch the long (200k-step) PFN reference run detached. Meant to be
# started once, now, via nohup+disown -- it does all the waiting itself, so
# nothing (not even the launching ssh/Claude session) needs to stay alive
# until 1am. Idempotent guard: refuses to start a second copy.
#
# Busy-check covers process name AND GPU memory, not just utilization --
# an earlier version only matched "train_pfn" and checked utilization<5%,
# which would have missed a real job (`ppfn.pipelines.train
# experiment=step4_pathway`, ~22GB) that doesn't match that process-name
# substring and can transiently show near-zero utilization between steps.
set -uo pipefail

REPO=~/PycharmProjects/Meta_FTPFN
LOG="$REPO/logs/scheduled_long_launch.log"
TRAIN_LOG="$REPO/logs/pfn_variable_dim5_long.log"
LOCKFILE="/tmp/pfn_variable_dim5_long.lock"

mkdir -p "$REPO/logs"

if [ -e "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "$(date -Is) already running as pid $(cat "$LOCKFILE"), refusing to start a second copy" >> "$LOG"
  exit 1
fi
echo $$ > "$LOCKFILE"

log() { echo "$(date -Is) $*" >> "$LOG"; }

TARGET_EPOCH=$(date -d "tomorrow 01:00:00" +%s)
NOW_EPOCH=$(date +%s)
SLEEP_S=$(( TARGET_EPOCH - NOW_EPOCH ))
log "sleeping ${SLEEP_S}s until $(date -d "@$TARGET_EPOCH" -Is) before checking the GPU"
sleep "$SLEEP_S"

log "woke up, checking whether the GPU is free"
MAX_WAIT_S=$((60*60))   # give any other job up to 1h past 01:00 to finish
MEM_THRESHOLD_MIB=500   # below this counts as "free" -- driver/idle overhead is a few MiB
WAITED=0
while true; do
  # Any python process under ppfn.pipelines (train_pfn.py OR train.py --
  # step4_pathway runs the latter) counts as busy, not just "train_pfn".
  BUSY_PROC=$(pgrep -af "ppfn\.pipelines\." | grep -v "run_scheduled_long" || true)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
  if [ -z "$BUSY_PROC" ] && [ "${GPU_MEM:-0}" -lt "$MEM_THRESHOLD_MIB" ]; then
    log "GPU free (mem_used=${GPU_MEM}MiB, no ppfn.pipelines process) -- launching"
    break
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_S" ]; then
    log "still busy after ${MAX_WAIT_S}s of waiting (mem_used=${GPU_MEM}MiB, proc='$BUSY_PROC') -- launching anyway"
    break
  fi
  log "GPU busy (mem_used=${GPU_MEM}MiB, proc='$BUSY_PROC') -- waiting 120s (${WAITED}/${MAX_WAIT_S}s so far)"
  sleep 120
  WAITED=$((WAITED + 120))
done

cd "$REPO" || { log "FATAL: could not cd to $REPO"; rm -f "$LOCKFILE"; exit 1; }
rm -f models/pfn_variable_dim5_long.pt

log "launching pfn_variable_dim5_long (200k steps) -- see $TRAIN_LOG"
uv run python -m ppfn.pipelines.train_pfn experiment=pfn_variable_dim5_long allow_dirty=true \
  > "$TRAIN_LOG" 2>&1
STATUS=$?
log "training process exited with status $STATUS"
rm -f "$LOCKFILE"
