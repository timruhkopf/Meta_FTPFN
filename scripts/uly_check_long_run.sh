#!/bin/bash
# Status check for tonight's scheduled long PFN run (run_scheduled_long.sh).
# Three questions, in order: is the scheduler still waiting, is training
# actually running, and what has it produced so far.
set -uo pipefail
REMOTE_HOST="${1:-ulysses}"
REPO='$HOME/PycharmProjects/Meta_FTPFN'

ssh "$REMOTE_HOST" "
REPO=${REPO}
echo '--- 1) is the scheduler process alive (waiting, GPU-polling, or has it handed off to training)? ---'
if [ -f /tmp/pfn_variable_dim5_long.lock ] && kill -0 \$(cat /tmp/pfn_variable_dim5_long.lock) 2>/dev/null; then
  echo \"scheduler/launcher pid \$(cat /tmp/pfn_variable_dim5_long.lock) is alive\"
  ps -o pid,etime,cmd -p \$(cat /tmp/pfn_variable_dim5_long.lock)
else
  echo 'no live lockfile pid -- either it has not been launched, or it already exited (check launch log below for which)'
fi
echo
echo '--- 2) launcher log (sleep countdown / GPU-busy-wait decisions) ---'
tail -n 10 \"\$REPO/logs/scheduled_long_launch.log\" 2>&1
echo
echo '--- 3) is the actual training process running right now (any ppfn.pipelines job, not just ours)? ---'
pgrep -af 'ppfn\.pipelines\.' || echo 'no ppfn.pipelines process running'
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo
echo '--- 4) latest training progress (last 5 logged steps) ---'
grep -E '^step' \"\$REPO/logs/pfn_variable_dim5_long.log\" 2>/dev/null | tail -5 || echo 'no step lines yet'
echo
echo '--- 5) checkpoint on disk (updated every 10k steps -- see checkpoint_every) ---'
ls -la \"\$REPO/models/pfn_variable_dim5_long.pt\" 2>&1
"
