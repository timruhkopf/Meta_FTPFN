#!/bin/bash
# Example: schedule the long PFN run via the standard Unix `at` one-shot
# scheduler, as an alternative to run_scheduled_long.sh's sleep+nohup
# approach (which is what's actually running tonight, 2026-09-09 -- see
# below for why). Run this LOCALLY; it submits the job to ulysses over ssh.
#
# Prerequisite on ulysses (checked 2026-09-09: NOT currently met there --
# `dpkg -l at` finds nothing, `systemctl status atd` reports "could not be
# found"):
#   ssh ulysses 'sudo apt-get install -y at && sudo systemctl enable --now atd'
#
# Why tonight's actual run doesn't use this: `at` needs the `atd` daemon
# installed and running, which ulysses doesn't have and installing it
# needs sudo -- not worth doing interactively for a one-off. The
# sleep+nohup script (run_scheduled_long.sh) needs nothing beyond bash, so
# it was the pragmatic choice for *this* run. Once atd is installed, `at`
# is the more standard tool for future one-shot schedules: it survives
# reboots (persisted by atd, not tied to a live process the way
# sleep+nohup is) and shows up in normal job-scheduling tooling (`atq`).
#
# Simpler than run_scheduled_long.sh on purpose: no GPU-busy-wait loop, no
# lockfile guard against a double-submit. Bolt those on around REMOTE_CMD
# below if reusing this for a real unattended run alongside someone else's
# job, the way tonight's actual run needed to.
#
# Usage:
#   scripts/uly_schedule_long_run_at.sh              # defaults to "01:00" (next occurrence)
#   scripts/uly_schedule_long_run_at.sh ulysses "01:00 tomorrow"
set -euo pipefail

REMOTE_HOST="${1:-ulysses}"
AT_TIME="${2:-01:00}"
REPO='$HOME/PycharmProjects/Meta_FTPFN'

REMOTE_CMD="cd ${REPO} && rm -f models/pfn_variable_dim5_long.pt && \
uv run python -m ppfn.pipelines.train_pfn experiment=pfn_variable_dim5_long allow_dirty=true \
  > ${REPO}/logs/pfn_variable_dim5_long.log 2>&1"

echo "submitting to ulysses's atd, firing at: ${AT_TIME}"
ssh "$REMOTE_HOST" "echo '${REMOTE_CMD}' | at ${AT_TIME}"

echo
echo "check it landed with:"
echo "  ssh ${REMOTE_HOST} atq                 # lists pending at-jobs (id, time, queue)"
echo "  ssh ${REMOTE_HOST} at -c <job-id>       # prints the exact command that will run"
echo "to cancel it:"
echo "  ssh ${REMOTE_HOST} atrm <job-id>"
