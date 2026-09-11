# Chain script silently "succeeded" after a CLI-breaking mid-run redeploy

commit: b968452

## What happened

While the `svm -> rf -> xgb -> nn` chain (`/tmp/hpo_warp_chain.sh` on
ulysses) was still waiting on `rf`, `run_pairs.py`'s CLI was changed to add
a required `--family` flag (to support the new `lcbench`/`taskset`
families alongside `hpobench`) and redeployed via `rsync` while the chain
was live. The reasoning at the time was "safe — an already-running Python
process keeps its old in-memory code; only a *future* subprocess picks up
the new files." True, but incomplete: the chain script's `for model in svm
rf xgb nn` loop invokes the CLI **with hardcoded old-style arguments**
(`--model "$model"`, no `--family`), so when it got to `xgb` and `nn` those
subprocesses failed instantly with an argparse usage error — and the loop
had no error check, so it printed `finished xgb`/`finished nn` and then
`all models done` anyway. A second chain script waiting on that exact log
line (`hpo_warp_chain2.sh`, correctly written *after* the CLI change, so it
used `--family lcbench` properly) started right on cue, on a false premise.

Caught by noticing `xgb`/`nn` each "finished" 2 seconds after starting —
implausible for jobs estimated at 30 min / 9 min — and checking the actual
log (`argparse` usage error) and result counts (0 shards, not the expected
380/56) before trusting the "all done" signal.

## Fix

Relaunched `xgb`/`nn` with the correct CLI (`--family hpobench --model
...`), chained to start after `lcbench` finishes rather than immediately,
so they don't compete with `lcbench`'s already-running 14-worker pool for
the same 16 cores.

## The general lesson

"An in-flight process keeps its old code" is true and was the right call
for the `rf` process itself. It does **not** extend to a static shell
script that re-invokes the CLI later in its own lifetime with arguments
baked in at write-time — that's exactly as exposed to a breaking interface
change as a fresh invocation would be. Changing a CLI's required arguments
while *any* script that calls it (not just any process currently running
it) is still pending should be treated as a breaking deploy, not a safe
one — and the failure mode here (a loop with no error check, printing
"finished" regardless of exit status) is worth generalizing: any chained
background script here should check exit codes and abort/flag on failure
rather than press on to the next stage.
