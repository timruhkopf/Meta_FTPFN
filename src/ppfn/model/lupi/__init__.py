"""LUPI-distilled alignment-aware multi-task PFN --
`docs/labbook/`'s 2026-09-14 LUPI build spec.

Distinct from `ppfn.model.registration` (the explicit-transport-head,
gated-cross-attention architecture CLAUDE.md's own invariants govern): this
model never outputs a transport estimate at all. It has two POSITIONING
MODES on one shared trunk -- student (infers alignment purely via
cross-attention against B) and oracle (the true B-frame position is injected
directly, a privileged-information signal only available at meta-training
time, spec §5.1) -- distilled together via a forward-KL term (spec §5.2).
None of CLAUDE.md's registration-model invariants (gated cross-attention,
layer-1 y-only, a transport head, etc.) apply here; this package follows the
LUPI build spec's own §4/§5 instead.
"""
