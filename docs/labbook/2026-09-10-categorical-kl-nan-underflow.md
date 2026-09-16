# Categorical KL NaN from float32 softmax underflow

commit: pending

## What was investigated

After the OOM fix ([prior entry](2026-09-10-arch-verification-oom-cloud-size.md))
let `arch_verification` train stably (15 clean epochs, ~113s/epoch, losses
and the three-way NLL comparison all moving sensibly), the run crashed:

```
FloatingPointError: non-finite loss at global_step=7851: nan
```

`RegistrationTrainer._train_step` only checks the *total* loss for
finiteness, so this didn't immediately say which term went NaN. No
diagnostic dump exists of the exact logits at the crash step, so the
following is a well-reasoned diagnosis from code inspection and a
reproduced pathological case, not a captured trace of the actual failure.

## Root cause

`_categorical_kl` (duplicated identically in `ppfn.loss.arch_verification_loss`
and `ppfn.loss.registration_loss`) computes, per token:

```python
log_p_t = torch.log_softmax(teacher_logits, dim=-1)
log_p_s = torch.log_softmax(student_logits, dim=-1)
p_t = log_p_t.exp()
return (p_t * (log_p_t - log_p_s)).sum(-1)
```

Two float32 underflow paths both produce `0 * inf`-shaped NaNs:

1. If the student's logits are spread wide enough, `softmax` can round a
   bin's probability to exactly 0, making `log_p_s == -inf` there. If the
   teacher assigns that same bin non-negligible mass (`p_t > 0`), the term
   becomes `p_t * (finite - (-inf)) = p_t * inf = inf` — and if the
   teacher's mass there is *also* underflowed to exactly 0 (a plausible
   correlated failure, since teacher and student share the same decoder
   weights in this self-distillation setup), it's `0 * inf = nan`.
2. Symmetrically, if the *teacher* underflows a bin to `p_t == 0` while
   `log_p_t` there is `-inf`, and the student's `log_p_s` at that bin is
   finite, the term is `0 * (-inf - finite) = 0 * -inf = nan` directly,
   even without any student-side underflow.

Reproduced in isolation (not from the actual crash's logits, since those
weren't captured — this just confirms the mechanism is real and that the
fix addresses it):

```python
teacher = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
student = torch.tensor([[-200.0, -200.0, 200.0, -200.0]])  # underflows to exactly 0/1
_categorical_kl(teacher, student)  # -> nan, pre-fix
```

`arch_verification` survived 7851 steps before hitting this because
`step4_pathway` (same underlying `_categorical_kl`) never ran that long —
it OOM'd within its first 500 steps (see the OOM entry), so this failure
mode had never had the chance to surface before. Not a new bug introduced
by anything in this session; a pre-existing latent fragility exposed by
finally training long enough to hit an unlucky batch.

## Fix

```python
log_p_s = torch.log_softmax(student_logits, dim=-1).clamp_min(-50.0)
p_t = log_p_t.exp()
term = p_t * (log_p_t - log_p_s)
term = torch.where(p_t > 0, term, torch.zeros_like(term))
```

- Clamping `log_p_s` at -50 (≈ 2e-22 probability, far below any bin that
  matters to the sum) turns failure mode 1's `inf` into a large-but-finite
  penalty instead of removing the signal.
- The `torch.where` guard makes failure mode 2 exact rather than relying on
  IEEE `0 * -inf = nan`: a bin the teacher assigns zero mass to
  contributes zero KL, by the standard `0 log(0/q) := 0` convention,
  regardless of what the student's log-prob is there.

Verified both pathological directions (student underflow, teacher
underflow) now return a finite value; full test suite and both loss
modules' own `__main__` demos still pass/run unchanged. Applied identically
to `registration_loss.py`'s copy of the same helper, since it has the exact
same vulnerability and will eventually run long enough to hit it too.

**Not yet relaunched/verified against a real recurrence** — the user asked
to hold off launching until this is reviewed. Whoever resumes training
should watch for another `FloatingPointError` past step ~7851 as the
remaining check that this was actually the cause (as opposed to, say, a
learning-rate or gradient-scale issue that happens to correlate with this
code path).
