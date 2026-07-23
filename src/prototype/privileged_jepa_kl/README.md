# Domain-shift JEPA/PFN skeleton

Implements the architecture from the design discussion:

- **Prior (`prior.py`)**: a shared random-MLP latent function `f` on `R^{d_z}`,
  observed through two independently-sampled invertible affine domain views
  `phi_A`, `phi_B`. `A_tr`/`A_test` and `B_tr` locations are drawn
  independently (no index correspondence). `B_in_A` reuses `B_tr`'s exact
  latent locations but pushes them through `phi_A` instead of `phi_B` — the
  privileged, domain-shift-free oracle view. Real Gaussian observation noise
  (`noise_std_range`) gives the teacher's NLL a genuine floor.

- **Model (`model.py`)**:
  - `SetEncoder` (`E`): shared self-attention context encoder, domain
    identity injected as a learned per-token embedding.
  - Student: `Z_A = E(A_tr, domain=A)`, `Z_B = E(B_tr, domain=B)` — separate
    passes, no fusion.
  - Teacher: `Z_c = E([A_tr, B_in_A], domain=A)` — single joint pass,
    single domain by construction.
  - `Predictor` (`P`): N-layer **joint** cross-attention decoder stack
    (not sequential refine-then-predict) with residual connections, so the
    query's original frame (`PE(A_test)`, i.e. `x_A_test` only, no `y`) is
    never fully overwritten by attending into one context group.
  - `BinnedHead` (`D`): shared decode head, both branches produce logits
    over the same fixed `y`-bins.

- **Training (`train.py`)**: `L = NLL(teacher) + alpha * KL(sg(p_teach) || p_stud) + beta * NLL(student)`.
  Includes context dropout on `B_in_A` for the teacher (`b_in_a_dropout`)
  and per-layer attention-mass logging split by context group, for
  monitoring the "forgetting Z_A" failure mode.

## Run

```
pip install torch --break-system-packages
python3 train.py
```

## Known open issues / what to check next (from the design discussion)

1. **Teacher degeneracy**: watch `teach_ent` over training. If it heads to
   ~0 rather than plateauing near the noise floor, increase
   `noise_std_range` in the prior, or raise `b_in_a_dropout`, or add label
   smoothing in `kl_teacher_student`.
2. **Ablation to run**: compare this teacher against an "A-only" teacher
   (drop `B_in_A` entirely) to confirm the oracle context is actually
   earning its keep, i.e. `L_nll_teach` is meaningfully lower with it than
   without.
3. **Attention collapse**: watch the logged `[Z_A, Z_B]` attention split.
   If it saturates near `[1, 0]` or `[0, 1]` early and stays there
   regardless of task difficulty, the joint predictor may not be using
   both context groups — consider an entropy regularizer on the split.
4. **KL direction**: currently reverse KL (mode-seeking, penalizes student
   under-confidence more than over-confidence). If teacher overconfidence
   becomes a problem per point 1, try forward KL or a symmetric variant.
5. **Cycle consistency**: not implemented here. If you add a "test point
   in B" notion, a round-trip consistency loss between an A-query
   attending into `Z_B` and a B-query attending into `Z_A` could further
   regularize the alignment away from shortcut solutions.
6. **Shared predictor weights** serve both the fused (`[Z_c]`) and unfused
   (`[Z_A, Z_B]`) regimes. If teacher-path gradients dominate early
   training (likely, since teacher NLL is a cleaner, more direct signal),
   consider a warmup schedule on `alpha` rather than a fixed value.
