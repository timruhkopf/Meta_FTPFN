"""
Rosetta-stone POC, corrected to respect TabPFN's native interface.

No fine-tuning. No custom heads. No hidden-state hooking. Y is NOT baked
into the feature matrix -- it stays in its own y-stream, as TabPFN expects.
Missing cross-domain features are left as literal NaN, since TabPFN v2 is
trained to handle missing values in X natively; we don't need to invent our
own masking side-channel for that.

The test: does conditioning on the block-diagonal [A ; B] context change the
model's predictive NLL on held-out A_test, relative to conditioning on A
alone? A negative NLL gap (block-design NLL < baseline NLL) is direct,
teacher-free evidence that the model is extracting something useful from B
about A's task -- the deployment-time analogue of the JEPA "bending energy"
signal discussed earlier, obtained without ever needing ground-truth
alignment or a privileged teacher.
"""

import torch
from copy import deepcopy


# --------------------------------------------------------------------------- #
# 1. Feature-only block-diagonal design (Y stays separate)
# --------------------------------------------------------------------------- #
#
# def build_block_design_features(X_A, Y_A, X_B, Y_B, pad_val: float = float("nan"), b=0):
#     """
#     X-only block-diagonal design:
#       [ X_A      , NaN(F_B) ]
#       [ NaN(F_A) , X_B      ]
#     Y is returned separately, never concatenated into X.
#
#     Returns:
#       X_design: (..., N_A+N_B, F_A+F_B)
#       y_design: (..., N_A+N_B, 1)
#     """
#
#     batch_shape = X_A.shape[:-2]
#     N_A, F_A = X_A.shape[-2], X_A.shape[-1]
#     N_B, F_B = X_B.shape[-2], X_B.shape[-1]
#     device, dtype = X_A.device, X_A.dtype
#
#     pad_A_for_B = torch.full((*batch_shape, N_A, F_B), pad_val, dtype=dtype, device=device)
#     pad_B_for_A = torch.full((*batch_shape, N_B, F_A), pad_val, dtype=dtype, device=device)
#
#     row_A = torch.cat([X_A, pad_A_for_B], dim=-1)
#     row_B = torch.cat([pad_B_for_A, X_B], dim=-1)
#
#     X_design = torch.cat([row_A, row_B], dim=0)  # 0 is now T dim
#     y_design = torch.cat([Y_A, Y_B], dim=0)
#     return X_design, y_design
#
#
# def pad_test_features(X_test_A: torch.Tensor, F_B: int, pad_val: float = float("nan")):
#     """Pad A_test's features with NaN in the F_B columns it doesn't have,
#     so it matches the block design's column count."""
#     pad = torch.full((*X_test_A.shape[:-1], F_B), pad_val, dtype=X_test_A.dtype, device=X_test_A.device)
#     return torch.cat([X_test_A, pad], dim=-1)
#
def build_block_design_features(X_A, Y_A, X_B, Y_B, pad_val: float = float("nan"), add_task_id: bool = False):
    """
    X-only block-diagonal design:
      [ X_A      , NaN(F_B) ]
      [ NaN(F_A) , X_B      ]
    Y is returned separately, never concatenated into X.

    Returns:
      X_design: (..., N_A+N_B, F_A+F_B (+ 1 if add_task_id))
      y_design: (..., N_A+N_B, 1)
    """
    batch_shape = X_A.shape[:-2]
    N_A, F_A = X_A.shape[-2], X_A.shape[-1]
    N_B, F_B = X_B.shape[-2], X_B.shape[-1]
    device, dtype = X_A.device, X_A.dtype

    pad_A_for_B = torch.full((*batch_shape, N_A, F_B), pad_val, dtype=dtype, device=device)
    pad_B_for_A = torch.full((*batch_shape, N_B, F_A), pad_val, dtype=dtype, device=device)

    row_A = torch.cat([X_A, pad_A_for_B], dim=-1)
    row_B = torch.cat([pad_B_for_A, X_B], dim=-1)

    if add_task_id:
        # A = 0, B = 1
        task_id_A = torch.zeros((*batch_shape, N_A, 1), dtype=dtype, device=device)
        task_id_B = torch.ones((*batch_shape, N_B, 1), dtype=dtype, device=device)

        row_A = torch.cat([row_A, task_id_A], dim=-1)
        row_B = torch.cat([row_B, task_id_B], dim=-1)

    X_design = torch.cat([row_A, row_B], dim=0)  # 0 is now T dim
    y_design = torch.cat([Y_A, Y_B], dim=0)

    return X_design, y_design


def pad_test_features(X_test_A: torch.Tensor, F_B: int, pad_val: float = float("nan"), add_task_id: bool = False,
                      task_id: int = 0):
    """Pad A_test's features with NaN in the F_B columns it doesn't have,
    so it matches the block design's column count.

    Args:
        task_id (int): 0 for A, 1 for B. Defaults to 0 since this expects X_test_A.
    """
    pad = torch.full((*X_test_A.shape[:-1], F_B), pad_val, dtype=X_test_A.dtype, device=X_test_A.device)

    components = [X_test_A, pad]

    if add_task_id:
        # Create a column of the assigned task ID (0 or 1)
        task_id_col = torch.full((*X_test_A.shape[:-1], 1), float(task_id), dtype=X_test_A.dtype,
                                 device=X_test_A.device)
        components.append(task_id_col)

    return torch.cat(components, dim=-1)

# --------------------------------
# Alternate Mixed effect design



def build_stacked_overlay_features(X_A, Y_A, X_B, Y_B, pad_val: float = 0.0):
    """
    Stacked Overlay + Interaction design:
      [ X_A_padded , 0 , 0          ]
      [ X_B_padded , 1 , X_B_padded ]

    Y is returned separately, never concatenated into X.

    One important caveat about what this design tests

    The stacked-overlay construction bakes in an assumption that the block-diagonal design deliberately avoided:
    it assumes column i of A and column i of B play the same physical role, just under a different value transform.
    That's true for your current transform family (per-feature shift/scale is diagonal and index-preserving),
    so the design is a great fit for it — but it means this architecture is testing "can the model apply a
    per-feature correction when told which feature is which," not "can the model discover which of B's
    columns corresponds to which of A's columns." The latter — genuine correspondence discovery under permutation,
    dimensionality mismatch, or cross-dimension mixing — is the harder problem the original coordinate-alignment
    framing was aimed at, and this design doesn't exercise it at all; it can't, since same-index-same-role is wired
    into the construction.

    results in the best performance so far:
    gap1 (distorted, block): mean=-1.1962  median=-1.2014  sem=0.0624  n_inf=0/100
    gap2 (aligned, block): mean=-2.0476  median=-2.1724  sem=0.0841  n_inf=0/100
    gap3 (oracle/concat): mean=-2.2350  median=-2.9120  sem=0.1814  n_inf=0/100

    i.e. the model is able to extract a lot of information from B about A's task, even when B is distorted,
    simply because it does not relies u

    Returns:
      X_design: (..., N_A+N_B, 2*F_max + 1)
      y_design: (..., N_A+N_B, 1)
    """
    batch_shape = X_A.shape[:-2]
    N_A, F_A = X_A.shape[-2], X_A.shape[-1]
    N_B, F_B = X_B.shape[-2], X_B.shape[-1]
    device, dtype = X_A.device, X_A.dtype

    F_max = max(F_A, F_B)

    # 1. Pad raw features up to F_max so they can be stacked vertically
    if F_A < F_max:
        pad_A = torch.full((*batch_shape, N_A, F_max - F_A), pad_val, dtype=dtype, device=device)
        X_A_pad = torch.cat([X_A, pad_A], dim=-1)
    else:
        X_A_pad = X_A

    if F_B < F_max:
        pad_B = torch.full((*batch_shape, N_B, F_max - F_B), pad_val, dtype=dtype, device=device)
        X_B_pad = torch.cat([X_B, pad_B], dim=-1)
    else:
        X_B_pad = X_B

    # 2. Base Shared Representation (X_raw)
    X_raw = torch.cat([X_A_pad, X_B_pad], dim=-2)

    # 3. Domain Indicator (I_B)
    I_A = torch.zeros((*batch_shape, N_A, 1), dtype=dtype, device=device)
    I_B_ind = torch.ones((*batch_shape, N_B, 1), dtype=dtype, device=device)
    I_B_col = torch.cat([I_A, I_B_ind], dim=-2)

    # 4. Interaction Term (X_raw * I_B)
    # Built explicitly to avoid NaN * 0 = NaN propagation if pad_val is NaN
    int_A = torch.zeros((*batch_shape, N_A, F_max), dtype=dtype, device=device)
    int_B = X_B_pad.clone()
    X_int = torch.cat([int_A, int_B], dim=-2)

    # 5. Final Assembly along the feature dimension
    X_design = torch.cat([X_raw, I_B_col, X_int], dim=-1)
    y_design = torch.cat([Y_A, Y_B], dim=0)

    return X_design, y_design


def format_test_features_stacked(X_test, F_other: int, task_id: int = 0, pad_val: float = 0.0):
    """
    Format test points to match the (2*F_max + 1) stacked interaction layout.

    Args:
        X_test: Test features (..., N_test, F_test)
        F_other (int): The number of features in the *other* domain (needed to compute F_max).
        task_id (int): 0 for A, 1 for B. Defaults to 0.
    """
    batch_shape = X_test.shape[:-2]
    N_test, F_test = X_test.shape[-2], X_test.shape[-1]
    device, dtype = X_test.device, X_test.dtype

    # Determine F_max based on which domain we are testing
    F_A = F_test if task_id == 0 else F_other
    F_B = F_other if task_id == 0 else F_test
    F_max = max(F_A, F_B)

    # 1. Pad to F_max
    if F_test < F_max:
        pad = torch.full((*batch_shape, N_test, F_max - F_test), pad_val, dtype=dtype, device=device)
        X_pad = torch.cat([X_test, pad], dim=-1)
    else:
        X_pad = X_test

    # 2. Indicator
    I_col = torch.full((*batch_shape, N_test, 1), float(task_id), dtype=dtype, device=device)

    # 3. Interaction Term
    if task_id == 0:
        X_int = torch.zeros((*batch_shape, N_test, F_max), dtype=dtype, device=device)
    else:
        X_int = X_pad.clone()

    return torch.cat([X_pad, I_col, X_int], dim=-1)

# --------------------------------------------------------------------------- #
# 2. Fit + score NLL under a given context, using the native TabPFN interface
# --------------------------------------------------------------------------- #

def fit_and_score_nll(X_train, y_train, X_test, y_test_true, device="cpu"):
    """
    Fits TabPFN natively on (X_train, y_train) and returns the per-point
    predictive NLL on (X_test, y_test_true).

    NOTE -- verify against your installed tabpfn version:
      - the exact regressor entry point (TabPFNRegressor / your existing
        get_tabpfn_model wrapper),
      - the kwarg/attribute names for retrieving raw distributional logits
        and the bar-distribution criterion used to score them.
    The shape of the pipeline (fit -> get distribution -> score NLL) is
    what matters here; the specific attribute names are a one-line fix once
    you check `dir(reg)` / `dir(reg.model_)` on your release.
    """
    from tabpfn import TabPFNRegressor  # TODO verify import path for v2.5

    reg = TabPFNRegressor(device=device)
    reg.fit(X_train, y_train)

    # TODO verify: kwarg name for getting raw distributional output instead
    # of a point prediction, and the attribute holding the bar-distribution
    # criterion used internally for NLL.
    output = reg.predict(X_test, output_type="full")
    # criterion = reg.model_.criterion

    logits, criterion = output["logits"], output["criterion"]
    nll_per_point = criterion(logits, y_test_true)  # shape (N_test,)
    return nll_per_point


# --------------------------------------------------------------------------- #
# 3. The actual comparison
# --------------------------------------------------------------------------- #

def evidence_gap_poc(X_A, Y_A, X_test_A: torch.Tensor, y_test_A: torch.Tensor, build_design_fn, device="cpu", *args, **kwargs):
    """
    Returns (nll_baseline, nll_block, evidence_gap) where
      evidence_gap = nll_block - nll_baseline
    evidence_gap < 0  -> block-design context (A+B) helped predict A_test
    evidence_gap ~ 0  -> B was ignored / uninformative for this task
    evidence_gap > 0  -> negative transfer: B actively hurt A_test predictions
    """

    F_B = train["X_B"].shape[-1]

    # ---- block design: A + B jointly in context ----
    X_design, y_design = build_design_fn(X_A, Y_A, X_B, Y_B, *args, **kwargs)
    if X_design.shape[-1] != X_A.shape[-1]:  # padding, if we have block design!
        X_test_A = format_test_features_stacked(X_test_A, F_B, task_id=0, *args, **kwargs)

    nll_block = fit_and_score_nll(X_design, y_design, X_test_A, y_test_A, device=device)

    return nll_block


def build_concat_design(X_A, Y_A, X_B, Y_B, *args, **kwargs):
    y_design = torch.cat([Y_A, Y_B], dim=0)
    return torch.cat([X_A, X_B], dim=0), y_design


def summarize_gaps(name: str, gaps: list[torch.Tensor]) -> dict:
    """
    gaps: list of per-test-point NLL-gap tensors, one entry per batch item
          (each tensor may contain +/-inf if the model assigned ~0 density
          to some test point under that context).

    Reports mean/median/SEM over FINITE per-item means only, plus the
    inf failure rate as its own explicit number rather than silently
    zeroing it out (zeroing asserts "no difference for this item", which
    is a much stronger and likely false claim than "undefined here").
    """
    per_item_mean = torch.stack([g.mean() for g in gaps])  # (B,)
    finite_mask = torch.isfinite(per_item_mean)
    finite = per_item_mean[finite_mask]

    n_total = per_item_mean.numel()
    n_inf = n_total - finite.numel()

    mean = finite.mean()
    median = finite.median()
    sem = finite.std(unbiased=True) / (finite.numel() ** 0.5)

    print(
        f"{name}: mean={mean:.4f}  median={median:.4f}  "
        f"sem={sem:.4f}  n_inf={n_inf}/{n_total}"
    )
    return {
        "per_item_mean": per_item_mean,
        "finite_mask": finite_mask,
        "mean": mean,
        "median": median,
        "sem": sem,
        "n_inf": n_inf,
        "n_total": n_total,
    }


def paired_alignment_cost(gaps_aligned: list[torch.Tensor], gaps_distorted: list[torch.Tensor]) -> dict:
    """
    Paired comparison of alignment cost: for each batch item b, both
    gaps_aligned[b] and gaps_distorted[b] share the same A_train/A_test
    draw, so differencing them per-item cancels most of the item-level
    variance before averaging -- much more powerful than differencing
    two independently-computed batch means.

    alignment_cost_b = distorted_gap_b - aligned_gap_b
    (i.e. how much MORE it costs, in nats, to have to infer the shift/scale
    on top of just paying the block-diagonal-format tax)
    """
    assert len(gaps_aligned) == len(gaps_distorted)

    per_item_aligned = torch.stack([g.mean() for g in gaps_aligned])
    per_item_distorted = torch.stack([g.mean() for g in gaps_distorted])

    diff = per_item_distorted - per_item_aligned  # (B,)
    finite_mask = torch.isfinite(diff)
    finite_diff = diff[finite_mask]

    n_total = diff.numel()
    n_inf = n_total - finite_diff.numel()

    mean = finite_diff.mean()
    median = finite_diff.median()
    sem = finite_diff.std(unbiased=True) / (finite_diff.numel() ** 0.5)

    print(
        f"paired alignment cost (distorted - aligned): mean={mean:.4f}  "
        f"median={median:.4f}  sem={sem:.4f}  n_inf={n_inf}/{n_total}"
    )
    print(f"  -> {'looks real' if abs(mean) > 2 * sem else 'NOT distinguishable from noise at ~2 SEM'}")
    return {
        "per_item_diff": diff,
        "finite_mask": finite_mask,
        "mean": mean,
        "median": median,
        "sem": sem,
        "n_inf": n_inf,
        "n_total": n_total,
    }


def cross_condition_inf_report(named_gap_lists: dict[str, list[torch.Tensor]]) -> None:
    """
    For each batch item, reports which condition(s) produced an inf mean,
    to check whether inf failures are shared across conditions (probably a
    genuinely degenerate prior draw) or unique to one condition (probably a
    condition-specific numerical issue, e.g. bar-distribution border
    misadjustment from an unusual y-range in that particular context).
    """
    names = list(named_gap_lists.keys())
    per_item_means = {
        name: torch.stack([g.mean() for g in gaps]) for name, gaps in named_gap_lists.items()
    }
    n_total = next(iter(per_item_means.values())).numel()

    for b in range(n_total):
        flags = {name: bool(torch.isinf(per_item_means[name][b])) for name in names}
        if any(flags.values()):
            print(f"item {b}: " + ", ".join(f"{name}={'INF' if v else 'ok'}" for name, v in flags.items()))


if __name__ == "__main__":
    from tqdm import tqdm
    from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream

    BATCH_SIZE = 50
    N_A, N_B = 10, 50
    device = 'cpu' # FIXME: install a newer cuda driver to use GPU acceleration (due to tabpfn's torch dependency)
    prior = HarmonicMixturePrior(warp=False)
    dataset = InfiniteHarmonicsStream(prior, batch_size=BATCH_SIZE, n_A=N_A, n_B=N_B)
    data_dict = next(iter(dataset))
    train, test = data_dict['train'], data_dict['test']

    gaps1 = []
    gaps2 = []
    gaps3 = []
    for b in tqdm(range(BATCH_SIZE)):
        X_A, Y_A = train["X_A"][:, b, :], train["Y_A"][:, b, 0]

        # A. DEFAULT Query
        X_test_A, y_test_A = test["X_A"][:, b], test["Y_A"][:, b, 0]

        # B. Alternate Query formulation. Rather than asking for
        # TODO consider tracking X_B_in_A, Y_B_in_A as additional measurement. This way in a fine-tuning scenario,
        #  we can guide the model with privileged information where the information matters most!
        # it's just Approach A's end-to-end NLL with extra, well-localized query rows added during training, using
        # privileged X-locations (available only in simulation) purely to sharpen where the loss is evaluated.
        # It's the cheapest possible way to inject privileged information without touching the architecture at all,
        # and it composes cleanly with everything else here.
        """
        # This is with B in A
        (distorted, block): mean=-0.0528  median=-0.0289  sem=0.0457  n_inf=0/100
        (aligned, block): mean=-0.2515  median=-0.2044  sem=0.0413  n_inf=0/100
        (oracle/concat): mean=-3.8902  median=-3.8073  sem=0.0654  n_inf=0/100
        
        paired alignment cost (distorted - aligned): mean=0.1987  median=0.1597  sem=0.0297  n_inf=0/100
          -> looks real
        """
        # X_test_A, y_test_A = train["X_B_in_A"][:, b], train["Y_B_in_A"][:, b, 0]

        # baseline:
        # only A_test | A_train, i.e. the unconditional model
        nll_baseline = fit_and_score_nll(X_A, Y_A, X_test_A, y_test_A, device=device)


        # Experiment 1:
        # the use case, where B is distorted, and we pass it as block diag with nan tokens for padding,
        # forcing the model to infer the alignment
        X_B, Y_B = train["X_B"][:, b], train["Y_B"][:, b, 0]
        nll_block2 = evidence_gap_poc(X_A, Y_A, X_test_A, y_test_A, build_stacked_overlay_features, device=device)
        gap1 = nll_block2 - nll_baseline

        # Experiment 2:
        # Simplified case, where B is still in the domain of A, but passed as a Block diag, meaning, the model
        # has to understand the block design in feature-space in order to undo it.
        X_B, Y_B = train["X_B_in_A"][:, b], train["Y_B_in_A"][:, b, 0]
        nll_block1 = evidence_gap_poc(X_A, Y_A, X_test_A, y_test_A, build_stacked_overlay_features, device=device)
        gap2 = nll_block1 - nll_baseline

        """
        Adding task id tokens results in  roughly the same gap
        
        # 1. With A_test query set: 
        (distorted, block): mean=-0.1518  median=-0.1409  sem=0.0313  n_inf=0/100
        (aligned, block): mean=-0.2076  median=-0.1656  sem=0.0284  n_inf=0/100
        (oracle/concat): mean=-2.2357  median=-2.7868  sem=0.1649  n_inf=2/100

        paired alignment cost (distorted - aligned): mean=2.0265  median=2.6955  sem=0.1614  n_inf=2/100
          -> looks real
        
        # 2. with B_in_A query set: 
        (distorted, block): mean=-0.1045  median=-0.0829  sem=0.0349  n_inf=0/100
        (aligned, block): mean=-0.2425  median=-0.2414  sem=0.0323  n_inf=0/100
        (oracle/concat): mean=-3.8095  median=-3.7777  sem=0.0548  n_inf=0/100
        
        paired alignment cost (distorted - aligned): mean=3.5670  median=3.5189  sem=0.0502  n_inf=0/100
          -> looks real
        """

        # Experiment 3:
        # the lower bound in nll is determined by having more data in A's domain (which is B in A)
        X_B, Y_B = train["X_B_in_A"][:, b], train["Y_B_in_A"][:, b, 0]
        nll_block3 = evidence_gap_poc(X_A, Y_A, X_test_A, y_test_A, build_concat_design, device=device)
        gap3 = nll_block3 - nll_baseline

        gaps1.append(gap1)
        gaps2.append(gap2)
        gaps3.append(gap3)

    summarize_gaps("gap1 (distorted, block)", gaps1)
    summarize_gaps("gap2 (aligned, block)", gaps2)
    summarize_gaps("gap3 (oracle/concat)", gaps3)
    print()

    paired_alignment_cost(gaps_aligned=gaps3, gaps_distorted=gaps2)
    """
    gap1 (distorted, block): mean=-2.1682  median=-2.8041  sem=0.1725  n_inf=2/100
    gap2 (aligned, block): mean=-0.1516  median=-0.1252  sem=0.0335  n_inf=0/100
    gap3 (oracle/concat) : mean=-0.2149  median=-0.2392  sem=0.0319  n_inf=0/100
    paired alignment cost (distorted - aligned): mean=0.0633  median=0.0416  sem=0.0194  n_inf=0/100
      -> looks real
    item 80: gap1=INF, gap2=ok, gap3=ok
    item 89: gap1=INF, gap2=ok, gap3=ok
        
    The alignment cost is real

    0.0633 ± 0.0194 nats is over 3 SEM from zero — small, but no longer dismissible as noise. 
    Combined with the block-format-only cost (gap3 − oracle, roughly 2 nats), you now have a solid three-way 
    decomposition: 
    1. the model pays a large tax for the block-diagonal/NaN presentation itself,
    2. and a small but genuine additional tax for actually needing to infer the shift/scale on top of that. 
    Proportionally that's still a ~30:1 ratio -- most of the headroom is in getting the pretrained model to parse 
    this input shape at all, not in the alignment reasoning per se.
    """
    print()

    cross_condition_inf_report({"gap1": gaps1, "gap2": gaps2, "gap3": gaps3})

