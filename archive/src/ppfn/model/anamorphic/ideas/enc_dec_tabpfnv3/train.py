"""Two-phase training pipeline.

Phase 0  Teacher. Stock TabPFN v2.5 on the A-task prior with *dense* contexts
         (A_train union A_pool). Warm-start from the released checkpoint; it
         converges fast. Then freeze.
Phase 1  Student. WarpAlignPFN against the frozen teacher.
Phase 2  Optional. Unfreeze the teacher at low LR, or switch to an EMA of the
         student's decoder. Only if Phase 1 plateaus.

A co-trained teacher is a moving target stacked on top of an already-hard
alignment problem, and the failure mode is hard to diagnose. If you want
co-adaptation, prefer the EMA route: no second free model, no instability.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import torch
from torch import nn

from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5, TabPFNV2p5Config

from .losses import LossWeights, compute_loss, effective_rank
from .model import WarpAlignConfig, WarpAlignPFN, make_borders


@dataclasses.dataclass
class MetaBatch:
    """One draw from your meta-task distribution. Fill this in from your sampler.

    All y are standardised with A_train's mean/std, so the fixed bar-distribution
    grid applies uniformly. Shapes follow TabPFN's row-major (R, B, C) layout.
    """

    x_A: torch.Tensor  # (R_A, B, C)
    y_A: torch.Tensor  # (R_A, B) -- only [:n_A_train] is fed to the model
    n_A_train: int
    x_B: torch.Tensor  # (R_B, B, C)
    y_B: torch.Tensor  # (R_B, B) -- B's own domain, distorted
    severity: torch.Tensor  # (B,) in [0, 1]; 0 = undistorted, 1 = unrelated

    # ---- privileged. Never an input; targets only. ----
    y_B_inA: torch.Tensor | None = None  # (R_B, B) B's rows, A's target domain
    x_A_pool: torch.Tensor | None = None  # (S, B, C) dense A draw, teacher only
    y_A_pool: torch.Tensor | None = None  # (S, B)

    @property
    def y_A_train(self) -> torch.Tensor:
        return self.y_A[: self.n_A_train]

    @property
    def y_A_test(self) -> torch.Tensor:
        return self.y_A[self.n_A_train :]


def severity_schedule(step: int, total: int, *, warmup_frac: float = 0.3) -> float:
    """Max severity to sample, annealed outward.

    Start near s=0 so the model first learns that extra context is usable at
    all. If it sees mostly-useless B early it learns to ignore B, and it does
    not come back.
    """
    return min(1.0, (step / max(total * warmup_frac, 1)))


def sample_severity(batch_size: int, s_max: float, *, p_zero: float = 0.15,
                    p_one: float = 0.15) -> torch.Tensor:
    """Uniform on [0, s_max] with atoms at both ends.

    The atom at s=1 (B independent of A) is what teaches the shared softmax to
    starve B. Without it the model trusts B unconditionally.
    """
    s = torch.rand(batch_size) * s_max
    u = torch.rand(batch_size)
    s = torch.where(u < p_zero, torch.zeros_like(s), s)
    return torch.where(u > 1 - p_one, torch.full_like(s, s_max), s)


# --------------------------------------------------------------------- teacher


def build_teacher(cfg: WarpAlignConfig, **kw) -> TabPFNV2p5:
    """Stock TabPFN, same bins as the student."""
    return TabPFNV2p5(
        config=TabPFNV2p5Config(
            emsize=cfg.emsize,
            nlayers=cfg.n_prologue_A + cfg.n_decoder,
            nhead=cfg.nhead,
            features_per_group=cfg.features_per_group,
            num_thinking_rows=cfg.num_thinking_rows,
        ),
        task_type="regression",
        n_out=cfg.n_bins,
        **kw,
    )


@torch.no_grad()
def teacher_logits(teacher: TabPFNV2p5, batch: MetaBatch) -> torch.Tensor:
    """Teacher sees A_train PLUS the dense privileged pool, as ordinary rows."""
    if batch.x_A_pool is None:
        x, y, n_ctx = batch.x_A, batch.y_A_train, batch.n_A_train
    else:
        x = torch.cat(
            [batch.x_A[: batch.n_A_train], batch.x_A_pool, batch.x_A[batch.n_A_train :]],
            dim=0,
        )
        y = torch.cat([batch.y_A_train, batch.y_A_pool], dim=0)
        n_ctx = batch.n_A_train + batch.x_A_pool.shape[0]
    del n_ctx
    return teacher(x, y, only_return_standard_out=True)


# --------------------------------------------------------------------- student


def train_student(
    student: WarpAlignPFN,
    teacher: TabPFNV2p5 | None,
    batches: Iterator[MetaBatch],
    *,
    total_steps: int,
    lr: float = 3e-4,
    weights: LossWeights | None = None,
    anneal_aux_from: float = 0.6,
    log_every: int = 100,
    device: str = "cuda",
) -> WarpAlignPFN:
    weights = weights or LossWeights()
    student.to(device).train()
    if teacher is not None:
        teacher.to(device).eval().requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps)

    for step, batch in zip(range(total_steps), batches):
        # Anneal the scaffolding down once the aligner works; the task loss and
        # the teacher KL should carry the end of training.
        decay = 1.0 if step < total_steps * anneal_aux_from else max(
            0.0, 1 - (step - total_steps * anneal_aux_from)
            / (total_steps * (1 - anneal_aux_from))
        )
        w = LossWeights(
            kl=weights.kl,
            translation=weights.translation * decay,
            variance=weights.variance,
            severity=weights.severity,
            mmd=weights.mmd * decay,
        )

        logits, aux = student(
            batch.x_A, batch.y_A_train, batch.x_B, batch.y_B, batch.n_A_train
        )
        t_logits = teacher_logits(teacher, batch) if teacher is not None else None

        loss, parts = compute_loss(
            criterion=student.criterion,
            student_logits_MBK=logits,
            y_A_test_MB=batch.y_A_test,
            aux=aux,
            weights=w,
            teacher_logits_MBK=t_logits,
            y_B_inA_RB=batch.y_B_inA,
            severity_B=batch.severity,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0:
            diag = {} if aux is None else {
                "eff_rank": float(effective_rank(aux.b_hat_rows_BRE))
            }
            print(step, parts, diag)

    return student


# ----------------------------------------------------------------- diagnostics


@torch.no_grad()
def utility_of_B(
    student: WarpAlignPFN, batch: MetaBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example delta-NLL from including B, and the predicted severity.

    THE headline diagnostic. Plot delta against true severity. If it does not
    decay as severity rises, the model has not learned to detect distortion --
    it has learned to trust B unconditionally, and you will get sharp,
    confident, wrong bar distributions on shifted tasks while your averaged
    metrics look fine.

    This is a better instrument than raw attention mass and needs no hooks: it
    measures what B actually buys, not where the softmax happened to point.
    """
    student.eval()
    with_B, aux = student(
        batch.x_A, batch.y_A_train, batch.x_B, batch.y_B, batch.n_A_train, use_B=True
    )
    without_B, _ = student(
        batch.x_A, batch.y_A_train, None, None, batch.n_A_train, use_B=False
    )
    c = student.criterion
    delta = (c(without_B, batch.y_A_test) - c(with_B, batch.y_A_test)).mean(0)
    return delta, torch.sigmoid(aux.severity_logit_B1).squeeze(-1)


@torch.no_grad()
def oracle_gap(
    student: WarpAlignPFN, batch: MetaBatch, x_B_inA: torch.Tensor
) -> dict[str, float]:
    """Floor / student / ceiling on identical tasks.

    Raw NLL is not interpretable here. Report the fraction of the
    oracle-alignment gap recovered: (floor - student) / (floor - ceiling).
    """
    c = student.criterion
    student.eval()

    def nll(xb, yb, use_B=True):
        lg, _ = student(
            batch.x_A, batch.y_A_train, xb, yb, batch.n_A_train, use_B=use_B
        )
        return float(c(lg, batch.y_A_test).mean())

    floor = nll(None, None, use_B=False)
    mid = nll(batch.x_B, batch.y_B)
    ceiling = nll(x_B_inA, batch.y_B_inA)
    denom = floor - ceiling
    return {
        "floor_no_B": floor,
        "student": mid,
        "ceiling_oracle_B_inA": ceiling,
        "gap_recovered": (floor - mid) / denom if abs(denom) > 1e-8 else float("nan"),
    }


def make_borders_for(cfg: WarpAlignConfig) -> torch.Tensor:
    return make_borders(cfg.n_bins)
