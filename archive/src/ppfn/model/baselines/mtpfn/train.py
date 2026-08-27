"""
Meta-training pipeline for MTPFN.

Standard "epochless" PFN training: every optimizer step draws a fresh
batch of synthetic multi-task episodes straight from the Algorithm A.1
prior (no fixed dataset, no epochs -- the prior IS the training set).

Section 5's reported training configuration for the full MTPFN:
    - 23-layer hierarchical transformer: 12 intra-task + 11 inter-task
      layers, 4 attention heads, hidden size 512.
    - ~50,000,000 synthetically generated datasets.
    - batch size 16.
    - AdamW, learning rate 1e-4, cosine annealing.
These are used as the defaults below (`TrainConfig`, `full_paper_model_kwargs`).

Loss: Gaussian NLL (Figure 2's (mu, sigma^2) output head), evaluated only
at the target task's held-out query point(s) -- matching the paper's
stated NLL objective L_NLL = E_{D~p(D|h)}[-log f_theta(y_test | x_test, D_train)].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ppfn.model.baselines.mtpfn.gaussian_head import gaussian_nll, split_head_output
from ppfn.model.baselines.mtpfn.model import MTPFN
from ppfn.model.baselines.mtpfn.prior import MultiTaskICMPrior, MTPFNPriorConfig


@dataclass
class TrainConfig:
    # Paper: ~50M sampled datasets / batch size 16 -> ~3,125,000 steps.
    # Default here is drastically smaller so the script is actually runnable
    # for testing; override `steps` to match the paper for a real run.
    steps: int = 3_125_000
    batch_size: int = 16
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    log_every: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


def full_paper_model_kwargs() -> dict:
    """The exact architecture hyperparameters reported in Section 5."""
    return dict(d_model=512, n_heads=4, d_ff=2048, num_intra_layers=12, num_inter_layers=11)


def _cosine_lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_mtpfn(
    prior_cfg: MTPFNPriorConfig | None = None,
    train_cfg: TrainConfig | None = None,
    model_kwargs: dict | None = None,
):
    prior_cfg = prior_cfg or MTPFNPriorConfig()
    train_cfg = train_cfg or TrainConfig()
    model_kwargs = model_kwargs or full_paper_model_kwargs()

    prior_cfg.device = train_cfg.device
    prior = MultiTaskICMPrior(prior_cfg)

    model = MTPFN(input_dim=prior_cfg.input_dim, **model_kwargs).to(train_cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _cosine_lr(s, train_cfg.warmup_steps, train_cfg.steps)
    )
    use_amp = train_cfg.amp and train_cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    running_loss = 0.0
    for step in range(1, train_cfg.steps + 1):
        batch = prior.sample_batch(batch_size=train_cfg.batch_size)
        x = batch["x"].to(train_cfg.device)
        y = batch["y"].to(train_cfg.device)
        valid_mask = batch["valid_mask"].to(train_cfg.device)
        query_mask = batch["query_mask"].to(train_cfg.device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if train_cfg.device == "cuda" else "cpu", enabled=use_amp):
            raw = model(x, y, valid_mask, query_mask)  # [B,T,L,2]
            mu, log_var = split_head_output(raw)
            nll = gaussian_nll(mu, log_var, y)  # [B,T,L]
            # supervise only the target task's (task 0) held-out query point(s)
            loss = nll[query_mask].mean()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()

        running_loss += loss.item()
        if step % train_cfg.log_every == 0:
            avg = running_loss / train_cfg.log_every
            print(f"step {step:9d} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e}")
            running_loss = 0.0

    return model


if __name__ == "__main__":
    # A small, fast smoke-test configuration -- NOT the paper's real setup.
    # See `full_paper_model_kwargs()` and the `steps`/`batch_size` defaults
    # in `TrainConfig` for the values actually reported in Section 5.
    train_mtpfn(
        prior_cfg=MTPFNPriorConfig(num_tasks=4, input_dim=1, seq_len=40),
        train_cfg=TrainConfig(steps=200, batch_size=8, log_every=20, device="cpu", amp=False),
        model_kwargs=dict(d_model=64, n_heads=4, d_ff=128, num_intra_layers=4, num_inter_layers=3),
    )
