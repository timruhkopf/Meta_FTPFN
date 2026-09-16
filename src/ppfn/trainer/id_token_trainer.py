"""Training loop for `ppfn.model.baselines.id_token_pfn.IDTokenPFN` --
deliberately a smaller sibling of `ppfn.trainer.registration_trainer.RegistrationTrainer`,
not a reuse of it: that trainer's `_train_step` always calls
`self.model(batch, transport_override=...)` and threads a curriculum
`progress` value into `self.criterion(...)`, both specific to
`RegistrationPFN`/`RegistrationLoss`. This baseline has neither a
transport head nor a weight schedule, so forcing it through that interface
would mean stubbing arguments that mean nothing here rather than a real
reduction in code.

Mirrors `RegistrationTrainer` wherever the underlying decision is the same
one: a fixed, held-out validation batch built once at a seed disjoint from
training (so numbers are comparable to `RegistrationTrainer`'s own
validation, same prior, same seed convention -- ARCHITECTURE.md §5.5 "All
evaluated on the same fixed validation set"), the same
`epochs x steps`-per-epoch convention, the same minimal checkpoint schema.
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ppfn.callbacks.abstract_callback import CallbackHandler
from ppfn.prior.lupi.dataset import (
    LUPIBatch,
    build_training_item,
    collate_lupi_batch,
)
from ppfn.utils.gracefull_exit import GracefulExit, signal_handler

logger = logging.getLogger(__name__)


def _move_batch(batch: LUPIBatch, device: torch.device) -> LUPIBatch:
    moved = {
        f.name: (
            getattr(batch, f.name).to(device)
            if isinstance(getattr(batch, f.name), torch.Tensor)
            else getattr(batch, f.name)
        )
        for f in dataclasses.fields(batch)
    }
    return LUPIBatch(**moved)


def _upcast_logits(output: dict) -> dict:
    """Keep bar-distribution ops in fp32 when model forward used bf16 autocast."""
    return {
        k: (v.float() if isinstance(v, torch.Tensor) and "logits" in k else v)
        for k, v in output.items()
    }


class IDTokenTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        criterion: nn.Module,
        optimizer,
        scheduler,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        use_bf16: bool = False,
        val_size: int = 256,
        val_seed: int = 999_999,
        val_s_max: float = 0.1,
        val_force_rho_zero: bool = False,
        callbacks: dict | None = None,
        checkpoint_dir: str | None = None,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.use_bf16 = use_bf16
        self.verbose = verbose
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)

        self.callbacks = callbacks or {}
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        # Same val-set convention as RegistrationTrainer (seed/size/s_max),
        # INCLUDING reading d/n_a_range/n_b_range off the training dataset
        # rather than defaulting -- build_training_item's own defaults are
        # (8,256)/(256,1024), far wider than this baseline's capped training
        # ranges, and a pooled [A_ctx;B] context at the default n_b (up to
        # 1024) is an OOM waiting to happen (confirmed: this is exactly what
        # crashed the first run of this trainer, 2026-09-10 -- see
        # docs/labbook/ for the writeup). RegistrationTrainer was fixed for
        # the same reason; this trainer just hadn't picked that fix up yet.
        train_dataset = getattr(train_loader, "dataset", None)
        # Advances the rho curriculum with training progress, matching
        # RegistrationTrainer -- without this, sample_rho_curriculum never
        # sees progress > 0 and stays frozen at its easiest (Beta(1,5),
        # mean~0.17) setting for the entire run, never reaching the
        # intended uniform Beta(1,1) spread by 30% of training. Confirmed
        # missing here and fixed 2026-09-15 -- see docs/labbook/.
        self.dataset_progress = getattr(train_dataset, "progress", None)
        if self.dataset_progress is None:
            logger.warning(
                "train_loader.dataset has no `.progress` (SharedProgress) attribute -- "
                "the rho curriculum will not advance with training progress; every draw "
                "will use whatever progress value the dataset was constructed with."
            )
        val_d = getattr(train_dataset, "d", None)
        val_n_a_range = getattr(train_dataset, "n_a_range", (8, 256))
        val_n_b_range = getattr(train_dataset, "n_b_range", (256, 1024))
        val_warp_grid_n = getattr(train_dataset, "warp_grid_n", 4)
        val_n_qry_range = getattr(train_dataset, "n_qry_range", (8, 128))
        val_frac_uniform = getattr(train_dataset, "frac_uniform", 0.4)
        val_frac_near_b = getattr(train_dataset, "frac_near_b", 0.4)
        val_query_eps_std = getattr(train_dataset, "query_eps_std", 0.03)
        val_beta_override = getattr(train_dataset, "beta_override", None)
        val_rng = np.random.default_rng(val_seed)
        val_items = [
            build_training_item(
                val_rng,
                progress=1.0,
                s_max=val_s_max,
                force_rho_zero=val_force_rho_zero,
                d=val_d,
                warp_grid_n=val_warp_grid_n,
                n_a_range=val_n_a_range,
                n_b_range=val_n_b_range,
                n_qry_range=val_n_qry_range,
                frac_uniform=val_frac_uniform,
                frac_near_b=val_frac_near_b,
                query_eps_std=val_query_eps_std,
                beta_override=val_beta_override,
            )
            for _ in range(val_size)
        ]
        self.val_batch = _move_batch(collate_lupi_batch(val_items), self.device)

        self.config = None
        self.global_step = 0
        self.epochs = None
        self.steps = None

        self.callback_handler.on_event("on_trainer_init")

    def fit(self, epochs: int, steps: int) -> None:
        self.epochs = epochs
        self.steps = steps
        total_steps = max(epochs * steps, 1)
        logger.info(f"Starting training: {epochs} epochs x {steps} steps.")

        self.callback_handler.on_event("on_train_start")
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)

        loader_iter = iter(self.train_loader)
        try:
            for epoch in range(epochs):
                self.callback_handler.on_event("on_epoch_start", epoch=epoch)
                epoch_start = time.time()
                last_metrics: dict = {}

                for step in range(steps):
                    progress = self.global_step / max(total_steps - 1, 1)
                    if self.dataset_progress is not None:
                        self.dataset_progress.set(progress)

                    batch = _move_batch(next(loader_iter), self.device)
                    metrics = self._train_step(batch, progress)
                    metrics["train/progress"] = progress
                    last_metrics = metrics
                    self.global_step += 1
                    self.callback_handler.on_event(
                        "on_step_end", epoch=epoch, step=step, metrics=metrics
                    )

                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(self.val_batch)
                    # progress=1.0 (full curriculum weight) regardless of
                    # where training actually is -- val/loss/total should
                    # track the true, curriculum-independent objective
                    # across epochs, not a moving target. The individual
                    # val/loss/nll_student etc. are unweighted regardless
                    # (see LUPIIDTokenLoss's own metrics dict), so this only
                    # affects val/loss/total's value, not the checkpoint
                    # monitor (val/loss/nll_student).
                    val_loss, val_metrics = self.criterion(
                        self.model, self.val_batch, val_output, progress=1.0
                    )
                self.model.train()

                epoch_metrics = {
                    **last_metrics,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "time": time.time() - epoch_start,
                }
                feedback = self.callback_handler.on_event(
                    "on_epoch_end", epoch=epoch, metrics=epoch_metrics
                )
                epoch_metrics.update(feedback)
                self.callback_handler.on_event(
                    "log_on_epoch_end", epoch=epoch, eon=0, metrics=epoch_metrics
                )

                if self.verbose:
                    # Generic, not hardcoded to one loss's key names -- this
                    # trainer is shared across IDTokenLoss (loss/pred_nll)
                    # and LUPIIDTokenLoss (loss/total, loss/nll_student, ...),
                    # and hand-picking one convention's keys silently prints
                    # a placeholder "nan" under the other (not an actual NaN
                    # loss -- caught via ppfn.model.baselines.lupi_id_token_pfn's
                    # own debug run, see docs/labbook/).
                    extra = "  ".join(
                        f"{k}={v:.4f}" for k, v in epoch_metrics.items()
                        if k != "time" and isinstance(v, (int, float))
                    )
                    logger.info(f"epoch {epoch:4d} | {extra} | time={epoch_metrics['time']:.1f}s")

                if self.checkpoint_dir is not None:
                    self._save_checkpoint(epoch)

                if feedback.get("stop_training", False):
                    logger.info("Early stopping triggered.")
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        except GracefulExit as e:
            logger.warning(f"Training interrupted by SLURM: {e}")
        finally:
            self.callback_handler.on_event("on_train_end")
            self.callback_handler.on_event("log_on_train_end")

        logger.info("Training complete.")
        self.epochs = None
        self.steps = None

    def _train_step(self, batch: LUPIBatch, progress: float) -> dict:
        device_type = self.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=(self.use_bf16 and device_type == "cuda"),
        ):
            output = _upcast_logits(self.model(batch))

        loss, metrics = self.criterion(self.model, batch, output, progress=progress)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(
                f"non-finite loss at global_step={self.global_step}: {loss.item()}"
            )

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            metrics["train/grad_norm"] = float(grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        metrics["train/lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """A third, deliberately minimal checkpoint scheme, same rationale
        as `RegistrationTrainer._save_checkpoint` -- see
        `.claude/rules/checkpoints.md`: `PPFNTrainer`'s own scheme and
        `CheckpointCallback` are documented as mutually incompatible and
        carry assumptions (eon bookkeeping, AMP scaler state) this trainer
        doesn't share. `checkpoint_dir` is never actually passed by the Hydra
        experiment configs (`CheckpointCallback`, wired via `configs/
        callbacks/checkpoint.yaml`, is what actually saves the checkpoints
        loaded elsewhere in this repo, e.g. by the comparison notebook) --
        this path exists for direct/manual trainer use outside that config
        path, not because it's dead code left over by mistake."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / "checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
