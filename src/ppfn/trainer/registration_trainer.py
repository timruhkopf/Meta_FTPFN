"""Training loop for `ppfn.model.registration.RegistrationPFN` --
ARCHITECTURE.md §4. A drop-in for `src/train.py`'s existing calling
convention (`instantiate(cfg.trainer.trainer_class, model=..., train_loader=...,
optimizer=..., scheduler=..., device=...)`, then `trainer.fit(epochs=...,
steps=...)`), registered under `configs/trainer/registration.yaml` rather
than modifying the existing (unrelated, and currently broken --
`_get_next_batch` is referenced but never defined) `ppfn.trainer.trainer.PPFNTrainer`.

`epochs`/`steps` keep the existing repo convention (see that trainer's own
`fit()` docstring): `steps` training steps make up one "epoch", at the end
of which the §4.5 monitor set is computed on the fixed validation batch and
logged -- so `epochs` here means "number of monitor/logging cycles", not a
pass over a finite dataset (there isn't one; the prior is resampled every
step).

Reuses `ppfn.trainer.callbacks.mlflow_cb.MLflowCallback` for the run
lifecycle (git-hash tagging, hydra-override params, `log_on_epoch_end`'s
`step = eon*epochs+epoch` formula) via the existing
`AbstractCallback`/`CallbackHandler` machinery -- tested, and the right
level of reuse for "start/stop an MLflow run," which is a different concern
from the flexible per-metric registry in `ppfn.monitor.registry` (see that
module's docstring for why the metric side is NOT built on
`AbstractCallback`).
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ppfn.monitor.registry import MonitorContext, compute_all_monitors
from ppfn.prior.registration.dataset import (
    RegistrationBatch,
    build_training_item,
    collate_registration_batch,
)
from ppfn.callbacks.abstract_callback import CallbackHandler
from ppfn.utils.gracefull_exit import GracefulExit, signal_handler

logger = logging.getLogger(__name__)


def _move_batch(batch: RegistrationBatch, device: torch.device) -> RegistrationBatch:
    moved = {
        f.name: (
            getattr(batch, f.name).to(device)
            if isinstance(getattr(batch, f.name), torch.Tensor)
            else getattr(batch, f.name)
        )
        for f in dataclasses.fields(batch)
    }
    return RegistrationBatch(**moved)


class RegistrationTrainer:
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
        force_identity_transport: bool = False,
        callbacks: dict | None = None,
        checkpoint_dir: str | None = None,
        verbose: bool = True,
        monitor_fn: Callable | None = None,
    ):
        """`force_identity_transport`: bypass the affine head and Delta_l
        entirely, using t_i = x-tilde_i^A (the raw decoder coordinates) at
        every layer -- ARCHITECTURE.md build order step 3 / CLAUDE.md's
        go/no-go: "transport residual disabled (t_i = x_i^A, exactly
        correct there)", trained together with `prior.force_rho_zero=true`
        (`configs/prior/p0_identity.yaml`) via `configs/experiment/step4_pathway.yaml`.
        Reuses the same `Decoder.transport_override` plumbing built for
        oracle-teacher-forcing (`loss.registration_loss`'s `L_distil`).

        `monitor_fn`: `MonitorContext -> dict[str, float]`, called at every
        epoch boundary on the fixed validation batch. Defaults to
        `ppfn.monitor.registry.compute_all_monitors` (the full §4.5
        dashboard); pass a narrower function (e.g.
        `ppfn.monitor.arch_verification.compute_arch_verification_metrics`)
        for an experiment that doesn't want that whole monitor set logged --
        CLAUDE.md: "nothing gets logged that isn't declared" applies per
        experiment, not just globally."""
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.monitor_fn = monitor_fn or compute_all_monitors
        self.criterion = criterion
        self.grad_clip = grad_clip
        self.use_bf16 = use_bf16
        self.verbose = verbose
        self.force_identity_transport = force_identity_transport
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.dataset_progress = getattr(
            getattr(train_loader, "dataset", None), "progress", None
        )
        if self.dataset_progress is None:
            logger.warning(
                "train_loader.dataset has no `.progress` (SharedProgress) attribute -- "
                "the rho curriculum will not advance with training progress; every draw "
                "will use whatever progress value the dataset was constructed with."
            )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)

        self.callbacks = callbacks or {}
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        # Fixed, held-out validation batch -- ARCHITECTURE.md §4.1: "Pre-generate
        # a fixed validation set of 2048 pairs... and never train on it." Built
        # once, eagerly, at a seed disjoint from the training stream's worker
        # seeds, so it's genuinely frozen for the life of this trainer.
        val_rng = np.random.default_rng(val_seed)
        val_items = [
            build_training_item(
                val_rng,
                progress=1.0,
                s_max=val_s_max,
                force_rho_zero=val_force_rho_zero,
            )
            for _ in range(val_size)
        ]
        self.val_batch = _move_batch(collate_registration_batch(val_items), self.device)

        self.config = None
        self.global_step = 0
        self.epochs = None
        self.steps = None

        self.callback_handler.on_event("on_trainer_init")

    def fit(self, epochs: int, steps: int) -> None:
        self.epochs = epochs
        self.steps = steps
        total_steps = max(epochs * steps, 1)
        logger.info(
            f"Starting training: {epochs} epochs x {steps} steps ({total_steps} total steps)."
        )

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
                    last_metrics = metrics
                    self.global_step += 1
                    self.callback_handler.on_event(
                        "on_step_end", epoch=epoch, step=step, metrics=metrics
                    )

                self.model.eval()
                monitor_metrics = self.monitor_fn(
                    MonitorContext(model=self.model, val_batch=self.val_batch)
                )
                self.model.train()

                epoch_metrics = {
                    **last_metrics,
                    **monitor_metrics,
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
                    # gate/mean_abs and bounds/transfer_gap are only produced
                    # by the full ppfn.monitor.registry dashboard -- skip
                    # them rather than print NaN for a monitor_fn (e.g.
                    # ppfn.monitor.arch_verification) that doesn't compute
                    # them.
                    extra = "".join(
                        f"| {key}={epoch_metrics[key]:.4f} "
                        for key in ("gate/mean_abs", "bounds/transfer_gap")
                        if key in epoch_metrics
                    )
                    logger.info(
                        f"epoch {epoch:4d} | loss/total={epoch_metrics.get('loss/total', float('nan')):.4f} "
                        f"{extra}| time={epoch_metrics['time']:.1f}s"
                    )

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

    def _train_step(self, batch: RegistrationBatch, progress: float) -> dict:
        device_type = self.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=(self.use_bf16 and device_type == "cuda"),
        ):
            transport_override = (
                (batch.dec_ctx_x, batch.dec_qry_x)
                if self.force_identity_transport
                else None
            )
            output = self.model(batch, transport_override=transport_override)
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
        metrics["train/progress"] = progress
        return metrics

    def _save_checkpoint(self, epoch: int) -> None:
        """A third, deliberately minimal checkpoint scheme for this
        pipeline -- see `.claude/rules/checkpoints.md`: the two existing
        schemes (`PPFNTrainer._save_checkpoint`, `CheckpointCallback`) are
        documented as incompatible with each other and carry their own
        assumptions (epoch/eon bookkeeping, AMP scaler state) this
        step-based, no-AMP-scaler trainer doesn't share -- reusing either
        without a deliberate refactor would be the wrong kind of reuse."""
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
