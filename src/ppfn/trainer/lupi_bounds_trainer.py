"""Training loop for `ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN` -- a
near-verbatim sibling of `ppfn.trainer.lupi_trainer.LUPITrainer` (same
prior/dataset, same fixed-validation-batch convention); the only
LUPI-trainer-specific thing that doesn't carry over is the console log
line's metric names (`loss/lower_nll`/`loss/upper_nll` here vs.
`loss/student_nll`/`loss/oracle_nll` there) -- kept as a separate file
rather than a shared base class per this repo's established convention of
independent per-model trainer siblings (IDTokenTrainer, LUPITrainer, ...).
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


class BoundsTrainer:
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
        val_n_a_range: tuple[int, int] = (8, 256),
        val_n_b_range: tuple[int, int] = (256, 1024),
        val_n_qry_range: tuple[int, int] = (8, 128),
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

        val_rng = np.random.default_rng(val_seed)
        val_items = [
            build_training_item(
                val_rng,
                progress=1.0,
                s_max=val_s_max,
                force_rho_zero=val_force_rho_zero,
                n_a_range=val_n_a_range,
                n_b_range=val_n_b_range,
                n_qry_range=val_n_qry_range,
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
                    batch = _move_batch(next(loader_iter), self.device)
                    metrics = self._train_step(batch)
                    last_metrics = metrics
                    self.global_step += 1
                    self.callback_handler.on_event(
                        "on_step_end", epoch=epoch, step=step, metrics=metrics
                    )

                self.model.eval()
                with torch.no_grad():
                    val_loss, val_metrics = self.criterion(self.model, self.val_batch)
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
                    logger.info(
                        f"epoch {epoch:4d} | loss/lower_nll={epoch_metrics.get('loss/lower_nll', float('nan')):.4f} "
                        f"| loss/upper_nll={epoch_metrics.get('loss/upper_nll', float('nan')):.4f} "
                        f"| val/loss/lower_nll={epoch_metrics.get('val/loss/lower_nll', float('nan')):.4f} "
                        f"| val/loss/upper_nll={epoch_metrics.get('val/loss/upper_nll', float('nan')):.4f} "
                        f"| time={epoch_metrics['time']:.1f}s"
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

    def _train_step(self, batch: LUPIBatch) -> dict:
        device_type = self.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=(self.use_bf16 and device_type == "cuda"),
        ):
            loss, metrics = self.criterion(self.model, batch)

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
