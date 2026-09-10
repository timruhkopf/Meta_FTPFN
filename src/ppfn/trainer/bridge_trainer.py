"""Training loop for `ppfn.model.bridge.model.BridgePFN` -- deliberately
minimal: no curriculum, no `ppfn.monitor.registry` dashboard, no
transport/coupling/affine loss terms. Just NLL on query tokens, plus a free
severed-vs-bridged comparison on the held-out validation batch
(`BridgePFN.forward`'s `use_encoder` toggle) so it's directly visible
whether the cross-attention pathway is doing anything at all -- CLAUDE.md's
"never a bare NLL" in its cheapest form, one level below the go/no-go run's
three-way comparison.

Same shape as `ppfn.trainer.registration_trainer.RegistrationTrainer`
(callback lifecycle, `fit(epochs, steps)` convention, `MLflowCallback`/
`CheckpointCallback` compatibility) so it's a drop-in for
`pipelines/train.py`'s existing calling convention -- just without a
separate `criterion` module, since the loss here is one line.
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import time

import numpy as np
import torch
import torch.nn as nn

from ppfn.callbacks.abstract_callback import CallbackHandler
from ppfn.prior.bridge.dataset import (
    BridgeBatch,
    build_bridge_item,
    collate_bridge_batch,
)
from ppfn.utils.gracefull_exit import GracefulExit, signal_handler

logger = logging.getLogger(__name__)


def _move_batch(batch: BridgeBatch, device: torch.device) -> BridgeBatch:
    moved = {
        f.name: (
            getattr(batch, f.name).to(device)
            if isinstance(getattr(batch, f.name), torch.Tensor)
            else getattr(batch, f.name)
        )
        for f in dataclasses.fields(batch)
    }
    return BridgeBatch(**moved)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


class BridgeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        optimizer,
        scheduler,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        val_size: int = 256,
        val_seed: int = 999_999,
        val_d: int | None = None,
        val_n_a_range: tuple[int, int] = (8, 100),
        val_n_b_range: tuple[int, int] = (8, 100),
        callbacks: dict | None = None,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.grad_clip = grad_clip
        self.verbose = verbose

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)

        self.callbacks = callbacks or {}
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        # Fixed, held-out validation batch, same convention as
        # RegistrationTrainer's (ARCHITECTURE.md §4.1) -- built once, at a
        # seed disjoint from the training stream's worker seeds.
        val_rng = np.random.default_rng(val_seed)
        val_items = [
            build_bridge_item(
                val_rng, d=val_d, n_a_range=val_n_a_range, n_b_range=val_n_b_range
            )
            for _ in range(val_size)
        ]
        self.val_batch = _move_batch(collate_bridge_batch(val_items), self.device)

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
                    batch = _move_batch(next(loader_iter), self.device)
                    metrics = self._train_step(batch)
                    last_metrics = metrics
                    self.global_step += 1
                    self.callback_handler.on_event(
                        "on_step_end", epoch=epoch, step=step, metrics=metrics
                    )

                self.model.eval()
                val_metrics = self._compute_val_metrics()
                self.model.train()

                epoch_metrics = {
                    **last_metrics,
                    **val_metrics,
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
                        f"epoch {epoch:4d} | train/nll={epoch_metrics.get('train/nll', float('nan')):.4f} "
                        f"| val/nll_bridged={epoch_metrics.get('val/nll_bridged', float('nan')):.4f} "
                        f"| val/nll_severed={epoch_metrics.get('val/nll_severed', float('nan')):.4f} "
                        f"| time={epoch_metrics['time']:.1f}s"
                    )

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

    def _train_step(self, batch: BridgeBatch) -> dict:
        output = self.model(batch, use_encoder=True)
        nll = self.model.predictive_dist(output["predictive_logits"], batch.y_qry)
        loss = _masked_mean(nll, batch.dec_qry_mask)

        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(
                f"non-finite loss at global_step={self.global_step}: {loss.item()}"
            )

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = 0.0
        if self.grad_clip:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            )
        self.optimizer.step()
        self.scheduler.step()

        return {
            "train/nll": loss.item(),
            "train/grad_norm": grad_norm,
            "train/lr": self.scheduler.get_last_lr()[0],
        }

    def _compute_val_metrics(self) -> dict:
        batch = self.val_batch
        with torch.no_grad():
            out_bridged = self.model(batch, use_encoder=True)
            nll_bridged = self.model.predictive_dist(
                out_bridged["predictive_logits"], batch.y_qry
            )
            out_severed = self.model(batch, use_encoder=False)
            nll_severed = self.model.predictive_dist(
                out_severed["predictive_logits"], batch.y_qry
            )
        return {
            "val/nll_bridged": _masked_mean(nll_bridged, batch.dec_qry_mask).item(),
            "val/nll_severed": _masked_mean(nll_severed, batch.dec_qry_mask).item(),
        }
