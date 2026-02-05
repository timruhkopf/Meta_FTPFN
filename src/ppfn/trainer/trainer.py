"""
Trainer for PPFN models with MLflow integration and callback support.

Follows the patterns from pfns4hpo/train.py but adapted for the
PPFN (Pre-conditioned Prior Fitted Network) architecture.
"""

from __future__ import annotations


import time
import warnings
from typing import Dict, List
from tqdm import tqdm
import signal

import torch
import torch.nn as nn
from torch import amp

from pfns4hpo.priors import Batch

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback, CallbackHandler


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GracefulExit(Exception):
    """Custom exception to trigger clean shutdown on signals."""
    pass

def signal_handler(signum, frame):
    signame = signal.Signals(signum).name
    logger.info(f"Signal {signame} received. Triggering graceful shutdown...")
    raise GracefulExit(f"Received {signame}")


class PPFNTrainer:
    """
    Trainer for PPFN models with MLflow integration.

    Designed to train frozen pre-trained models augmented with trainable
    interleaved cross-attention layers.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader,
            criterion: nn.Module,
            device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
            use_amp: bool = False,
            grad_clip: float = 1.0,
            aggregate_k_gradients: int = 1,
            callbacks: list[AbstractCallback] | None = None,
            verbose: bool = False,
            optimizer=None,
            scheduler=None,
            description_template: str | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PPFN model to train
            train_loader: DataLoader for training
            criterion: Loss function
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            grad_clip: Gradient clipping norm (None to disable)
            aggregate_k_gradients: Number of gradient accumulation steps
            callbacks: List of callback objects
            verbose: Whether to print progress
            optimizer: Partial optimizer callable - will be called with trainable_params
            scheduler: Partial scheduler callable - will be called with optimizer
            description_template: Template for epoch description in progress bar,
                e.g. "Epoch {epoch} | Time: {time:.2f}s | LR: {lr:.6f}"
                will allow keys provided in epoch metrics (which in turn are created by the
                callbacks).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp

        # Get trainable parameters from the model
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)


        self.callbacks = callbacks or []
        # TODO callbackhandler will need to be ddp rank aware to avoid multiple logging
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        self.verbose = verbose
        self.config = None # Placeholder, we pass this from outside if needed

        # Mixed precision & gradient settings
        self.scaler = amp.GradScaler(device=self.device) if use_amp else None
        self.grad_clip = grad_clip
        self.aggregate_k_gradients = aggregate_k_gradients

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        self.description_template = description_template or (
            "Epoch {epoch:3d} | Time: {time:6.2f}s | LR: {train/lr:.8f}"
        )

        self.callback_handler.on_event("on_trainer_init")


    def fit(self, epochs: int, steps: int):
        """
        Train the model for a given number of epochs.

        Args:
            epochs: Number of epochs to train
        """
        logger.info("Starting training...")
        self.callback_handler.on_event("on_train_start")

        # prepare sigterm handler for slurm
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)

        # notice that we can trigger the signal during debugging:
        # Sends SIGUSR1 to the most recent Python process running train.py
        # kill -USR1 $(pgrep -n -f train.py)
        # Then check the resulting log!

        try:
            iterator = tqdm(range(epochs), disable=not self.verbose)

            if hasattr(self.train_loader, '__iter__'):
                # classical trainloader
                self.train_loader = iter(self.train_loader)

            for epoch in iterator:
                self.current_epoch = epoch

                self.callback_handler.on_event("on_epoch_start", epoch=epoch)

                epoch_metrics = self.train_epoch(steps)

                feedback = self.callback_handler.on_event(
                    "on_epoch_end", epoch=epoch, metrics=epoch_metrics
                )

                epoch_metrics.update(feedback)

                self.callback_handler.on_event(
                    "log_on_epoch_end", epoch=epoch, metrics=epoch_metrics
                )

                if feedback.get('stop_training', False):
                    print("Early stopping triggered. Terminating training.")
                    break

                if self.verbose:
                    try:
                        # We merge epoch into the metrics dict or pass it as a kwarg
                        description = self.description_template.format(
                            epoch=epoch,
                            **epoch_metrics
                        )
                        iterator.set_description(description)
                    except KeyError as e:
                        # Fallback or warning if user provided a key that doesn't exist
                        iterator.set_description(f"Epoch {epoch} (Template Error: Missing {e})")

        except KeyboardInterrupt:
            print("Training interrupted by user")
        except GracefulExit as e:
            # This block specifically catches the Slurm Timeout / USR1
            logger.warning(f"Training interrupted by Slurm: {e}")

        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
            raise e
        finally:
            logger.info("Reached end of training...")
            self.callback_handler.on_event("on_train_end")

            # e.g. terminate mlflow run
            self.callback_handler.on_event("log_on_train_end")

        logger.info("Training complete.")

    def train_epoch(self, n_steps) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {
            "num_batches": 0,
            "time": 0.0,
        }

        epoch_start = time.time()

        for step in range(n_steps):

            if hasattr(self.train_loader, 'get_batch'):
                # PRIORDATALOADER Legacy support
                batch = self.train_loader.get_batch(device=self.device)
            else:
                # standard dataloader
                batch = next(self.train_loader)
                assert isinstance(batch, List) and len(batch) == 1, (
                    "The PPFNTrainer expects that the dataset class already provides batches, "
                    "so the loader must have batch_size=1."
                )
                batch = batch[0]  # since we expect batch_size=1 with collate_fn=lambda x: x[0]

                batch = batch.to(self.device)

            if batch.single_eval_pos is None:
                seq_len = torch.tensor(batch.x.shape[1])
                # sample according to the PriorDataLoader default
                single_eval_pos = int(torch.floor(
                    torch.exp(torch.rand(1) * torch.log(seq_len + 1))
                ) - 1)
                warnings.warn("single_eval_pos not set in batch; using random value.")
            else:
                single_eval_pos = batch.single_eval_pos

            step_metrics = self._train_step(step, batch, single_eval_pos=single_eval_pos)

            # Accumulate metrics per step
            for key, value in step_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value

            epoch_metrics["num_batches"] += 1

            # Call step callbacks
            self.callback_handler.on_event(
                "on_step_end", epoch=self.current_epoch, step=step, metrics=step_metrics
            )

            self.global_step += 1

        # Average metrics for epoch (except time)
        num_batches = epoch_metrics.pop("num_batches")
        for key in epoch_metrics:
            if key != "time":
                epoch_metrics[key] /= num_batches

        epoch_metrics["time"] = time.time() - epoch_start
        self.scheduler.step()

        return epoch_metrics

    def _train_step(self, step, batch: Batch, single_eval_pos) -> Dict[str, float]:
        device_type = self.device.type if hasattr(self, "device") else "cuda"
        loss, step_metrics = self._forward_pass(batch, single_eval_pos=single_eval_pos)

        # Scale loss for gradient accumulation
        loss_scaled = loss / self.aggregate_k_gradients

        if self.scaler and device_type == "cuda":
            self.scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # Update weights if gradient accumulation is complete
        if (self.global_step + 1) % self.aggregate_k_gradients == 0:
            if self.scaler and device_type == "cuda":
                self.scaler.unscale_(self.optimizer)

            if self.grad_clip:

                feedack = self.callback_handler.on_event(
                    "on_clipping", epoch=self.current_epoch, step=step, metrics=step_metrics
                )
                step_metrics.update(feedack)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if self.scaler and device_type == "cuda":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        step_metrics.update(
            {
                "nll/batch_loss": loss.detach().cpu().item(),
                "train/lr": self.scheduler.get_last_lr()[0],
            }
        )
        if self.verbose:
            step_metrics.update({
                "train/single_eval_pos": single_eval_pos
            })

        return step_metrics

    def _forward_pass(self, batch: Batch, single_eval_pos, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single forward pass and compute loss."""
        # Unpack batch; adapt based on your batch structure
        # if isinstance(batch, (tuple, list)):
        # batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)

        with (amp.autocast(device_type="cuda", enabled=self.use_amp)):
            output = self.model(batch, single_eval_pos=single_eval_pos, **kwargs)
            targets = batch.y[single_eval_pos:, ...]

            loss, metrics = self.criterion(output, targets, batch=batch, single_eval_pos=single_eval_pos)

        return loss, metrics

    def _save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, filename)
        # mlflow.log_artifact(filename)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]



class DistributedTrainer(PPFNTrainer):
    """Trainer with distributed training support (DDP)."""

    def __init__(self, *args, using_dist: bool = False, rank: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.using_dist = using_dist
        self.rank = rank

        if using_dist:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                broadcast_buffers=False,
            )

    def _log(self, message: str):
        """Log only from rank 0."""
        if self.rank == 0:
            print(message)
