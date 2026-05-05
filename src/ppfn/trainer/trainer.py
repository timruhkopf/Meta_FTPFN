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
import math

import torch
import torch.nn as nn
from torch import amp

from pfns4hpo.priors import Batch

from ppfn.utils.gracefull_exit import GracefulExit, signal_handler
from ppfn.trainer.callbacks.abstract_callback import AbstractCallback, CallbackHandler

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            callbacks: dict[AbstractCallback] | None = None,
            verbose: bool = False,
            optimizer=None,
            scheduler=None,
            description_template: str | None = None,
            eons=1,
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
        self.criterion = criterion.to(device)
        self.device = device
        self.use_amp = use_amp
        self.eons = eons  # this is a hack to avoid the overhead of generating and storing!! all that prior data; instead this is a revert
        # to the classical training loop where we just iterate over the dataloader multiple times

        # Get trainable parameters from the model
        if hasattr(model, 'get_trainable_params'):
            trainable_params = model.get_trainable_params(optimizer.keywords['weight_decay'])
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)

        self.callbacks = callbacks or {}
        # TODO callbackhandler will need to be ddp rank aware to avoid multiple logging
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)

        self.verbose = verbose
        self.config = None  # Placeholder, we pass this from outside if needed

        # Mixed precision & gradient settings
        self.scaler = amp.GradScaler(device=self.device, enabled=(device == "cuda")) if self.use_amp else None
        self.grad_clip = grad_clip
        self.aggregate_k_gradients = aggregate_k_gradients

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        self.description_template = description_template or (
            "Epoch {epoch:3d} | Time: {time:6.2f}s | LR: {train/lr:.8f} | nll/C-A_related: {nll/C-A_related:.3}"
        )

        self.callback_handler.on_event("on_trainer_init")

    def fit(self, epochs: int, steps: int):
        """
        The dataset is potentially a stream dataset.

        * To still have regular metric logging intervals, the number of steps define an "epoch"
        * Eons are one entire pass over the dataset. Since the dataset is expensive to generate (and is generated in
        advance) for dev purposes we introduce eons; i.e. passes over the stream
        """
        self.epochs = epochs  # helper for eon logging in mlflow callback
        self.steps = steps
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
            for eon in range(self.eons):
                pbar = tqdm(range(epochs), disable=not self.verbose)

                if self.eons > 1:
                    logger.info(f"Starting eon {eon + 1}/{self.eons}...")

                # Check if the underlying dataset has the shuffle method
                # FIXME: this is a code-smell
                dataset = getattr(self.train_loader, 'dataset', self.train_loader)
                if hasattr(dataset, 'shuffle'):
                    dataset.shuffle()

                # Create a FRESH iterator for this eon without overwriting the loader
                self.train_iter = iter(self.train_loader)

                for epoch in pbar:
                    self.current_epoch = epoch
                    self.callback_handler.on_event("on_epoch_start", epoch=epoch)

                    epoch_metrics = self.train_epoch(steps)

                    feedback = self.callback_handler.on_event(
                        "on_epoch_end", epoch=epoch, metrics=epoch_metrics
                    )

                    epoch_metrics.update(feedback)

                    self.callback_handler.on_event(
                        "log_on_epoch_end", epoch=epoch, eon=eon, metrics=epoch_metrics
                    )

                    if feedback.get("stop_training", False):
                        print("Early stopping triggered. Terminating training.")
                        break

                    if self.verbose:
                        try:
                            # We merge epoch into the metrics dict or pass it as a kwarg
                            description = self.description_template.format(
                                epoch=epoch, **epoch_metrics
                            )
                            pbar.set_description(description)
                        except KeyError as e:
                            # Fallback or warning if user provided a key that doesn't exist
                            pbar.set_description(f"Epoch {epoch} (Template Error: Missing {e})")

        except KeyboardInterrupt:
            print("Training interrupted by user")

        except GracefulExit as e:
            # This block specifically catches the Slurm Timeout / USR1
            logger.warning(f"Training interrupted by Slurm: {e}")

        except Exception:
            # This will now print the full traceback to your logs/terminal
            logger.exception("An error occurred during training:")
            raise

        finally:
            logger.info("Reached end of training...")
            self.callback_handler.on_event("on_train_end")

            # e.g. terminate mlflow run
            self.callback_handler.on_event("log_on_train_end")

        logger.info("Training complete.")
        self.epochs = None
        self.steps = None

    def _get_next_batch(self):

        # FIXME Unfortunately the PFN's inability to deal with paddings^* (for both train and test) enforces a rigid design choice:
        #  The context train test split must be at the exact same location in the entire batch.
        #  Since we need to sample over different train context sizes, which makes collating and stacking batches without padding
        #  impossible, made them precompute, store and fix the batch, causing this ugly design. Another consideration is,
        #  that the independent item sampling in the batch with varying task complexity enforces looping over randomly sampled
        #  MLP instances (both in size and weights). This forces pre-computing, because otherwise the GPU starves waiting for data.
        #  ^* that themselves are an inefficiency

        if hasattr(self.train_loader, "get_batch"):
            # Prior dataloader legacy support
            return self.train_loader.get_batch(device=self.device)

        # Standard loader path
        batch = next(self.train_iter)
        # If the loader collated it into a list/tuple of length 1
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            batch = batch[0]

        if batch.single_eval_pos == 0:
            # in this edge case, both A and B are empty, we cannot meta-learn
            # FIXME: this needs to removed from the dataset generation for efficiency
            # Fixme: we want to default to A's unconditional!
            logger.error(
                "Received batch with single_eval_pos=0 at step {step}, skipping this batch due to meta-learning minimal requirements.")
            batch = self._get_next_batch()

        kwargs = {'single_eval_pos': batch.single_eval_pos}

        return batch.to(self.device), kwargs

    def train_epoch(self, n_steps=None) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {
            "num_batches": 0,
            "time": 0.0,
        }

        epoch_start = time.time()
        if n_steps is None:
            steps = range(len(self.train_loader))
        else:
            steps = range(n_steps)

        for step in steps:
            try:
                batch, fwd_kwargs = self._get_next_batch()

                step_metrics = self._train_step(step, batch, **fwd_kwargs)

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

            except StopIteration:
                logger.warning("DataLoader exhausted before reaching n_steps. Ending epoch early.")
                break

        # Average metrics for epoch (except time)
        num_batches = epoch_metrics.pop("num_batches")

        if num_batches > 0:
            for key in epoch_metrics:
                if key != "time":
                    epoch_metrics[key] /= num_batches
        else:
            logger.warning("Epoch finished with 0 batches processed.")

        epoch_metrics["time"] = time.time() - epoch_start


        return epoch_metrics

    def _train_step(self, step, batch: Batch, **fwd_kwargs) -> Dict[str, float]:
        device_type = self.device.type if hasattr(self, "device") else "cuda"
        # Consider: dtype=torch.bfloat16: and not use the gard scaler at all?
        with amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda" and self.use_amp)):
            output = self.model(batch, **fwd_kwargs)

        # loss calculation outside of autocast for stable training
        loss, step_metrics = self.criterion(output, batch=batch, **fwd_kwargs)

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
                step_metrics.update(
                    self.callback_handler.on_event(
                        "on_clipping",
                        epoch=self.current_epoch,
                        step=step,
                        metrics=step_metrics,
                    ))

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if self.scaler and device_type == "cuda":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

        step_metrics.update(
            {
                "nll/batch_loss": loss.detach().cpu().item(),
                "train/lr": self.scheduler.get_last_lr()[0],
            }
        )

        if any(math.isnan(v) for v in step_metrics.values() if isinstance(v, (int, float))):
            # Handle the error (e.g., logging a warning or skipping the log)
            print(step_metrics)
            print("Warning: NaN detected in step_metrics")

        if self.verbose and 'single_eval_pos' in fwd_kwargs.keys():
            step_metrics.update({"train/single_eval_pos": fwd_kwargs['single_eval_pos']})

        return step_metrics

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
