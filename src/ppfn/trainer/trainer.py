"""
Trainer for PPFN models with MLflow integration and callback support.

Follows the patterns from pfns4hpo/train.py but adapted for the
PPFN (Pre-conditioned Prior Fitted Network) architecture.
"""

from __future__ import annotations

import subprocess
import os
import time
from typing import Dict

import torch
import torch.nn as nn
from torch import amp

from tqdm import tqdm

import mlflow
from omegaconf import OmegaConf

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback, CallbackHandler

import pfns4hpo.utils as utils
from pfns4hpo.priors import Batch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        logger.warning("Could not retrieve git hash.")
        return "not-a-git-repo"

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
    
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
        experiment_name: str = "ppfn_training",
        run_name: str | None = None,
        verbose: bool = True,
        optimizer=None,
        scheduler=None,
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
            experiment_name: MLflow experiment name
            run_name: MLflow run name (auto-generated if None)
            config: OmegaConf config dict to log
            verbose: Whether to print progress
            optimizer: Partial optimizer callable - will be called with trainable_params
            scheduler: Partial scheduler callable - will be called with optimizer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp

        # Get trainable parameters from the model

        # FIXME: trainable parameters!
        if hasattr(model, "trainable_parameters"):
            trainable_params = model.trainable_parameters()
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = optimizer(trainable_params)
        self.scheduler = scheduler(self.optimizer)

        self.grad_clip = grad_clip
        self.aggregate_k_gradients = aggregate_k_gradients
        self.callbacks = callbacks or []
        self.callback_handler = CallbackHandler(self.callbacks, trainer=self)
        self.verbose = verbose

        # MLflow 
        logger.info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = mlflow.start_run(run_name=run_name)

        # 1. Track Git Hash as a Tag
        mlflow.set_tag("mlflow.source.git.commit", get_git_hash())

        # Mixed precision
        # Mixed precision
        self.scaler = amp.GradScaler(device=self.device) if use_amp else None
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

    def log_config(self, cfg):
         
        # Convert to container and resolve interpolations (e.g. ${model.lr})
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(config_dict, "config.yaml")

    def fit(self, epochs: int, steps: int):
        """
        Train the model for a given number of epochs.

        Args:
            epochs: Number of epochs to train
        """
        try:
            iterator = tqdm(range(epochs), disable=not self.verbose)
            for epoch in iterator:
                self.current_epoch = epoch

                # Epoch start callbacks
                self.callback_handler.on_event("on_epoch_start", epoch=epoch)

                # Train
                epoch_metrics = self.train_epoch(steps)

                # Epoch end callbacks
                feedback = self.callback_handler.on_event(
                    "on_epoch_end", epoch=epoch, metrics=epoch_metrics
                )

                if feedback.get('stop_training', False):
                    print("Early stopping triggered. Terminating training.")
                    break

                # Log to MLflow
                for key, value in epoch_metrics.items():
                    mlflow.log_metric(key, value, step=epoch)

                # Track best loss
                # FIXME: obsolete with working EarlyStopping callback
                if epoch_metrics["loss"] < self.best_loss and epoch > 100:
                    self.best_loss = epoch_metrics["loss"]
                    self._save_checkpoint(f"best_model.pt")

                if self.verbose:
                    # update tqdm description
                    iterator.set_description(
                        f"Epoch {epoch:3d} | "
                        f"Loss: {epoch_metrics['loss']:7.4f} | "
                        f"NLL: {epoch_metrics.get('nll_diff_uncond_cond', 0):7.4f} | "
                        f"Time: {epoch_metrics['time']:6.2f}s | "
                        f"LR: {epoch_metrics.get('lr', 0):.6f}"
                    )

        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            # Train end callbacks
            self.callback_handler.on_event(
                "on_train_end", best_loss=self.best_loss, epochs=self.current_epoch
            )

            # Log final model
            # self._save_checkpoint("final_model.pt")

    def train_epoch(self, n_steps) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "num_batches": 0,
            "time": 0.0,
        }

        epoch_start = time.time()

        for step in range(n_steps):
            batch = self.train_loader.get_batch(device=self.device)
            step_metrics = self._train_step(batch)

            # Accumulate metrics
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

        # Average metrics
        num_batches = epoch_metrics.pop("num_batches")
        for key in epoch_metrics:
            if key != "time":
                epoch_metrics[key] /= num_batches

        epoch_metrics["time"] = time.time() - epoch_start
        self.scheduler.step()

        return epoch_metrics

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        """Execute a single training step."""
        device_type = self.device.type if hasattr(self, "device") else "cuda"
        losses, step_metrics = self._forward_pass(batch)

        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if self.scaler and device_type == "cuda":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        step_metrics.update(
            {
                "loss": loss.detach().cpu().item(),
                "lr": self.scheduler.get_last_lr()[0],
                "nan_share": nan_share,
                "single_eval_pos": batch.single_eval_pos
                if hasattr(batch, "single_eval_pos")
                else -1,
            }
        )

        return step_metrics

    def _forward_pass(self, batch: Batch) -> tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single forward pass and compute loss."""
        # Unpack batch; adapt based on your batch structure
        # if isinstance(batch, (tuple, list)):
        # batch = tuple(b.to(self.device) if torch.is_tensor(b) else b for b in batch)

        with amp.autocast(device_type="cuda", enabled=self.use_amp):
            output = self.model(batch, single_eval_pos=batch.single_eval_pos)

            targets = batch.y[batch.single_eval_pos :, ...]

            feedback = self.callback_handler.on_event(
                "on_forward_end", batch=batch, output=output, targets=targets
            )

            output = feedback.get("output", output)
            targets = feedback.get("targets", targets)

            loss = self.criterion(output, targets)

            feedback = self.callback_handler.on_event(
                "on_loss_end", batch=batch, output=output, targets=targets, loss=loss
            )

            loss = feedback.get("loss", loss)

        # 6. Metrics extraction (to satisfy the type hint)
        # Often callbacks return extra metrics (accuracy, etc.)
        metrics = feedback

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

    def end_run(self):
        """End the MLflow run."""
        mlflow.end_run()


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

