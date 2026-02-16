import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import logging
import math

import time
import threading
from concurrent.futures import ThreadPoolExecutor

import mlflow
import torch

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback

logger = logging.getLogger(__name__)


class CheckpointCallback(AbstractCallback):
    """
    Refined Checkpoint Callback for PFN Training.

    - Atomic saving (via .tmp and rename) to prevent corruption.
    - JSON sidecar with serializable metrics for fast inspection.
    - Pathlib for cross-platform safety.
    - MLflow integration triggered at the end of training.
    """

    def __init__(
            self,
            save_dir: str = "checkpoints",
            monitor: Union[str, List] = "val/nll_diff",
            mode: str = "min",
            name: str = "nll_best",
            resume_from: Optional[Union[str, Path]] = None,  # New: Path to checkpoint
            read_only: bool = False,  # New: If True, skip saving
            min_save_interval: int = 600,
            **kwargs,
    ):
        """
        Args:
            save_dir (str): Directory to save checkpoints.
            monitor (str | List): Metric(s) to monitor for improvement.
            mode (str): "min" or "max" to indicate if lower or higher is better.
            name (str): Base name for saved checkpoint files.
            resume_from (str | Path, optional): Path to a checkpoint to resume from.
            read_only (bool): If True, disables saving checkpoints.
        """
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self.resume_path = Path(resume_from) if resume_from else None
        self.read_only = read_only

        if isinstance(monitor, str):
            monitor = [monitor]

        self.monitor = monitor
        self.mode = mode
        self.name = name
        self.best_score = float("inf") if mode == "min" else float("-inf")

        # Ensure directory exists immediately
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.min_save_interval = min_save_interval
        self.last_save_time = 0
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single worker for sequential saves
        self._pending_snapshot = None  # Store the latest "best" in memory
        self._needs_saving = False  # Track if the RAM version is newer than Disk version

    def on_trainer_init(self):
        """
        Logic to load an existing checkpoint before training begins.
        """
        if self.resume_path and self.resume_path.exists():
            logger.info(f"🔄 Loading checkpoint from: {self.resume_path}")

            # Map location to CPU first to avoid OOM issues before moving to device
            checkpoint = torch.load(self.resume_path, map_location='cpu', weights_only=False)

            # 1. Restore Model & Optimizer state
            self.trainer.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 2. Restore Scheduler (if it exists)
            if "scheduler_state_dict" in checkpoint and hasattr(self.trainer, 'scheduler'):
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # 3. Restore Scaler (for mixed precision)
            if "scaler_state_dict" in checkpoint and getattr(self.trainer, 'scaler', None):
                self.trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # 4. Sync Callback & Trainer state
            self.trainer.start_epoch = checkpoint.get("epoch", 0) + 1
            self.trainer.global_step = checkpoint.get("global_step", 0)
            self.best_score = checkpoint.get("best_score", self.best_score)

            logger.info(f"✅ Resumed from epoch {checkpoint.get('epoch')} "
                        f"with best score {self.best_score:.4f}")
        elif self.resume_path:
            logger.warning(f"⚠️ Checkpoint path {self.resume_path} was provided but does not exist.")

    def log_on_epoch_end(self, epoch: int, metrics: Dict[str, Any]):
        """Decides if we should save, then triggers background task."""
        # 1. Calculate Score
        current_scores = [metrics.get(m) for m in self.monitor if metrics.get(m) is not None]
        if not current_scores: return
        current_score = sum(current_scores) / len(current_scores)

        # 2. Check Logic (Improvement + Time Guard)
        is_best = (self.mode == "min" and current_score < self.best_score) or \
                  (self.mode == "max" and current_score > self.best_score)

        now = time.time()
        time_since_last = now - self.last_save_time

        if is_best:
            self.best_score = current_score
            # Always snapshot the best state to CPU memory immediately
            # This ensures we have the best version even if we don't write to disk yet
            self._pending_snapshot = self._prepare_snapshot(epoch, metrics)
            self._needs_saving = True  # <--- Mark as "dirty" (needs disk sync)
            logger.info(f"✨ New best captured in memory (Epoch {epoch}: {current_score:.4f})")

            # 3. Time Guard for Disk I/O
        now = time.time()
        time_since_last = now - self.last_save_time

        if self._needs_saving and time_since_last >= self.min_save_interval:
            self._trigger_save(now)

    def _trigger_save(self, timestamp: float):
        """Internal helper to push pending snapshot to the background thread."""
        if self._pending_snapshot:
            snapshot = self._pending_snapshot
            # We clear the pending ref (or keep it, but trigger_save handles the logic)
            self._executor.submit(self._save_local_worker, snapshot)
            self.last_save_time = timestamp
            self._needs_saving = False  # <--- Disk is now catching up to RAM
            # We can keep _pending_snapshot as a reference, but
            # we've successfully offloaded this version to the worker.
            logger.info(f"💾 Async disk save started for epoch {snapshot['epoch']}")

    def _prepare_snapshot(self, epoch: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Snapshots current state into CPU memory. Runs on MAIN thread."""
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                serializable_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                serializable_metrics[k] = v.detach().cpu().item()

        # Deep copy/move weights to CPU to prevent race conditions during training
        snapshot = {
            "epoch": epoch,
            "global_step": self.trainer.global_step,
            "model_state_dict": {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in self.trainer.model.state_dict().items()
            },
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),  # Usually small/CPU
            "metrics": serializable_metrics,
            "best_score": self.best_score,
            "filename": f"best_{self.name}.pt"
        }

        if hasattr(self.trainer, 'scheduler'):
            snapshot["scheduler_state_dict"] = self.trainer.scheduler.state_dict()
        if getattr(self.trainer, 'scaler', None):
            snapshot["scaler_state_dict"] = self.trainer.scaler.state_dict()

        return snapshot

    def _save_local_worker(self, snapshot: Dict[str, Any]):
        """The actual I/O logic. Runs on BACKGROUND thread."""
        if self.read_only: return

        try:
            file_path = self.save_dir / snapshot["filename"]

            # 1. Atomic Save of .pt
            temp_path = file_path.with_suffix('.tmp')
            torch.save(snapshot, temp_path)
            temp_path.replace(file_path)

            # 2. Save Sidecar JSON
            meta_path = file_path.with_suffix('.json')
            meta_payload = {
                "name": self.name,
                "best_score": snapshot["best_score"],
                "epoch": snapshot["epoch"],
                "step": snapshot["global_step"],
                "metrics_at_save": snapshot["metrics"]
            }
            meta_path.write_text(json.dumps(meta_payload, indent=4))
        except Exception as e:
            logger.error(f"❌ Background save failed: {e}")

    def on_train_end(self, **kwargs):
        """
        Triggered at the end of training (ideally in a 'finally' block).
        Uploads the best local versions to MLflow.
        """
        # 1. If we have a pending best that never hit the disk due to the timer:
        if self._needs_saving and self._pending_snapshot:
            logger.info("💾 Final sync: Saving the last 'best' that was throttled by the timer.")
            self._save_local_worker(self._pending_snapshot)

        # 2. Standard shutdown logic
        logger.info("⏳ Training ended. Waiting for final background saves...")
        self._executor.shutdown(wait=True)

        if not mlflow.active_run():
            return

        # Locate the best model and its sidecar
        best_pt = self.save_dir / f"best_{self.name}.pt"
        best_json = self.save_dir / f"best_{self.name}.json"
        if best_pt.exists() and best_json.exists() and not self.read_only:
            logger.info(f"📤 Uploading best {self.name} artifacts to MLflow...")
            # We use artifact_path to keep the MLflow UI organized
            mlflow.log_artifact(str(best_pt), artifact_path="final_checkpoints")
            mlflow.log_artifact(str(best_json), artifact_path="final_checkpoints")
        else:
            logger.warning(f"⚠️ No best checkpoints found for {self.name} to upload.")
