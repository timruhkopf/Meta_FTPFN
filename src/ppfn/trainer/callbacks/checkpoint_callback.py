import json
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
import math

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
            name: str = "nll_best"
    ):
        """
        Args:
            save_dir (str): Directory to save checkpoints.
            monitor (str | List): Metric(s) to monitor for improvement.
            mode (str): "min" or "max" to indicate if lower or higher is better.
            name (str): Base name for saved checkpoint files.
        """
        self.save_dir = Path(save_dir)

        if isinstance(monitor, str):
            monitor = [monitor]

        self.monitor = monitor
        self.mode = mode
        self.name = name
        self.best_score = float("inf") if mode == "min" else float("-inf")

        # Ensure directory exists immediately
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def log_on_epoch_end(self, epoch: int, metrics: Dict[str, Any], global_step: int, trainer: Any):
        """Checks for improvement and saves locally."""

        # Mean over multiple monitored metrics if list is provided
        current_scores = [metrics.get(m) for m in self.monitor]
        current_score = sum(current_scores) / len(current_scores)

        # Guard against missing metrics or NaNs
        if current_score is None or math.isnan(current_score):
            return

        # Check for improvement
        is_best = (self.mode == "min" and current_score < self.best_score) or \
                  (self.mode == "max" and current_score > self.best_score)

        if is_best:
            self.best_score = current_score
            self._save_local(epoch, global_step, f"best_{self.name}.pt", metrics, trainer)
            logger.info(f"✨ New best [{self.name}] | {self.monitor}: {current_score:.4f}")

    def _save_local(self, epoch: int, global_step: int, filename: str, metrics: Dict[str, Any],
                    trainer: Any):
        file_path = self.save_dir / filename

        # 1. Filter metrics for JSON (Tensors -> floats)
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                serializable_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                serializable_metrics[k] = v.item()

        # 2. Construct Checkpoint Payload
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if hasattr(trainer, 'scheduler') else None,
            "metrics": serializable_metrics,
            "best_score": self.best_score,
            "monitor": self.monitor
        }

        if getattr(trainer, 'scaler', None):
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

        # 3. Atomic Save (Anti-Corruption)
        temp_path = file_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.replace(file_path)

        # 4. Save Sidecar Metadata
        meta_path = file_path.with_suffix('.json')
        meta_payload = {
            "name": self.name,
            "monitor": self.monitor,
            "best_score": self.best_score,
            "epoch": epoch,
            "step": global_step,
            "metrics_at_save": serializable_metrics
        }
        meta_path.write_text(json.dumps(meta_payload, indent=4))

    def log_on_train_end(self, **kwargs):
        """
        Triggered at the end of training (ideally in a 'finally' block).
        Uploads the best local versions to MLflow.
        """
        if not mlflow.active_run():
            return

        # Locate the best model and its sidecar
        best_pt = self.save_dir / f"best_{self.name}.pt"
        best_json = self.save_dir / f"best_{self.name}.json"

        if best_pt.exists() and best_json.exists():
            logger.info(f"📤 Uploading best {self.name} artifacts to MLflow...")
            # We use artifact_path to keep the MLflow UI organized
            mlflow.log_artifact(str(best_pt), artifact_path="final_checkpoints")
            mlflow.log_artifact(str(best_json), artifact_path="final_checkpoints")
        else:
            logger.warning(f"⚠️ No best checkpoints found for {self.name} to upload.")
