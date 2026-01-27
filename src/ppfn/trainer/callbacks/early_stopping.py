from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from typing import Dict

import logging

logger = logging.getLogger(__name__)


class EarlyStopping(AbstractCallback):
    CKPT_WARMUP_EPOCHS = 100

    def __init__(
            self,
            monitor: str = "val_loss",
            patience: int = 5,
            min_delta: float = 0.0,
            mode: str = "min",
            checkpoint_name: str = "best_model.pt"
    ):
        """
        Args:
            monitor: The metric name to track (stored in trainer.metrics or similar).
            patience: How many epochs to wait after last time the monitor improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: One of {"min", "max"}. In "min" mode, training stops when the quantity
                  monitored has stopped decreasing.
            checkpoint_name: Filename for the saved checkpoint.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_name = checkpoint_name

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict = None) -> Dict:

        # 1. Retrieve the current metric value
        # Assuming your trainer stores current epoch metrics in a 'logs' dict or similar
        current_score = metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"EarlyStopping monitored metric '{self.monitor}' not found.")
            return {}

        # 2. Check if the score has improved
        if self.best_score is None and epoch > self.CKPT_WARMUP_EPOCHS:
            self._save_and_update(current_score)
        else:
            if self.mode == "min":
                improved = current_score < (self.best_score - self.min_delta)
            else:
                improved = current_score > (self.best_score + self.min_delta)

            if improved and epoch > self.CKPT_WARMUP_EPOCHS:
                self._save_and_update(current_score)
                self.counter = 0  # Reset counter
            else:
                self.counter += 1
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

                if self.counter >= self.patience:
                    logger.info("Early stopping triggered. Terminating training.")
                    # We set a flag on the trainer that the training loop should check
                    return {'stop_training': True}

    def _save_and_update(self, score):
        """Helper to update best score and trigger trainer's checkpointing."""
        self.best_score = score
        logger.info(f"Metric {self.monitor} improved. Saving checkpoint: {self.checkpoint_name}")

        # Calling the specific method you mentioned
        self.trainer._save_checkpoint(filename=self.checkpoint_name)