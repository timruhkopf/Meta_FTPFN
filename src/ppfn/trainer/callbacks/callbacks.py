from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
import mlflow
from typing import Dict


class MLflowCallback(AbstractCallback):
    """Log metrics to MLflow."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        if step % self.log_frequency == 0:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=epoch * 1000 + step)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        for key, value in metrics.items():
            mlflow.log_metric(f"epoch_{key}", value, step=epoch)


class PrintCallback(AbstractCallback):
    """Print training progress."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        print(f"Epoch {epoch}: {metrics}")