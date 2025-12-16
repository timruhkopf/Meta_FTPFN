
from typing import Dict


class AbstractCallback:
    """Base class for training callbacks."""

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass
