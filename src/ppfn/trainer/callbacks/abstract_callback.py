
from typing import Dict


class AbstractCallback:
    """Base class for training callbacks."""

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_forward_end(self, batch, output, targets) -> Dict:
        pass

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass


class CallbackHandler:
    """On event occurence, call all registered callbacks."""

    def __init__(self, callbacks, trainer):
        self.callbacks = callbacks
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_event(self, event_name: str, *args, **kwargs):
        feedback = {}
        for callback in self.callbacks:
            
            method = getattr(callback, event_name, None)
            if callable(method):
                D = method(*args, **kwargs)
                if D is not None:
                    feedback.update(D)

        return feedback