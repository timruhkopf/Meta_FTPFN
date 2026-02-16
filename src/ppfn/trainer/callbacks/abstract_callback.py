
from typing import Dict


class AbstractCallback:
    """Base class for training callbacks."""
    def __init__(self, verbose:bool= False):
        self.verbose = verbose
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_trainer_init(self, **kwargs):
        """Called at the end of trainer initialization; allows e.g. the checkpoint callback to load the trainer checkpoint."""
        pass

    def on_train_start(self, **kwargs):
        pass

    def on_epoch_start(self, epoch: int, **kwargs):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of an epoch. Can return a dict of metrics to log.
        These will be new entries to the trainer's metrics dict after all on_epoch_end
        calls are done. The final and complete dict will be passed to log_on_epoch_end."""
        pass

    def log_on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        """This allows us to separate logging from other epoch end actions; this way
        we know for sure, that the on_epoch_end computations are done before logging."""
        pass

    def on_forward_end(self, batch, single_eval_pos, output, targets) -> Dict:
        pass

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def log_on_train_end(self, **kwargs):
        pass

    def on_clipping(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs) -> Dict:
        pass



class CallbackHandler:
    """On event occurrence, call only callbacks with actual implementations."""

    def __init__(self, callbacks: dict, trainer):
        # At this point, 'callbacks' is already { 'mlflow': MLflowCallbackObj, ... }
        self.trainer = trainer

        # We store the instances. We use .values() because the keys ('mlflow', etc.)
        # are useful for config but the handler cares about the objects.
        self.callbacks = list(callbacks.values())

        for cb in self.callbacks:
            # Initialize the trainer on the callback
            cb.set_trainer(self.trainer)

        # Pre-build the cache
        self._method_cache = self._build_cache()

    def _build_cache(self):
        """Scans all callbacks once to map implemented events."""
        all_events = [
            method_name for method_name in dir(AbstractCallback)
            if callable(getattr(AbstractCallback, method_name))
               and not method_name.startswith("_")
               and method_name != "set_trainer"
        ]

        cache = {}
        for cb in self.callbacks:
            for event_name in all_events:
                cb_method = getattr(cb, event_name)
                # We need the function object from the base class for comparison
                base_method = getattr(AbstractCallback, event_name)

                # Robust check:
                # Use __func__ for regular methods to see if they point to the same code
                if getattr(cb_method, "__func__", None) is not base_method:
                    cache.setdefault(event_name, []).append(cb_method)

        return cache

    def on_event(self, event_name: str, *args, **kwargs):
        feedback = {}

        # O(1) lookup: No inspection, no getattr, just execution
        methods = self._method_cache.get(event_name, [])

        for method in methods:
            res = method(*args, **kwargs)
            if isinstance(res, dict):
                feedback.update(res)

        return feedback