import torch

import threading
from typing import Mapping, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class ForwardMetaContext:
    """
    Thread-local storage for side-channeling data to adapters.

    The frozen PFN model has certain expertation about its input format and dataflow,
    which is not directly compatible with the needs of the interleaved adapter layers.
    To bridge this gap, MetaContext provides a thread-local storage mechanism that allows us to
    store and retrieve out-of-band data (like the hyperparameter coordinates, the position of the single evaluation task, etc.)
    and store attention statistics for telemetry purposes, without modifying the standard dataflow of the frozen model.

    That is, this class allows us to sidestep the standard input/output of the frozen model and provide the necessary context to the adapter layers.
    """
    _state = threading.local()

    @classmethod
    def set(cls, **kwargs):
        for k, v in kwargs.items():
            setattr(cls._state, k, v)

    @classmethod
    def get(cls, key, default=None):
        return getattr(cls._state, key, default)

    @classmethod
    def clear(cls):
        cls._state.__dict__.clear()

    # (Telemetry) Logging utilities for attention statistics -----------
    @classmethod
    def log_stats(cls, layer_name, stats_dict):
        if not hasattr(cls._state, 'attention_stats'):
            cls._state.attention_stats = {}
        cls._state.attention_stats[layer_name] = stats_dict

    @classmethod
    def get_stats(cls):
        return getattr(cls._state, 'attention_stats', {})
