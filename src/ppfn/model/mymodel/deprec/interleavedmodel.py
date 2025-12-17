import torch

import torch.nn as nn
from typing import Mapping


class InterleavedModel(nn.Module):
    """
    Model that interleaves frozen pre-trained layers with trainable cross-attention layers.
    """

    def __init__(
        self,
        frozen_model: nn.Module,
        interleaved_layers: Mapping[str, nn.Module],
        pre_hook: bool = True,
    ):
        """
        :param frozen_model: Transformer model to be frozen and injected with new layers
        :param interleaved_layers: Dictionary mapping of layer names ('addresses') of frozen_model with new layers to be interleaved. Make sure the order is correct; the first layer in the dict will recieve is_first=True.
        :param pre_hook: If True, use forward pre-hooks to insert interleaved layers. Otherwise, use forward hooks (post layer application).
        """
        super().__init__()
        self.frozen_model = frozen_model
        self._single_eval_pos = None

        # Store interleaved layers as a ModuleList to register them properly
        self.interleaved_layers = nn.ModuleList(interleaved_layers.values())

        # Keep mapping from target module to interleaved layer
        self._layer_mapping = {}
        self.hook_handles = []
        for i, (name, interleaved_layer) in enumerate(
            zip(interleaved_layers.keys(), self.interleaved_layers)
        ):
            target_module = dict(self.frozen_model.named_modules())[name]
            self._layer_mapping[id(target_module)] = interleaved_layer

            if pre_hook:
                # Register forward pre-hook
                handle = target_module.register_forward_pre_hook(
                    self._make_pre_hook(interleaved_layer, is_first=(i == 0))
                )
            else:
                # Register forward hook
                handle = target_module.register_forward_hook(
                    self._make_hook(interleaved_layer, is_first=(i == 0))
                )

            self.hook_handles.append(handle)

    @property
    def single_eval_pos(self):
        return self._single_eval_pos

    @single_eval_pos.setter
    def single_eval_pos(self, value):
        for layer in self.interleaved_layers:
            if hasattr(layer, 'single_eval_pos'):
                layer.single_eval_pos = value

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def _make_pre_hook(self, interleaved_layer, is_first=False):
        """Create a forward pre-hook that applies the interleaved layer."""
        interleaved_layer.is_first = is_first # post instantiation
        def hook(module, input):
           return interleaved_layer(input)
            
        return hook

    def _make_hook(self, interleaved_layer, is_first=False):
        """Create a forward hook that applies the interleaved layer."""
        interleaved_layer.is_first = is_first
        def hook(module, input, output):
            return interleaved_layer(output)
        return hook

    def trainable_parameters(self):
        """Iterator over trainable parameters (i.e., interleaved layers)."""
        for layer in self.interleaved_layers:
            yield from layer.parameters()

    def forward(
        self,
        *args,
        **kwargs,
    ):
        """
        Forward pass through the interleaved model.

        Args:
            *args: Arguments for the frozen model
            **kwargs: Additional arguments for the frozen model

        Returns:
            output: Final output from the augmented model
        """
        # communicate this batch's single_eval_pos to interleaved layers, since pre-hooks don't get **kwargs
        self.single_eval_pos = kwargs.get('single_eval_pos', None)
        result =  self.frozen_model(*args, **kwargs)
        self.single_eval_pos = None # reset after forward
        
        return result
    
