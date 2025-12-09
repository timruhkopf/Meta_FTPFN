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
        dual_path: bool = True,
    ):
        """
        :param frozen_model: Transformer model to be frozen and injected with new layers
        :param interleaved_layers: Dictionary mapping of layer names ('addresses') of frozen_model with new layers to be interleaved
        """
        super().__init__()
        self.frozen_model = frozen_model
        self._dual_path = dual_path

        # Store interleaved layers as a ModuleList to register them properly
        self.interleaved_layers = nn.ModuleList(interleaved_layers.values())

        # Keep mapping from target module to interleaved layer
        self._layer_mapping = {}
        for i, (name, interleaved_layer) in enumerate(
            zip(interleaved_layers.keys(), self.interleaved_layers)
        ):
            target_module = dict(self.frozen_model.named_modules())[name]
            self._layer_mapping[id(target_module)] = interleaved_layer

            # Register forward hook
            target_module.register_forward_hook(
                self._make_hook(interleaved_layer, is_first=(i == 0))
            )

    # def _make_hook(self, interleaved_layer):
    #     """Create a forward hook that applies the interleaved layer."""
    #     def hook(module, input, output):
    #         return interleaved_layer(output)
    #     return hook

    def _make_hook(self, interleaved_layer, is_first=False):
        """Create a forward hook that applies the interleaved layer.
        Added twist: if dual_path is enabled, concatenate original and augmented outputs."""

        def hook(module, input, output):
            if self._dual_path:
                inp = output[:, : output.shape[1] // 2, :] if not is_first else output
                augmented = interleaved_layer(inp)
                return torch.cat([output, augmented], dim=1)
            else:
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
        return self.frozen_model(*args, **kwargs)


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from ifbo.surrogate import FTPFN

    load_dotenv(dotenv_path=Path(__file__).parents[4] / ".env")

    model_path = os.getenv("MODELDIR") + "pfn_ckpt"
    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device="cpu"
    ).model

    criterion = frozen_model.criterion
    print(f"Loaded PFN model with criterion: {criterion}")

    T, B, D = 32, 8, 5  # example: sequence length, batch size, feature dim
    assert D >= 1 and D <= 11  # 1 int + up to 10 float features

    # First feature: integer in [0, 1000]
    ints = torch.randint(low=0, high=1001, size=(T, B, 1))  # [T, B, 1][web:1]

    # Remaining features: floats in [0, 1)
    num_float_feats = D - 1
    if num_float_feats > 0:
        floats = torch.rand(T, B, num_float_feats)  # [T, B, D-1][web:5]
        x = torch.cat([ints.float(), floats], dim=-1)  # [T, B, D][web:3]
    else:
        x = ints.float()  # [T, B, 1]

    # x has shape [T, B, D] with desired constraints
    print(x.shape, x.min(), x.max())

    target_sentence = x[:, :1, :]
    batch_of_related_sentences = x[:, 1:, :]

    print(f"\nTesting forward pass with dummy data...")
    print(f"Target shape: {target_sentence.shape}")
    print(f"Context shape: {batch_of_related_sentences.shape}")
