import torch
import torch.nn as nn
from typing import Mapping, Union, List

from ppfn.model.mymodel.stream_parser import StreamParser
from ppfn.model.mymodel.layers.adapter_wrapper import AdapterWrapper
from ppfn.model.mymodel.meta_context import ForwardMetaContext

from ppfn.utils.mybatch import MyBatch

import logging

logger = logging.getLogger(__name__)


class PPFN(nn.Module):
    def __init__(
            self,
            frozen_model: nn.Module,
            interleaved_layers: Union[Mapping[str, nn.Module], list],
            stream_parser=StreamParser(),
            pass_hp_from_frozen: bool = False,
            pass_hp_as_rawpaded: bool = False,
            seq_len=1000,
    ):
        super().__init__()
        self.frozen_model = frozen_model
        self.seq_len = seq_len
        self.stream_parser = stream_parser
        self.pass_hp_from_frozen = pass_hp_from_frozen
        self.pass_hp_as_rawpaded = pass_hp_as_rawpaded

        assert not (
                    self.pass_hp_from_frozen and self.pass_hp_as_rawpaded), "Cannot pass HP from frozen model and as raw padded input at the same time. Choose one."

        # Check paddability
        self.paddable = not hasattr(self.frozen_model, "TransformerModel")

        # Freeze the pre-trained model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

            # Normalize interleaved_layers format
        if isinstance(interleaved_layers, list):
            self.interleaved_layers = {item["name"]: item["layer"] for item in interleaved_layers}
        else:
            self.interleaved_layers = interleaved_layers

        # add a backlink to the adapter layer for telemetry purposes
        for name, layer in self.interleaved_layers.items():
            layer.address = name

        self._params_registry = nn.ModuleList(self.interleaved_layers.values())
        self._inject_wrappers()

        logger.info \
            (f"FT_PPFN initialized. Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _inject_wrappers(self):
        """Replaces target modules with our Context-Aware Wrappers."""
        frozen_modules = dict(self.frozen_model.named_modules())
        for name, adapter_layer in self.interleaved_layers.items():
            if name not in frozen_modules:
                raise ValueError(f"Target module '{name}' not found in the frozen model.")

            original_module = frozen_modules[name]
            wrapped_module = AdapterWrapper(original_module, adapter_layer)

            # Set the wrapped module back
            parts = name.split(".")
            parent = self.frozen_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], wrapped_module)

    def forward(self, batch, src_key_padding_mask=None, **kwargs):
        """
        1. Parse the batch into A/B/C streams and apply any specified mutations (e.g., ForceSameQuery, AppendATrainToBTest).
        2. Store necessary context (like single_eval_pos, HP coordinates) in the MetaContext for the adapters to access during their forward pass.
        3. Pass the modified batch through the frozen model, which will now include the injected AdapterWrappers that can utilize the MetaContext to
            manipulate the streams as needed.
        """
        # Clear any existing meta context at the start of forward to avoid leakage between batches
        ForwardMetaContext.clear()

        hp = None
        if self.pass_hp_from_frozen:
            hp = self.frozen_model.encoder.configuration_enc(batch.x[:, :, 2:]) if self.pass_hp_from_frozen else None

        if self.pass_hp_as_rawpaded:
            hp = batch.x[:, :, 2:]  # Assuming the last D-2 dimensions are HP coordinates
            hp = torch.nn.functional.pad(hp, (0, 10 - hp.shape[2]))  # Pad to a fixed size of 10 if needed

        # Parse into A, B, C streams and maniputate as needed forcing the same query
        batch, src_key_padding_mask, hp_tuple = self.stream_parser(
            batch,
            hp=hp,
            src_key_padding_mask=src_key_padding_mask if self.paddable else None,
        )

        # provide meta context to the adapter forward, that would break the pfn's forward signature
        ForwardMetaContext.set(
            single_eval_pos=batch.single_eval_pos,
            # FIXME: @amir: this is indexing is FT-PFN specific!!!
            # rather than the projection, we can directly take the raw HP of variable size
            # and zero pad it to a fixed size (compare VariableNumFeaturesEncoder)
            hp=hp_tuple
        )


        kwargs['single_eval_pos'] = batch.single_eval_pos  # Ensure this is available for the criterion if needed
        if self.paddable:
            kwargs['src_key_padding_mask'] = src_key_padding_mask

        x = (batch.x, batch.y)
        output = self.frozen_model(x, **kwargs)
        output = self.stream_parser.splice_at_fwd_end(output, batch)
        return output

    @property
    def criterion(self):
        return self.frozen_model.criterion

    def joint_prediction(self, batch: MyBatch, **kwargs):
        """
        Override joint_prediction to ensure we return the correct part of the output
        corresponding to the target conditional predictions (Stream C).
        """
        output = self.forward(batch, **kwargs)

        attn_scores = ForwardMetaContext.get_stats()

        # Extract Stream C (the last R tasks in the batch)
        B = output.shape[1]
        R = B // 3
        stream_c_logits = output[:, -R:, ...]  # Stream C

        # 1. Convert to probabilities (softmax over the 1k classes)
        probs = torch.softmax(stream_c_logits, dim=-1)

        # 2. Average over the R related tasks (dim=1)
        joint_probs = probs.mean(dim=1)  # shape: [T_test, 1000]

        # 3. (Optional) convert back to log-probs if your pipeline expects them
        joint_log_probs = torch.log(joint_probs + 1e-8)

        return joint_log_probs

    def state_dict(self, *args, **kwargs):
        """
        Gathers state_dicts from all interleaved layers into a nested dictionary.
        """
        return {
            name: layer.state_dict(*args, **kwargs)
            for name, layer in self.interleaved_layers.items()
        }

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Distributes the nested state_dict back to the individual layers.
        """
        # Clean the state_dict keys if they come from a DDP save
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        for name, layer_state in state_dict.items():
            if name in self.interleaved_layers:
                self.interleaved_layers[name].load_state_dict(
                    layer_state, strict=strict
                )
            elif strict:
                # If we are loading a full trainer checkpoint, it might have
                # other keys; we only care about the keys in our registry.
                logger.warning(
                    f"Key '{name}' in state_dict not found in interleaved_layers."
                )

    def save(self, path: str):
        """Saves only the relevant interleaved weights."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """Loads weights from a file into the current instance."""
        state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_checkpoint(
            cls,
            path: str,
            frozen_model: nn.Module,
            interleaved_layers: Union[Mapping[str, nn.Module], List],
    ):
        """Factory method: Creates the wrapper and loads weights immediately."""
        instance = cls(frozen_model, interleaved_layers)
        instance.load(path)  # Calls the instance method above
        return instance


if __name__ == '__main__':

    import torch
    import torch.nn as nn
    from dataclasses import dataclass

    from ppfn.model.mymodel.layers.nw_adapter import NadarayaWatsonAdapter
    from ppfn.model.mymodel.stream_mutations import ForceSameQueryMutation, AppendATrainToBTestMutation


    # --- 1. DUMMY DEPENDENCIES ---
    @dataclass
    class MyBatch:
        x: torch.Tensor
        y: torch.Tensor
        target_y: torch.Tensor
        single_eval_pos: int


    class DummyEncoder(nn.Module):
        def configuration_enc(self, x):
            # Mocks generating HP coordinates
            T, B, D = x.shape
            # the pfn has encoder.configuration_enc(batch.x[:, :, 2:]) so we need to patch here +2 to avoid size mismatch in the test
            return torch.randn(T, B, D + 2)


    class DummyFrozenTransformer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.encoder = DummyEncoder()
            # A simple mock sequential layer to represent the transformer blocks
            self.layer_1 = nn.Linear(d_model, d_model)

        def forward(self, inputs, **kwargs):
            x, y = inputs
            # The frozen model just processes the concatenated batch normally
            return self.layer_1(x)


    device = torch.device("cpu")
    seq_len = 100
    batch_size = 12  # Must be even for train interleaving
    d_model = 14
    sep = 50

    # 1. Create Dummy Data
    print("Generating batch...")
    x = torch.randn(seq_len, batch_size, d_model)
    y = torch.randn(seq_len, batch_size)
    batch = MyBatch(x=x, y=y, target_y=y, single_eval_pos=sep)

    # 2. Initialize Models
    print("Initializing architecture...")
    frozen_model = DummyFrozenTransformer(d_model=d_model)

    # Instantiate your actual adapter
    nw_adapter = NadarayaWatsonAdapter(
        d_model=d_model, n_heads=2, seq_len=seq_len
    )
    nw_adapter.layer_name = "adapter_layer_1"  # Name it for telemetry

    # Map the adapter to inject it at "layer_1" of the frozen model
    interleaved_layers = {"layer_1": nw_adapter}

    model = PPFN(
        frozen_model=frozen_model,
        interleaved_layers=interleaved_layers,
        stream_parser=StreamParser(
            stream_mutations=[ForceSameQueryMutation(), AppendATrainToBTestMutation()]
        ),
        pass_hp_from_frozen=True,
        seq_len=seq_len
    )

    model.train()  # Sets self.training = True

    # 3. Forward Pass
    print("\nExecuting Forward Pass...")
    out = model(batch)
    print(f"Output shape: {out.shape}")  # Expected: (Seq_len, 3 * (Batch/2), d_model) -> (100, 6, 16)

    # 4. Extract telemetry
    stats = ForwardMetaContext.get_stats()
    print("\nExtracted Attention Statistics:")
    for layer, metrics in stats.items():
        print(f"  Layer: {layer}")
        for metric_name, tensor in metrics.items():
            print(f"    - {metric_name}: shape {tensor.shape}")
