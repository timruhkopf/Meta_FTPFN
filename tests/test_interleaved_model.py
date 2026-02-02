import pytest
import torch
import torch.nn as nn
from ppfn.model.mymodel.cross_fusion import CrossFusion

from ppfn.model.mymodel.interleaved_model import HierarchicalPFN, MyModuleList

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from copy import deepcopy


# Mocking the forward-pass requirement for your specific Batch structure
class MockBatch:
    def __init__(self, x):
        self.x = x
        self.y = torch.zeros(1)
        self.single_eval_pos = 0


@pytest.fixture
def frozen_backbone():
    """A simple backbone where we target 'layer1' for adaptation."""

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(5, 5)
            self.layer2 = nn.Linear(5, 2)

        def forward(self, x, **kwargs):
            # Your HierarchicalPFN expects (x, y) as input tuple
            x_data, y_data = x
            return self.layer2(self.layer1(x_data))

    model = Backbone()
    model.eval()
    return model


def test_checkpoint_integrity(frozen_backbone, tmp_path):
    # 1. Setup Layers
    # We target 'layer1' specifically
    interleaved = {"layer1": nn.Linear(5, 5)}
    model = HierarchicalPFN(frozen_backbone, interleaved)

    # 2. Capture Initial State
    # Note: Because of your wrapping, model.frozen_model.layer1
    # is now a MyModuleList where [0] is adapter, [1] is original
    original_frozen_weight = model.frozen_model.layer1[1].weight.clone()

    # Randomize the adapter to ensure we aren't just loading zeros/defaults
    with torch.no_grad():
        model.interleaved_layers["layer1"].weight.fill_(42.0)

    # 3. Save
    ckpt_path = tmp_path / "adapter.pt"
    model.save(str(ckpt_path))

    # 4. Load into a fresh instance
    new_interleaved = {"layer1": nn.Linear(5, 5)}
    new_model = HierarchicalPFN(frozen_backbone, new_interleaved)

    # Before loading, adapter weights should be different
    assert not torch.allclose(new_model.interleaved_layers["layer1"].weight,
                              model.interleaved_layers["layer1"].weight)

    new_model.load(str(ckpt_path))

    # 5. Assertions
    # A) Adapter weights were restored
    assert torch.allclose(new_model.interleaved_layers["layer1"].weight,
                          torch.tensor(42.0))

    # B) Frozen weights remained identical (index [1] in the wrapper)
    assert torch.allclose(new_model.frozen_model.layer1[1].weight,
                          original_frozen_weight)

    # C) Verify forward pass consistency
    test_input = MockBatch(torch.randn(1, 5))
    with torch.no_grad():
        assert torch.allclose(model(test_input), new_model(test_input))


def test_frozen_parameters_remain_frozen(frozen_backbone):
    interleaved = {"layer1": nn.Linear(5, 5)}
    model = HierarchicalPFN(frozen_backbone, interleaved)

    # The backbone parameters should have requires_grad = False
    assert model.frozen_model.layer1[1].weight.requires_grad is False
    assert model.frozen_model.layer2.weight.requires_grad is False

    # The adapter should have requires_grad = True
    assert model.interleaved_layers["layer1"].weight.requires_grad is True



class MockAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, dim)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def frozen_model():
    """A simple 'frozen' backbone."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    model.eval()
    return model


@pytest.fixture
def interleaved_layers():
    """Definition of the adapters to inject."""
    # We target the first linear layer named '0'
    return {"0": MockAdapter(10)}


def test_hierarchical_pfn_save_load(frozen_model, interleaved_layers, tmp_path):
    # 1. Initialize the model
    model = HierarchicalPFN(frozen_model, interleaved_layers)
    save_path = tmp_path / "adapter_only.pt"

    # 2. Modify adapter weights so they aren't default (to prove loading works)
    with torch.no_grad():
        for param in model.interleaved_layers["0"].parameters():
            param.add_(1.0)

            # Capture state for comparison
    original_adapter_state = {k: deepcopy(v) for k, v in model.state_dict().items()}
    original_frozen_weight = model.frozen_model[0][1].weight.clone()  # Accessing the wrapped layer

    # 3. Save only the adapters
    model.save(str(save_path))

    # 4. Create a fresh model instance (simulating a new process)
    # We pass the same backbone and fresh adapter layers
    new_interleaved = {"0": MockAdapter(10)}
    new_model = HierarchicalPFN(frozen_model, new_interleaved)

    # Verify new model has different adapter weights initially
    with torch.no_grad():
        assert not torch.equal(
            new_model.interleaved_layers["0"].layer.weight,
            model.interleaved_layers["0"].layer.weight
        )

    # 5. Load the saved weights
    new_model.load(str(save_path))

    # 6. Assertions
    # Check that adapter weights match the saved ones
    for name in model.interleaved_layers:
        for p1, p2 in zip(model.interleaved_layers[name].parameters(),
                          new_model.interleaved_layers[name].parameters()):
            assert torch.equal(p1, p2), f"Adapter {name} weights did not match after loading."

    # Check that frozen weights are still what they were (not corrupted/overwritten)
    assert torch.equal(new_model.frozen_model[0][1].weight, original_frozen_weight)

    # 7. Functional Check
    dummy_input = torch.randn(1, 10)

    # Mocking the batch structure your forward expects
    class MockBatch:
        def __init__(self, x):
            self.x = x
            self.y = torch.zeros(1)
            self.single_eval_pos = 0

    batch = MockBatch(dummy_input)

    with torch.no_grad():
        out1 = model(batch)
        out2 = new_model(batch)
        assert torch.allclose(out1, out2), "Model outputs differ after save/load."

#
# @pytest.fixture
# def model_wrapped(get_ft_pfn):
#     """
#     Fixture that wraps the real frozen model with CrossFusion layers.
#     Note: 'ft_pfn' is assumed to be provided by your existing test suite.
#     """
#     ft_pfn = get_ft_pfn
#     interleaved_layers = {
#         "transformer_encoder.layers.0.linear1": CrossFusion(d_model=512, num_heads=8),
#         "transformer_encoder.layers.2.linear1": CrossFusion(d_model=512, num_heads=8),
#     }
#     return HierarchicalPFN(frozen_model=ft_pfn, interleaved_layers=interleaved_layers)
#
# def test_integration_constancy_eval(model_wrapped,get_ft_pfn, ft_batch_factory):
#     """
#     Checks if Stream A and B survive the ENTIRE HierarchicalPFN pass
#     completely unchanged during evaluation.
#     """
#
#     (x, y), single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
#     # In eval mode: B=9 -> R = (9-1)//2 = 4
#     # Stream A: [:, 0, :]
#     # Stream B: [:, 1:5, :]
#     # Stream C: [:, 5:, :]
#     ft_pfn = get_ft_pfn
#     ft_pfn.eval()
#     with torch.no_grad():
#         frozen_out = ft_pfn((x, y), single_eval_pos=single_eval_pos)
#
#     model_wrapped.eval()
#     with torch.no_grad():
#
#
#         # Run full Hierarchical PFN pass
#         # The output of the PFN is usually a tensor (logits or embeddings)
#         output = model_wrapped((x, y), single_eval_pos=single_eval_pos)
#
#         # Check if output is a tuple or tensor depending on your PFN head
#         out_x = output[0] if isinstance(output, tuple) else output
#
#     # CRITICAL TEST:
#     # Even after many layers of PFN, the first 1+R indices of the batch
#     # should be identical to the input because CrossFusion shields them.
#     R = (x.shape[1] - 1) // 2
#
#     # Check Stream A
#     torch.testing.assert_close(out_x[:, :1, :], frozen_out[:, :1, :],
#                                msg="Target Task (Stream A) was corrupted by the PFN pass.")
#
#     # Check Stream B
#     torch.testing.assert_close(out_x[:, 1:R+1, :], frozen_out[:, 1:R+1, :],
#                                msg="Related Marginals (Stream B) were corrupted by the PFN pass.")
#
#     # Check Stream C (Should be different)
#     assert not torch.allclose(out_x[:, R+1:, :], frozen_out[:, R+1:, :]), \
#         "Workspace (Stream C) was not updated by the PFN pass."
#
# def test_integration_constancy_train(model_wrapped, get_ft_pfn, ft_batch_factory):
#     """Checks for constancy in 3R training mode."""
#     (x, y), single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
#     # Batch is 9, so R=3 for training [A:3, B:3, C:3]
#     R = x.shape[1] // 3
#
#     ft_pfn = get_ft_pfn
#     ft_pfn.train()
#     output_frozen = ft_pfn((x, y), single_eval_pos=single_eval_pos)
#
#     model_wrapped.train()
#
#
#     output = model_wrapped((x, y), single_eval_pos=single_eval_pos)
#     out_x = output[0] if isinstance(output, tuple) else output
#
#     # Check Streams A and B (indices 0 to 2*R)
#     torch.testing.assert_close(out_x[:, :2*R, :], output_frozen[:, :2*R, :],
#                                msg="Streams A or B corrupted during training forward pass.")
#
# def test_single_eval_pos_reset(model_wrapped, ft_batch_factory):
#     """Ensures that single_eval_pos is cleaned up to prevent side effects."""
#     batch_data, single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
#
#     model_wrapped.eval()
#     _ = model_wrapped(batch_data, single_eval_pos=single_eval_pos)
#
#     assert model_wrapped.single_eval_pos is None
#     for layer in model_wrapped.interleaved_layers.values():
#         assert layer.single_eval_pos is None
#
# def test_pfn_param_freezing(model_wrapped):
#     """Verify that only CrossFusion layers have gradients."""
#
#     # Check an arbitrary frozen weight
#     intercepted = model_wrapped.interleaved_layers.keys()
#     for name, param in model_wrapped.frozen_model.named_parameters():
#         if any(name.startswith(n) for n in intercepted):
#             continue
#
#         assert not param.requires_grad, f"Parameter {name} was not frozen!"
#
#     # Check CrossFusion weight
#     for layer in model_wrapped.interleaved_layers.values():
#         for param in layer.parameters():
#             assert param.requires_grad, "CrossFusion layer was accidentally frozen!"