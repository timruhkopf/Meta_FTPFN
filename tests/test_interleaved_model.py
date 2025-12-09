# tests/test_interleaved_model.py
from copy import deepcopy
import pytest
import torch
import torch.nn as nn
from ppfn.model.ppfn.cross_attn import InterleavedModel

class FrozenEncoder(nn.Module):
    """Dummy Model to simulate frozen layers and inject new layers"""
    def __init__(self, d_model=64, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TinyMLP(nn.Module):
    """Interleaving linear layers with residual connection"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        return self.net(x) + x
    
@pytest.fixture
def raw_model():
    """Setup raw frozen model without interleaving"""
    torch.manual_seed(42)
    d_model = 64
    model = FrozenEncoder(d_model)
    return model, d_model

@pytest.fixture
def setup_model():
    """Setup interleaved model with one trainable layer"""
    torch.manual_seed(42)
    d_model = 64
    frozen = FrozenEncoder(d_model)
    interleaved_layers = {"layers.1": TinyMLP(d_model)}
    
    model = InterleavedModel(frozen, interleaved_layers)
    return model, d_model

@pytest.fixture
def optimizer(setup_model):
    """Setup optimizer for trainable parameters"""
    model, _ = setup_model
    optim = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    return optim

def test_gradients_only_interleaved(setup_model, optimizer):
    """Verify gradients flow ONLY to interleaved layers"""
    model, d_model = setup_model
    
    state = deepcopy(model.state_dict())

    optimizer.zero_grad()

    # Forward + backward
    x = torch.randn(10, 32, d_model, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    optimizer.step()
    for name, param in model.named_parameters():
        if 'interleaved_layers' in name:
            # Check that interleaved layers updated
            assert not torch.equal(state[name], model.state_dict()[name]), f"Interleaved layer {name} did not update!"
        else: 
            # Check frozen layers have no gradients
            assert torch.equal(state[name], model.state_dict()[name]), f"Frozen layer {name} updated!"

def test_different_outputs_dual_path(setup_model, raw_model):
    """Verify outputs differ correctly when dual_path=True"""
    model, d_model = setup_model
    model._dual_path = True  # Explicitly set dual path mode
    
    x = torch.randn(10, 32, d_model)

    frozen_out = raw_model[0](x)
    interleaved_out = model(x)

    half = interleaved_out.shape[1] // 2
    assert torch.allclose(
        frozen_out, interleaved_out[:, :half, :]
    ), "First half outputs differ; dual path failed!"
    assert not torch.allclose(
        frozen_out, interleaved_out[:, half:, :]
    ), "Second half outputs identical; interleaving had no effect!"


def test_different_outputs_single_path(setup_model, raw_model):
    """Verify outputs differ correctly when dual_path=False"""
    model, d_model = setup_model
    model._dual_path = False  # Disable dual path

    x = torch.randn(10, 32, d_model)

    frozen_out = raw_model[0](x)
    interleaved_out = model(x)

    assert not torch.allclose(
        frozen_out, interleaved_out
    ), "Outputs are identical; interleaving had no effect!"


def test_forward_shape_preserved(setup_model):
    """Verify output shape unchanged after interleaving"""
    model, d_model = setup_model
    model._dual_path = False  # Test shape preservation in single path mode
    
    x = torch.randn(10, 32, d_model)
    out = model(x)
    
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

# Run with: pytest tests/test_interleaved_model.py -v
