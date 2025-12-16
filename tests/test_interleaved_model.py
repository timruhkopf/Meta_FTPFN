# tests/test_interleaved_model.py
from copy import deepcopy
import pytest
import torch
import torch.nn as nn
from ppfn.model.mymodel.interleavedmodel import InterleavedModel


class FrozenEncoder(nn.Module):
    """Dummy Model to simulate frozen layers and inject new layers"""

    def __init__(self, d_model=64, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TinyMLP(nn.Module):
    """Interleaving linear layers with residual connection"""

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, inp):
        x = inp[0]
        return self.net(x) + x


@pytest.fixture
def d_model():
    """Dimension of model embeddings"""
    return 64


@pytest.fixture
def raw_model(d_model):
    """Setup raw frozen model without interleaving"""
    torch.manual_seed(42)
    model = FrozenEncoder(d_model)
    return model, d_model


@pytest.fixture
def setup_model(d_model):
    """Setup interleaved model with one trainable layer"""
    torch.manual_seed(42)
    frozen = FrozenEncoder(d_model)

    model = InterleavedModel(frozen, interleaved_layers={"layers.1": TinyMLP(d_model)})
    return model, d_model


@pytest.fixture
def optimizer(setup_model):
    """Setup optimizer for trainable parameters"""
    model, _ = setup_model
    optim = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
    return optim


@pytest.fixture
def dummy_input(d_model, batch_size=10, seq_len=32):
    """Generate dummy input tensor"""
    torch.manual_seed(42)
    return torch.randn(seq_len, batch_size, d_model)



def test_different_outputs(dummy_input, raw_model):
    """Verify outputs differ correctly"""
    

    frozen_out = raw_model[0](dummy_input)

    model = InterleavedModel(raw_model[0], interleaved_layers={"layers.1": TinyMLP(raw_model[1])})
    interleaved_out = model(dummy_input)

    assert not torch.equal(frozen_out, interleaved_out), (
        "outputs are the same; interleaving failed!"
    )



def test_gradients_only_interleaved(setup_model, optimizer, ft_pfn, dummy_input):
    """Verify gradients flow ONLY to interleaved layers"""
    model, d_model = setup_model

    state = deepcopy(model.state_dict())

    optimizer.zero_grad()

    # Forward + backward
    out = model(dummy_input)
    loss = out.sum()
    loss.backward()

    optimizer.step()
    for name, param in model.named_parameters():
        if "interleaved_layers" in name:
            # Check that interleaved layers updated
            assert not torch.equal(state[name], model.state_dict()[name]), (
                f"Interleaved layer {name} did not update!"
            )
        else:
            # Check frozen layers have no gradients
            assert torch.equal(state[name], model.state_dict()[name]), (
                f"Frozen layer {name} updated!"
            )


# Run with: pytest tests/test_interleaved_model.py -v
