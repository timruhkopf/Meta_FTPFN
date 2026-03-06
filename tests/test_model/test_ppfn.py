import os
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch

from ppfn.model.mymodel.ppfn_model import PPFN



# --- Mocking the Frozen Backbone ---

class DummyFrozenModel(nn.Module):
    """A lightweight mock of the FTPFN backbone to test layer injection."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        # We need target modules that match the names we want to inject into
        self.encoder.layer1 = nn.Linear(10, 10)
        self.encoder.layer2 = nn.Linear(10, 10)
        self.criterion = nn.MSELoss()

    def forward(self, x, **kwargs):
        return x


@pytest.fixture
def mock_frozen_model():
    """Fixture to bypass the heavy file I/O of the real PFN during unit testing."""
    with patch("ppfn.utils.load_ftpfn.load_frozen_model") as mock_load:
        # Replace the heavy loaded model with our dummy model
        mock_load.return_value = DummyFrozenModel()
        yield mock_load.return_value


@pytest.fixture
def adapter_layers():
    """Creates dummy adapters to interleave."""
    return {
        "encoder.layer1": nn.Linear(10, 10),
        "encoder.layer2": nn.Linear(10, 10)
    }


@pytest.fixture
def ppfn_model(mock_frozen_model, adapter_layers):
    """Initializes the PPFN orchestrator with the mocked frozen model."""
    # Note: Using pass_hp_as_rawpaded=False to bypass the need for a complex batch in basic I/O tests
    return PPFN(
        frozen_model=mock_frozen_model,
        interleaved_layers=adapter_layers,
        pass_hp_from_frozen=False,
        pass_hp_as_rawpaded=False
    )


# --- Test Cases ---

def test_custom_state_dict_isolation(ppfn_model):
    """
    Intent: Verify that PPFN.state_dict() only extracts the weights of the
    interleaved adapter layers and completely ignores the frozen backbone.
    """
    state = ppfn_model.state_dict()

    # 1. Ensure the state dict is nested by adapter name
    assert "encoder.layer1" in state
    assert "encoder.layer2" in state

    # 2. Ensure it contains the adapter weights, NOT the wrapper or frozen weights
    assert "weight" in state["encoder.layer1"]
    assert "bias" in state["encoder.layer1"]

    # 3. Ensure the frozen backbone parameters (which would normally populate
    # a native state_dict) are entirely absent.
    assert "frozen_model.encoder.layer1.weight" not in state


def test_save_and_load_cycle(ppfn_model, adapter_layers, tmp_path):
    """
    Intent: Verify that saving to disk and loading from disk successfully
    restores the adapter weights without crashing or missing keys.
    """
    ckpt_path = tmp_path / "test_adapters.pt"

    # 1. Save original weights
    original_weight = ppfn_model.interleaved_layers["encoder.layer1"].weight.clone()
    ppfn_model.save(str(ckpt_path))

    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    # 2. Mutate the weights in memory to simulate a fresh model
    with torch.no_grad():
        ppfn_model.interleaved_layers["encoder.layer1"].weight.fill_(99.0)

    assert not torch.equal(ppfn_model.interleaved_layers["encoder.layer1"].weight, original_weight)

    # 3. Load from disk
    ppfn_model.load(str(ckpt_path))

    # 4. Verify weights were perfectly restored
    restored_weight = ppfn_model.interleaved_layers["encoder.layer1"].weight
    assert torch.equal(restored_weight, original_weight)


def test_from_checkpoint_factory(mock_frozen_model, adapter_layers, tmp_path):
    """
    Intent: Ensure the @classmethod `from_checkpoint` instantiates the model
    and loads the weights cleanly in one step.
    """
    ckpt_path = tmp_path / "factory_adapters.pt"

    # Create a source model and save it
    source_model = PPFN(mock_frozen_model, adapter_layers)

    # Force a recognizable weight
    with torch.no_grad():
        source_model.interleaved_layers["encoder.layer2"].weight.fill_(42.0)
    source_model.save(str(ckpt_path))

    # Use factory to load
    new_adapters = {
        "encoder.layer1": nn.Linear(10, 10),
        "encoder.layer2": nn.Linear(10, 10)
    }
    loaded_model = PPFN.from_checkpoint(str(ckpt_path), mock_frozen_model, new_adapters)

    # Verify the specific injected weight was loaded
    assert torch.all(loaded_model.interleaved_layers["encoder.layer2"].weight == 42.0)


def test_ddp_module_prefix_cleaning(ppfn_model):
    """
    Intent: Verify that if a checkpoint is saved via DistributedDataParallel
    (which prepends 'module.' to all keys), `load_state_dict` successfully
    strips the prefix and routes the weights to the correct adapters.
    """
    # Create a fake DDP state dict
    fake_ddp_state = {
        "module.encoder.layer1": {
            "weight": torch.full((10, 10), 7.0),
            "bias": torch.zeros(10)
        }
    }

    # Load it (strict=False so we don't fail on missing layer2 in this dummy dict)
    ppfn_model.load_state_dict(fake_ddp_state, strict=False)

    # Verify the prefix was stripped and the weight was applied
    assert torch.all(ppfn_model.interleaved_layers["encoder.layer1"].weight == 7.0)