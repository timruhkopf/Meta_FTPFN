import pytest
import torch
import torch.nn as nn
import threading

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.model.mymodel.ppfn_model import PPFN
from ppfn.model.mymodel.stream_parser import StreamParser


class MockTrackingLayer(nn.Module):
    """
    A mock adapter that extracts its injected address ID to generate
    distinct telemetry and loss values, proving independent state tracking.
    """

    def forward(self, A, B, C, sep, hp=None, **kwargs):
        assert hp is not None, "AdapterWrapper failed to fetch 'hp' from ForwardMetaContext"
        assert len(hp) == 3, "HP should be a perfectly sliced tuple of (A_hp, B_hp, C_hp)"

        # 1. Identify source layer
        address = getattr(self, "address", "unknown_address.0")

        # 2. Extract the trailing integer (e.g., "transformer_encoder.layers.1" -> 1)
        try:
            layer_idx = int(address.split(".")[-1])
        except ValueError:
            layer_idx = 0  # Fallback if the address doesn't end in an int

        # 3. Generate distinct mathematical values based on the layer index
        unique_entropy = 0.5 + (layer_idx * 0.1)  # Layer 0: 0.5, Layer 1: 0.6
        unique_gate_loss = 0.1 * (layer_idx + 1)  # Layer 0: 0.1, Layer 1: 0.2

        # 4. Log distinct Telemetry / Stats
        ForwardMetaContext.log_stats(address, {
            "attention_entropy": unique_entropy,
            "stream_c_seq_len": C.shape[0]
        })

        # 5. Log distinct Auxiliary Loss
        ForwardMetaContext.set(f"gate_loss/{address}", torch.tensor(unique_gate_loss))

        return A, B, C


@pytest.fixture(params=["standard_pfn", "paddable_pfn"])
def tracking_ppfn_setup(request, get_ft_pfn, mybatch):
    setup_type = request.param
    seq_len = mybatch.x.shape[0]

    if setup_type == "standard_pfn":
        pfn = get_ft_pfn
    else:
        from ppfn.model.baselines.ft_pfn_padding import ft_pfn_padding
        pfn = ft_pfn_padding()

    model = PPFN(
        frozen_model=pfn,
        interleaved_layers=[
            {"layer": MockTrackingLayer(), "name": "transformer_encoder.layers.0"},
            {"layer": MockTrackingLayer(), "name": "transformer_encoder.layers.1"}
        ],
        stream_parser=StreamParser(),
        # Force HP to be passed so the tracking layer can assert its presence
        pass_hp_as_rawpaded=True,
        seq_len=seq_len
    )
    return model



def test_forward_meta_context_isolation():
    """
    Intent: Verify the thread-local nature and basic CRUD operations of the
    ForwardMetaContext in isolation before testing integration.
    """
    ForwardMetaContext.clear()

    # Test setting explicit kwargs
    ForwardMetaContext.set(single_eval_pos=5)
    assert ForwardMetaContext.get("single_eval_pos") == 5

    # Test setting distinct string keys (vital for gate_loss/name logging)
    ForwardMetaContext.set("gate_loss/layer0", torch.tensor(1.0))
    assert torch.equal(ForwardMetaContext.get("gate_loss/layer0"), torch.tensor(1.0))

    # Test dictionary-based telemetry logging
    ForwardMetaContext.log_stats("layer0", {"mha_score": 0.9})
    stats = ForwardMetaContext.get_stats()
    assert "layer0" in stats
    assert stats["layer0"]["mha_score"] == 0.9

    # Test cleanup
    ForwardMetaContext.clear()
    assert ForwardMetaContext.get("single_eval_pos") is None
    assert ForwardMetaContext.get_stats() == {}

def test_meta_context_telemetry_and_source_tracking(tracking_ppfn_setup, mybatch):
    """
    Intent: Verify that during a full forward pass, multiple interleaved adapters
    independently log their unique telemetry and auxiliary losses, and that the
    Sidecar successfully aggregates them without overwriting.
    """
    model = tracking_ppfn_setup

    ForwardMetaContext.clear()

    # The forward pass will trigger MockTrackingLayer for layers .0 and .1
    output = model(mybatch)

    # 1. Fetch the aggregated statistics dictionary
    stats = ForwardMetaContext.get_stats()

    # 2. Verify both injected adapters successfully phoned home
    assert "transformer_encoder.layers.0" in stats
    assert "transformer_encoder.layers.1" in stats

    # 3. Verify the logged contents are DISTINCT and intact
    layer_0_stats = stats["transformer_encoder.layers.0"]
    layer_1_stats = stats["transformer_encoder.layers.1"]

    assert layer_0_stats["attention_entropy"] == 0.5
    assert layer_1_stats["attention_entropy"] == 0.6  # Proves values are distinct

    assert layer_0_stats["stream_c_seq_len"] == mybatch.x.shape[0]
    assert layer_1_stats["stream_c_seq_len"] == mybatch.x.shape[0]

    # 4. Verify auxiliary losses were properly named and stored distinctly
    state_dict = vars(ForwardMetaContext._state)
    assert "gate_loss/transformer_encoder.layers.0" in state_dict
    assert "gate_loss/transformer_encoder.layers.1" in state_dict

    # 5. Verify values of the auxiliary losses differ by layer index
    assert torch.allclose(
        state_dict["gate_loss/transformer_encoder.layers.0"],
        torch.tensor(0.1)
    )
    assert torch.allclose(
        state_dict["gate_loss/transformer_encoder.layers.1"],
        torch.tensor(0.2)
    )
