import pytest
import torch.nn as nn

from ppfn.model.mymodel.multistream_objective import MultiStreamObjective
from ppfn.model.mymodel.ppfn_model import PPFN
from ppfn.model.mymodel.stream_parser import StreamParser


class MockLayer(nn.Module):
    def forward(self, A, B, C, sep, **kwargs):
        # TODO check that ForwardMetaContext is properly used and does not cause issues,
        # ensure that we have access to everything we can ask of the
        return A, B, C


# 1. Parameterize a single setup fixture to yield both model configurations
@pytest.fixture(params=["standard_pfn", "paddable_pfn"])
def ppfn_setup(request, get_ft_pfn, mybatch):
    setup_type = request.param
    seq_len = mybatch.x.shape[0]

    # Select the correct backbone based on the parameter
    if setup_type == "standard_pfn":
        pfn = get_ft_pfn
    else:
        from ppfn.model.baselines.ft_pfn_padding import ft_pfn_padding
        pfn = ft_pfn_padding()

    # Build the model wrapper
    model = PPFN(
        frozen_model=pfn,
        interleaved_layers=[
            {"layer": MockLayer(), "name": "transformer_encoder.layers.0"},
            {"layer": MockLayer(), "name": "transformer_encoder.layers.1"}
        ],
        stream_parser=StreamParser(),
        seq_len=seq_len
    )

    # Build the objective using the specific backbone's criterion
    objective = MultiStreamObjective(
        criterion=pfn.criterion,
        stream_parser=StreamParser(),
        lambda_sparsity=0.
    )

    return model, objective


# 2. Update the tests to unpack the tuple from the new fixture
def test_ppfn_output_shape(ppfn_setup, mybatch):
    model, _ = ppfn_setup  # We only need the model here
    output = model(mybatch)

    # we expect (T_test, 3*R, 1k) since the output is interleaved A/B/C and we have 3*R total streams, and the output dim is 1000 for logits
    expecteddim = (mybatch.x.shape[0] - mybatch.single_eval_pos, int((mybatch.y.shape[1] / 2) * 3), 1000)
    assert output.shape == expecteddim, \
        f"PPFN Output shape is unexpected. Expected (T_test, 3*R, output_dim) {expecteddim}, got {output.shape}"


def test_zero_loss_on_C_minus_A_with_empty_adapter(ppfn_setup, mybatch):
    """Since we mock the layer, streams A and C are identical, so the loss should be zero."""
    model, objective = ppfn_setup

    output = model(mybatch)

    # Make sure to pass batch=mybatch as a kwarg since your objective signature expects it!
    loss, metrics = objective(output, single_eval_pos=mybatch.single_eval_pos, batch=mybatch)

    assert metrics['nll/A'] == metrics['nll/C'], "Since A and C are identical, their NLL should be the same."
    assert loss != 0
    assert loss == metrics['nll/A'], "The loss should be equal to the NLL of stream A, since the outputs for C and A are identical"