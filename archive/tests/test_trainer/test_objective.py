import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import patch

from ppfn.model.mymodel.multistream_objective import MultiStreamObjective




@dataclass
class MyBatch:
    x: torch.Tensor
    y: torch.Tensor
    single_eval_pos: int
    style: torch.Tensor = None


class DummyCriterion(nn.Module):
    """A simple MSE loss that simulates your PFN criterion."""

    def forward(self, pred, target):
        # pred shape: [T_test, R, D], target shape: [T_test, R]
        # We'll just squeeze D for testing to do a direct comparison
        return torch.nn.functional.mse_loss(pred.squeeze(-1), target, reduction='none')


@pytest.fixture
def mock_nanmean():
    """Mocks torch_nanmean to bypass external dependencies."""
    with patch("pfns4hpo.utils.torch_nanmean") as mock_func:
        # returns (optimization_loss, nan_share)
        def dummy_nanmean(tensor, axis=0, return_nanshare=True):
            return tensor.mean(axis), torch.tensor(0.0)

        mock_func.side_effect = dummy_nanmean
        yield mock_func


@pytest.fixture
def objective():
    # Using StreamParser directly as it handles the routing logic we want to test
    from ppfn.model.mymodel.stream_parser import StreamParser
    return MultiStreamObjective(
        criterion=DummyCriterion(),
        verbose=True,
        lambda_sparsity=0.1,
        stream_parser=StreamParser()
    )


@pytest.fixture
def setup_context():
    """Cleans and prepares the ForwardMetaContext for each test."""
    from ppfn.model.mymodel.meta_context import  ForwardMetaContext
    ForwardMetaContext.clear()
    yield ForwardMetaContext
    ForwardMetaContext.clear()


def test_target_alignment_and_loss(objective, mock_nanmean, setup_context):
    """
    Intent: Verify that the objective function correctly pairs the truncated
    T_test model outputs with the sliced T_test targets from the raw batch.
    """
    objective.stream_parser.train()

    T_full = 10
    sep = 4
    T_test = T_full - sep  # 6
    R = 2  # 2 Target, 2 Related in train interleaved batch

    # 1. Create a raw batch (Shape: T_full=10, Batch=4, D=1)
    # Target values = 10.0, Related values = 20.0
    x = torch.zeros((T_full, 4, 1))
    y = torch.zeros((T_full, 4))
    y[:, ::2] = 10.0  # Stream A/C targets
    y[:, 1::2] = 20.0  # Stream B targets

    batch = MyBatch(x=x, y=y, single_eval_pos=sep)

    # 2. Create mock model output already truncated to T_test (Shape: T_test=6, 3R=6, D=1)
    # Stream A output = 10.0 (Perfect match)
    # Stream B output = 20.0
    # Stream C output = 12.0 (Off by 2.0 from target 10.0)
    output = torch.zeros((T_test, 6, 1))
    output[:, 0:2, :] = 10.0  # A
    output[:, 2:4, :] = 20.0  # B
    output[:, 4:6, :] = 12.0  # C

    # for debugging purposes, let's look at the internal streams, that the objective will have aceess to
    streams = objective.stream_parser.get_raw_streams(batch, None)
    o_streams = objective.stream_parser.parse_output_streams(output, sep)

    opt_loss, metrics = objective(output=output, single_eval_pos=sep, batch=batch)

    # Stream A loss should be 0.0 (pred 10.0 vs target 10.0)
    # Stream C loss should be 4.0 (pred 12.0 vs target 10.0, squared error)
    assert metrics["nll/A"] == 0.0
    assert metrics["nll/C"] == 4.0
    assert metrics["nll/C-A"] == 4.0

    # Opt loss should equal Stream C loss (4.0) since there's no aux loss injected yet
    assert opt_loss.item() == 4.0


def test_auxiliary_loss_integration(objective, mock_nanmean, setup_context):
    """
    Intent: Ensure that gate losses stored in MetaContext are correctly
    retrieved, scaled by lambda_sparsity, and added to the optimization loss.
    """
    objective.stream_parser.train()

    # Inject fake aux losses into context
    setup_context.set(**{
        "gate_loss/layer1": torch.tensor(10.0),
        "gate_loss/layer2": torch.tensor(20.0)
    })

    T_test, sep = 6, 4
    batch = MyBatch(x=torch.zeros((10, 4, 1)), y=torch.zeros((10, 4)), single_eval_pos=sep)
    output = torch.zeros((T_test, 6, 1))

    opt_loss, _ = objective(output=output, single_eval_pos=sep, batch=batch)

    # Main loss is 0.0 (outputs and targets are both 0)
    # Aux loss: mean(10, 20) = 15.0.
    # Scaled by lambda_sparsity (0.1) = 1.5
    assert opt_loss.item() == 1.5


def test_empty_auxiliary_loss_device_safety(objective, mock_nanmean, setup_context):
    """
    Intent: Verify that when no aux losses exist, the fallback tensor defaults
    to 0.0 and safely matches the device of the main optimization loss.
    """
    objective.stream_parser.train()

    T_test, sep = 6, 4
    # Put tensors on a specific mock device if possible, or just check CPU fallback
    batch = MyBatch(x=torch.zeros((10, 4, 1)), y=torch.zeros((10, 4)), single_eval_pos=sep)
    output = torch.zeros((T_test, 6, 1))

    # Context is empty
    opt_loss, _ = objective(output=output, single_eval_pos=sep, batch=batch)

    assert opt_loss.item() == 0.0


def test_style_metric_parsing_train(objective, mock_nanmean, setup_context):
    """
    Intent: Verify the batch.style tensor is correctly sliced to categorize
    similar vs. unrelated tasks during training.
    """
    objective.stream_parser.train()
    T_test, sep = 6, 4

    # Batch size = 4 (R=2).
    # Target 0 is unrelated (style 0), Target 1 is similar (style 1)
    style = torch.tensor([[0], [0], [1], [1]])
    batch = MyBatch(x=torch.zeros((10, 4, 1)), y=torch.zeros((10, 4)), single_eval_pos=sep, style=style)

    # Output:
    # Stream A is perfect (loss 0)
    # Stream C: Target 0 (unrelated) off by 2.0 -> loss 4.0. Target 1 (similar) off by 3.0 -> loss 9.0
    output = torch.zeros((T_test, 6, 1))
    output[:, 4, :] = 2.0  # C, Target 0
    output[:, 5, :] = 3.0  # C, Target 1

    _, metrics = objective(output=output, single_eval_pos=sep, batch=batch)

    assert metrics["nll/unrelated_task"] == 4.0
    assert metrics["nll/similar_task"] == 9.0