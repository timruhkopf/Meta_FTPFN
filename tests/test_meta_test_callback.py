import pytest
import torch
from unittest.mock import MagicMock

# Replace 'ppfn.trainer.callbacks.meta_test_callback' with your actual import path
from ppfn.trainer.callbacks.meta_test import MetaTestCallback


@pytest.fixture
def mock_dataset():
    """Mocks a dataset with the required attributes for MetaTestCallback."""
    dataset = MagicMock()
    dataset.name = "toy_benchmark"
    dataset.single_eval_pos = 10

    # Mocking __len__ and __getitem__ so DataLoader works
    dataset.__len__.return_value = 2
    # Returns a dummy batch dict
    dataset.__getitem__.return_value = {"x": torch.randn(1, 5), "y": torch.randn(1)}
    return dataset


@pytest.fixture
def mock_trainer():
    """Mocks the trainer and its _forward_pass logic."""
    trainer = MagicMock()
    # Mock _forward_pass to return (loss, metrics_dict)
    # We use a dummy loss and a dictionary of metrics
    trainer._forward_pass.return_value = (torch.tensor(0.5), {"mse": 0.1, "nll": 0.2})
    # Ensure the model mock supports train() and eval() calls
    trainer.model = MagicMock(spec=torch.nn.Module)
    return trainer


@pytest.fixture
def callback(mock_dataset, mock_trainer):
    """Initializes the MetaTestCallback with mocks."""
    cb = MetaTestCallback(
        dataset=mock_dataset, frequency=2, device="cpu", switch_to_eval=True
    )
    cb.set_trainer(mock_trainer)
    return cb


# =================================================================
# TEST CASES
# =================================================================


def test_init_validation():
    """Ensures the callback raises an error if dataset has no 'name' attribute."""
    invalid_ds = MagicMock(spec=[])  # Explicitly empty spec, no 'name'
    with pytest.raises(AssertionError, match="Dataset must have a 'name' attribute"):
        MetaTestCallback(dataset=invalid_ds)


def test_frequency_logic(callback, mock_trainer):
    """Ensures evaluation only runs at the specified frequency."""
    metrics = {"train/loss": 0.5}

    # Epoch 0 (1st epoch): (0+1) % 2 != 0 -> Should return None
    result = callback.on_epoch_end(epoch=0, metrics=metrics)
    assert result is None
    mock_trainer._forward_pass.assert_not_called()

    # Epoch 1 (2nd epoch): (1+1) % 2 == 0 -> Should run evaluation
    result = callback.on_epoch_end(epoch=1, metrics=metrics)
    assert result is not None
    # Dataset length is 2, so _forward_pass should be called twice
    assert mock_trainer._forward_pass.call_count == 2


def test_evaluation_metric_aggregation(callback, mock_trainer):
    """Checks if metrics are correctly averaged and renamed with dataset suffix."""
    # Setup mock to return different metric values per step to test averaging
    mock_trainer._forward_pass.side_effect = [
        (None, {"mse": 0.2}),
        (None, {"mse": 0.4}),
    ]

    results = callback.on_epoch_end(epoch=1, metrics={})

    # Expected average: (0.2 + 0.4) / 2 = 0.3
    # Expected key format: '{metric}:{dataset_name}'
    expected_key = "mse:toy_benchmark"
    assert expected_key in results
    assert results[expected_key] == pytest.approx(0.3)


def test_mode_switching(callback, mock_trainer):
    """Ensures model is toggled to eval mode then back to train mode."""
    callback.switch_to_eval = True

    callback.on_epoch_end(epoch=1, metrics={})

    # Verify the sequence of calls
    mock_trainer.model.eval.assert_called
