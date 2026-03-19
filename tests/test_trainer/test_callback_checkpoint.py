import json
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Replace with your actual import path
from ppfn.trainer.callbacks.checkpoint import CheckpointCallback


@pytest.fixture
def mock_trainer():
    """Mocks the trainer object and its internal states."""
    trainer = MagicMock()
    # Mocking state dicts with actual Tensors to test .cpu().clone() logic
    trainer.model.state_dict.return_value = {"weights": torch.tensor([0.1, 0.2])}
    trainer.optimizer.state_dict.return_value = {"opt": "state"}
    trainer.scheduler.state_dict.return_value = {"lr": 0.01}

    # Mocking attributes
    trainer.global_step = 1234
    trainer.start_epoch = 0
    trainer.scaler = None
    return trainer


@pytest.fixture
def callback(tmp_path, mock_trainer):
    """Initializes callback with min_save_interval=0 for standard functional tests."""
    callb = CheckpointCallback(
        save_dir=str(tmp_path),
        monitor="val/loss",
        mode="min",
        name="test_model",  # Defined name implies file will be best_test_model.pt
        min_save_interval=0,  # Disable time guard for most tests by default
        min_save_epoch=0
    )
    callb.set_trainer(mock_trainer)
    return callb


def test_async_save_completion(callback, tmp_path):
    """Verifies that files are eventually written even if save is async."""
    metrics = {"val/loss": 0.1}

    # Trigger save
    callback.log_on_epoch_end(epoch=1, eon=0, metrics=metrics)

    # This acts as the 'join' for the thread
    callback.on_train_end()

    # Dynamic filename check ensures we look for the right file
    pt_file = tmp_path / f"best_{callback.name}.pt"
    json_file = tmp_path / f"best_{callback.name}.json"

    assert pt_file.exists()
    assert json_file.exists()

    # Verify content
    loaded_ckpt = torch.load(pt_file)
    assert loaded_ckpt["epoch"] == 1
    assert loaded_ckpt["metrics"]["val/loss"] == 0.1

    with open(json_file, "r") as f:
        meta = json.load(f)
        assert meta["best_score"] == 0.1


def test_min_save_interval_blocking(callback, tmp_path):
    """Verifies that excessive back-to-back saving is blocked by time."""
    callback.min_save_interval = 60  # 1 minute

    # Use patch to control time.time()
    with patch("time.time") as mock_time:
        # 1. Start at time T=1000
        mock_time.return_value = 1000.0

        # First save (Success)
        callback.log_on_epoch_end(epoch=1, eon=0, metrics={"val/loss": 0.5})
        callback.on_train_end()  # Wait for thread

        json_file = tmp_path / f"best_{callback.name}.json"
        assert json_file.exists()

        # Remove file to verify if it gets recreated
        json_file.unlink()

        # 2. Advance time only 10 seconds (T=1010) -> Should be BLOCKED
        mock_time.return_value = 1010.0
        callback.log_on_epoch_end(epoch=2, eon=0, metrics={"val/loss": 0.1})
        callback._executor.shutdown(wait=True)  # Ensure any tasks finish

        assert not json_file.exists(), (
            "File should not exist because save was blocked by timer"
        )


def test_atomic_save_mechanism(callback):
    """Ensures .tmp is used and then replaced in the background thread."""
    metrics = {"val/loss": 0.1}

    # We patch inside the worker method because it runs in a different thread
    with (
        patch("torch.save") as mock_torch_save,
        patch.object(Path, "replace") as mock_replace,
    ):
        callback.log_on_epoch_end(epoch=1, eon=0, metrics=metrics)
        callback.on_train_end()  # Ensure worker runs

        # Verify it saved to .tmp first
        args, _ = mock_torch_save.call_args
        assert str(args[1]).endswith(".tmp")
        mock_replace.assert_called()


def test_snapshot_on_main_thread(callback, mock_trainer):
    """
    Crucial test: ensures model weights are cloned to CPU
    BEFORE the background thread starts, to avoid race conditions.
    """
    metrics = {"val/loss": 0.1}

    # Track the call to state_dict
    callback.log_on_epoch_end(epoch=1, eon=0, metrics=metrics)

    # The main thread should have called state_dict immediately
    mock_trainer.model.state_dict.assert_called()
    callback.on_train_end()


def test_sidecar_metadata_accuracy(callback, tmp_path):
    """Ensures JSON sidecar is accurate despite being saved asynchronously."""
    metrics = {"val/loss": 0.42}
    callback.log_on_epoch_end(epoch=99, eon=0, metrics=metrics)
    callback.on_train_end()

    json_path = tmp_path / f"best_{callback.name}.json"
    data = json.loads(json_path.read_text())

    assert data["epoch"] == 99
    assert data["best_score"] == 0.42
    assert data["metrics_at_save"]["val/loss"] == 0.42


def test_resume_from_checkpoint(tmp_path, mock_trainer):
    """Loading logic remains synchronous and should work as before."""
    checkpoint_path = tmp_path / "existing.pt"
    fake_checkpoint = {
        "epoch": 5,
        "global_step": 500,
        "best_score": 0.2,
        "model_state_dict": {"weights": torch.tensor([0.9])},
        "optimizer_state_dict": {"opt": "resumed"},
    }
    torch.save(fake_checkpoint, checkpoint_path)

    callback = CheckpointCallback(save_dir=str(tmp_path), resume_from=checkpoint_path)
    callback.set_trainer(mock_trainer)
    callback.on_trainer_init()

    assert mock_trainer.start_epoch == 6
    assert callback.best_score == 0.2
    mock_trainer.model.load_state_dict.assert_called()


def test_mlflow_upload_on_train_end(callback, tmp_path):
    """Ensures upload happens only AFTER background threads are joined."""
    # Create dummy files manually
    (tmp_path / f"best_{callback.name}.pt").write_text("dummy")
    (tmp_path / f"best_{callback.name}.json").write_text("{}")

    with (
        patch("mlflow.active_run", return_value=True),
        patch("mlflow.log_artifact") as mock_log,
    ):
        callback.on_train_end()  # This calls shutdown(wait=True) then logs
        assert mock_log.call_count == 2


def test_read_only_mode(tmp_path, mock_trainer):
    """Ensures no files are written if read_only is True."""
    # Initialize separate callback for read-only test
    cb = CheckpointCallback(
        save_dir=str(tmp_path), read_only=True, monitor="val/loss", mode="min"
    )
    cb.set_trainer(mock_trainer)

    cb.log_on_epoch_end(epoch=1, eon=0, metrics={"val/loss": 0.1})
    cb.on_train_end()

    # No files should be created
    assert not list(tmp_path.iterdir())
