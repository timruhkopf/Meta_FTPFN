import json
import pytest
import torch
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
from ppfn.trainer.callbacks.checkpoint_callback import CheckpointCallback


@pytest.fixture
def mock_trainer():
    """Mocks the trainer object and its internal states."""
    trainer = MagicMock()
    # Mocking state dicts
    trainer.model.state_dict.return_value = {"weights": [0.1, 0.2]}
    trainer.optimizer.state_dict.return_value = {"opt": "state"}
    trainer.scheduler.state_dict.return_value = {"lr": 0.01}

    # Mocking attributes
    trainer.global_step = 1234
    trainer.start_epoch = 0
    trainer.scaler = None
    return trainer


@pytest.fixture
def callback(tmp_path, mock_trainer):
    """Initializes callback with a temporary directory."""
    callb = CheckpointCallback(
        save_dir=str(tmp_path),
        monitor="val/loss",
        mode="min",
        name="test_model"
    )
    callb.set_trainer(mock_trainer)
    return callb



def test_resume_from_checkpoint(tmp_path, mock_trainer):
    """Tests the loading logic in on_trainer_init."""
    checkpoint_path = tmp_path / "existing.pt"

    # Create a fake checkpoint file
    fake_checkpoint = {
        "epoch": 5,
        "global_step": 500,
        "best_score": 0.2,
        "model_state_dict": {"weights": [0.9]},
        "optimizer_state_dict": {"opt": "resumed"},
        "scheduler_state_dict": {"lr": 0.001}
    }
    torch.save(fake_checkpoint, checkpoint_path)

    # Initialize callback with resume path
    callback = CheckpointCallback(
        save_dir=str(tmp_path),
        resume_from=checkpoint_path,
        monitor="val/loss"
    )
    callback.set_trainer(mock_trainer)

    # Trigger loading
    callback.on_trainer_init()

    # Verify trainer state was updated
    assert mock_trainer.start_epoch == 6
    assert mock_trainer.global_step == 500
    assert callback.best_score == 0.2
    mock_trainer.model.load_state_dict.assert_called_with(fake_checkpoint["model_state_dict"])


def test_atomic_save_mechanism(callback):
    """Ensures .tmp is used and then replaced (atomic saving)."""
    metrics = {"val/loss": 0.1}

    with patch("torch.save") as mock_torch_save, \
            patch.object(Path, "replace") as mock_replace:
        callback.log_on_epoch_end(epoch=1, metrics=metrics)

        # Verify it saved to .tmp first
        args, _ = mock_torch_save.call_args
        assert str(args[1]).endswith(".tmp")

        # Verify replace was called to finalize the move
        mock_replace.assert_called()


def test_read_only_mode(callback, tmp_path):
    """Ensures no files are written if read_only is True."""
    callback.read_only = True
    metrics = {"val/loss": 0.0001}  # Definitely an improvement

    with patch("torch.save") as mock_torch_save:
        callback.log_on_epoch_end(epoch=1, metrics=metrics)

        # Verify torch.save was never called
        mock_torch_save.assert_not_called()
        # Verify no json sidecar was created
        assert not (Path(tmp_path) / "best_test_model.json").exists()


def test_monitor_list_averaging(callback):
    """Verifies that multiple metrics are averaged correctly."""
    callback.monitor = ["loss_a", "loss_b"]
    callback.best_score = 10.0
    metrics = {"loss_a": 2.0, "loss_b": 4.0}  # Average is 3.0

    with patch("torch.save"), patch.object(Path, "replace"):
        callback.log_on_epoch_end(epoch=1, metrics=metrics)
        assert callback.best_score == 3.0


def test_sidecar_metadata_accuracy(callback):
    """Ensures the sidecar JSON matches the training state exactly."""
    metrics = {"val/loss": 0.42}

    with patch("torch.save"), patch.object(Path, "replace"):
        callback.log_on_epoch_end(epoch=99, metrics=metrics)

        json_path = callback.save_dir / f"best_{callback.name}.json"
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["epoch"] == 99
        assert data["step"] == 1234  # From mock_trainer
        assert data["best_score"] == 0.42
        assert data["metrics_at_save"]["val/loss"] == 0.42


def test_mlflow_upload_on_train_end(callback):
    """Ensures artifacts are uploaded only if they exist and not read-only."""
    # Create dummy files
    pt_path = callback.save_dir / "best_test_model.pt"
    json_path = callback.save_dir / "best_test_model.json"
    pt_path.write_text("dummy")
    json_path.write_text("{}")

    with patch("mlflow.active_run", return_value=True), \
            patch("mlflow.log_artifact") as mock_log:
        callback.on_train_end()
        assert mock_log.call_count == 2
        mock_log.assert_any_call(str(pt_path), artifact_path="final_checkpoints")


def test_nan_guard(callback):
    """Ensures NaNs do not trigger a save or update best_score."""
    callback.best_score = 0.5
    metrics = {"val/loss": float("nan")}

    with patch("torch.save") as mock_save:
        callback.log_on_epoch_end(epoch=1, metrics=metrics)
        mock_save.assert_not_called()
        assert callback.best_score == 0.5