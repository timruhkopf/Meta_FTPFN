from pathlib import Path

import pytest
import json

from unittest.mock import MagicMock, patch
from ppfn.trainer.callbacks.checkpoint_callback import CheckpointCallback


@pytest.fixture
def mock_trainer():
    """Mocks the trainer object and its internal states."""
    trainer = MagicMock()
    trainer.model.state_dict.return_value = {"weights": [0.1, 0.2]}
    trainer.optimizer.state_dict.return_value = {"opt": "state"}
    trainer.global_step = 100
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
    callb.best_score = float("inf")  # Start with worst possible score
    callb.global_step = 10  # Mock global step
    return callb


def test_save_on_improvement(callback):
    metrics = {"val/loss": 0.5}

    # Define a side effect that creates the file torch.save is supposed to create
    def fake_save(obj, path):
        Path(path).touch()

    with patch("torch.save", side_effect=fake_save):
        callback.log_on_epoch_end(epoch=1, metrics=metrics)

        pt_path = callback.save_dir / "best_test_model.pt"
        json_path = callback.save_dir / "best_test_model.json"

        assert pt_path.exists()
        assert json_path.exists()
        assert callback.best_score == 0.5


def test_no_save_on_worse_metric(callback):
    callback.best_score = 0.1  # Set a very good initial score
    metrics = {"val/loss": 0.5}

    with patch("torch.save") as mock_torch_save:
        callback.log_on_epoch_end(epoch=1, metrics=metrics)

        pt_path = callback.save_dir / "best_test_model.pt"
        assert not pt_path.exists()
        assert callback.best_score == 0.1


def test_mlflow_upload_on_train_end(callback):
    # Setup: Create dummy local files first
    pt_path = callback.save_dir / "best_test_model.pt"
    json_path = callback.save_dir / "best_test_model.json"
    pt_path.write_text("dummy model data")
    json_path.write_text("{}")

    with patch("mlflow.active_run", return_value=True), \
            patch("mlflow.log_artifact") as mock_log:
        callback.log_on_train_end()

        # Verify log_artifact was called twice (one for .pt, one for .json)
        assert mock_log.call_count == 2

        # Check if it targeted the right artifact path
        mock_log.assert_any_call(str(pt_path), artifact_path="final_checkpoints")
        mock_log.assert_any_call(str(json_path), artifact_path="final_checkpoints")


def test_monitor_list_averaging(callback):
    callback.monitor = ["loss_a", "loss_b"]
    callback.best_score = 10.0
    metrics = {"loss_a": 2.0, "loss_b": 4.0}

    def fake_save(obj, path):
        Path(path).touch()

    with patch("torch.save", side_effect=fake_save):
        callback.log_on_epoch_end(epoch=1, metrics=metrics)
        assert callback.best_score == 3.0


def test_nan_guard(callback):
    metrics = {"val/loss": float("nan")}

    with patch("torch.save") as mock_torch_save:
        callback.log_on_epoch_end(epoch=1, metrics=metrics)
        assert not (callback.save_dir / "best_test_model.pt").exists()


def test_maximization_mode(callback):
    """Ensures Benchmark Accuracy (max) logic is correct."""
    callback.mode = "max"
    callback.best_score = 0.70
    callback.monitor = ["val/accuracy"]

    # Define fake_save to prevent FileNotFoundError
    def fake_save(obj, path): Path(path).touch()

    with patch("torch.save", side_effect=fake_save):
        # Should NOT save (0.65 < 0.70)
        callback.log_on_epoch_end(1, {"val/accuracy": 0.65})
        assert callback.best_score == 0.70

        # SHOULD save (0.85 > 0.70)
        callback.log_on_epoch_end(2, {"val/accuracy": 0.85})
        assert callback.best_score == 0.85


def test_sidecar_metadata_accuracy(callback):
    """Ensures the sidecar matches the training state exactly."""
    metrics = {"val/loss": 0.42}

    def fake_save(obj, path): Path(path).touch()

    with patch("torch.save", side_effect=fake_save):
        callback.log_on_epoch_end(epoch=99, metrics=metrics)

        json_path = callback.save_dir / f"best_{callback.name}.json"
        data = json.loads(json_path.read_text())

        assert data["epoch"] == 99
        assert data["step"] == 1234
        assert data["monitor"] == ["val/loss"]
