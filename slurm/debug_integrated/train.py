# train.py
import hydra
import mlflow
import time
import os
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Set the tracking URI provided by the shell script
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        print(f"Running on device: {cfg.device}")
        print(f"Seed set to: {cfg.seed}")
        print(f"MLflow URI: {mlflow.get_tracking_uri()}")

        # Log a dummy parameter and metric
        mlflow.log_param("device", cfg.device)
        mlflow.log_param("seed", cfg.seed)

        print("Training (simulated)...")
        for i in range(10):
            mlflow.log_metric("accuracy", 0.1 * i)
            time.sleep(2)  # Give you time to test 'scancel --signal=USR1'

        print("Done!")


if __name__ == "__main__":
    my_app()