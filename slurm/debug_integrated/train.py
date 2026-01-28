# train.py
import hydra
import mlflow
import time
import os
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Set the tracking URI provided by the shell script
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        logger.info(f"Running on device: {cfg.device}")
        logger.info(f"Seed set to: {cfg.seed}")
        logger.info(f"MLflow URI: {mlflow.get_tracking_uri()}")

        # Log a dummy parameter and metric
        mlflow.log_param("device", cfg.device)
        mlflow.log_param("seed", cfg.seed)

        logger.info("Training (simulated)...")
        for i in range(10):
            mlflow.log_metric("accuracy", 0.1 * i, step=i)
            time.sleep(100)  # Give you time to test 'scancel --signal=USR1'

        logger.info("Done!")


if __name__ == "__main__":
    my_app()
    "/bigwork/nhwpruht/Meta_FTPFN/mlruns/156515304721359357/372e1c4d40c04c138c10673b00cbea5c/metrics"