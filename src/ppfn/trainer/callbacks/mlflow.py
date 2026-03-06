import mlflow
import os

from typing import Dict
from ppfn.trainer.callbacks.abstract_callback import AbstractCallback

import logging

from ppfn.utils.git_hash import get_git_hash

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MLflowCallback(AbstractCallback):
    def __init__(
        self,
        experiment_name: str = "ppfn_training",
        run_name: str | None = None,
        mlflow_tracking_uri: str | None = None,
        log_system_metrics: bool = True,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.run = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.log_system_metrics = log_system_metrics

    def on_train_start(self, **kwargs):
        logger.info("Setting up MLflow tracking...")
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        else:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(
            run_name=self.run_name, log_system_metrics=self.log_system_metrics
        )

        # TODO log run_name and run_id

        # Log Git Metadata
        mlflow.set_tag("mlflow.folder", os.getcwd())
        mlflow.set_tag("mlflow.source.git.commit", get_git_hash())

        # Log Hydra Overrides if available
        try:
            from hydra.core.hydra_config import HydraConfig

            overrides = HydraConfig.get().overrides.task
            params = {
                o.strip("+").split("=")[0]: o.split("=")[1]
                for o in overrides
                if "=" in o
            }
            mlflow.log_params(params)
        except Exception:
            logger.warning("Could not log Hydra overrides.")

        # Log Config Dict
        if self.trainer.config is not None:
            mlflow.log_dict(self.trainer.config, "config.yaml")

    def log_on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        # Log metrics to MLflow
        mlflow.log_metrics(metrics, step=epoch)

    def log_on_train_end(self, **kwargs):
        if mlflow.active_run():
            mlflow.end_run()
