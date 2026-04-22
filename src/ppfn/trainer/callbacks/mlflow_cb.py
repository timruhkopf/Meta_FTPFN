import hydra
import os
import time
import subprocess
import tempfile
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional

import mlflow
from hydra.core.hydra_config import HydraConfig

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback

import logging

from ppfn.utils.git_hash import get_git_hash

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"



def get_safe_parent_run(experiment_id, experiment_name, job_id, mlruns_path):
    lock_base = Path(mlruns_path).parent / ".mlflow_locks"
    lock_base.mkdir(exist_ok=True)

    safe_exp_name = str(experiment_name).replace("/", "_").replace(" ", "_")
    lock_path = lock_base / f"lock_{job_id}_{safe_exp_name}"
    id_file = lock_path / "parent_id.txt"

    parent_id = None

    try:
        lock_path.mkdir()
        logger.info(f"LEADER: Creating parent run for Job {job_id}")
        with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"Multirun_{job_id}",
                tags={"job_id": job_id, "mode": "parent"}
        ) as r:
            parent_id = r.info.run_id
        id_file.write_text(parent_id)
        return parent_id
    except FileExistsError:
        logger.info("FOLLOWER: Waiting for Leader to write Parent ID...")
        for _ in range(420):
            if id_file.exists():
                potential_id = id_file.read_text().strip()
                try:
                    mlflow.get_run(potential_id)
                    return potential_id
                except Exception:
                    time.sleep(0.5)
            time.sleep(0.5)
    raise RuntimeError("Parent run resolution timed out.")


def get_dynamic_run_name(default_prefix="task"):
    try:
        hc = HydraConfig.get()
        task_overrides = hc.overrides.task
        sweep_dir = Path(hc.runtime.output_dir).parent.absolute()
        multirun_yaml_path = sweep_dir / "multirun.yaml"

        swept_keys = set()
        if multirun_yaml_path.exists():
            with open(multirun_yaml_path, "r") as f:
                multirun_cfg = yaml.safe_load(f)
            sweep_task_overrides = multirun_cfg.get("hydra", {}).get("overrides", {}).get("task", [])
            for override in sweep_task_overrides:
                if "=" in override:
                    key, val = override.split("=", 1)
                    if "," in val:
                        swept_keys.add(key.lstrip("+~"))

        dynamic_parts = []
        for override in task_overrides:
            clean_override = override.lstrip("+~")
            key = clean_override.split("=")[0] if "=" in clean_override else clean_override

            if swept_keys:
                if key in swept_keys:
                    dynamic_parts.append(clean_override.replace("=", "_").replace("/", "_"))
            elif key not in ("experiment_name", "run_name", "nested"):
                dynamic_parts.append(clean_override.replace("=", "_").replace("/", "_"))

        if not dynamic_parts:
            return f"{default_prefix}_{hc.job.num}"
        return "-".join(dynamic_parts)[:97]
    except Exception as e:
        logger.error(f"Name gen failed: {e}")
        return f"{default_prefix}_unknown"

class MLflowCallback(AbstractCallback):  # Inherit from your AbstractCallback
    def __init__(
            self,
            sweep_id: str = None,
            experiment_name: str = "ppfn_training",
            run_name: Optional[str] = None,
            mlflow_tracking_uri: Optional[str] = None,
            log_system_metrics: bool = True,
    ):
        super().__init__()
        self.sweep_id = sweep_id
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.log_system_metrics = log_system_metrics

    def _setup_experiment(self):
        """Robust experiment initialization for file backends."""
        exp_id = None
        for _ in range(5):
            try:
                exp = mlflow.get_experiment_by_name(self.experiment_name)
                exp_id = exp.experiment_id if exp else mlflow.create_experiment(self.experiment_name)
                break
            except Exception:
                time.sleep(1)
        return exp_id

    def _get_mlruns_path(self, uri: str) -> str:
        """Extracts the physical path from the tracking URI for the lock mechanism."""
        if uri.startswith("file://"):
            return uri.replace("file://", "")
        return os.path.abspath("./mlruns")  # Fallback

    def on_train_start(self, **kwargs):
        logger.info("Setting up MLflow tracking...")
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        exp_id = self._setup_experiment()

        # Check if we are running in Slurm. Submitit sets SLURM_JOB_ID.
        # If SLURM_ARRAY_JOB_ID is set, it's definitely a batch.
        slurm_id = os.environ.get("SLURM_JOB_ID")
        is_sweep = "sweep" in str(HydraConfig.get().runtime.output_dir)

        if slurm_id and is_sweep:
            # --- DISTRIBUTED NESTED LOGIC ---
            sweep_identifier = self.sweep_id  # Passed from config

            # 1. Look for an existing Parent Run for this sweep
            parent_run = self._get_or_create_parent(exp_id, sweep_identifier)

            # 2. Start the Parent (to make it active)
            mlflow.start_run(run_id=parent_run.info.run_id)

            # 3. Start the Child (Nested)
            dynamic_name = get_dynamic_run_name()
            self.run = mlflow.start_run(
                run_name=dynamic_name,
                nested=True,
                log_system_metrics=self.log_system_metrics
            )
            self._log_task_metadata()

        else:
            # --- SINGLE RUN LOGIC ---
            self.run = mlflow.start_run(
                experiment_id=exp_id,
                run_name=self.run_name,
                log_system_metrics=self.log_system_metrics
            )
            self._log_global_metadata()
            self._log_task_metadata()

    def _get_or_create_parent(self, exp_id: str, sweep_id: str):
        """Uses MLflow API to find the parent, avoiding file locks."""
        query = f"tags.sweep_id = '{sweep_id}' and tags.mode = 'parent'"
        existing_runs = mlflow.search_runs(experiment_ids=[exp_id], filter_string=query)

        if not existing_runs.empty:
            return mlflow.get_run(existing_runs.iloc[0].run_id)

        # If no parent exists, try to create one.
        # (Wrap in try/except in case of a race condition between two workers)
        try:
            with mlflow.start_run(
                    experiment_id=exp_id,
                    run_name=f"Sweep_{sweep_id}",
                    tags={"sweep_id": sweep_id, "mode": "parent"}
            ) as r:
                self._log_global_metadata()  # Log git diffs to parent
                return r
        except Exception:
            # If it failed, another worker probably beat us to it. Fetch it again.
            time.sleep(10)
            existing_runs = mlflow.search_runs(experiment_ids=[exp_id], filter_string=query)
            return mlflow.get_run(existing_runs.iloc[0].run_id)

    def _log_global_metadata(self):
        """Logs heavyweight or identical metadata only once per Job/Sweep."""
        mlflow.set_tag("mlflow.source.git.commit", get_git_hash())

        try:
            # Capture the current uncommitted changes
            git_diff = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode("utf-8")
            if git_diff.strip():
                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
                    tmp.write(git_diff)
                    tmp_path = tmp.name

                mlflow.log_artifact(tmp_path, artifact_path="scripts")
                Path(tmp_path).unlink()
                logger.info("Uncommitted git changes logged as diff.patch to Parent Run.")
        except Exception as e:
            logger.warning(f"Failed to capture git diff: {e}")

    def _log_task_metadata(self):
        """Logs configuration specific to this specific worker/task."""
        mlflow.set_tag("mlflow.folder", os.getcwd())

        if self.run_name:
            mlflow.set_tag("mlflow.runName", self.run_name)

        if HydraConfig.initialized():
            try:
                overrides = HydraConfig.get().overrides.task
                params = {o.strip("+").split("=")[0]: o.split("=")[1] for o in overrides if "=" in o}
                mlflow.log_params(params)
            except Exception:
                logger.warning("Could not log Hydra overrides.")




    def log_on_epoch_end(self, epoch: int, eon: int, metrics: Dict[str, float], **kwargs):
        global_step = (eon * self.trainer.epochs) + epoch
        mlflow.log_metrics(metrics, step=global_step)

    def log_on_train_end(self, **kwargs):
        """Pops all active runs off the MLflow stack."""
        # A while loop is required here!
        # If we are nested, we have 2 active runs. This closes Child, then Parent.
        while mlflow.active_run():
            logger.info(f"Closing active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()


# class MLflowCallback(AbstractCallback):
#     def __init__(
#             self,
#             experiment_name: str = "ppfn_training",
#             run_name: str | None = None,
#             mlflow_tracking_uri: str | None = None,
#             log_system_metrics: bool = True,
#     ):
#         super().__init__()
#         self.experiment_name = experiment_name
#         self.run_name = run_name
#
#         self.run = None
#         self.mlflow_tracking_uri = mlflow_tracking_uri
#         self.log_system_metrics = log_system_metrics
#
#     def on_train_start(self, **kwargs):
#         logger.info("Setting up MLflow tracking...")
#         if self.mlflow_tracking_uri:
#             mlflow.set_tracking_uri(self.mlflow_tracking_uri)
#         else:
#             mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
#
#         mlflow.set_experiment(self.experiment_name)
#         self.run = mlflow.start_run(
#             run_name=self.run_name, log_system_metrics=self.log_system_metrics
#         )
#
#         # TODO log run_name and run_id
#
#         # Log Git Metadata
#         mlflow.set_tag("mlflow.folder", os.getcwd())
#         mlflow.set_tag("mlflow.runName", self.run_name)
#         mlflow.set_tag("mlflow.source.git.commit", get_git_hash())
#
#
#         # Log Hydra Overrides if available
#         try:
#             from hydra.core.hydra_config import HydraConfig
#
#             overrides = HydraConfig.get().overrides.task
#             params = {
#                 o.strip("+").split("=")[0]: o.split("=")[1]
#                 for o in overrides
#                 if "=" in o
#             }
#             mlflow.log_params(params)
#
#         except Exception:
#             logger.warning("Could not log Hydra overrides.")
#
#         # Log Git Metadata & Diff
#         current_hash = get_git_hash()
#         mlflow.set_tag("mlflow.folder", os.getcwd())
#         mlflow.set_tag("mlflow.source.git.commit", current_hash)
#
#         try:
#             # Capture the current uncommitted changes
#             git_diff = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode("utf-8")
#
#             if git_diff.strip():
#                 with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
#                     tmp.write(git_diff)
#                     tmp_path = tmp.name
#
#                 mlflow.log_artifact(tmp_path, artifact_path="scripts")
#                 # Clean up the local temp file
#                 Path(tmp_path).unlink()
#                 logger.info("Uncommitted git changes logged as diff.patch")
#             else:
#                 logger.info("No uncommitted git changes detected.")
#         except Exception as e:
#             logger.warning(f"Failed to capture git diff: {e}")
#
#         # Log Config Dict
#         if self.trainer.config is not None:
#             mlflow.log_dict(self.trainer.config, "config.yaml")
#
#     def log_on_epoch_end(self, epoch: int, eon: int, metrics: Dict[str, float], **kwargs):
#         # Formula: (current_eon * total_epochs_in_one_eon) + current_epoch
#         global_step = (eon * self.trainer.epochs) + epoch
#
#         # Log metrics to MLflow using the continuous step
#         mlflow.log_metrics(metrics, step=global_step)
#
#     def log_on_train_end(self, **kwargs):
#         if mlflow.active_run():
#             mlflow.end_run()


class MockTrainer:
    def __init__(self, config):
        self.config = config
        self.epochs = 3

from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="/home/ruhkopf/PycharmProjects/Meta_FTPFN/configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Setup Mock Framework
    trainer = MockTrainer(dict(cfg))
    callback = MLflowCallback(experiment_name=cfg.experiment_name)
    callback.trainer = trainer

    # Simulate Training Loop
    callback.on_train_start()

    for epoch in range(trainer.epochs):
        logger.info(f"Simulating Epoch {epoch}...")
        metrics = {
            "loss": 1.0 / (epoch + 1),
            "accuracy": cfg.lr * (epoch + 1)
        }
        callback.log_on_epoch_end(epoch, eon=0, metrics=metrics)


    callback.log_on_train_end()


if __name__ == "__main__":
    def githash(*args, **kwargs) -> str:
        try:
            import subprocess

            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
            return git_hash
        except Exception as e:
            logger.warning(f"Could not retrieve git hash: {e}")
            return "unknown"


    OmegaConf.register_new_resolver("githash", githash)

    main()