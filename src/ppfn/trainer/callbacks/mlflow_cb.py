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
from ppfn.utils.git_hash import get_git_hash

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def get_dynamic_run_name(default_prefix="run"):
    """Generates a run name by appending swept parameters to the prefix."""
    try:
        if not HydraConfig.initialized():
            return default_prefix

        hc = HydraConfig.get()
        task_overrides = hc.overrides.task

        # Determine if we are in a multirun by looking for multirun.yaml
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

            # --- Extract the short key and format the value ---
            if "=" in clean_override:
                full_key, val = clean_override.split("=", 1)
                # Grab only the final string after the last period
                short_key = full_key.split(".")[-1]
                safe_val = val.replace("/", "_")
                formatted_part = f"{short_key}_{safe_val}"
            else:
                full_key = clean_override
                # Handle flags that don't have an equals sign
                formatted_part = full_key.split(".")[-1]

                # Filter against swept_keys using the full_key, but append the formatted_part
            if swept_keys:
                if full_key in swept_keys:
                    dynamic_parts.append(formatted_part)
            elif full_key not in ("experiment_name", "run_name", "nested"):
                dynamic_parts.append(formatted_part)

        # Construct final name
        suffix = "-".join(dynamic_parts)[:97]
        if suffix:
            return f"{default_prefix}_{suffix}"

        # Fallback to job number if no dynamic parts
        job_num = hc.job.get("num", "")
        if job_num != "":
            return f"{default_prefix}_{job_num}"

        return default_prefix

    except Exception as e:
        logger.error(f"Name gen failed: {e}")
        return f"{default_prefix}_unknown"

class MLflowCallback(AbstractCallback):
    def __init__(
            self,
            sweep_id: Optional[str] = None,  # Left here just in case config still passes it
            experiment_name: str = "ppfn_training",
            run_name: Optional[str] = None,
            mlflow_tracking_uri: Optional[str] = None,
            log_system_metrics: bool = True,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.log_system_metrics = log_system_metrics

    def _setup_experiment(self):
        """Robust experiment initialization bypassing NFS cache via ID."""
        # 1. Check if the login node explicitly passed us the resolved ID
        env_exp_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
        if env_exp_id:
            logger.info(f"Inherited Experiment ID {env_exp_id} from Leader Node. Bypassing NFS check.")
            return env_exp_id

        # 2. Fallback for local debugging
        exp_id = None
        for _ in range(5):
            try:
                exp = mlflow.get_experiment_by_name(self.experiment_name)
                exp_id = exp.experiment_id if exp else mlflow.create_experiment(self.experiment_name)
                break
            except Exception:
                time.sleep(1)
        return exp_id

    def on_train_start(self, **kwargs):
        logger.info("Setting up MLflow tracking...")

        # 1. Connect
        uri = self.mlflow_tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
        if uri:
            mlflow.set_tracking_uri(uri)

        # 2. Resolve Experiment (Crucially, using the ID from slim.sh)
        exp_id = self._setup_experiment()

        # 3. Resolve dynamic Run Name (e.g. Baseline-n_A_dataset_n_A_10)
        prefix = self.run_name or "run"
        final_run_name = get_dynamic_run_name(default_prefix=prefix)

        # 4. Start the single flat run
        self.run = mlflow.start_run(
            experiment_id=exp_id,
            run_name=final_run_name,
            log_system_metrics=self.log_system_metrics
        )
        logger.info(f"Flat Run Active: {self.run.info.run_id} under name {final_run_name}")

        # 5. Log Metadata
        self._log_global_metadata()
        self._log_task_metadata()

    def _log_global_metadata(self):
        """Logs heavyweight metadata."""
        mlflow.set_tag("mlflow.source.git.commit", get_git_hash())
        try:
            git_diff = subprocess.check_output(["git", "diff"], stderr=subprocess.STDOUT).decode("utf-8")
            if git_diff.strip():
                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as tmp:
                    tmp.write(git_diff)
                    tmp_path = tmp.name

                mlflow.log_artifact(tmp_path, artifact_path="scripts")
                Path(tmp_path).unlink()
                logger.info("Uncommitted git changes logged as diff.patch.")
        except Exception as e:
            logger.warning(f"Failed to capture git diff: {e}")

    def _log_task_metadata(self):
        """Logs configuration specific to this specific worker/task."""
        mlflow.set_tag("mlflow.folder", os.getcwd())

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
        """Ends the active MLflow run."""
        if mlflow.active_run():
            logger.info(f"Closing active MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()