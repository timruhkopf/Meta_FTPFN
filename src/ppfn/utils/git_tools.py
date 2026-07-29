import subprocess

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def githash(*args, **kwargs) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def assert_clean_tree_for_real_runs(cfg):
    """Hard rule: baseline/sweep runs must come from a committed tree, so
    the git_sha tag logged to MLflow is always trustworthy. Debug runs are
    exempt — that's what 00-debug is for."""
    if str(cfg.experiment_name).startswith(("02-baseline", "03-sweep")):
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return  # not a git repo yet (e.g. fresh scaffold) — don't block
        if dirty:
            raise RuntimeError(
                "Uncommitted changes detected. Commit before logging a "
                "baseline/sweep run — use mlflow_experiment=00-debug-* instead."
            )