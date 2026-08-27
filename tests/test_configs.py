"""Config-composition sanity checks for configs/.

There's no active model/objective yet (see configs/model/README.md,
configs/trainer/default.yaml), so these tests can't run a real training step.
What they do verify: every group that DOES have a real implementation
(prior=bnn, optimizer=adamw, scheduler=*, deployment=local/slurm, callbacks)
actually instantiates via Hydra, and the deliberate placeholders (`model`,
`benchmark`, `prior.dataset_class`, `trainer.trainer_class.criterion`) fail in
the intended, loud way rather than silently.
"""

import pytest
import torch
import torch.nn as nn
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf.errors import MissingMandatoryValue

# `model`/`benchmark` are mandatory (`???`) with no config file to select yet —
# every compose() call needs to opt out with `~model`/`~benchmark`.
BASE_OVERRIDES = ["~model", "~benchmark", "experiment_name=00-debug-configtest"]


def test_root_config_requires_model_and_benchmark(compose_cfg):
    with pytest.raises(ConfigCompositionException):
        compose_cfg(["experiment_name=00-debug-configtest"])


def test_root_config_composes_with_placeholders_opted_out(compose_cfg):
    cfg = compose_cfg(BASE_OVERRIDES)
    assert cfg.prior.prior_class._target_ == "ppfn.prior.bnn.bnn_prior.BNNPrior"
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.trainer.trainer_class._target_ == "ppfn.trainer.trainer.PPFNTrainer"


def test_prior_dataset_class_is_a_loud_placeholder(compose_cfg):
    """No stream/IterableDataset wrapper around BNNPrior exists yet — accessing
    it should raise, not silently resolve to something wrong."""
    cfg = compose_cfg(BASE_OVERRIDES)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.prior.dataset_class


def test_prior_bnn_instantiates(compose_cfg):
    cfg = compose_cfg(BASE_OVERRIDES)
    prior = instantiate(cfg.prior.prior_class)
    assert prior.num_inputs == cfg.prior.num_inputs
    assert prior.num_outputs == cfg.prior.num_outputs


def test_optimizer_adamw_instantiates_and_applies(compose_cfg):
    cfg = compose_cfg(BASE_OVERRIDES)
    optimizer_partial = instantiate(cfg.optimizer)
    optimizer = optimizer_partial(nn.Linear(2, 2).parameters())
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == cfg.optimizer.weight_decay


@pytest.mark.parametrize("scheduler_name", ["cosine_with_warmup", "cosine", "constant"])
def test_scheduler_instantiates(compose_cfg, scheduler_name):
    cfg = compose_cfg(BASE_OVERRIDES + [f"scheduler={scheduler_name}"])
    scheduler_partial = instantiate(cfg.scheduler)
    dummy_optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    scheduler = scheduler_partial(dummy_optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)


def test_trainer_criterion_is_a_loud_placeholder(compose_cfg):
    """No active loss/objective module exists yet (MultiStreamObjective is
    archived) — instantiating the trainer should fail on the missing criterion,
    not silently construct with criterion=None."""
    cfg = compose_cfg(BASE_OVERRIDES)
    optimizer_partial = instantiate(cfg.optimizer)
    scheduler_partial = instantiate(cfg.scheduler)
    with pytest.raises(MissingMandatoryValue):
        instantiate(
            cfg.trainer.trainer_class,
            model=nn.Linear(1, 1),
            train_loader=None,
            optimizer=optimizer_partial,
            scheduler=scheduler_partial,
            device="cpu",
        )


def test_callbacks_instantiate(compose_cfg):
    cfg = compose_cfg(BASE_OVERRIDES)
    mlflow_cb = instantiate(cfg.callbacks.mlflow)
    clip_cb = instantiate(cfg.callbacks.clip)
    assert type(mlflow_cb).__name__ == "MLflowCallback"
    assert clip_cb.frequency == 1


def test_deployment_local_selects_submitit_local_launcher(compose_cfg):
    cfg = compose_cfg(BASE_OVERRIDES, return_hydra_config=True)
    assert cfg.hydra.launcher._target_.endswith("LocalLauncher")


def test_deployment_slurm_selects_submitit_slurm_launcher(compose_cfg):
    cfg = compose_cfg(
        ["~model", "~benchmark", "experiment=slurm", "experiment_name=03-sweep-configtest", "run_name=x"],
        return_hydra_config=True,
    )
    assert cfg.hydra.launcher._target_.endswith("SlurmLauncher")
    assert cfg.hydra.launcher.partition == "gpu"
