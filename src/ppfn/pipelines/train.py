from __future__ import annotations


import numpy as np
import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(cfg: DictConfig, device: torch.device) -> None:
    logger.info(f"Set random seed to {cfg.seed}")

    # Create dataloaders
    logger.info("Creating dataloaders...")

    # Sampling the prior and storing it if required.
    # This is only needed once and is the entry point to the meta_batch functions
    dataset = instantiate(cfg.prior.dataset_class)

    # Create a simple DataLoader around the dataset
    loader = instantiate(
        cfg.prior.dataloader_class, dataset=dataset
    )

    # Load frozen model and get criterion from it
    logger.info("Loading frozen model...")
    model = instantiate(cfg.model.model_class).to(device)

    # Instantiate optimizer and scheduler as partials
    # They will be called with model params and optimizer respectively in trainer.__init__
    logger.info("Setting up optimizer and scheduler...")
    optimizer_partial = instantiate(cfg.optimizer)
    scheduler_partial = instantiate(cfg.scheduler)

    # Create trainer using Hydra instantiate
    logger.info("Initializing trainer...")
    trainer = instantiate(
        cfg.trainer.trainer_class,
        model=model,
        train_loader=loader,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
        device=device,
    )

    # dictconfig cannot be passed directly; neither a dict with _target_ key.
    # Read by MLflowCallback._log_task_metadata to log the full resolved
    # config as params (CLAUDE.md: "Log the flattened resolved config as
    # params"), not just the raw CLI overrides.
    trainer.config = OmegaConf.to_container(cfg, resolve=True)

    logger.info(f"Starting training for {cfg.trainer.epochs} epochs...")
    trainer.fit(epochs=cfg.trainer.epochs, steps=cfg.trainer.steps)

    logger.info("Training completed!")


@hydra.main(version_base="1.1", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.

    Args:
        cfg: Hydra config from configs/config.yaml and experiment override
    """
    assert_clean_tree_for_real_runs(cfg)

    # Pretty print config
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # log hydra overrides in .err files for easier debugging:
    logger.error(f"Overrides: {hydra.core.hydra_config.HydraConfig.get()['overrides']['task']}")

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # set seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    run(cfg, device)

    return 0



if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    from ppfn.utils.git_tools import assert_clean_tree_for_real_runs
    from ppfn.utils.resolvers import register_resolvers

    load_dotenv(dotenv_path=Path(__file__).parents[3] / ".env")

    register_resolvers()

    main()
