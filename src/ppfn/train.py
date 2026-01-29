from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base='1.1', config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    
    Args:
        cfg: Hydra config from configs/config.yaml and experiment override
    """

    # Pretty print config
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # set seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)

    # Create dataloaders
    logger.info("Creating dataloaders...")

    # Sampling the prior and storing it if required. 
    # This is only needed once and is the entry point to the get_batch functions
    dataset = instantiate(cfg.dataset.dataset_class)
    if cfg.dataset.get("sample_prior", False):
        logger.info("Storing prior samples...")
        dataset.store_prior(**instantiate(cfg.dataset.store_prior))

        # store the generating yaml config alongside the prior samples
        with open(dataset.storage_path / 'generating_config.yaml', 'w') as f:
            OmegaConf.save(config=cfg.dataset, f=f)
        return 0  # exit after storing prior

    # Create a simple DataLoader around the dataset
    loader = instantiate(
        cfg.dataset.dataloader_class,
        dataset=dataset,
        collate_fn=lambda x: x
    )
    # next(iter(loader))  # sanity check

    # FIXME: this is the old API for pfns4bo.utils.PriorDataLoader: remove
    # loader = instantiate(cfg.dataset.dataloader)  # PriorDataLoader / DistributedPriorDataLoader
    # if cfg.dataset.dataloader.get("store", True):
    #     logger.info("Storing prior samples...")
    #
    #     (Path(cfg.dataset.dataloader.load_path) / 'partition_0').mkdir(parents=True, exist_ok=True)
    #
    #     # store the generating yaml config alongside the prior samples
    #     with open( os.path.join(cfg.dataset.dataloader.load_path, 'generating_config.yaml'), 'w') as f:
    #         OmegaConf.save(config=cfg.dataset, f=f)
    #
    #     loader.store_prior(**instantiate(cfg.dataset.store_prior))
    #     loader._load_chunk(0)
    #
    #     return 0 # exit after storing prior

    # Load frozen model and get criterion from it
    logger.info("Loading frozen model...")
    model = instantiate(cfg.model.model_class).to(device)
    criterion = model.criterion

    if hasattr(cfg.trainer, "objective"):
        logger.info("Wrapping criterion with objective...")
        # this is a wrapper objective around the model's criterion
        criterion = instantiate(
            cfg.trainer.objective,
            criterion=criterion,
            model=model
        )

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
        criterion=criterion,
        device=device,
    )
    # dictconfig cannot be passed directly; neither a dict with _target_ key
    trainer.config = OmegaConf.to_container(cfg, resolve=True),

    logger.info(f"Starting training for {cfg.trainer.epochs} epochs...")
    trainer.fit(epochs=cfg.trainer.epochs, steps=cfg.trainer.steps)

    logger.info("Training completed!")

    return 0


if __name__ == "__main__":

    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")


    def githash(*args, **kwargs) -> str:
        try:
            import subprocess
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            return git_hash
        except Exception as e:
            logger.warning(f"Could not retrieve git hash: {e}")
            return "unknown"


    OmegaConf.register_new_resolver("mod", lambda x, y: x % y)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y))
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("githash", githash)
    main()
