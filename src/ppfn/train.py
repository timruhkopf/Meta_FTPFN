from __future__ import annotations

import os
from typing import Tuple

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
    
    # Load frozen model and get criterion from it
    logger.info("Loading frozen model...")
    model = instantiate(cfg.model.model_class).to(device)
    criterion = model.criterion
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    loader = instantiate(cfg.dataset.dataloader) # PriorDataLoader / DistributedPriorDataLoader
    # Sampling the prior and storing it if required. 
    # This is only needed once and is the entry point to the get_batch functions
    if cfg.dataset.dataloader.get("store", True):
        logger.info("Storing prior samples...")

        (Path(cfg.dataset.dataloader.load_path) / 'partition_0').mkdir(parents=True, exist_ok=True)

        # store the generating yaml config alongside the prior samples
        with open( os.path.join(cfg.dataset.dataloader.load_path, 'generating_config.yaml'), 'w') as f:
            OmegaConf.save(config=cfg.dataset, f=f)

        loader.store_prior(**instantiate(cfg.dataset.store_prior))
        loader._load_chunk(0)

        return 0 # exit after storing prior
    
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
        # mycfg=cfg, # dictconfig cannot be passed directly 
        experiment_name=cfg.experiment_name,
        run_name=cfg.get("run_name", None),
    )

    trainer.log_config(cfg)
    
    logger.info(f"Starting training for {cfg.trainer.epochs} epochs...")
    trainer.fit( epochs=cfg.trainer.epochs, steps=cfg.trainer.steps )
    
    # End MLflow run
    trainer.end_run()
    
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

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("githash", githash)
    main()
