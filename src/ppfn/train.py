from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow

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
    loader = instantiate(cfg.dataset.dataloader)
    loader.store_prior(**instantiate(cfg.dataset.store_prior))
    loader._load_chunk(0)
    
    # Instantiate optimizer and scheduler as partials
    # They will be called with model params and optimizer respectively in trainer.__init__
    logger.info("Setting up optimizer and scheduler...")
    optimizer_partial = instantiate(cfg.optimizer)
    scheduler_partial = instantiate(cfg.scheduler)
    
    # Create callbacks using Hydra instantiate
    # callbacks = []
    # if hasattr(cfg.trainer, 'callbacks'):
    #     for callback_cfg in cfg.trainer.callbacks:
    #         callbacks.append(instantiate(callback_cfg))
    
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
        # callbacks=callbacks,
        experiment_name=cfg.experiment_name,
        run_name=cfg.get("run_name", None),
    )
    
    # Set MLflow config logging if applicable
    # if hasattr(trainer, 'mlflow_run') and trainer.mlflow_run:
    #     config_dict = OmegaConf.to_container(cfg, resolve=True)
    #     if isinstance(config_dict, dict):
    #         mlflow.log_params({str(k): str(v) for k, v in config_dict.items()})
    
    # Train
    logger.info(f"Starting training for {cfg.trainer.epochs} epochs...")
    trainer.fit( epochs=cfg.trainer.epochs, steps=cfg.trainer.steps )
    
    # End MLflow run
    trainer.end_run()
    
    logger.info("Training completed!")





if __name__ == "__main__":
    
    from pathlib import Path
    from dotenv import load_dotenv
    
    load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env")
    main()
