from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow






def create_data_loaders(cfg: DictConfig, device: torch.device) -> Tuple[DataLoader, DataLoader | None]:
    """Create train and validation dataloaders."""
    
    if cfg.dataset.type == "paired_extrapolation":
        # Generate paired extrapolation batches
        from ppfn.dataset.x_prior import generate_paired_extrapolation_batch
        batch_data = generate_paired_extrapolation_batch(
            batch_size=cfg.dataset.batch_size,
            num_features=cfg.dataset.num_features,
            support_points=cfg.dataset.support_points,
            gap=tuple(cfg.dataset.gap),
            device=device,
        )
        
        # For simplicity, create a dataset that repeats batches
        # In practice, you'd generate multiple batches offline or use a custom DataLoader
        train_dataset = TensorDataset(
            batch_data["support_a_x"],
            batch_data["support_a_y"],
            batch_data["support_b_x"],
            batch_data["support_b_y"],
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # Each "batch" is already a full paired extrapolation batch
            shuffle=cfg.dataset.get("shuffle", True),
            num_workers=cfg.dataset.get("num_workers", 0),
        )
        
        val_loader = None  # Optionally create a validation loader
        
    elif cfg.dataset.type == "dummy":
        # Create dummy dataset for testing
        num_samples = cfg.dataset.get("num_samples", 100)
        num_features = cfg.dataset.num_features
        seq_len = cfg.dataset.get("seq_len", 32)
        
        x = torch.randn(num_samples, seq_len, num_features, device=device)
        y = torch.randn(num_samples, seq_len, 1, device=device)
        
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.dataset.batch_size,
            shuffle=cfg.dataset.get("shuffle", True),
            num_workers=0,
        )
        
        val_loader = None
    
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.type}")
    
    return train_loader, val_loader


@hydra.main(version_base='1.1', config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entry point.
    
    Args:
        cfg: Hydra config from configs/config.yaml and experiment override
    """
    
    # Pretty print config
    print(OmegaConf.to_yaml(cfg))
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load frozen model and get criterion from it
    print("Loading frozen model...")
    model = instantiate(cfg.model.model_class).to(device)
    criterion = model.criterion
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_data_loaders(cfg, device)
    
    # Instantiate optimizer and scheduler as partials
    # They will be called with model params and optimizer respectively in trainer.__init__
    print("Setting up optimizer and scheduler...")
    optimizer_partial = instantiate(cfg.optimizer)
    scheduler_partial = instantiate(cfg.scheduler)
    
    # Create callbacks using Hydra instantiate
    callbacks = []
    if hasattr(cfg.trainer, 'callbacks'):
        for callback_cfg in cfg.trainer.callbacks:
            callbacks.append(instantiate(callback_cfg))
    
    # Create trainer using Hydra instantiate
    print("Initializing trainer...")
    trainer = instantiate(
        cfg.trainer.trainer_class,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        experiment_name=cfg.experiment_name,
        run_name=cfg.get("run_name", None),
    )
    
    # Set MLflow config logging if applicable
    if hasattr(trainer, 'mlflow_run') and trainer.mlflow_run:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(config_dict, dict):
            mlflow.log_params({str(k): str(v) for k, v in config_dict.items()})
    
    # Train
    print(f"Starting training for {cfg.trainer.epochs} epochs...")
    trainer.fit(
        epochs=cfg.trainer.epochs,
        val_loader=val_loader,
        val_frequency=cfg.trainer.get("val_frequency", 10),
    )
    
    # End MLflow run
    trainer.end_run()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
