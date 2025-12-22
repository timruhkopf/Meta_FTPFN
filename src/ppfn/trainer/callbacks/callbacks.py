# from ifbo.priors.prior_bag import get_batch
from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
import mlflow
import torch
from typing import Dict


class MLflowCallback(AbstractCallback):
    """Log metrics to MLflow."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency

    def on_step_end(self, epoch: int, step: int, metrics: Dict[str, float], **kwargs):
        if step % self.log_frequency == 0:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=epoch * 1000 + step)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        for key, value in metrics.items():
            mlflow.log_metric(f"epoch_{key}", value, step=epoch)


class MLflowValidationCallback:
    def __init__(self, val_config, get_batch, n,  frequency=1, mlflow_run_id=None):
        """
        :param val_config: Dictionary containing arguments for get_batch 
                           (batch_size, seq_len, num_features, single_eval_pos, etc.)
        :param mlflow_run_id: Optional existing run ID to log to
        """
        self.val_config = val_config
        self.run_id = mlflow_run_id
        self.frequency = frequency
        self.get_batch = get_batch
        self.n = n

    @torch.no_grad()
    def on_epoch_end(self, model, step, prefix="val"):
        """
        Executes validation and logs to MLflow.
        :param model: The torch model being trained
        :param step: The current training step or epoch
        :param prefix: Prefix for the MLflow metric key (e.g., 'val' or 'test')
        """
        if step % self.frequency != 0:
            avg_loss = self.evaluation_loop(model)

            if mlflow.active_run():
                mlflow.log_metric(f"{prefix}_nll", avg_loss, step=step)
        
            
    def evaluation_loop(self, model):
        model.eval()

        losses = []
        for _ in range(self.n):
        
            # 1. Generate synthetic batch using your function
            batch = self.get_batch(**self.val_config)
            
            # 2. Forward pass
            # Adjusting inputs based on typical Transformer/PFN architectures
            # Usually: model((x, y), single_eval_pos=...)
            output = model((batch.x, batch.y), single_eval_pos=self.val_config['single_eval_pos'])
            
            # 3. Calculate NLL using the model's internal criterion
            # Based on your note: model.criterion.__call__ gives the NLL score
            # Note: target_y is usually indexed at single_eval_pos for PFNs
            target = batch.target_y[self.val_config['single_eval_pos']:]
            nll_loss = model.criterion(output, target)
            
            val_loss = nll_loss.item()
            losses.append(val_loss)

        avg_loss = sum(losses) / len(losses)
        
        model.train()
        return avg_loss