from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from typing import Dict
import torch

import logging

logger = logging.getLogger(__name__)


class GradientClippingCallback(AbstractCallback):
    def __init__(self, frequency: int = 100):
        self.frequency = frequency

    def on_clipping(
        self, epoch: int, step: int, metrics: Dict[str, float], **kwargs
    ) -> Dict:
        if (step + 1) % self.frequency == 0:
            grads = []
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().flatten())

            # Concatenate all gradients into one large vector
            all_grads = torch.cat(grads)

            # 1. Sparsity: percentage of near-zero gradients
            sparsity = (all_grads.abs() < 1e-7).float().mean().item()

            # 2. Statistics for SNR
            mean_grad = all_grads.mean().item()
            std_grad = all_grads.std().item()
            # Adding a small epsilon to avoid division by zero
            gsnr = (mean_grad**2) / (std_grad**2 + 1e-8)

            return {
                "train/grad/sparsity": sparsity,
                "train/grad/gsnr": gsnr,
                "train/grad/mean": mean_grad,
                "train/grad/std": std_grad,
            }
