import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def update_ema(student_model: nn.Module, teacher_model: nn.Module, momentum: float = 0.996):
    """
    Updates the Teacher network using Exponential Moving Average.
    """
    with torch.no_grad():
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

def compute_jepa_loss(prediction: Tensor, target: Tensor, mode: str = 'cosine') -> Tensor:
    """
    Target MUST be detached. Tensors expected in shape: (Time, Batch, Dim).
    """
    if mode == 'mse':
        return F.mse_loss(prediction, target)
    elif mode == 'cosine':
        # Cosine distance across the embedding dimension
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
        return 1.0 - (prediction * target).sum(dim=-1).mean()
    else:
        raise ValueError("Mode must be 'mse' or 'cosine'")


import torch
import torch.nn as nn
import torch.nn.functional as F
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class TriJepaNLLLoss(nn.Module):
    def __init__(self, criterion,  share_unrelated=0.2):
        super().__init__()
        self.criterion_backend = criterion  # e.g., FullSupportBarDistribution



        # for class imbalance!
        # Calculate pos_weight for gate loss
        pos_weight = share_unrelated / (1 - share_unrelated)
        self.register_buffer('global_pos_weight', torch.tensor([pos_weight]))

        # state variables to be changed by the trainer
        self.train_jointly = False
        self.is_warmup = False

    def forward(self, output, batch, **kwargs):
        logits_A, logits_B, logits_C = output
        device = logits_A.device

        # Extract targets
        Y_test_A = batch['test']['Y_A']
        Y_test_B = batch['test']['Y_B']
        is_unrelated = batch['params']['is_unrelated']

        # --- Base NLL Losses ---
        loss_A = self.criterion_backend(logits_A, Y_test_A).mean()
        loss_B = self.criterion_backend(logits_B, Y_test_B).mean()
        loss_C = self.criterion_backend(logits_C, Y_test_A).mean()  # C targets A

        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            "nll/A": loss_A.item(),
            "nll/B": loss_B.item(),
            "nll/C": loss_C.item(),
        }

        if self.train_jointly:
            total_loss = loss_A + loss_B + loss_C
        elif self.is_warmup:
            total_loss = loss_A + loss_B
        else:
            total_loss = loss_C # + loss_guided_attn

        metrics["nll/Total"] = total_loss.clone().item()

        # --- Auxiliary Losses ---
        aux =  ForwardMetaContext.get('aux_loss/jepa')
        if aux is not None:
            total_loss += aux


        # FIXME: move to trainer?
        metrics.update({k:v for k,v in ForwardMetaContext._state.__dict__.items() if k.startswith('Telemetry/')})

        aux = ForwardMetaContext.get('B_in_A_domain')
        if aux is not None:
            if 'kl_loss' in aux: total_loss += 1e-5 * aux['kl_loss']
            if 'cycle_loss' in aux: total_loss += 1e-5 * aux['cycle_loss']

        # --- Custom Metric Calculations (Diffs & Gates) ---
        with torch.no_grad():
            loss_C_unreduced = self.criterion_backend(logits_C, Y_test_A)
            loss_A_unreduced = self.criterion_backend(logits_A, Y_test_A)
            diff_unreduced = loss_C_unreduced - loss_A_unreduced


        metrics["nll/C-A_all"] = diff_unreduced.mean().item()

        if (~is_unrelated).any():
            metrics["nll/C-A_related"] = diff_unreduced[:, ~is_unrelated].mean().item()
        if is_unrelated.any():
            metrics["nll/C-A_unrelated"] = diff_unreduced[:, is_unrelated].mean().item()

        return total_loss, metrics