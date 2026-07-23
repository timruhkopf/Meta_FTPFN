import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def update_ema_dict_modules(student_module: nn.Module, teacher_module: nn.Module, momentum: float = 0.996):
    """
    Updates the teacher module parameters using EMA based on the student module.
    Works for any nn.Module (Linear, PFNStack, LayerNorm, etc).
    """
    with torch.no_grad():
        for s_param, t_param in zip(student_module.parameters(), teacher_module.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

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
    def __init__(self, criterion, share_unrelated=0.2):
        super().__init__()
        self.criterion_backend = criterion

        pos_weight = share_unrelated / (1 - share_unrelated)
        self.register_buffer('global_pos_weight', torch.tensor([pos_weight]))

        self.train_jointly = False
        self.is_warmup = False

    def forward(self, output, batch, **kwargs):
        logits_A, logits_B, logits_C = output

        logits_A, logits_B, logits_C = logits_A.float(), logits_B.float(), logits_C.float()

        logits_teacher = ForwardMetaContext.get("logits_teacher_QA")
        device = logits_A.device

        Y_test_A = batch['test']['Y_A']
        Y_test_B = batch['test']['Y_B']
        is_unrelated = batch['params']['is_unrelated']

        # --- Base NLL Losses ---
        loss_A = self.criterion_backend(logits_A, Y_test_A).mean()
        loss_B = self.criterion_backend(logits_B, Y_test_B).mean()

        # loss_C is calculated but intentionally excluded from the main backprop graph
        # It serves purely as our baseline for transfer telemetry
        loss_C = self.criterion_backend(logits_C, Y_test_A).mean()

        metrics = {
            "nll/A": loss_A.item(),
            "nll/B": loss_B.item(),
            "nll/hatC": loss_C.item(),
        }

        # --- The New Core Logic: Anchor + Align ---
        # The marginals (A and B) constantly anchor the network to geometric reality.
        total_loss = loss_A + loss_B

        # --- Auxiliary Losses (JEPA Distillation) ---
        # The Predictor trains EXCLUSIVELY by mimicking the Teacher via this aux loss.
        if self.train_jointly or not self.is_warmup:
            aux_jepa = ForwardMetaContext.get('aux_loss/jepa')
            if aux_jepa is not None:
                total_loss += aux_jepa

        metrics["nll/Total"] = total_loss.clone().item()

        # Telemetry ingestion
        metrics.update({k: v for k, v in ForwardMetaContext._state.__dict__.items() if k.startswith('Telemetry/')})

        aux_domain = ForwardMetaContext.get('B_in_A_domain')
        if aux_domain is not None:
            if 'kl_loss' in aux_domain: total_loss += 1e-5 * aux_domain['kl_loss']
            if 'cycle_loss' in aux_domain: total_loss += 1e-5 * aux_domain['cycle_loss']

        # --- Custom Metric Calculations (Transfer Gains) ---
        with torch.no_grad():
            loss_A_unreduced = self.criterion_backend(logits_A, Y_test_A)
            loss_hatC_unreduced = self.criterion_backend(logits_C, Y_test_A)
            loss_teacher_unreduced = self.criterion_backend(logits_teacher, Y_test_A)

            metrics["nll/C"] = loss_teacher_unreduced.mean().item()

            diff_hatC_A = loss_hatC_unreduced - loss_A_unreduced  # Student Transfer
            diff_C_A = loss_teacher_unreduced - loss_A_unreduced  # Oracle Transfer

        metrics["nll/hatC-A_all"] = diff_hatC_A.mean().item()
        metrics["nll/C-A_all"] = diff_C_A.mean().item()

        if (~is_unrelated).any():
            metrics["nll/hatC-A_related"] = diff_hatC_A[:, ~is_unrelated].mean().item()
            metrics["nll/C-A_related"] = diff_C_A[:, ~is_unrelated].mean().item()

        if is_unrelated.any():
            metrics["nll/hatC-A_unrelated"] = diff_hatC_A[:, is_unrelated].mean().item()
            metrics["nll/C-A_unrelated"] = diff_C_A[:, is_unrelated].mean().item()

        return total_loss, metrics