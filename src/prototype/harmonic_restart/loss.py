import torch
import torch.nn as nn
import torch.nn.functional as F
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class TriHarmonicLoss(nn.Module):
    def __init__(self, criterion, use_attn_bonus=False, share_unrelated=0.2):
        super().__init__()
        self.criterion_backend = criterion  # e.g., FullSupportBarDistribution
        self.use_attn_bonus = use_attn_bonus


        # for class imbalance!
        # Calculate pos_weight for gate loss
        pos_weight = share_unrelated / (1 - share_unrelated)
        self.register_buffer('global_pos_weight', torch.tensor([pos_weight]))

        # state variables to be changed by the trainer
        self.train_jointly = False
        self.is_warmup = False

    def forward(self, output, batch,  **kwargs):
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

        # --- Guided Attention Loss ---
        loss_guided_attn = torch.tensor(0.0, device=device)
        raw_attn_weights = ForwardMetaContext.get('cross_attn_weights')

        if self.use_attn_bonus and raw_attn_weights is not None:
            # ... (Insert your existing Guided Attention "Bonus Hill" logic here) ...
            # Assign result to loss_guided_attn
            metrics["metrics/guided_Attn_Loss"] = loss_guided_attn.item()

        # --- Loss Routing based on Warmup Phase ---
        # We will let the Trainer decide if we are in warmup via a kwarg flag

        if self.train_jointly:
            total_loss = loss_A + loss_B + loss_C
        elif self.is_warmup:
            total_loss = loss_A + loss_B
        else:
            total_loss = loss_C + loss_guided_attn

        # --- Gate Loss ---
        gate_logits_val = ForwardMetaContext.get('gate_logits')
        if gate_logits_val is not None:
            gate_logits = gate_logits_val.squeeze(-1).float()
            ideal_gate = (~is_unrelated).float()
            loss_gate = F.binary_cross_entropy_with_logits(
                gate_logits, ideal_gate, pos_weight=self.global_pos_weight
            )
            total_loss += loss_gate
            metrics["metrics/Gate_Loss"] = loss_gate.item()

        # --- Auxiliary Losses ---
        aux = ForwardMetaContext.get('B_in_A_domain')
        if aux is not None:
            if 'kl_loss' in aux: total_loss += 1e-5 * aux['kl_loss']
            if 'cycle_loss' in aux: total_loss += 1e-5 * aux['cycle_loss']

        # --- Custom Metric Calculations (Diffs & Gates) ---
        loss_C_unreduced = self.criterion_backend(logits_C, Y_test_A)
        loss_A_unreduced = self.criterion_backend(logits_A, Y_test_A)
        diff_unreduced = loss_C_unreduced - loss_A_unreduced

        metrics["nll/Total"] = total_loss.item()
        metrics["nll/C-A_all"] = diff_unreduced.mean().item()

        if (~is_unrelated).any():
            metrics["nll/C-A_related"] = diff_unreduced[:, ~is_unrelated].mean().item()
        if is_unrelated.any():
            metrics["nll/C-A_unrelated"] = diff_unreduced[:, is_unrelated].mean().item()

        return total_loss, metrics