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

        # --- Guided Attention Loss ---
        loss_guided_attn = torch.tensor(0.0, device=device)
        raw_attn_weights = ForwardMetaContext.get('cross_attn_weights')

        if self.use_attn_bonus and raw_attn_weights is not None:
            if raw_attn_weights.dim() == 4:
                attn_weights = raw_attn_weights.mean(dim=1)
            else:
                attn_weights = raw_attn_weights

            X_C = torch.cat([batch['train']['X_A'], batch['test']['X_A']], dim=0)
            X_C_clean = torch.nan_to_num(X_C, nan=0.0).transpose(0, 1)

            X_B_train = batch['train']['X_B']
            X_B_clean = torch.nan_to_num(X_B_train, nan=0.0).transpose(0, 1)

            h_shift = batch['params']['h_shift_A'].to(device).view(-1, 1)
            X_target_b = X_C_clean - h_shift

            dist_sq = (X_target_b.unsqueeze(2) - X_B_clean.unsqueeze(1)) ** 2
            sigma = 0.5
            bonus_landscape = torch.exp(-dist_sq / (2 * sigma ** 2))

            Seq_B = X_B_clean.shape[1]
            attn_to_B = attn_weights[:, :, :Seq_B]
            expected_bonus = (attn_to_B * bonus_landscape).sum(dim=-1)
            bonus_loss = -torch.log(expected_bonus + 1e-8)

            # NOTE: model.layer.use_B_attn_sink is hard to access from the loss class directly.
            # Assuming it's False here based on standard config, or you can pass it via kwargs.
            loss_per_batch = torch.where(is_unrelated.unsqueeze(1), torch.zeros_like(bonus_loss), bonus_loss)

            loss_guided_attn = loss_per_batch.mean()
            metrics["metrics/guided_Attn_Loss"] = loss_guided_attn.item()

        # --- Loss Routing based on Warmup Phase ---
        # We will let the Trainer decide if we are in warmup via a kwarg flag

        if self.train_jointly:
            total_loss = loss_A + loss_B + loss_C
        elif self.is_warmup:
            total_loss = loss_A + loss_B
        else:
            total_loss = loss_C + loss_guided_attn

        # --- Auxiliary Losses ---
        aux = ForwardMetaContext.get('B_in_A_domain')
        if aux is not None:
            if 'kl_loss' in aux: total_loss += 1e-5 * aux['kl_loss']
            if 'cycle_loss' in aux: total_loss += 1e-5 * aux['cycle_loss']

        # Capture Total NLL BEFORE Gate Loss (Matches original script)
        metrics["nll/Total"] = total_loss.item()

        # --- Gate Loss & Gate Metrics ---
        gate_logits_val = ForwardMetaContext.get('gate_logits')
        if gate_logits_val is not None:
            gate_logits = gate_logits_val.squeeze(-1).float()
            ideal_gate = (~is_unrelated).float()
            loss_gate = F.binary_cross_entropy_with_logits(
                gate_logits, ideal_gate, pos_weight=self.global_pos_weight
            )
            total_loss += loss_gate
            metrics["metrics/Gate_Loss"] = loss_gate.item()

            # Recreate Gate/Related and Gate/Unrelated metrics
            gate_vals = ForwardMetaContext.get('gate')
            if gate_vals is not None:
                gate_per_batch = gate_vals.squeeze(-1)
                gate_unrelated = gate_per_batch[is_unrelated].mean().item() if is_unrelated.any() else 0.0
                gate_related = gate_per_batch[~is_unrelated].mean().item() if (~is_unrelated).any() else 0.0

                metrics["Gate/Related"] = gate_related
                metrics["Gate/Unrelated"] = gate_unrelated

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