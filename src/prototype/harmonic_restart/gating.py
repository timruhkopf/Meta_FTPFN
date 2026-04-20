import torch
import torch.nn.functional as F
import torch.nn as nn

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class GumbelGate(nn.Module):
    def __init__(self, input_dim, hard=True):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, 1)
        self.hard = hard
        # Initialize bias so it starts "undecided" or slightly open
        nn.init.constant_(self.gate_linear.bias, 0.5)

    def forward(self, x, tau=1.0, training=True):
        logits = self.gate_linear(x)  # (Batch, 1)

        if training:
            # 1. Sample Gumbel noise
            unif = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(unif + 1e-20) + 1e-20)

            # 2. Apply Gumbel-Sigmoid trick
            # We treat the single logit as the difference between two gumbel samples
            y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

            if self.hard:
                # 3. Straight-Through Estimator
                # Forward pass: 0 or 1. Backward pass: gradient of y_soft
                y_hard = (y_soft > 0.5).float()
                gate = y_hard - y_soft.detach() + y_soft
            else:
                gate = y_soft
        else:
            # Inference: Just a deterministic threshold or sigmoid
            gate = (torch.sigmoid(logits) > 0.5).float() if self.hard else torch.sigmoid(logits)

        return gate, logits


class HardSigmoidGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d_model, 1)
        # Initialize bias to 0.5 to 1.0.
        # This ensures gate output starts around 0.6 - 0.7 (inside the linear gradient zone)
        nn.init.constant_(self.gate_linear.bias, 0.8)

    def forward(self, x):
        # F.hardsigmoid is built into modern PyTorch (1.7+)
        logits = self.gate_linear(x)
        ForwardMetaContext.set('gate_logits', logits)
        return F.hardsigmoid(logits)
