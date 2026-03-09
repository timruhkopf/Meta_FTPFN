import torch
import torch.nn as nn

from ppfn.dataset.get_batch.transforms.ae_warping import visualize_functional_warp, visualize_vector_field


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, input_dim, z_dim, mask, hidden_dim=64):
        super().__init__()
        self.register_buffer('mask', mask)

        pass_dim = mask.sum().item()
        trans_dim = input_dim - pass_dim

        # The networks now take (x_pass + z_shift) as input!
        combined_dim = pass_dim + z_dim

        self.net_s = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trans_dim)
        )

        self.net_t = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trans_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # We can use standard initialization here.
        # The alpha parameter will automatically anchor us to the identity mapping.
        for net in [self.net_s, self.net_t]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.5)
                    nn.init.zeros_(m.bias)

    def forward(self, x, z_shift, alpha):
        x_pass = x[:, self.mask]
        x_trans = x[:, ~self.mask]

        # Ensure z_shift matches the batch size of x
        if z_shift.dim() == 1:
            z_shift = z_shift.unsqueeze(0).expand(x.shape[0], -1)

        # Condition on the global bottleneck
        condition = torch.cat([x_pass, z_shift], dim=-1)

        # Tanh bounds the scaling so the space doesn't explode
        s = torch.tanh(self.net_s(condition))
        t = self.net_t(condition)

        # Apply the alpha-scaled warp
        y_trans = x_trans * torch.exp(alpha * s) + alpha * t

        y = torch.empty_like(x)
        y[:, self.mask] = x_pass
        y[:, ~self.mask] = y_trans
        return y

    def inverse(self, y, z_shift, alpha):
        """ The mathematical inverse for the Corrector to undistort Task B """
        y_pass = y[:, self.mask]
        y_trans = y[:, ~self.mask]

        if z_shift.dim() == 1:
            z_shift = z_shift.unsqueeze(0).expand(y.shape[0], -1)

        condition = torch.cat([y_pass, z_shift], dim=-1)

        s = torch.tanh(self.net_s(condition))
        t = self.net_t(condition)

        # Exact mathematical reversal
        x_trans = (y_trans - alpha * t) * torch.exp(-alpha * s)

        x = torch.empty_like(y)
        x[:, self.mask] = y_pass
        x[:, ~self.mask] = x_trans
        return x


class SharedLatentINNPrior(nn.Module):
    def __init__(self, input_dim=2, z_dim=2, depth=4, hidden_dim=64):
        """
        z_dim is your strict global bottleneck complexity.

        Critically, we don't sample the inn instance, but only the latent z_shift that conditions our warping
        """
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            mask = torch.zeros(input_dim, dtype=torch.bool)
            # Alternate the mask
            if i % 2 == 0:
                mask[:input_dim // 2] = True
            else:
                mask[input_dim // 2:] = True

            self.layers.append(ConditionalAffineCoupling(input_dim, z_dim, mask, hidden_dim))

    def forward(self, hp, z_shift, alpha=1.0):
        # 1. Map to unbounded space
        x = torch.clamp(hp, 1e-5, 1.0 - 1e-5)
        z_flow = torch.logit(x)

        # 2. Flow forward (Warping Task A -> Task B)
        for layer in self.layers:
            z_flow = layer(z_flow, z_shift, alpha)

        # 3. Map back to [0, 1]
        return torch.sigmoid(z_flow)

    def inverse(self, hp_warped, z_shift, alpha=1.0):
        # 1. Map warped HPs to unbounded space
        y = torch.clamp(hp_warped, 1e-5, 1.0 - 1e-5)
        z_flow = torch.logit(y)

        # 2. Flow backward in reverse order (Undistorting Task B -> Task A)
        for layer in reversed(self.layers):
            z_flow = layer.inverse(z_flow, z_shift, alpha)

        # 3. Map back to [0, 1]
        return torch.sigmoid(z_flow)


if __name__ == '__main__':
    from functools import partial

    Z_DIM = 2
    ALPHA = 1.0

    # 1. Instantiate the fixed architecture (e.g., 2D HP space, Rank-1 Bottleneck)
    model = SharedLatentINNPrior(input_dim=2, z_dim=Z_DIM, depth=2, hidden_dim=10)

    # 2. Sample a random task-shift vector from a standard normal prior
    z_shift = torch.randn(Z_DIM)  # This single scalar drives the entire 2D warp!

    # 3. Generate Task B (alpha controls how far along the z_shift trajectory we move)
    # hp_task_B = model(hp_tensor, z_shift=z_shift, alpha=1.0)

    m = partial(model, z_shift=z_shift)  # This is the function we inject into Task B for warping

    visualize_vector_field(m, alpha=ALPHA)
    visualize_functional_warp(m, alpha=ALPHA)
