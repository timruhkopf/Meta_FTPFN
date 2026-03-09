import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, mask, hidden_dim=32):
        super().__init__()
        self.register_buffer('mask', mask)
        pass_dim = mask.sum().item()
        trans_dim = input_dim - pass_dim

        self.net_s = nn.Sequential(
            nn.Linear(pass_dim, hidden_dim),
            nn.LeakyReLU(0.1),  # LeakyReLU helps gradients flow better near zero
            nn.Linear(hidden_dim, trans_dim)
        )

        self.net_t = nn.Sequential(
            nn.Linear(pass_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, trans_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for net in [self.net_s, self.net_t]:
            for m in net:
                if isinstance(m, nn.Linear):
                    # Slightly higher gain to encourage more initial warping
                    nn.init.xavier_normal_(m.weight, gain=1.2)
                    # Small random biases prevent the (0,0) fixed-point trap
                    nn.init.normal_(m.bias, mean=0.0, std=0.05)

    def forward(self, x, alpha):
        x_pass = x[:, self.mask]
        x_trans = x[:, ~self.mask]

        # Using softplus or a scaled tanh for s to ensure more expressive scaling
        s = torch.tanh(self.net_s(x_pass))
        t = self.net_t(x_pass)

        y_trans = x_trans * torch.exp(alpha * s) + (alpha * t)

        y = torch.empty_like(x)
        y[:, self.mask] = x_pass
        y[:, ~self.mask] = y_trans
        return y


class INNWarping(nn.Module):
    def __init__(self, input_dim, depth=4, hidden_dim=32):
        super().__init__()
        # NEW: Global latent shift and scale to break center symmetry
        self.latent_shift = nn.Parameter(torch.zeros(input_dim))
        self.latent_scale = nn.Parameter(torch.ones(input_dim))

        self.layers = nn.ModuleList([
            AffineCouplingLayer(
                input_dim,
                self._create_mask(input_dim, i),
                hidden_dim
            ) for i in range(depth)
        ])

    def _create_mask(self, input_dim, i):
        mask = torch.zeros(input_dim, dtype=torch.bool)
        # For 2D, this simply alternates [True, False] and [False, True]
        if i % 2 == 0:
            mask[:input_dim // 2] = True
        else:
            mask[input_dim // 2:] = True
        return mask

    def forward(self, hp, alpha=1.0):
        # 1. Map to unbounded R space
        x = torch.clamp(hp, 1e-5, 1.0 - 1e-5)
        z = torch.logit(x)

        # 2. Apply global latent transformation
        # This allows the "center" of the warp to move away from 0.5, 0.5
        z = (z + (alpha * self.latent_shift)) * (1 + alpha * (self.latent_scale - 1))

        # 3. Apply flows
        for layer in self.layers:
            z = layer(z, alpha)

        return torch.sigmoid(z)


class INNWarpingTransform:
    """
    Wraps the target task with an Invertible Neural Network (INN)
    to create complex, topology-preserving spatial warps in the HP space.
    """

    def __init__(self, alpha_dist_params=(1, 3), depth=4, hidden_dim=32, resample_y0_ymax=True):
        self.alpha_params = alpha_dist_params
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.resample_y0_ymax = resample_y0_ymax

    def __call__(self, target_task):
        related_task = target_task.clone()

        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        # Edge Case Defense: Coupling layers require at least 2 dimensions to split.
        # If a 1D task is passed, we default to a perfect identity mapping.
        if num_inputs < 2:
            return related_task, 1.0

        # Instantiate a fresh INN so the warp is unique for this specific related task
        inn = INNWarping(input_dim=num_inputs, depth=self.depth, hidden_dim=self.hidden_dim)
        inn.eval()  # Set to eval mode as a best practice

        # Sample the "power" of the warp.
        alpha = np.random.beta(self.alpha_params[0], self.alpha_params[1])

        def inn_get_marginal_curve(hyperparams):
            # hyperparams shape: (num_configs, num_params)
            # We don't need gradients for the data generation step
            with torch.no_grad():
                warped_hp = inn(hyperparams, alpha=alpha)

            # Pass the warped configurations into the original task's evaluator
            return ancestor_get_curve_target(warped_hp)

        # Overwrite the related task's evaluator
        related_task.get_marginal_curve = inn_get_marginal_curve

        # Calculate relatedness:
        # alpha=0 implies identical task (relatedness=1.0)
        # Higher alpha implies more severe warping (lower relatedness)
        relatedness = 1.0 - alpha

        return related_task, relatedness

if __name__ == '__main__':
    from ppfn.dataset.get_batch.transforms.ae_warping import plot

    model = INNWarping(input_dim=2, depth=2, hidden_dim=5)
    alpha = 1
    # alpha_dist_params = (1, 3)
    # alpha = np.random.beta(alpha_dist_params[0], alpha_dist_params[1])

    # print(f"Sampled alpha for INN warping: {alpha:.4f} (Relatedness: {1.0 - alpha:.4f})")

    plot(model, alpha=alpha)
