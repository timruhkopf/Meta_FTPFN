import torch


def random_rotation(batch, d, device):
    """Batched random orthogonal matrices via QR of a Gaussian."""
    a = torch.randn(batch, d, d, device=device)
    q, _ = torch.linalg.qr(a)
    return q


class DomainMapPrior:
    """Samples a per-task invertible affine domain view: x = R @ z * s + t."""

    def __init__(self, d_z, scale_range=(0.5, 2.0), translate_scale=2.0):
        self.d_z = d_z
        self.scale_range = scale_range
        self.translate_scale = translate_scale

    def sample(self, batch, device):
        R = random_rotation(batch, self.d_z, device)
        s = torch.empty(batch, 1, self.d_z, device=device).uniform_(*self.scale_range)
        t = torch.randn(batch, 1, self.d_z, device=device) * self.translate_scale
        return {"R": R, "s": s, "t": t}

    def apply(self, params, z):
        # z: [B, n, d_z] -> x: [B, n, d_z]
        x = torch.einsum("bij,bnj->bni", params["R"], z)
        x = x * params["s"] + params["t"]
        return x


class RandomMLPFunctionPrior:
    """Samples a random-weight MLP per task as the shared ground-truth
    latent function f: R^{d_z} -> R. Same f underlies both domain views."""

    def __init__(self, d_z, hidden=32, weight_scale=1.0):
        self.d_z = d_z
        self.hidden = hidden
        self.weight_scale = weight_scale

    def sample(self, batch, device):
        d_z, h = self.d_z, self.hidden
        W1 = torch.randn(batch, h, d_z, device=device) * (self.weight_scale / d_z ** 0.5)
        b1 = torch.randn(batch, h, device=device) * 0.1
        W2 = torch.randn(batch, 1, h, device=device) * (self.weight_scale / h ** 0.5)
        b2 = torch.randn(batch, 1, device=device) * 0.1
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def apply(self, params, z):
        # z: [B, n, d_z] -> y: [B, n]
        h = torch.tanh(torch.einsum("bhj,bnj->bnh", params["W1"], z) + params["b1"].unsqueeze(1))
        y = torch.einsum("boh,bnh->bno", params["W2"], h) + params["b2"].unsqueeze(1)
        return y.squeeze(-1)


class DomainShiftRegressionPrior:
    """
    Full task sampler for the domain-shift JEPA/PFN experiment.

    A shared random function f lives on a canonical latent space R^{d_z}.
    Domains A and B are two independently-sampled invertible views of that
    latent space (phi_A, phi_B). A-locations and B-locations are sampled
    INDEPENDENTLY -- there is no index correspondence between an A point
    and a B point.

    Because phi_A, phi_B and f are all known at generation time, we can
    additionally construct "B_in_A": the exact same latent locations that
    underlie B_tr, pushed through phi_A instead of phi_B. This is the
    privileged, domain-shift-free oracle view the teacher gets -- it is
    single-domain by construction, not a cross-domain fusion problem.
    """

    def __init__(self, d_z=2, mlp_hidden=32, noise_std_range=(0.05, 0.3),
                 domain_map_kwargs=None):
        self.d_z = d_z
        self.fn_prior = RandomMLPFunctionPrior(d_z, hidden=mlp_hidden)
        self.map_prior = DomainMapPrior(d_z, **(domain_map_kwargs or {}))
        self.noise_std_range = noise_std_range

    def sample_batch(self, batch_size, n_A_tr, n_A_test, n_B_tr, device="cpu"):
        f = self.fn_prior.sample(batch_size, device)
        phi_A = self.map_prior.sample(batch_size, device)
        phi_B = self.map_prior.sample(batch_size, device)
        noise_std = torch.empty(batch_size, 1, device=device).uniform_(*self.noise_std_range)

        def draw(n):
            return torch.randn(batch_size, n, self.d_z, device=device)

        def make(z, phi):
            x = self.map_prior.apply(phi, z)
            y = self.fn_prior.apply(f, z) + noise_std * torch.randn(batch_size, z.shape[1], device=device)
            return x, y

        z_A_tr, z_A_test, z_B_tr = draw(n_A_tr), draw(n_A_test), draw(n_B_tr)

        x_A_tr, y_A_tr = make(z_A_tr, phi_A)
        x_A_test, y_A_test = make(z_A_test, phi_A)
        x_B_tr, y_B_tr = make(z_B_tr, phi_B)
        # oracle: same latent locations as B_tr, observed through A's domain map
        x_BinA_tr, y_BinA_tr = make(z_B_tr, phi_A)

        return {
            "A_tr": (x_A_tr, y_A_tr),
            "A_test": (x_A_test, y_A_test),
            "B_tr": (x_B_tr, y_B_tr),
            "BinA_tr": (x_BinA_tr, y_BinA_tr),
            "noise_std": noise_std,
        }


if __name__ == "__main__":
    prior = DomainShiftRegressionPrior(d_z=2)
    batch = prior.sample_batch(batch_size=4, n_A_tr=20, n_A_test=10, n_B_tr=40)
    for k, v in batch.items():
        if isinstance(v, tuple):
            print(k, v[0].shape, v[1].shape)
        else:
            print(k, v.shape)
