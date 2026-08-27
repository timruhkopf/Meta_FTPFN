import torch
import torch.nn as nn
from typing import Tuple, Optional


class KumaraswamyInputWarper(nn.Module):
    """
    Coordinate-wise monotonic input warper for [0, 1]^D domains.
    Matches the functional vocabulary of Snoek et al. (2014) while being
    fully vectorized and GPU-native for PFN prior generation.
    """

    def __init__(
            self,
            dim: int,
            log_std: float = 0.65,
            min_param: float = 0.1,
            max_param: float = 10.0,
            eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.log_std = log_std
        self.min_param = min_param
        self.max_param = max_param
        self.eps = eps

    def sample_parameters(
            self,
            batch_size: int = 1,
            device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples warp shape parameters (a, b) ~ LogNormal(0, log_std^2)
        Shape: (batch_size, 1, dim) for broadcasting over (B, N, D).
        """
        device = device or torch.device("cpu")

        # Sample log-normal around identity (0 mean in log space = 1 in linear space)
        log_a = torch.randn(batch_size, 1, self.dim, device=device) * self.log_std
        log_b = torch.randn(batch_size, 1, self.dim, device=device) * self.log_std

        # Clip to prevent extreme underflow / step-function collapses
        a = torch.clamp(torch.exp(log_a), self.min_param, self.max_param)
        b = torch.clamp(torch.exp(log_b), self.min_param, self.max_param)
        return a, b

    def forward(
            self,
            x: torch.Tensor,
            a: Optional[torch.Tensor] = None,
            b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies g(x; a, b) = 1 - (1 - x^a)^b coordinate-wise.

        Args:
            x: Tensor of shape (batch_size, n_points, dim) in [0, 1]
            a: Optional pre-sampled 'a' parameter tensor
            b: Optional pre-sampled 'b' parameter tensor
        """
        if a is None or b is None:
            a, b = self.sample_parameters(batch_size=x.shape[0], device=x.device)

        # Numerical stabilization at boundary values
        x_safe = torch.clamp(x, self.eps, 1.0 - self.eps)

        # Kumaraswamy CDF warp
        inner = 1.0 - torch.pow(x_safe, a)
        inner_safe = torch.clamp(inner, self.eps, 1.0)
        warped_x = 1.0 - torch.pow(inner_safe, b)

        return warped_x