"""
Gaussian predictive output head.

Figure 2 of the paper shows the Output Layer producing (mu, sigma^2)
directly -- i.e. MTPFN parameterizes a Normal predictive distribution
p(y_test | x_test, D) = N(mu(D, x_test), sigma^2(D, x_test)), rather than
the discretized "Riemann/bar" distribution used in some other PFN papers
(e.g. PFNs4BO). We therefore train with a closed-form Gaussian negative
log-likelihood, and get closed-form means/variances for BO acquisition
functions for free.
"""

from __future__ import annotations

import math

import torch


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor,
                  min_std: float = 1e-4) -> torch.Tensor:
    """Elementwise Gaussian NLL. Clamps the predicted std for stability."""
    log_var = log_var.clamp(min=2 * math.log(min_std))
    var = log_var.exp()
    return 0.5 * (log_var + math.log(2 * math.pi) + (y - mu) ** 2 / var)


def split_head_output(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """raw: [..., 2] -> (mu, log_var), with a softplus on the variance channel
    for a numerically friendlier parameterization than raw log-variance."""
    mu, raw_var = raw.unbind(dim=-1)
    log_var = torch.nn.functional.softplus(raw_var).clamp_min(1e-6).log()
    return mu, log_var
