"""
Using a trained MTPFN as a Bayesian-optimization surrogate.

Matches Appendix A.1's multi-task BO setup: task 0 is the live target task
being optimized; auxiliary tasks 1..K-1 supply historical/related data in
context. Since the output head is Gaussian (mu, sigma^2), Expected
Improvement has the usual closed form (Jones et al., 1998) rather than
needing Monte Carlo integration over a discretized posterior.
"""

from __future__ import annotations

import math

import torch

from .gaussian_head import split_head_output
from .model import MTPFN

_NORMAL = torch.distributions.Normal(0.0, 1.0)


class MTPFNSurrogate:
    def __init__(self, model: MTPFN, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def _build_episode(self, x_query: torch.Tensor, task_data: list[dict]):
        """task_data[0] is the target task's history so far; task_data[1:]
        are auxiliary tasks. Appends x_query as extra (masked) query slots
        onto task 0's sequence."""
        T = len(task_data)
        Q, d = x_query.shape
        n0 = task_data[0]["x"].shape[0]
        max_len = max(t["x"].shape[0] for t in task_data)
        max_len = max(max_len, n0 + Q)

        x = torch.zeros(1, T, max_len, d, device=self.device)
        y = torch.zeros(1, T, max_len, device=self.device)
        valid_mask = torch.zeros(1, T, max_len, dtype=torch.bool, device=self.device)
        query_mask = torch.zeros(1, T, max_len, dtype=torch.bool, device=self.device)

        for t, data in enumerate(task_data):
            n = data["x"].shape[0]
            x[0, t, :n] = data["x"].to(self.device)
            y[0, t, :n] = data["y"].to(self.device)
            valid_mask[0, t, :n] = True
            if t == 0:
                x[0, t, n:n + Q] = x_query.to(self.device)
                valid_mask[0, t, n:n + Q] = True
                query_mask[0, t, n:n + Q] = True

        return x, y, valid_mask, query_mask, n0, Q

    @torch.no_grad()
    def predict(self, x_query: torch.Tensor, task_data: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) for x_query on the target task, each [Q]."""
        x, y, valid_mask, query_mask, n0, Q = self._build_episode(x_query, task_data)
        raw = self.model(x, y, valid_mask, query_mask)
        mu, log_var = split_head_output(raw)
        mu_q = mu[0, 0, n0:n0 + Q]
        std_q = log_var[0, 0, n0:n0 + Q].mul(0.5).exp()
        return mu_q, std_q

    @torch.no_grad()
    def expected_improvement(self, x_query: torch.Tensor, task_data: list[dict],
                              best_f: float, maximize: bool = True) -> torch.Tensor:
        """Closed-form Expected Improvement (Jones et al., 1998)."""
        mu, std = self.predict(x_query, task_data)
        std = std.clamp_min(1e-9)
        sign = 1.0 if maximize else -1.0
        z = sign * (mu - best_f) / std
        ei = std * (z * _NORMAL.cdf(z) + torch.exp(_NORMAL.log_prob(z)))
        return ei.clamp_min(0.0)
