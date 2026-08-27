"""
Multi-task data generating process (DGP / "prior") for MTPFN.

This is a direct implementation of Algorithm A.1 ("Robust Isotropic
Full-Rank ICM") from the paper, with Algorithm A.2 (ARD variant, per-input
-dimension lengthscales) available via `MTPFNPriorConfig(ard=True)`.

Paper's Algorithm A.1, verbatim structure:

    Require: sequence length n, number of tasks T, unrelated task prob. p
    # Input sampling and task assignment
    1: sample inputs {x_i}_{i=1}^n ~ Uniform([0,1]^d)
    2: sample task proportions pi ~ Dirichlet(alpha)
    3: for i = 1..n: sample task id t_i ~ Categorical(pi)
    # Isotropic ICM covariance structure
    4: sample task covariance matrix K_T ~ LKJ(eta=1)
    5: sample input lengthscale l ~ Gamma(3, 6)
    6: define input covariance K_X on {x_i} as RBF kernel with lengthscale l
    7: compute full ICM kernel K = K_T (x) K_X   [indexed/gathered by task id]
    8: sample y ~ N(0, K)
    # Sampling unrelated tasks
    9: for each source task j (j != target task 0):
   10:   with probability p:
   11:     sample new lengthscale l_j ~ Gamma(3, 6)
   12:     define RBF kernel K_X^(j) on {x_i : t_i = j} using l_j
   13:     resample y^(j) ~ N(0, K_X^(j))     # overwrite, single-task GP draw

Two implementation notes, both called out explicitly below because they are
NOT literally in the paper's pseudocode and are only needed to get a
well-posed, batchable training signal:

  (a) Task 0 (the target task) is exempt from the "resample independently"
      step -- the paper's loop is explicitly "for each SOURCE task j", and
      task 0 is the target by definition (Appendix A.1: "target task
      (denoted by task ID 0)").
  (b) Because step 3 assigns every point's task id independently via a
      single multinomial draw, a naive implementation can (rarely) give the
      target task zero points, which breaks the "at least one context +
      one query point" requirement needed for meta-training. We guard
      against this with a minimum-count re-draw; this is a practical
      necessity, not something described in the paper.

For batching in PyTorch, tasks end up with a variable number of points
(exactly as the Dirichlet/Categorical construction implies), so episodes
are represented as fixed-length, task-padded tensors with a validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.distributions import LKJCholesky, Dirichlet, Gamma


@dataclass
class MTPFNPriorConfig:
    num_tasks: int = 4              # T: 1 target (id 0) + (T-1) auxiliary/source tasks
    input_dim: int = 1              # d
    seq_len: int = 60                # n: total points across ALL tasks in the episode
    min_points_per_task: int = 4     # practical guard, see note (b) above
    n_query_target: int = 1          # how many of task 0's points are held out as query/test
    dirichlet_alpha: float = 1.0     # symmetric Dirichlet(alpha) over task proportions pi
    lkj_eta: float = 1.0              # LKJ concentration for the task covariance K_T
    lengthscale_gamma_concentration: float = 3.0   # Gamma(3, 6), BoTorch v1.11 default prior
    lengthscale_gamma_rate: float = 6.0
    p_independent: float = 0.5        # p: prob. a given SOURCE task is resampled independently
    ard: bool = False                 # False -> Algorithm A.1 (isotropic); True -> Algorithm A.2 (ARD)
    obs_noise_std: float = 1e-3       # tiny observation noise for numerical conditioning
    jitter: float = 1e-6
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class MultiTaskICMPrior:
    """Batched sampler implementing Algorithm A.1 / A.2."""

    def __init__(self, cfg: MTPFNPriorConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    @staticmethod
    def _rbf(x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
        """x1: [n, d], x2: [m, d], lengthscale: scalar or [d] (ARD)."""
        d2 = torch.cdist(x1 / lengthscale, x2 / lengthscale, p=2) ** 2
        return torch.exp(-0.5 * d2)

    def _sample_lengthscale(self, gen: torch.Generator | None = None) -> torch.Tensor:
        cfg = self.cfg
        dist = Gamma(cfg.lengthscale_gamma_concentration, cfg.lengthscale_gamma_rate)
        shape = (cfg.input_dim,) if cfg.ard else ()
        return dist.sample(shape).to(cfg.device, cfg.dtype).clamp_min(1e-3)

    # ------------------------------------------------------------------
    def _sample_one_episode(self):
        cfg = self.cfg
        device, dtype = cfg.device, cfg.dtype
        T, n, d = cfg.num_tasks, cfg.seq_len, cfg.input_dim

        # --- Steps 1-3: inputs and task assignment ---
        for _ in range(50):  # note (b): rejection guard for degenerate task counts
            x = torch.rand(n, d, device=device, dtype=dtype)
            pi = Dirichlet(torch.full((T,), cfg.dirichlet_alpha, device=device, dtype=dtype)).sample()
            task_id = torch.multinomial(pi, n, replacement=True)
            counts = torch.bincount(task_id, minlength=T)
            if counts.min().item() >= cfg.min_points_per_task:
                break

        # --- Step 4: task covariance from LKJ ---
        K_T = LKJCholesky(dim=T, concentration=cfg.lkj_eta).sample().to(device, torch.float64)
        K_T = K_T @ K_T.T  # [T, T] correlation matrix

        # --- Steps 5-6: shared lengthscale, input kernel over ALL n points ---
        # (float64 throughout: the Hadamard product of two PSD matrices is
        # PSD in exact arithmetic (Schur product theorem), but only
        # borderline-PSD in float32, so we do the kernel/Cholesky math in
        # double precision and cast back down afterwards.)
        ell = self._sample_lengthscale().double()
        K_X = self._rbf(x.double(), x.double(), ell)  # [n, n]

        # --- Step 7: full ICM kernel, gathered by each point's task id ---
        K = K_T[task_id][:, task_id] * K_X
        jitter = cfg.jitter
        L = None
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(n, device=device, dtype=torch.float64))
                break
            except torch._C._LinAlgError:
                jitter *= 10
        if L is None:
            raise RuntimeError("Cholesky failed even after increasing jitter; check prior config.")

        # --- Step 8: joint sample y ~ N(0, K) ---
        z = torch.randn(n, 1, device=device, dtype=torch.float64)
        y = (L @ z).squeeze(-1).to(dtype)

        # --- Steps 9-13: resample SOURCE tasks independently w.p. p ---
        for j in range(1, T):  # j=0 is the target task and is never resampled
            if torch.rand(()).item() < cfg.p_independent:
                idx = (task_id == j).nonzero(as_tuple=True)[0]
                ell_j = self._sample_lengthscale().double()
                K_j = self._rbf(x[idx].double(), x[idx].double(), ell_j)
                jitter_j = cfg.jitter
                Lj = None
                for _ in range(6):
                    try:
                        Lj = torch.linalg.cholesky(
                            K_j + jitter_j * torch.eye(len(idx), device=device, dtype=torch.float64)
                        )
                        break
                    except torch._C._LinAlgError:
                        jitter_j *= 10
                if Lj is None:
                    raise RuntimeError("Cholesky failed for a source task even after increasing jitter.")
                zj = torch.randn(len(idx), 1, device=device, dtype=torch.float64)
                y[idx] = (Lj @ zj).squeeze(-1).to(dtype)

        y = y + cfg.obs_noise_std * torch.randn_like(y)
        return x, y, task_id

    # ------------------------------------------------------------------
    def sample_batch(self, batch_size: int):
        """Returns task-padded tensors:
            x            : [B, T, L, d]
            y            : [B, T, L]
            valid_mask   : [B, T, L]   True where a real point occupies this slot
            query_mask   : [B, T, L]   True at task-0 slots held out as the query/test point(s)
        where L = max points assigned to any single task across the batch.
        """
        cfg = self.cfg
        xs, ys, tids = [], [], []
        for _ in range(batch_size):
            x, y, task_id = self._sample_one_episode()
            xs.append(x)
            ys.append(y)
            tids.append(task_id)

        T = cfg.num_tasks
        max_len = max(int(torch.bincount(t, minlength=T).max()) for t in tids)

        x_out = torch.zeros(batch_size, T, max_len, cfg.input_dim, device=cfg.device, dtype=cfg.dtype)
        y_out = torch.zeros(batch_size, T, max_len, device=cfg.device, dtype=cfg.dtype)
        valid_mask = torch.zeros(batch_size, T, max_len, dtype=torch.bool, device=cfg.device)
        query_mask = torch.zeros(batch_size, T, max_len, dtype=torch.bool, device=cfg.device)

        for b in range(batch_size):
            x, y, task_id = xs[b], ys[b], tids[b]
            for t in range(T):
                idx = (task_id == t).nonzero(as_tuple=True)[0]
                k = len(idx)
                x_out[b, t, :k] = x[idx]
                y_out[b, t, :k] = y[idx]
                valid_mask[b, t, :k] = True
                if t == 0:
                    # hold out the last `n_query_target` of the target task's
                    # points as the query/test set to be predicted
                    n_q = min(cfg.n_query_target, k - 1)
                    if n_q > 0:
                        query_mask[b, t, k - n_q:k] = True

        return {
            "x": x_out,
            "y": y_out,
            "valid_mask": valid_mask,
            "query_mask": query_mask,
        }
