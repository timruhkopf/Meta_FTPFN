"""
Problem-instance sampler for the Hidden Harmonic Mixture transfer prior.

Each instance yields three context sets and one query set:

    DEPLOYMENT-VISIBLE
      train.X_A,     train.Y_A       sparse, windowed, canonical frame
      train.X_B_obs, train.Y_B_obs   dense, global, DRIFTED frame

    QUERY
      test.X_A,      test.Y_A        full canonical domain  <- what to predict

    TRAINING-ONLY (oracle / auxiliary supervision)
      train.X_B_in_A, train.Y_B_in_A   B undrifted:  (tau^-1, g^-1) of B_obs
      train.X_A_in_B, train.Y_A_in_B   A drifted:    (tau, g) of A

By construction

    (X_B_in_A, Y_B_in_A) == prior.to_A_frame(X_B_obs, Y_B_obs, T)

exactly, to float precision -- the oracle context is a deterministic bijection
of the deployment data given the drift parameters, and nothing more.

The upper bound you want to chase is a model given concat([A, B_in_A]) as
context; the deployable model gets concat([A, B_obs]) and must close the gap
by inferring the drift. Sanity floor: A alone. On trap instances (is_unrelated)
the ordering flips and the oracle is worse than A alone -- that is the point,
and those instances must be scored separately or the two effects cancel.

Padding follows the nn.MultiheadAttention convention: mask True == ignore.
Padded coordinate/value entries are zeroed after the mask is built.
"""

import torch
from torch.utils.data import IterableDataset

from ppfn.prior.harmonics_fix.harmonic_mixture_prior import HarmonicMixturePrior


class InfiniteHarmonicsStream(IterableDataset):

    def __init__(self, prior: HarmonicMixturePrior,
                 batch_size=32,
                 n_A_range=(3, 10),
                 n_B_range=(40, 80),
                 n_test=128,
                 x_range=(-5.0, 5.0),
                 a_window_range=(2.0, 5.0),
                 noisy_test_targets=False):
        """
        Args:
            n_A_range:      per-instance target context size. Keep the upper
                            bound at or below 2K so A alone stays
                            under-determined -- f_theta is linear in 2K
                            coefficients given the frequencies, so n_A >= 2K
                            makes A self-sufficient and the transfer signal
                            vanishes.
            n_B_range:      per-instance source context size. Dense.
            a_window_range: width of the sub-interval A's context is drawn
                            from. Narrow windows force extrapolation, which is
                            where B's global coverage actually pays.
            n_test:         query points, drawn over the FULL canonical domain.
            noisy_test_targets: if True, test.Y_A is a noisy draw; otherwise it
                            is the latent f_A. Both are always returned
                            (test.Y_A_clean / test.Y_A_noisy) -- this flag only
                            selects which one lands in test.Y_A.
        """
        super().__init__()
        self.prior = prior
        self.batch_size = batch_size
        self.n_A_range = n_A_range
        self.n_B_range = n_B_range
        self.n_test = n_test
        self.min_x, self.max_x = x_range
        self.a_window_range = a_window_range
        self.noisy_test_targets = noisy_test_targets

    # ------------------------------------------------------------------ utils

    def _lengths_and_mask(self, n_range, n_max, B):
        """Per-instance context length -> padding mask [n_max, B], True == pad."""
        lo, hi = n_range
        lengths = torch.randint(lo, hi + 1, (B,), device=self.prior.device)
        idx = torch.arange(n_max, device=self.prior.device).unsqueeze(1)
        return lengths, idx >= lengths.unsqueeze(0)

    @staticmethod
    def _sort_valid_first(X, pad_mask):
        """Sort along the sequence axis with padded entries pushed to the end."""
        X = torch.where(pad_mask, torch.full_like(X, float('inf')), X)
        X, order = torch.sort(X, dim=0)
        return X, order

    def _sample_A_coordinates(self, B, n_max, pad_mask):
        """Uniform inside a random sub-window of the canonical domain."""
        w_lo, w_hi = self.a_window_range
        width = torch.empty(B, device=self.prior.device).uniform_(w_lo, w_hi)
        span = (self.max_x - self.min_x) - width
        start = self.min_x + torch.rand(B, device=self.prior.device) * span

        X = start.unsqueeze(0) + torch.rand(n_max, B, device=self.prior.device) * width.unsqueeze(0)
        X, _ = self._sort_valid_first(X, pad_mask)
        X = torch.where(pad_mask, torch.zeros_like(X), X)
        return X, start, width

    def _sample_B_coordinates(self, B, n_max, pad_mask):
        """Uniform over the full canonical domain -- B is the global observer."""
        X = torch.empty(n_max, B, device=self.prior.device).uniform_(self.min_x, self.max_x)
        X, _ = self._sort_valid_first(X, pad_mask)
        X = torch.where(pad_mask, torch.zeros_like(X), X)
        return X

    # ----------------------------------------------------------------- sampler

    def _sample_batch(self):
        B = self.batch_size
        n_A_max, n_B_max = self.n_A_range[1], self.n_B_range[1]

        params_A, params_B, n_shared = self.prior.sample_blueprints(B)
        T = self.prior.sample_transform(B)
        sigma = self.prior.sample_noise(B)
        is_unrelated = (n_shared == 0)

        len_A, mask_A = self._lengths_and_mask(self.n_A_range, n_A_max, B)
        len_B, mask_B = self._lengths_and_mask(self.n_B_range, n_B_max, B)

        X_A, win_start, win_width = self._sample_A_coordinates(B, n_A_max, mask_A)
        X_B_lat = self._sample_B_coordinates(B, n_B_max, mask_B)

        # --- canonical frame: signal + noise, both tasks -----------------
        Y_A = self.prior.eval_function(X_A, params_A)
        Y_A = Y_A + torch.randn_like(Y_A) * sigma.unsqueeze(0)
        Y_A = torch.where(mask_A, torch.zeros_like(Y_A), Y_A)

        Y_B_lat = self.prior.eval_function(X_B_lat, params_B)
        Y_B_lat = Y_B_lat + torch.randn_like(Y_B_lat) * sigma.unsqueeze(0)
        Y_B_lat = torch.where(mask_B, torch.zeros_like(Y_B_lat), Y_B_lat)

        # --- push B through the drift; this is all the model ever sees ----
        X_B_obs, Y_B_obs = self.prior.to_B_frame(X_B_lat, Y_B_lat, T)
        X_B_obs = torch.where(mask_B, torch.zeros_like(X_B_obs), X_B_obs)
        Y_B_obs = torch.where(mask_B, torch.zeros_like(Y_B_obs), Y_B_obs)

        # --- A pushed the other way, for alignment supervision ------------
        X_A_in_B, Y_A_in_B = self.prior.to_B_frame(X_A, Y_A, T)
        X_A_in_B = torch.where(mask_A, torch.zeros_like(X_A_in_B), X_A_in_B)
        Y_A_in_B = torch.where(mask_A, torch.zeros_like(Y_A_in_B), Y_A_in_B)

        # --- queries: full canonical domain -------------------------------
        X_test = torch.empty(self.n_test, B, device=self.prior.device).uniform_(self.min_x, self.max_x)
        Y_test_clean = self.prior.eval_function(X_test, params_A)
        Y_test_noisy = Y_test_clean + torch.randn_like(Y_test_clean) * sigma.unsqueeze(0)
        Y_test = Y_test_noisy if self.noisy_test_targets else Y_test_clean

        # B's latent function on the query grid: what an alignment head that
        # correctly undrifted B should believe. Equals Y_test_clean iff related.
        Y_test_B_lat = self.prior.eval_function(X_test, params_B)

        def seq(t):
            return t.unsqueeze(-1)                                       # [T, B, 1]

        return {
            'params': {
                'params_A': params_A,
                'params_B': params_B,
                'transform': T,
                'sigma': sigma,
                'n_shared': n_shared,
                'is_unrelated': is_unrelated,
                'len_A': len_A, 'len_B': len_B,
                'a_window': (win_start, win_width),
            },
            'train': {
                # deployment-visible
                'X_A': seq(X_A), 'Y_A': seq(Y_A), 'mask_A': mask_A,
                'X_B_obs': seq(X_B_obs), 'Y_B_obs': seq(Y_B_obs), 'mask_B': mask_B,
                # training-only oracle
                'X_B_in_A': seq(X_B_lat), 'Y_B_in_A': seq(Y_B_lat),
                'X_A_in_B': seq(X_A_in_B), 'Y_A_in_B': seq(Y_A_in_B),
            },
            'test': {
                'X_A': seq(X_test), 'Y_A': seq(Y_test),
                'Y_A_clean': seq(Y_test_clean), 'Y_A_noisy': seq(Y_test_noisy),
                'Y_B_lat': seq(Y_test_B_lat),
            },
        }

    def __iter__(self):
        while True:
            yield self._sample_batch()
