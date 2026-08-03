import torch
from torch.utils.data import IterableDataset

from ppfn.prior.harmonics.harmnoic_mixture_prior import HarmonicMixturePrior


class InfiniteHarmonicsStream(IterableDataset):
    """
    Creates problem instances using a given Prior.
    Handles coordinate sampling, sequence padding, and dataset batching.
    """

    def __init__(self, prior: HarmonicMixturePrior, batch_size=32,
                 n_A=10, n_B=50, n_test=200, x_range=(-5, 5)):
        super().__init__()
        self.prior = prior
        self.batch_size = batch_size
        self.n_A = n_A
        self.n_B = n_B
        self.n_test = n_test
        self.min_x, self.max_x = x_range

    def _sample_x_coordinates(self):
        """Samples X coordinates for Train and Test sets."""
        B = self.batch_size

        # B's domain is elongated here to ensure a valid target for the test point projection.
        X_train_B, _ = torch.sort(torch.empty(self.n_B, B).uniform_(self.min_x, self.max_x), dim=0)
        X_train_A, _ = torch.sort(torch.empty(self.n_A, B).uniform_(self.min_x, self.max_x), dim=0)

        X_test_B = torch.empty(self.n_test, B).uniform_(self.min_x, self.max_x)
        X_test_A = torch.empty(self.n_test, B).uniform_(self.min_x, self.max_x)

        return X_train_A, X_test_A, X_train_B, X_test_B

    def _pad_task_a(self, X_train_A, Y_train_A):
        """Pads Task A to match Task B's sequence length."""
        B = self.batch_size
        pad_size = self.n_B - self.n_A

        if pad_size > 0:
            pad = torch.full((pad_size, B), float('nan'))
            X_train_A_padded = torch.cat([X_train_A, pad], dim=0)
            Y_train_A_padded = torch.cat([Y_train_A, pad], dim=0)
        else:
            X_train_A_padded = X_train_A
            Y_train_A_padded = Y_train_A

        padding_mask_A = torch.isnan(X_train_A_padded).transpose(1, 0)

        # Replace NaNs with 0.0 after mask creation to prevent gradient issues
        X_train_A_padded = torch.nan_to_num(X_train_A_padded, nan=0.0)
        Y_train_A_padded = torch.nan_to_num(Y_train_A_padded, nan=0.0)

        return X_train_A_padded, Y_train_A_padded, padding_mask_A

    def _sample_batch(self):
        """Orchestrates dataset generation for a single batch."""

        # 1. Generate underlying truth parameters via Prior
        params_A, params_B, is_unrelated = self.prior.generate_blueprints(self.batch_size)
        shifts, scale_A, warps = self.prior.generate_transformations(self.batch_size)

        # 2. Sample coordinates
        X_train_A, X_test_A, X_train_B_in_A, X_test_B_in_A = self._sample_x_coordinates()

        # --- THE TRUTH (Canonical Domain A) ---
        Y_train_A = self.prior.eval_function(X_train_A, *params_A) + (
                    torch.randn_like(X_train_A) * self.prior.noise_std)
        Y_test_A = self.prior.eval_function(X_test_A, *params_A)

        # B in A
        Y_train_B_in_A = self.prior.eval_function(X_train_B_in_A, *params_A) + (
                    torch.randn_like(X_train_B_in_A) * self.prior.noise_std)
        Y_test_B_in_A = self.prior.eval_function(X_test_B_in_A, *params_A)

        # --- THE TARGET (Domain B mapped into A's curve) ---
        X_train_B, Y_train_B = self.prior.warp_and_evaluate(X_train_B_in_A, params_B, shifts, scale_A, warps)
        X_test_B, Y_test_B = self.prior.warp_and_evaluate(X_test_B_in_A, params_B, shifts, scale_A, warps)

        # === INJECT HERE: A mapped into B's distorted coordinate/value space ===
        X_train_A_in_B, Y_train_A_in_B = self.prior.warp_and_evaluate(X_train_A, params_B, shifts, scale_A, warps)
        X_test_A_in_B, Y_test_A_in_B = self.prior.warp_and_evaluate(X_test_A, params_B, shifts, scale_A, warps)

        # 3. Pad Task A to match Task B shapes
        # X_train_A_pad, Y_train_A_pad, padding_mask_A = self._pad_task_a(X_train_A, Y_train_A)

        # 4. Compile and return dictionary
        return {
            'params': {
                'params_A': params_A,
                'params_B': params_B,
                'shifts': shifts,
                'scale_A': scale_A,
                'warps': warps,
                'is_unrelated': is_unrelated
            },
            'train': { # [T, B, D]
                'X_B': X_train_B.unsqueeze(-1), 'Y_B': Y_train_B.unsqueeze(-1),
                'X_A': X_train_A.unsqueeze(-1), 'Y_A': Y_train_A.unsqueeze(-1),
                'X_B_in_A': X_train_B_in_A.unsqueeze(-1), 'Y_B_in_A': Y_train_B_in_A.unsqueeze(-1),
                'X_A_in_B': X_train_A_in_B.unsqueeze(-1), 'Y_A_in_B': Y_train_A_in_B.unsqueeze(-1),
                # 'padding_mask_A': padding_mask_A,
            },
            'test': {
                'X_B': X_test_B.unsqueeze(-1), 'Y_B': Y_test_B.unsqueeze(-1),
                'X_A': X_test_A.unsqueeze(-1), 'Y_A': Y_test_A.unsqueeze(-1),
                'X_B_in_A': X_test_B_in_A.unsqueeze(-1), 'Y_B_in_A': Y_test_B_in_A.unsqueeze(-1),
                'X_A_in_B': X_test_A_in_B.unsqueeze(-1), 'Y_A_in_B': Y_test_A_in_B.unsqueeze(-1),

            }
        }

    def __iter__(self):
        while True:
            yield self._sample_batch()

if __name__ == '__main__':

    prior = HarmonicMixturePrior(num_components=4, noise_std=0.05, share_unrelated=0.)
    dataset = InfiniteHarmonicsStream(prior=prior, batch_size=32)

    d = next(iter(dataset))

    d