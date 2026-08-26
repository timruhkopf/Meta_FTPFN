import torch
from torch.utils.data import IterableDataset


class HarmonicMixturePrior:
    """
    Formulates the mathematical problem: The "Hidden Harmonic Mixture" Prior.
    Responsible for generating base functions, transformations, and evaluations.
    """

    def __init__(self, num_components=4, noise_std=0.05, share_unrelated=0.2,
                 scale=True, shift=True, warp=True):
        self.num_components = num_components
        self.noise_std = noise_std
        self.share_unrelated = share_unrelated
        self.scale = scale
        self.shift = shift
        self.warp = warp

    @staticmethod
    def apply_spatial_warp(X, w_amp, w_freq, w_phase):
        """Applies mild non-linear spatial warping to coordinates."""
        w_amp_ext = w_amp.unsqueeze(0)
        w_freq_ext = w_freq.unsqueeze(0)
        w_phase_ext = w_phase.unsqueeze(0)
        return w_amp_ext * torch.sin(2 * torch.pi * w_freq_ext * X + w_phase_ext)

    @staticmethod
    def eval_function(X, amps, freqs, phases):
        """Evaluates the sum of sinusoids based on explicit parameters."""
        X_ext = X.unsqueeze(0)
        a_ext = amps.unsqueeze(1)
        f_ext = freqs.unsqueeze(1)
        p_ext = phases.unsqueeze(1)
        terms = a_ext * torch.sin(2 * torch.pi * f_ext * X_ext + p_ext)
        return terms.sum(dim=0)

    def generate_blueprints(self, batch_size):
        """Generates the base sinusoidal parameters for Tasks A and B."""
        K = self.num_components

        # Task A Blueprint
        amps_A = torch.empty(K, batch_size).uniform_(0.5, 2.0)
        freqs_A = torch.empty(K, batch_size).uniform_(0.1, 0.5)
        phases_A = torch.empty(K, batch_size).uniform_(0, 2 * torch.pi)

        # Task B Blueprint (Clone A, then modify unrelated traps)
        amps_B, freqs_B, phases_B = amps_A.clone(), freqs_A.clone(), phases_A.clone()

        num_unrelated = int(batch_size * self.share_unrelated)
        is_unrelated = torch.zeros(batch_size, dtype=torch.bool)

        if num_unrelated > 0:
            is_unrelated[-num_unrelated:] = True
            amps_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.5, 2.0)
            freqs_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.1, 0.5)
            phases_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0, 2 * torch.pi)

        params_A = (amps_A, freqs_A, phases_A)
        params_B = (amps_B, freqs_B, phases_B)

        return params_A, params_B, is_unrelated

    def generate_transformations(self, batch_size):
        """Generates affine and spatial warping transformations."""
        # Affine Shifts
        if self.shift:
            v_shift_A = torch.empty(batch_size).uniform_(-2.0, 2.0)
            h_shift_A = torch.empty(batch_size).uniform_(-1.5, 1.5)
        else:
            v_shift_A, h_shift_A = torch.zeros(batch_size), torch.zeros(batch_size)

        # Scale
        scale_A = torch.empty(batch_size).uniform_(0.3, 1.3) if self.scale else torch.ones(batch_size)

        # Spatial Warp
        if self.warp:
            warp_amp_A = torch.empty(batch_size).uniform_(0.0, 0.7)
            warp_freq_A = torch.empty(batch_size).uniform_(0.0, 0.2)
            warp_phase_A = torch.empty(batch_size).uniform_(0, 2 * torch.pi)
        else:
            warp_amp_A, warp_freq_A, warp_phase_A = torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(
                batch_size)

        shifts = (v_shift_A, h_shift_A)
        warps = (warp_amp_A, warp_freq_A, warp_phase_A)

        return shifts, scale_A, warps

    def warp_and_evaluate(self, X, params, shifts, scale, warps):
        v_shift, h_shift = shifts
        X_obs = X - h_shift + self.apply_spatial_warp(X, *warps)  # observed coordinate
        Y = scale * self.eval_function(X, *params) + v_shift  # evaluated in latent frame
        Y += torch.randn_like(Y) * self.noise_std
        return X_obs, Y
