from functools import wraps

from ppfn.dataset.get_batch.transforms.abstract_transform import TaskTransform
from ppfn.dataset.prior import MultiFidelityTask

import torch
import numpy as np


# FIXME in the original code, they sampled the fidelity levels in the get_batch function per task


class InputWarpingTransform(TaskTransform):
    """
    Applies a non-linear power transformation to the Hyperparameter (HP) space.

    Intent:
        To create a related task where the "optimum" and the shape of the response
        surface are shifted or squeezed in the input space.

    Mechanism:
        Applies x_new = x^strength to the input features before they are passed
        to the BNN. This warps the topology of the search space (e.g., making a
        quadratic bowl look asymmetrical).

    Meta-Learning Intuition:
        Helps the meta-learner handle "feature shift," where different tasks might
        have different sensitivities to the same hyperparameters.
    """

    def __init__(self, strength=0.2, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax
        self.strength = strength

    def __call__(self, target_task: MultiFidelityTask):
        # 1. Capture the original (ancestor) method
        related_task = target_task.clone()  # Create a separate instance to avoid side effects
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()
        ancestor_get_curve = target_task.get_marginal_curve
        warp_power = np.random.uniform(1 - self.strength, 1 + self.strength)

        # 2. Define the overwritten behavior
        @wraps(ancestor_get_curve)
        def warped_get_marginal_curve(hyperparams):
            # Transform inputs before passing to the ancestor
            warped_hps = torch.pow(hyperparams, warp_power)

            # Call the ancestor's original functionality
            return ancestor_get_curve(warped_hps)

        # 3. Patch the method onto the cloned instance
        related_task.get_marginal_curve = warped_get_marginal_curve
        return related_task, 1.0 - abs(warp_power - 1)  # Relatedness score based on how much warping was applied


class VectorizedInputWarpingTransform(TaskTransform):
    """
    Applies a unique non-linear power transformation to each hyperparameter dimension.

    Intent:
        To simulate tasks where the influence of specific hyperparameters is
        shifted independently, creating a non-uniform distortion of the
        response surface.

    Mechanism:
        Generates a vector 'alpha' of size (num_inputs,). For each input x_i,
        the transformation is x_i' = x_i^{alpha_i}.
    """

    def __init__(self, strength=0.5, resample_y0_ymax=True, p_drop=0.5):
        self.resample_y0_ymax = resample_y0_ymax
        self.strength = strength
        self.p_drop = p_drop

    def plot_warp_distribution(self, num_inputs=2, num_samples=1000):
        import matplotlib.pyplot as plt
        all_warps = []
        for _ in range(num_samples):
            warp_powers = np.ones(num_inputs)
            mask = np.random.choice([0, 1], size=num_inputs, p=[self.p_drop, 1 - self.p_drop])
            warp_powers += mask * np.random.uniform(-self.strength, self.strength, size=num_inputs)
            all_warps.append(warp_powers)

        all_warps = np.array(all_warps)
        plt.figure(figsize=(12, 6))
        for i in range(num_inputs):
            plt.hist(all_warps[:, i], bins=30, alpha=0.5, label=f'Input {i}')
        plt.title("Distribution of Warp Powers Across Inputs")
        plt.xlabel("Warp Power")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def __call__(self, target_task: MultiFidelityTask):
        related_task = target_task.clone()
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()

        ancestor_get_curve = target_task.get_marginal_curve

        # 1. Generate a unique warp power for EACH dimension
        # Shape: (1, num_inputs) for easy broadcasting with (N, num_inputs) tensors
        D = target_task.num_inputs
        warp_powers = np.ones(D)

        # Only warp a subset of dimensions
        mask = np.random.choice([0, 1], size=D, p=[self.p_drop, 1 - self.p_drop])
        warp_powers += mask * np.random.uniform(-self.strength, self.strength, size=D)
        warp_powers_torch = torch.tensor(warp_powers, dtype=torch.float32)

        @wraps(ancestor_get_curve)
        def warped_get_marginal_curve(hyperparams):
            # 2. Apply element-wise power transformation via broadcasting
            # hyperparams shape: (N, num_inputs)
            warped_hps = torch.pow(hyperparams, warp_powers_torch.to(hyperparams.device))

            return ancestor_get_curve(warped_hps)

        related_task.get_marginal_curve = warped_get_marginal_curve

        # Relatedness is now the average deviation across all dimensions
        avg_deviation = np.mean(np.abs(warp_powers - 1))
        return related_task, 1.0 - avg_deviation


