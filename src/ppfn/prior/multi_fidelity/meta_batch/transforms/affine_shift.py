import torch
import numpy as np
from copy import deepcopy


class AffineShiftTransform:  # Assuming inherits from TaskTransform
    """
    Applies an affine transformation to both the HP input space and performance output space.
    f_B(x) = alpha * f_A(W*x + b) + c

    This applies a linear transformation to both the Hyperparameter space
     ($W\mathbf{x} + \mathbf{b}$) and the Output performance space ($\alpha y + c$).
    """

    def __init__(self, hp_scale_range=(0.8, 1.2), hp_shift_range=(-0.1, 0.1),
                 out_scale_range=(0.8, 1.2), out_shift_range=(-0.1, 0.1)):
        self.hp_scale_range = hp_scale_range
        self.hp_shift_range = hp_shift_range
        self.out_scale_range = out_scale_range
        self.out_shift_range = out_shift_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        # DO NOT call related_task.sample_task()! We use Task A's underlying BNN.

        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        # 1. Sample transformation parameters
        # W acts as the diagonal of our scaling matrix
        W = torch.empty(num_inputs).uniform_(*self.hp_scale_range)
        b = torch.empty(num_inputs).uniform_(*self.hp_shift_range)
        alpha = np.random.uniform(*self.out_scale_range)
        c = np.random.uniform(*self.out_shift_range)

        # 2. Overwrite with transformed wrapper
        def affine_get_marginal_curve(hyperparams):
            # Apply HP transformation: Wx + b
            # Broadcast W and b across the batch of configs
            shifted_hp = hyperparams * W + b

            # HP spaces are usually [0, 1] bounded
            shifted_hp = torch.clamp(shifted_hp, 0.0, 1.0)

            # Get the base curves evaluated at the shifted HPs
            base_curve_fn = ancestor_get_curve_target(shifted_hp)

            def blended_curve(x, cid=0, noise=True):
                y_base = base_curve_fn(x, cid, noise=noise)
                # Apply output transformation: alpha * y + c
                y_new = alpha * y_base + c
                # Ensure we don't violate performance bounds
                return np.clip(y_new, 0.0, 1.0)

            return blended_curve

        related_task.get_marginal_curve = affine_get_marginal_curve

        # Calculate a rough relatedness score (1.0 = identical, 0.0 = highly shifted)
        # Using the max deviation from the identity transformation
        w_dev = torch.max(torch.abs(W - 1.0)).item() / 0.2
        a_dev = abs(alpha - 1.0) / 0.2
        relatedness = max(0.0, 1.0 - max(w_dev, a_dev))

        return related_task, relatedness


import torch
import numpy as np


class LogitAffineShiftTransform:
    """
    Applies an affine transformation in logit space to avoid clamping dead-zones.
    To fix the dead gradients and artificial plateaus caused by torch.clamp, we move the affine transformation into
     logit space. By projecting the $[0, 1]$ bounded hyperparameters to $[-\infty, \infty]$, scaling/shifting them,
     and projecting back via a sigmoid, we ensure smooth, non-zero gradients everywhere while strictly respecting
     the bounds.
    """

    def __init__(self, hp_scale_range=(0.5, 2.0), hp_shift_range=(-1.0, 1.0),
                 out_scale_range=(0.8, 1.2), out_shift_range=(-0.1, 0.1)):
        self.hp_scale_range = hp_scale_range
        self.hp_shift_range = hp_shift_range
        self.out_scale_range = out_scale_range
        self.out_shift_range = out_shift_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        W = torch.empty(num_inputs).uniform_(*self.hp_scale_range)
        b = torch.empty(num_inputs).uniform_(*self.hp_shift_range)
        alpha = np.random.uniform(*self.out_scale_range)
        c = np.random.uniform(*self.out_shift_range)

        def logit_affine_get_marginal_curve(hyperparams):
            eps = 1e-6
            # 1. Safe project to logit space
            x_safe = torch.clamp(hyperparams, eps, 1.0 - eps)
            x_logit = torch.log(x_safe / (1.0 - x_safe))

            # 2. Apply affine shift
            x_shifted = x_logit * W + b

            # 3. Project back to [0, 1]
            x_new = torch.sigmoid(x_shifted)

            base_curve_fn = ancestor_get_curve_target(x_new)

            def blended_curve(x, cid=0, noise=True):
                y_base = base_curve_fn(x, cid, noise=noise)
                # Softplus can be used here instead of clip for outputs if desired,
                # but standard clipping on the final performance metric is usually acceptable.
                return np.clip(alpha * y_base + c, 0.0, 1.0)

            return blended_curve

        related_task.get_marginal_curve = logit_affine_get_marginal_curve
        return related_task, 1.0  # Implement your relatedness logic here