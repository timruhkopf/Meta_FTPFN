import torch
from torch.distributions.beta import Beta


class WarpingTransform:
    """
    Non-linearly warps the HP space using a Beta CDF.
    Maintains topology but stretches/compresses regions of the search space.

    This stretches and compresses the HP axes non-linearly using the Cumulative Distribution Function (CDF) of the
    Beta distribution, which is perfect for warping variables bounded between [0, 1].
    """

    def __init__(self, alpha_range=(0.5, 2.0), beta_range=(0.5, 2.0)):
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        # Sample Beta shape parameters for each HP dimension
        a = torch.empty(num_inputs).uniform_(*self.alpha_range)
        b = torch.empty(num_inputs).uniform_(*self.beta_range)
        beta_dist = Beta(a, b)

        def warped_get_marginal_curve(hyperparams):
            # Apply Beta CDF to warp the [0,1] space
            # cdf() handles the batched hyperparams seamlessly
            warped_hp = beta_dist.cdf(hyperparams)

            # Pass the warped HPs to the base task
            # The output space is untouched
            return ancestor_get_curve_target(warped_hp)

        related_task.get_marginal_curve = warped_get_marginal_curve

        # Relatedness: how far are a and b from 1.0 (which is the uniform identity distribution)
        max_dev = max(torch.max(torch.abs(a - 1.0)).item(), torch.max(torch.abs(b - 1.0)).item())
        relatedness = max(0.0, 1.0 - (max_dev / 1.0))

        return related_task, relatedness


class MobiusWarpingTransform:
    """
    Uses a Möbius transformation to smoothly warp the HP space,
    allowing local optima to shift left or right without breaking topology.


    The Beta CDF was too rigid because it couldn't shift the location of the optima without changing the underlying
    distribution family. The Möbius transformation is a classic rational function that maps $[0,1] \to [0,1]$ but
    smoothly shifts the median (and the optima) either left or right depending on a single parameter $\alpha$.
    """
    def __init__(self, log_alpha_range=(-1.5, 1.5)):
        # alpha = 1 is the identity. alpha > 1 bulges to 0, alpha < 1 bulges to 1.
        self.log_alpha_range = log_alpha_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        # Sample from log-uniform to maintain symmetry around the identity (1.0)
        log_alpha = torch.empty(num_inputs).uniform_(*self.log_alpha_range)
        alpha = torch.exp(log_alpha)

        def mobius_get_marginal_curve(hyperparams):
            # Möbius transformation: f(x) = x / (x + alpha * (1 - x))
            warped_hp = hyperparams / (hyperparams + alpha * (1.0 - hyperparams))
            return ancestor_get_curve_target(warped_hp)

        related_task.get_marginal_curve = mobius_get_marginal_curve
        return related_task, 1.0


