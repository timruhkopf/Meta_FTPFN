import numpy as np
import torch


class MonotonicTransform:
    """
    Applies a monotonic power transformation (y^gamma) to the outputs.
    Preserves global rank consistency while altering absolute performance values.

    This leaves the HP space entirely alone and strictly modifies the output performance using a power-law
    (Gamma correction). It guarantees that if Config 1 > Config 2 on Task A, it will
     remain Config 1 > Config 2 on Task B.
    """

    def __init__(self, gamma_range=(0.3, 3.0)):
        self.gamma_range = gamma_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve

        # gamma > 1 suppresses curves; gamma < 1 elevates curves
        gamma = np.random.uniform(*self.gamma_range)

        def monotonic_get_marginal_curve(hyperparams):
            base_curve_fn = ancestor_get_curve_target(hyperparams)

            def blended_curve(x, cid=0, noise=True):
                y_base = base_curve_fn(x, cid, noise=noise)
                # Apply power law (assuming y_base is in [0, 1])
                return np.clip(np.power(y_base, gamma), 0.0, 1.0)

            return blended_curve

        related_task.get_marginal_curve = monotonic_get_marginal_curve

        # Relatedness: distance of gamma from 1.0 in log space
        relatedness = max(0.0, 1.0 - abs(np.log(gamma)))

        return related_task, relatedness


from torch.distributions.beta import Beta


class BetaRankTransform:
    """
    Applies a Beta CDF to the output performance to guarantee rank consistency
    while drastically altering the absolute performance distribution.

    Instead of a simple $y^\gamma$ power law that has no effect at $y=0$ or $y=1$, we apply the Beta CDF to the
    output performance. This guarantees monotonic rank consistency while allowing us to fundamentally change the
    density of the performance values. An $S$-shaped Beta CDF will push mid-tier performances towards the extremes,
    while a bell-shaped Beta CDF will cluster extreme performances towards the middle.
    """

    def __init__(self, alpha_range=(0.5, 3.0), beta_range=(0.5, 3.0)):
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve

        a = torch.empty(1).uniform_(*self.alpha_range).item()
        b = torch.empty(1).uniform_(*self.beta_range).item()
        # We use scipy's beta CDF because it operates easily on numpy arrays (which y_base is)
        from scipy.stats import beta as scipy_beta
        beta_dist = scipy_beta(a, b)

        def beta_rank_get_marginal_curve(hyperparams):
            base_curve_fn = ancestor_get_curve_target(hyperparams)

            def blended_curve(x, cid=0, noise=True):
                y_base = base_curve_fn(x, cid, noise=noise)
                # Map the [0, 1] performance through the Beta CDF
                return beta_dist.cdf(y_base)

            return blended_curve

        related_task.get_marginal_curve = beta_rank_get_marginal_curve
        return related_task, 1.0
