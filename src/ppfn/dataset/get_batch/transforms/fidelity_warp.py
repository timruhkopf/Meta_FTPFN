from functools import wraps

import numpy as np

from ppfn.dataset.prior import MultiFidelityTask


class FidelityWarpTransform:
    """
    Warps the fidelity (resource/time) axis using a power transform.

    Intent:
        To simulate different convergence behaviors across tasks.

    Mechanism:
        Modifies the fidelity input 't' such that t_new = t^alpha.
        - alpha > 1: The task is "slow-start," reaching high performance late in the budget.
        - alpha < 1: The task is "fast-start," reaching a plateau very quickly.

    Meta-Learning Intuition:
        Crucial for Multi-Fidelity optimization. It trains the model to predict
        the final (t=1.0) value based on early-curve shapes that may be compressed
        or stretched.
    """

    def __init__(self, alpha=None, sample_alpha_fn=None, resample_y0_ymax=True):
        self.alpha = alpha
        # Log-normal sampling: ensures alpha=0.5 and alpha=2.0 are equally likely
        self.sample_alpha_fn = sample_alpha_fn or (lambda: np.exp(np.random.normal(0, 0.7)))
        self.resample_y0_ymax = resample_y0_ymax

    def plot_alpha_distribution(self, num_samples=1000):
        import matplotlib.pyplot as plt
        alphas = [self.sample_alpha_fn() for _ in range(num_samples)]
        plt.hist(alphas, bins=30, density=True)
        plt.title("Sampled Alpha Distribution")
        plt.xlabel("Alpha")
        plt.ylabel("Density")
        plt.show()

    def __call__(self, target_task: 'MultiFidelityTask'):
        related_task = target_task.clone()
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()
        # 1. Capture the original method
        ancestor_get_curve = target_task.get_marginal_curve

        if self.alpha is None:
            alpha = self.sample_alpha_fn()
        else:
            alpha = self.alpha

        @wraps(ancestor_get_curve)
        def warped_get_marginal_curve(hyperparams):
            # Get the original curve function (which expects fidelities t)
            base_curve_fn = ancestor_get_curve(hyperparams)

            # 2. Define a new function that warps the input 't'
            def warped_curve_fn(fidelities, cid=None, noise=True):
                # Ensure fidelities is a numpy array for the power operation
                t = np.asanyarray(fidelities)

                # Apply the warping: t' = t^alpha
                # We clip to [0, 1] just in case, though fidelities usually are.
                warped_t = np.power(t, alpha)

                # Pass the warped fidelities into the original curve
                return base_curve_fn(warped_t, cid=cid, noise=noise)

            return warped_curve_fn

        # 3. Patch the instance
        related_task.get_marginal_curve = warped_get_marginal_curve
        return related_task, 1.0 - abs(alpha - 1)  # Relatedness score based on how much warping was applied
