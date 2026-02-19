from functools import wraps

from ppfn.dataset.prior import MultiFidelityTask
from ppfn.dataset.prior.bnn_prior import BNNPrior
from ppfn.dataset.prior.bnn_link_fn import VectorizedParameterLinker

import torch
import numpy as np


# FIXME in the original code, they sampled the fidelity levels in the get_batch function per task

class TaskTransform:
    def __call__(self, target_task: MultiFidelityTask):
        raise NotImplementedError("This is an abstract base class for task transformations.")


class SameTaskTransform(TaskTransform):
    """
    Returns the task exactly as it is.
    Useful as a control group or baseline for meta-learning.
    """
    def __init__(self, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax

    def __call__(self, target_task: MultiFidelityTask):
        # 1. Capture the original method (Ancestor)
        related_task = target_task.clone()  # Create a separate instance to avoid side effects
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()
        return related_task


class InputWarpingTransform(TaskTransform):
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
        return related_task


class FidelityWarpTransform:
    """
    Warps the fidelity axis (time/resource) using a power transform.
    t_new = t^alpha

    alpha > 1: Task reaches high performance LATER (stretched).
    alpha < 1: Task reaches high performance EARLIER (compressed).
    """

    def __init__(self, alpha=None, sample_alpha_fn=None, resample_y0_ymax=True):
        self.alpha = alpha
        self.sample_alpha_fn = sample_alpha_fn or (lambda: np.random.uniform(0, 1))
        self.resample_y0_ymax = resample_y0_ymax

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
            def warped_curve_fn(fidelities, cid=None, noise=True ):
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
        return related_task


class LatentInputTransform(TaskTransform):
    def __init__(self, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax

    def __call__(self, target_task: MultiFidelityTask):
        # 1. We want a BNN that is 1 dimension wider than the HPs
        latent_width = target_task.num_inputs + 1

        # 2. SURGICALLY update the BNN without touching target_task.num_inputs
        # This prevents the setter from resetting the model later
        target_task.bnn_prior = BNNPrior(latent_width, target_task.num_outputs)
        target_task.model = target_task.bnn_prior.sample()
        target_task.linker = VectorizedParameterLinker(target_task.bnn_prior)
        target_task.sample_y0_ymax()

        # 3. Clone for the related task (inherits the wide BNN)
        related_task = target_task.clone()
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()

        # 4. Latents
        latent_target = np.random.uniform(0, 1)
        latent_related = np.random.uniform(0, 1)

        def inject_latent(task_instance, latent_value):
            # Capture the wide model specifically
            wide_model = task_instance.model

            def wrapped_get_marginal_curve(hyperparams):
                num_configs = hyperparams.shape[0]
                context_col = torch.full(
                    (num_configs, 1), latent_value,
                    device=hyperparams.device, dtype=hyperparams.dtype
                )

                # Input is now (N, HP_dim + 1)
                extended_hps = torch.cat([hyperparams, context_col], dim=-1)

                with torch.no_grad():
                    # This call will now succeed because wide_model is size HP_dim + 1
                    bnn_outputs = wide_model(extended_hps)

                return task_instance.linker.curve_factory(
                    bnn_outputs.numpy(), task_instance.y0, task_instance.ymax
                )

            task_instance.get_marginal_curve = wrapped_get_marginal_curve

        # 5. Apply context
        inject_latent(target_task, latent_target)
        inject_latent(related_task, latent_related)

        # target_task.num_inputs is still the original value!
        # Plotting will work, and the BNN is safely wide.
        return related_task


class OutputInterpolationTransform(TaskTransform):
    def __init__(self, alpha=None, sample_alpha_fn=None, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax
        self.alpha = alpha
        self.sample_alpha_fn = sample_alpha_fn or (lambda: np.random.uniform(0, 1))

    def __call__(self, target_task: MultiFidelityTask):
        # 1. related_task is our "Task B"
        related_task = target_task.clone()
        y0, ymax = target_task.y0, target_task.ymax
        related_task.sample_task()

        if not self.resample_y0_ymax:
            related_task.y0 = y0
            related_task.ymax = ymax


        # 2. CAPTURE the original methods before overwriting
        # This is the key to preventing recursion!
        ancestor_get_curve_target = target_task.get_marginal_curve
        ancestor_get_curve_related = related_task.get_marginal_curve

        if self.alpha is None:
            alpha = self.sample_alpha_fn()
        else:
            alpha = self.alpha

        # 3. Overwrite with an interpolating wrapper
        def interpolated_get_marginal_curve(hyperparams):
            # Call the SAVED ancestor methods, NOT the instance methods
            curve_fn_a = ancestor_get_curve_target(hyperparams)
            curve_fn_b = ancestor_get_curve_related(hyperparams)

            def blended_curve(x, cid=0, noise=True):
                y_a = curve_fn_a(x, cid, noise=noise)
                y_b = curve_fn_b(x, cid, noise=noise)
                return (1 - alpha) * y_a + alpha * y_b

            return blended_curve

        # Patch the related_task instance
        related_task.get_marginal_curve = interpolated_get_marginal_curve
        return related_task


if __name__ == '__main__':

    def debug_plot_transformation(transform, num_features=3):
        """
        Creates a comparison plot for a Target Task and its Related Task.
        """
        import matplotlib.pyplot as plt
        # 1. Initialize and Sample
        target = MultiFidelityTask(num_features, 23)
        target.sample_task()
        target.sample_y0_ymax()

        # 2. Clone and Transform
        if transform:
            related = transform(target)
        else:
            related = target.clone()
            related.sample_task()

        # 3. Plotting
        fig = plt.figure(figsize=(16, 7))

        ax1 = fig.add_subplot(121, projection='3d')
        target.plot_surface(ax=ax1, title="Target Task (Original)")

        ax2 = fig.add_subplot(122, projection='3d')
        related.plot_surface(ax=ax2, title=f"Related Task ({type(transform).__name__})")

        plt.tight_layout()
        plt.show()


    # for alpha in [0.1, 0.3, 0.7, 1.5, 2.0]:
    alpha = 0.5
    debug_plot_transformation(FidelityWarpTransform(alpha=alpha, resample_y0_ymax=False))
    debug_plot_transformation(LatentInputTransform(resample_y0_ymax=False))
    debug_plot_transformation(InputWarpingTransform(resample_y0_ymax=False))
    debug_plot_transformation(OutputInterpolationTransform(alpha=alpha, resample_y0_ymax=True))
    debug_plot_transformation(SameTaskTransform(resample_y0_ymax=False))
