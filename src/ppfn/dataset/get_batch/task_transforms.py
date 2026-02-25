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
    Returns a clone of the target task, optionally resampling its vertical scaling.

    Intent:
        To provide a baseline for meta-learning where the underlying objective
        function (the BNN response surface) remains identical.

    Mechanism:
        Clones the Task instance. If 'resample_y0_ymax' is True, it keeps the same
        relative performance curve but shifts the absolute range (e.g., shifting
        accuracy from 0.7-0.9 to 0.1-0.3).

    Meta-Learning Intuition:
        Teaches the model to be invariant to the absolute scale of the outputs
        while recognizing identical response surfaces.
    """

    def __init__(self, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax

    def __call__(self, target_task: MultiFidelityTask):
        # 1. Capture the original method (Ancestor)
        related_task = target_task.clone()  # Create a separate instance to avoid side effects
        if self.resample_y0_ymax:
            related_task.sample_y0_ymax()
        return related_task, 1.0  # Return the related task and a relatedness score of 1.0


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


class LatentInputTransform(TaskTransform):
    """
    Injects a hidden "context" variable into the BNN by expanding the input dimensionality.

    Intent:
        To create tasks that are fundamentally part of the same "family" (sharing
        the same BNN weights) but differ based on an unobserved latent parameter.

    Mechanism:
        Surgically expands the BNN input layer from D to D+1. Both the target
        and related tasks use the same BNN, but are passed different constant
        values in the (D+1)-th dimension.

    Meta-Learning Intuition:
        Simulates real-world scenarios where tasks are related by a hidden factor
        (e.g., the same model architecture trained on different but similar datasets).
    """

    def __init__(self, sigma=0.6, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax
        self.sigma = sigma  # Controls how different the latent contexts are between the target and related tasks

    def plot_latent_distribution(self, num_samples=1000):
        import matplotlib.pyplot as plt
        latents = []
        for _ in range(num_samples):
            latent_target = np.random.uniform(0, 1)
            noise = np.random.normal(0, self.sigma)
            latent_related = np.clip(latent_target + noise, 0.0, 1.0)
            latents.append((latent_target, latent_related))

        latents = np.array(latents)
        plt.scatter(latents[:, 0], latents[:, 1], alpha=0.5)
        plt.xlabel("Target Latent")
        plt.ylabel("Related Latent")
        plt.title("Latent Context Distribution")
        plt.grid(True)
        plt.show()

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
        latent_target = np.random.uniform(0.0, 1)  # Stay away from edges for cleaner steps
        noise = np.random.normal(0, self.sigma)
        latent_related = np.clip(latent_target + noise, 0.0, 1.0)

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
        return related_task, 1.0 - abs(latent_target - latent_related)  # Relatedness score based on latent distance


class OutputInterpolationTransform(TaskTransform):
    """
    Creates a new task by linearly blending the outputs of the target task and a random task.

    Intent:
        To create a "smooth transition" or "task-mashing" effect, similar to
        MixUp augmentation in computer vision.

    Mechanism:
        Samples a completely new random task (Task B) and returns a weighted
        average of the target task (Task A) and Task B: y_new = (1-α)y_A + αy_B.

    Meta-Learning Intuition:
        Forces the meta-learner to handle "noisy" or "hybrid" tasks, improving
        robustness by populating the gaps between discrete points in the task prior.
    """

    def __init__(self, alpha=None, sample_alpha_fn=None, resample_y0_ymax=True):
        self.resample_y0_ymax = resample_y0_ymax
        self.alpha = alpha
        self.sample_alpha_fn = sample_alpha_fn or (lambda: np.random.beta(1, 4))

    def plot_alpha_distribution(self, num_samples=1000):
        import matplotlib.pyplot as plt
        alphas = [self.sample_alpha_fn() for _ in range(num_samples)]
        plt.hist(alphas, bins=30, density=True)
        plt.title("Sampled Alpha Distribution")
        plt.xlabel("Alpha")
        plt.ylabel("Density")
        plt.show()

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
        return related_task, 1.0 - alpha


if __name__ == '__main__':

    from ppfn.dataset.get_batch.get_related_batch import get_batch
    # Example usage
    batch = get_batch(
        batch_size=4,
        seq_len=32,
        num_features=3,
        single_eval_pos=16,
        device="cpu",
        transform=OutputInterpolationTransform(alpha=0.3, resample_y0_ymax=False),
        share_unrelated=0.5,  # 50% of pairs will be unrelated
    )

    batch.style  # This will be half 0 half != 0 since we have a transform and a share_unrelated of 0.5.


    def debug_plot_transformation(transform, num_features=3):
        """
        Creates a comparison plot for a Target Task and its Related Task.
        """
        import matplotlib.pyplot as plt
        # 1. Initialize and Sample
        target = MultiFidelityTask(num_features, 23)
        target.sample_task()

        # 2. Clone and Transform
        related, relatedness = transform(target)

        # 3. Plotting
        fig = plt.figure(figsize=(16, 7))

        ax1 = fig.add_subplot(121, projection='3d')
        target.plot_surface(ax=ax1, title="Target Task (Original)")

        ax2 = fig.add_subplot(122, projection='3d')
        related.plot_surface(ax=ax2, title=f"Related Task ({type(transform).__name__}), Relatedness: {relatedness:.2f}")

        plt.tight_layout()
        plt.show()


    # for alpha in [0.1, 0.3, 0.7, 1.5, 2.0]:
    alpha = 0.3
    debug_plot_transformation(OutputInterpolationTransform(alpha=alpha, resample_y0_ymax=False))
    debug_plot_transformation(SameTaskTransform(resample_y0_ymax=False))
    debug_plot_transformation(FidelityWarpTransform(alpha=alpha, resample_y0_ymax=False))
    debug_plot_transformation(LatentInputTransform(resample_y0_ymax=False))
    debug_plot_transformation(InputWarpingTransform(resample_y0_ymax=False))
    debug_plot_transformation(VectorizedInputWarpingTransform(resample_y0_ymax=False, strength=0.8))

    OutputInterpolationTransform(alpha=0.5, resample_y0_ymax=True).plot_alpha_distribution()
    FidelityWarpTransform(alpha=None, resample_y0_ymax=True).plot_alpha_distribution()
    LatentInputTransform(sigma=0.3, resample_y0_ymax=True).plot_latent_distribution()
    VectorizedInputWarpingTransform(strength=0.8, resample_y0_ymax=True).plot_warp_distribution(num_inputs=4)
