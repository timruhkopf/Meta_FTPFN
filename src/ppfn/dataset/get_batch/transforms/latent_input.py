import numpy as np
import torch

from ppfn.dataset.get_batch.transforms.abstract_transform import TaskTransform
from ppfn.dataset.prior import MultiFidelityTask
from ppfn.dataset.prior.bnn_link_fn import VectorizedParameterLinker
from ppfn.dataset.prior.bnn_prior import BNNPrior


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
