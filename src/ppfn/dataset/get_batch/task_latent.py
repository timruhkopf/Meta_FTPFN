import torch
import torch.nn as nn
import numpy as np

from ppfn.dataset.prior import MultiFidelityTask, DimensionPrior, FidelityPrior, AllocationPrior
from ppfn.dataset.prior.bnn_prior import BNNPrior
from ppfn.model.mymodel.ft_ppfn import MyBatch


class LatentMultiFidelityTask(MultiFidelityTask):
    """Extension of MultiFidelityTask that uses a task latent vector."""

    def __init__(self, num_inputs, num_outputs, latent_dim=1):
        # We increase the BNN input dimension to accommodate the latent vector
        self.latent_dim = latent_dim
        super().__init__(num_inputs + latent_dim, num_outputs)

        # Current active latent for this task instance
        self.current_latent = torch.zeros(latent_dim)

    def sample_task(self):
        """Samples both BNN weights and a unique task latent."""
        super().sample_task()
        # Sample a random point in latent space (the 'personality' of this task)
        self.current_latent = torch.randn(self.latent_dim)

    def get_marginal_curve(self, hyperparams, latent_vec=None, noise=False):
        """
        Injects the latent vector into the hyperparameter input.
        """
        if latent_vec is None:
            latent_vec = self.current_latent

        # Expand latent to match batch size: [num_configs, latent_dim]
        z = latent_vec.unsqueeze(0).expand(hyperparams.size(0), -1)

        # Concatenate: [num_configs, num_inputs + latent_dim]
        bnn_input = torch.cat([hyperparams, z], dim=-1)

        with torch.no_grad():
            bnn_outputs = self.model(bnn_input)

        return self.linker.curve_factory(
            bnn_outputs.numpy(), self.y0, self.ymax, noise=noise
        )


@torch.no_grad()
def get_batch_latent(
        batch_size: int,
        seq_len: int,
        num_features: int,
        single_eval_pos: int,
        latent_dim: int = 1,
        num_params: int = None,
        n_levels: int = None,
        device='cpu',
        **kwargs,
):
    """
    Simplified batch generation where each task identity is purely
    defined by its location in the latent space Z.
    """
    if num_params is None:
        num_params = DimensionPrior(num_features).sample()
    if n_levels is None:
        n_levels = FidelityPrior().sample()

    # 1. Sample inputs for the whole batch
    all_configs = np.random.uniform(size=(seq_len, batch_size, num_params))

    parametrized_curves = []
    latents_used = []

    # 2. Generate tasks
    for i in range(batch_size):
        # We create a task instance.
        # Note: If you want all tasks in a batch to share 'physics' (BNN weights)
        # but have different 'personalities' (latents), move the Task init
        # outside this loop and just call sample_task() or change current_latent here.
        task_prior = LatentMultiFidelityTask(num_params, 23, latent_dim=latent_dim)
        task_prior.sample_task()

        # The latent is already sampled inside sample_task()
        z_i = task_prior.current_latent
        latents_used.append(z_i)

        # Generate the curve model using the sampled latent
        configs_torch = torch.from_numpy(all_configs[:, i, :]).float()
        curve_model = task_prior.get_marginal_curve(
            configs_torch,
            latent_vec=z_i,
            noise=True
        )
        parametrized_curves.append(curve_model)

    # 3. Allocation mapping
    x = []
    y = []
    for i in range(batch_size):
        allocation_prior = AllocationPrior(seq_len, n_levels)
        allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)

        x_i, y_i = allocation_prior.parse_allocation_into_sequence(
            all_configs[:, i, :],
            parametrized_curves[i],
            num_params,
            single_eval_pos,
            allocation
        )
        x.append(x_i)
        y.append(y_i)

    # 4. Final Batch Assembly
    y_tensor = torch.stack(y, dim=1).to(device).float()

    return MyBatch(
        x=torch.stack(x, dim=1).to(device).float(),
        y=y_tensor,
        target_y=y_tensor,
        single_eval_pos=single_eval_pos,
        # We can store the latents in 'style' if the model needs to see them
        style=torch.stack(latents_used).to(device)
    )

if __name__ == '__main__':

    get_batch_latent(
        batch_size=4,
        seq_len=1000,
        num_features=3,
        single_eval_pos=100,
        latent_dim=2,
        device='cpu'

    )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    def plot_latent_interpolation(steps=2):
        task = LatentMultiFidelityTask(num_inputs=1, num_outputs=23, latent_dim=1)
        task.sample_task()

        # Define two random points in unitcube latent space
        z_start = torch.rand(task.latent_dim)
        z_end = torch.rand(task.latent_dim)

        linspace = np.linspace(0, 1, 50)
        hp_input = torch.from_numpy(linspace).float().view(-1, 1)

        fig = make_subplots(
            rows=1, cols=steps,
            specs=[[{'type': 'surface'}] * steps],
            subplot_titles=[f"Interpolation {i / (steps - 1):.1f}" for i in range(steps)]
        )

        for i in range(steps):
            # Linear interpolation in latent space
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * z_start + alpha * z_end # FIXME: this should not be necesary
            print(z_interp)
            curve_model = task.get_marginal_curve(hp_input, latent_vec=z_interp, noise=False)

            z_values = np.zeros((len(linspace), len(linspace)))
            for j in range(len(linspace)):
                preds = curve_model(torch.from_numpy(linspace).float(), j)
                z_values[:, j] = np.array(preds).flatten()

            fig.add_trace(
                go.Surface(z=z_values, x=linspace, y=linspace, colorscale='Plasma', showscale=False),
                row=1, col=i + 1
            )

        fig.update_layout(height=500, width=1500, title_text="Latent Space Task Morphing")
        fig.show()


    # Run the visualization
    plot_latent_interpolation(steps=5)