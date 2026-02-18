"""
This module is a refactor of the previous DatasetPrior
(Meta_FTPFN/external/ifbo_icml2024/src/PFNs4HPO/pfns4hpo/priors/hpo_lc_pfn_bopfn_broken.py)
implementation, which has been renamed to RelationPrior to better reflect its purpose.

The functionallity and code are basically unchanged, except for more cleaner organization
into dedicated classes and methods. This will allow to make modifications more easily accessible.
"""

import torch
import numpy as np
from torch import vmap

from ppfn.dataset.prior.bnn_link_fn import VectorizedParameterLinker
from ppfn.dataset.prior.bnn_prior import BNNPrior


class MultiFidelityTask:
    """Container for a single MLP-based synthetic problem."""

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.y0 = None
        self.ymax = None

        self.bnn_prior = BNNPrior(num_inputs, num_outputs)

        self.model = self.bnn_prior.sample()
        self.linker = VectorizedParameterLinker(self.bnn_prior)

    def __call__(self, hyperparams, fidelities):
        """
        Evaluate the learning curve at given fidelities for the provided hyperparameter configurations.
        # just a convenience method

        Args:
            hyperparams (torch.Tensor): Hyperparameter configurations of shape (num_configs, num_params).
            fidelities (torch.Tensor): Fidelity levels at which to evaluate the curves of shape (num_fidelities,).

        """
        curve_model = self.get_marginal_curve(hyperparams)
        vectorized_model = vmap(curve_model)
        return vectorized_model(fidelities)  # Shape: (num_configs, num_fidelities)

    def get_marginal_curve(self, hyperparams, noise=True):
        """
        Maps hyperparameter configurations to functional learning curve evaluators.


        plugging the bnn outputs (curve parameters) into a weighted mixture to collect the functional
        learning curve equations, that can be evaluated at any fidelity level t

        Returns:
            Callable: specific_curve_model(t, config_idx)
                -> clipped [0, 1] performance prediction at fidelity t.
        """
        # Get the unbounded BNN outputs [-inf, +inf]
        with torch.no_grad():
            bnn_outputs = self.model(
                hyperparams
            )  # unbounded Bnn outputs are need to be bounded to the parameter ranges (and looked up in the y ecdf)

        parametrized_curve_model = self.linker.curve_factory(
            bnn_outputs.numpy(), self.y0, self.ymax, noise=noise
        )

        return parametrized_curve_model

    def sample_task(self):
        """
        Reinitialize dataset specific random variables:

        1. Sample a new BNN surrogate model that defines the mapping from hyperparameters to curve parameters.
        2. Sample initial performance y0 and maximum performance ymax for the curves in this dataset.

        :param self: Description
        """

        # reinit the parameters of the BNN
        self.model = self.bnn_prior.sample()
        # sample the performance range
        self.sample_y0_ymax()

    def sample_y0_ymax(self):
        # sample the performance range
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        self.y0 = min(u1, u2)
        self.ymax = max(u1, u2) if np.random.uniform() < 0.25 else 1.0


if __name__ == "__main__":
    import numpy as np
    import torch
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    from plotly.subplots import make_subplots

    def plot_blended_surfaces(linspace=np.linspace(0, 1, 50), alpha=0.5):
        # 1. Setup - Generate two distinct tasks
        def get_surface_data(linspace):
            task = MultiFidelityTask(num_inputs=1, num_outputs=23)
            task.sample_task()
            curve_model = task.get_marginal_curve(
                torch.from_numpy(linspace).float().view(-1, 1), noise=False
            )

            z = np.zeros((len(linspace), len(linspace)))
            for i in range(len(linspace)):
                with torch.no_grad():
                    preds = curve_model(torch.from_numpy(linspace).float(), i)
                    z[:, i] = np.array(preds).flatten()
            return z

        z1 = get_surface_data(linspace)
        z2 = get_surface_data(linspace)

        # 2. Compute the weighted average (the blend)
        # Equation: $Z_{blend} = (1 - \alpha)Z_1 + \alpha Z_2$
        z_blend = (1 - alpha) * z1 + alpha * z2

        # 3. Create Subplots (1 row, 3 columns)
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
            subplot_titles=("Task A", f"Blended (α={alpha})", "Task B"),
        )

        # Helper to add surfaces easily
        surfaces = [z1, z_blend, z2]
        for idx, z in enumerate(surfaces, start=1):
            fig.add_trace(
                go.Surface(
                    z=z, x=linspace, y=linspace, colorscale="Viridis", showscale=False
                ),
                row=1,
                col=idx,
            )

        # 4. Layout Updates
        fig.update_layout(
            title_text="Multi-Fidelity Task Blending",
            height=600,
            width=1500,
            scene=dict(xaxis_title="HP", yaxis_title="Fid", zaxis_title="Perf"),
            scene2=dict(xaxis_title="HP", yaxis_title="Fid", zaxis_title="Perf"),
            scene3=dict(xaxis_title="HP", yaxis_title="Fid", zaxis_title="Perf"),
        )

        fig.show()

    # Run it with alpha=0.5 for an equal mix
    plot_blended_surfaces(alpha=0.5)

    def plot_linesurface_2d_interactive(linspace=np.linspace(0, 1, 50)):
        # 1. Setup grids and task
        task = MultiFidelityTask(num_inputs=1, num_outputs=23)
        task.sample_task()
        curve_model = task.get_marginal_curve(
            torch.from_numpy(linspace).float().view(-1, 1), noise=False
        )
        # Initialize Z matrix: rows = fidelity, cols = hyperparameters
        z_values = np.zeros((len(linspace), len(linspace)))
        # 2. Evaluate the model
        for i in range(len(linspace)):
            with torch.no_grad():
                preds = curve_model(torch.from_numpy(linspace).float(), i)
                # Ensure we handle both tensor and numpy returns
                z_values[:, i] = np.array(preds).flatten()
        # 3. Create the Interactive Plot
        fig = go.Figure(
            data=[
                go.Surface(
                    z=z_values,
                    x=linspace,  # Hyperparameters
                    y=linspace,  # Fidelities
                    colorscale="Viridis",
                    colorbar_title="Performance",
                )
            ]
        )
        fig.update_layout(
            title="Multi-Fidelity Performance Surface",
            scene=dict(
                xaxis_title="Hyperparameter",
                yaxis_title="Fidelity",
                zaxis_title="Predicted Performance",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()
        # plot(fig)
        plt.clf()

    plot_linesurface_2d_interactive()
