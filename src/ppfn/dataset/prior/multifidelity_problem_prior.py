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
from copy import deepcopy

class MultiFidelityTask:
    """Container for a single MLP-based synthetic problem."""

    def __init__(self, num_inputs, num_outputs):
        self._num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.y0 = None
        self.ymax = None

        self.bnn_prior = BNNPrior(num_inputs, num_outputs)

        self.model = self.bnn_prior.sample()
        self.linker = VectorizedParameterLinker(self.bnn_prior)

    @property
    def num_inputs(self):
        return self._num_inputs

    @num_inputs.setter
    def num_inputs(self, value):
        self._num_inputs = value
        # Whenever num_inputs changes, we need to reinitialize the BNN prior and model
        self.bnn_prior = BNNPrior(self._num_inputs, self.num_outputs)
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

    def get_marginal_curve(self, hyperparams):
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
            bnn_outputs.numpy(), self.y0, self.ymax,
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

    def clone(self):
        """Creates a deep copy of the task state."""
        # We use a trick to create a new instance without re-initializing everything
        cls = self.__class__
        new_task = cls.__new__(cls)
        new_task.__dict__.update( deepcopy(self.__dict__))
        return new_task

    def plot_surface(self, ax=None, num_hp=40, num_fid=60, title="Task Surface"):
        """
        Plots the HP-Fidelity-Performance surface.
        Works even if the task is wrapped by a Transform.
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

        # 1. Define Grids (Assume we plot the first HP dimension)
        hp_axis = np.linspace(0, 1, num_hp)
        fid_axis = np.linspace(0.01, 1.0, num_fid)
        HP, FID = np.meshgrid(hp_axis, fid_axis)

        # 2. Prepare HP tensor for BNN (shape: num_hp, num_inputs)
        # We fill other dimensions with 0.5 if num_inputs > 1
        # FIXME: this is a hack to handle the case when num_inputs > 1, we should ideally have a more systematic way to visualize in that case (e.g. fix other HPs to some value or plot multiple surfaces)
        hp_configs = np.full((num_hp, self.num_inputs), 0.5)
        hp_configs[:, 0] = hp_axis
        hp_tensor = torch.from_numpy(hp_configs).float()

        # 3. Get curves (this triggers the wrapped logic if transformed!)
        # Disable noise for a clean surface plot
        curve_fn = self.get_marginal_curve(hp_tensor)

        # 4. Evaluate surface
        Z = np.zeros((num_fid, num_hp))
        for i in range(num_hp):
            Z[:, i] = curve_fn(fid_axis, cid=i)

        # 5. Visualization
        ax.plot_surface(
            HP, FID, Z,
            cmap="turbo",
            antialiased=True,
            alpha=0.8,
            rcount=num_fid, ccount=num_hp
        )

        ax.set_title(title)
        ax.set_xlabel("HP[0]")
        ax.set_ylabel("Fidelity")
        ax.set_zlabel("Perf")
        ax.set_zlim(0, 1)
        ax.view_init(elev=25, azim=-130)

        return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    task = MultiFidelityTask(num_inputs=1, num_outputs=23)
    task.sample_task()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    task.plot_surface(ax=ax, title="Sampled Multi-Fidelity Task Surface")
    plt.show()

