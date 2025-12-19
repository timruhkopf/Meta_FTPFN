"""
This module is a refactor of the previous DatasetPrior 
(Meta_FTPFN/external/ifbo_icml2024/src/PFNs4HPO/pfns4hpo/priors/hpo_lc_pfn_bopfn_broken.py)
implementation, which has been renamed to RelationPrior to better reflect its purpose.

The functionallity and code are basically unchanged, except for more cleaner organization
into dedicated classes and methods. This will allow to make modifications more easily accessible.
"""

import torch
import math
import numpy as np

from typing import Union



from ppfn.dataset.prior.base_curves import ECDFParameterLinker
from ppfn.dataset.prior.bnn_prior import BNNPrior




class MultiFidelityTask:
    """Container for a single MLP-based synthetic problem."""
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.y0 = None
        self.ymax = None

        self.bnn_prior = BNNPrior(num_inputs, num_outputs)

        self.model = self.bnn_prior.sample_mlp(num_inputs, num_outputs)
        self.linker = ECDFParameterLinker(self.bnn_prior)
       

    def __call__(self, hyperparams, fidelities):
        """
        Evaluate the learning curve at given fidelities for the provided hyperparameter configurations.
        # just a convenience method

        Args:
            hyperparams (torch.Tensor): Hyperparameter configurations of shape (num_configs, num_params).
            fidelities (torch.Tensor): Fidelity levels at which to evaluate the curves of shape (num_fidelities,).

        """
        curve_model = self.get_marginal_curve(hyperparams)
        results = []
        for t in fidelities:
            results.append(curve_model(t))
        return torch.stack(results, dim=1)  # Shape: (num_configs, num_fidelities)
    

    def get_marginal_curve(self, hyperparams, noise=True):
        """
        Maps hyperparameter configurations to functional learning curve evaluators.

 
        plugging the bnn outputs (curve parameters) into a weighted mixture to collect the functional 
        learning curve equations, taht can be evaluated at any fidelity level t

        Returns:
            Callable: specific_curve_model(t, config_idx) 
                -> clipped [0, 1] performance prediction at fidelity t.
        """        
        # Get the unbounded BNN outputs [-inf, +inf]
        with torch.no_grad():
            bnn_outputs = self.model(hyperparams)  # unbounded Bnn outputs are need to be bounded to the parameter ranges (and looked up in the y ecdf)
  
        specific_curve_model = self.linker.curve_factory(
            bnn_outputs.numpy(), self.y0, self.ymax, noise=noise
        )

       
        return specific_curve_model

                
    def sample_task(self):
        """
        Reinitialize dataset specific random variables: 

        1. Sample a new BNN surrogate model that defines the mapping from hyperparameters to curve parameters.
        2. Sample initial performance y0 and maximum performance ymax for the curves in this dataset.
        
        :param self: Description
        """
     
        # reinit the parameters of the BNN
        self.model = self.bnn_prior.sample_mlp(self.num_inputs, self.num_outputs)

        # sample the performance range
        u1 = np.random.uniform()
        u2 = np.random.uniform()
        self.y0 = min(u1, u2)
        self.ymax = max(u1, u2) if np.random.uniform() < 0.25 else 1.0


