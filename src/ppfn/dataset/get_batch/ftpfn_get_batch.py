"""
This Script is the original FT-PFN get_batch function, but adapted to the new structure.
"""


import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch

from ppfn.dataset.prior.other import DimensionPrior, FidelityPrior
from ppfn.dataset.prior.allocation_prior import AllocationPrior
from ppfn.dataset.prior.multifidelity_problem_prior import MultiFidelityTask


@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    device=default_device,
    hyperparameters=None,
    **kwargs,
):
 
    num_params = DimensionPrior(num_features).sample()

    dataset_prior = MultiFidelityTask(num_params, 23)

    x = []
    y = []

    for i in range(batch_size):
        

        # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
        n_levels = FidelityPrior().sample()
        # print(f"n_levels: {n_levels}")

        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        allocation_prior = AllocationPrior(seq_len, n_levels)
        
        # fix dataset specific random variables
        # i.e. get a new relation prior
        dataset_prior.sample_task()

        # determine config, x, y for every curve -----
        # (1) sample "available" hyperparameter configurations, these will later be subselected and
        # determined to be either observation or query points
        # FIXME: move this into the allocation prior, since it is basically an internal representation!
        curve_configs = np.random.uniform(size=(seq_len, num_params)) 

        # (2) get the curves for these configurations
        curves = dataset_prior.get_marginal_curve(torch.from_numpy(curve_configs).float())  # get callable to evaluate (hp, t) --> y

        # (3) map the allocation to actual (x,y) values
        x_i, y_i = allocation_prior.map_(curve_configs, curves, num_params, single_eval_pos)

        x.append(x_i)
        y.append(y_i)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)

if __name__ == "__main__":  

    import matplotlib.pyplot as plt

    # create plot with multiple curves based on the get_batch 
    batch = get_batch(
        batch_size=4,
        seq_len=32,
        num_features=5,
        single_eval_pos=16,
        device="cpu",
    )

    for b in range(batch.x.shape[1]):
        plt.figure()
        for i in range(batch.x.shape[0]):
            id_curve = batch.x[i, b, 0].item()
            epoch = batch.x[i, b, 1].item()
            y_val = batch.y[i, b].item()
            if id_curve == 0:
                plt.plot(epoch, y_val, "ro")  # query point
            else:
                plt.plot(epoch, y_val, "b.")  # observation point
        plt.title(f"Curve {b}")
        plt.xlabel("Fidelity (epoch)")
        plt.ylabel("Performance")
        plt.ylim(-0.1, 1.1)
        plt.show()
