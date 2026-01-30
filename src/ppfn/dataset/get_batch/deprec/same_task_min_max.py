import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch


from ppfn.dataset.prior import  AllocationPrior, DimensionPrior, FidelityPrior, MultiFidelityTask



   

# function producing batches for PFN training
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
    """
    This variant is proving the point, that cross attention will work despite 
    the tasks being shifted in the performance range (through y0, ymax).
    
    For every batch, we sample new relation, that is the same for all tasks in the batch.
    Main difference: The sampled Trajectory (hp and budget allocation) differs across tasks, 
    and also the performance range (y0, ymax) is resampled per task.
    
    :param batch_size: Description
    :param seq_len: Description
    :param num_features: Description
    :param single_eval_pos: Description
    :param device: Description
    :param hyperparameters: Description
    :param kwargs: Description
    """
 
    num_params = DimensionPrior(num_features).sample()

    dataset_prior = MultiFidelityTask(num_params, 23)
    dataset_prior.sample_task()

    # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
    n_levels = FidelityPrior().sample()
        # print(f"n_levels: {n_levels}")

    x = []
    y = []

    # FIXME: (low-prio) efficincy: since all is the same task, we could just do one single fwd (get_marginal curve)
    #  and collect all sequences at once. No looping requried.
    for i in range(batch_size):

    
        dataset_prior.sample_y0_ymax() # This is a distortion between the tasks

        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        allocation_prior = AllocationPrior(seq_len, n_levels)

        # determine config, x, y for every curve -----
        # (1) sample "available" hyperparameter configurations, these will later be subselected and
        # determined to be either observation or query points
        # FIXME: move this into the allocation prior, since it is basically an internal representation!
        curve_configs = np.random.uniform(size=(seq_len, num_params)) 

         # (2) get the curves for these configurations
        allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)

        # we need control over y0 and ymax here, so we overwrite the behaviour:
        dataset_prior.sample_y0_ymax()
        curves = dataset_prior.get_marginal_curve(torch.from_numpy(curve_configs).float())  # get callable to evaluate (hp, t) --> y

        # (3) map the allocation to actual (x,y) values
        x_i, y_i = allocation_prior.parse_allocation_into_sequence(
            curve_configs, curves, num_params, single_eval_pos, allocation
        )

        x.append(x_i)
        y.append(y_i)

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)



if __name__ == "__main__":  

    # import matplotlib.pyplot as plt

    # create plot with multiple curves based on the get_batch 
    batch = get_batch(
        batch_size=4,
        seq_len=32,
        num_features=5,
        single_eval_pos=16,
        device="cpu",
    )
