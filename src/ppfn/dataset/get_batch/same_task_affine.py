import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch


from ppfn.dataset.prior import  AllocationPrior, DimensionPrior, FidelityPrior, MultiFidelityTask
 
class AffineTransformPrior:
    """
    Applies an affine transformation to the outputs of a given prior.
    This class will wrap around the NN and apply sampled affine transformations to 
    the outputs of the BNN based on a scaling factor
    """
    def __init__(self, base_prior: MultiFidelityTask, scale_range=(0.5, 1.5), shift_range=(-0.2, 0.2)):
        self.base_prior = base_prior
        self.scale_range = scale_range
        self.shift_range = shift_range

    def sample_transform(self):
        self.scale = np.random.uniform(*self.scale_range)
        self.shift = np.random.uniform(*self.shift_range)

    def apply_transform(self, y):
        # should be applied to the output of y=curve_\lambda(t)
        return self.scale * y + self.shift

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

    
        dataset_prior.sample_y0_ymax()

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
