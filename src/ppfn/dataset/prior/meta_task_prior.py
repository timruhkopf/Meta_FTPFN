import numpy as np
import torch
import torch.nn as nn

from ppfn.dataset.prior.multifidelity_problem_prior import MultiFidelityTask


class MetaTaskPrior(MultiFidelityTask):
    """A prior over meta-tasks, i.e., over multifidelity problems."""

    def __init__(self, n_tasks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_tasks = n_tasks
        self.beta_params = (2.0, 5.0)  # parameters for relatedness sampling
        self.tasks = []
        self.sample_meta_tasks()

    def sample_meta_tasks(self):
        self.tasks = [
            self.sample_task() 
            for _ in range(self.n_tasks)
            ]
        self.relatedness = [
            np.random.beta(size=self.n_tasks, *self.beta_params) 
            for _ in range(self.n_tasks)
            ]

    def target_task_fwd(self, hyperparams):
        """Forward pass through the target task."""
        task = self.tasks[0]
        return task.evaluate_unbounded(hyperparams)
    
    def related_task_fwd(self, task_idx, relatedness, hyperparams):
        """Forward pass through a related task."""
        task = self.tasks[task_idx]
        related_output = task.evaluate_unbounded(hyperparams)
    
        # TODO consider, that relatedness should be 0,1 bounded. We can define a prior over it, that is a parametrized
        # beta distributed or similar.
        output = related_output * relatedness + self.target_task_fwd(hyperparams) * (1-relatedness)

        # TODO rescale? otherwise we may get out of bounds! You will need to consider y0, ymax here!!!
        # (i.e. y0 >= 0, ymax <= 1 and y0 < ymax)
        return output
    
    def crossed_task_fwd(self, hyperparams):
        """
        The idea is that during training, we want to have paired examples; always one target task and one related task.
        
        So to simplify batching, any task can be considered a target task once and a related task another time.
        We just need to roll and sample a relatedness for each task.
        """

        batch_size = hyperparams.shape[0]
        outputs = []
        for i in range(self.n_tasks):
            relatedness = np.random.beta(*self.beta_params)
            rolled_idx = (i + 1) % self.n_tasks
            # FIXME
            raise NotImplementedError("Fix crossed_task_fwd batching logic!")
        return torch.cat(outputs, dim=0)  # Shape: (n_tasks, batch_size, output_dim)

        

        


