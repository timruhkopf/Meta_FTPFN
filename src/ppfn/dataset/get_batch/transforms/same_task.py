from ppfn.dataset.get_batch.transforms.abstract_transform import TaskTransform
from ppfn.dataset.prior import MultiFidelityTask


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
