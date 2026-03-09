from ppfn.dataset.prior import MultiFidelityTask


class TaskTransform:
    def __call__(self, target_task: MultiFidelityTask):
        raise NotImplementedError("This is an abstract base class for task transformations.")
