import numpy as np
import torch


class ActiveSubspaceTransform:
    """
    Masks out a random subset of hyperparameters by overriding them with a default value (0.5).
    Simulates tasks having different 'active' dimensions.
    This simulates the scenario where two tasks share the same underlying function,
    but Task B is completely insensitive to certain hyperparameters that Task A cares about.
    """

    def __init__(self, max_drop_fraction=0.5):
        self.max_drop_fraction = max_drop_fraction

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        # Determine how many and which dimensions to drop
        num_drop = np.random.randint(1, max(2, int(num_inputs * self.max_drop_fraction) + 1))
        drop_indices = np.random.choice(num_inputs, num_drop, replace=False)

        def subspace_get_marginal_curve(hyperparams):
            # Create a copy so we don't mutate the original batch tensor
            masked_hp = hyperparams.clone()

            # Override dropped dimensions with 0.5 (center of search space)
            masked_hp[:, drop_indices] = 0.5

            return ancestor_get_curve_target(masked_hp)

        related_task.get_marginal_curve = subspace_get_marginal_curve

        relatedness = 1.0 - (num_drop / num_inputs)

        return related_task, relatedness


class RandomAnchorSubspaceTransform:
    """
    Drops dimensions by fixing them to a randomly sampled 'anchor' value
    specific to this task, rather than a naive 0.5 center.

    (Fixing Pathological Centers)Instead of hard-coding dropped dimensions to $0.5$ (which might coincidentally be a
    terrible local minimum), this version samples a random "anchor" configuration for each task. If an HP is dropped,
    it is fixed to that random anchor value. This properly simulates a user making an arbitrary, but fixed, choice for
    a hyperparameter they decided not to tune.
    """

    def __init__(self, max_drop_fraction=0.5):
        self.max_drop_fraction = max_drop_fraction

    def __call__(self, target_task):
        related_task = target_task.clone()
        ancestor_get_curve_target = target_task.get_marginal_curve
        num_inputs = target_task.num_inputs

        num_drop = np.random.randint(1, max(2, int(num_inputs * self.max_drop_fraction) + 1))
        drop_indices = np.random.choice(num_inputs, num_drop, replace=False)

        # Sample a fixed random anchor point for this specific related task
        anchor_point = torch.rand(num_inputs)

        def subspace_get_marginal_curve(hyperparams):
            masked_hp = hyperparams.clone()
            # Override dropped dimensions with the task's specific anchor point
            masked_hp[:, drop_indices] = anchor_point[drop_indices]

            return ancestor_get_curve_target(masked_hp)

        related_task.get_marginal_curve = subspace_get_marginal_curve
        return related_task, 1.0 - (num_drop / num_inputs)