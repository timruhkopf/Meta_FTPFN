import pytest
import torch
import numpy as np
from copy import deepcopy

from ppfn.dataset.get_batch.task_transforms import FidelityWarpTransform
from ppfn.dataset.prior import MultiFidelityTask


class TestFidelityWarpTransform:

    @pytest.fixture
    def task_setup(self):
        task = MultiFidelityTask(num_inputs=2, num_outputs=23)
        task.sample_task()
        # Set alpha to 2.0: t -> t^2 (delaying progress)
        transform = FidelityWarpTransform(alpha=2.0)
        return task, transform

    def test_warp_delay_effect(self, task_setup):
        """
        Verify that alpha > 1 results in lower or equal performance at
        mid-fidelity because the 'clock' is slowed down.
        """
        task, transform = task_setup
        hp = torch.zeros((1, task.num_inputs))
        t_mid = np.array([0.5])

        # 1. Get original value at 0.5
        orig_fn = task.get_marginal_curve(hp, )
        val_orig = orig_fn(t_mid, cid=0, noise=False)

        # 2. Get original value at 0.25 (which is 0.5^2)
        val_at_squared_t = orig_fn(t_mid ** 2, cid=0, noise=False)

        # 3. Transform a CLONE and check value at 0.5
        cloned_task = task.clone()
        warped_task = transform(cloned_task)
        warped_fn = warped_task.get_marginal_curve(hp, )
        val_warped = warped_fn(t_mid, cid=0, noise=False)

        # The warped value at 0.5 should exactly match the original value at 0.25
        np.testing.assert_allclose(val_warped, val_at_squared_t, err_msg="Warping math is incorrect.")

        # Generally, for learning curves, performance at 0.5 should be > performance at 0.25
        assert val_warped <= val_orig, "Warped task should show later progress for alpha > 1"

    def test_endpoint_invariance(self, task_setup):
        """Fidelity 0 and 1 should remain unchanged regardless of alpha."""
        task, transform = task_setup
        hp = torch.zeros((1, task.num_inputs))
        endpoints = np.array([0.0, 1.0])

        orig_fn = task.get_marginal_curve(hp, )
        val_orig = orig_fn(endpoints, cid=0, noise=False)

        warped_task = transform(task.clone())
        warped_fn = warped_task.get_marginal_curve(hp, )
        val_warped = warped_fn(endpoints, cid=0, noise=False)

        np.testing.assert_allclose(val_warped, val_orig, atol=1e-5)

    def test_cloning_isolation(self, task_setup):
        """Verify that transforming a clone does NOT affect the original task."""
        task, transform = task_setup

        original_method = task.get_marginal_curve

        cloned_task = task.clone()
        transform(cloned_task)

        # The original task's method should still be the original class method
        assert task.get_marginal_curve == original_method
        # The cloned task's method should be the wrapped version
        assert cloned_task.get_marginal_curve != original_method
