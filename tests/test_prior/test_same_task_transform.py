import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from unittest.mock import patch

from ppfn.dataset.get_batch.task_transforms import SameTaskTransform
from ppfn.dataset.prior import MultiFidelityTask



class TestSameTaskTransform:

    @pytest.fixture
    def task_setup(self):
        """Initializes a standard task and the transform."""
        num_features = 4
        task = MultiFidelityTask(num_inputs=num_features, num_outputs=23)
        task.sample_task()
        transform = SameTaskTransform(resample_y0_ymax=False)
        return task, transform

    def test_identity_output_values(self, task_setup):
        """Verify that the transformed task produces identical values to the original."""
        task, transform = task_setup

        # 1. Create dummy input data
        num_configs = 5
        hp_configs = torch.rand((num_configs, task.num_inputs))
        fidelities = np.linspace(0.1, 1.0, 10)

        # 2. Get original predictions BEFORE transform
        # We use  to ensure deterministic comparison
        with patch("numpy.random.normal", return_value=np.zeros_like(hp_configs[:, 0])):
            original_curve_fn = task.get_marginal_curve(hp_configs, )
            original_results = np.array([original_curve_fn(fidelities, cid=i, noise=False) for i in range(num_configs)])

            # 3. Apply Transform
            transformed_task = transform(task)

            # 4. Get transformed predictions
            transformed_curve_fn = transformed_task.get_marginal_curve(hp_configs, )
            transformed_results = np.array([transformed_curve_fn(fidelities, cid=i, noise=False) for i in range(num_configs)])

        # 5. Assert equality
        np.testing.assert_allclose(
            original_results,
            transformed_results,
            err_msg="The SameTaskTransform altered the output values!"
        )

    def test_metadata_preservation(self, task_setup):
        """Verify that @wraps correctly preserved the function name and docstring."""
        task, transform = task_setup

        original_doc = task.get_marginal_curve.__doc__
        original_name = task.get_marginal_curve.__name__

        transformed_task = transform(task)

        assert transformed_task.get_marginal_curve.__doc__ == original_doc
        assert transformed_task.get_marginal_curve.__name__ == original_name

    def test_inplace_patching(self, task_setup):
        """Verify that the transform actually patches the instance it's given."""
        task, transform = task_setup

        # Reference to the original method object
        original_method_obj = task.get_marginal_curve

        transformed_task = transform(task)

        # Check that the method object has changed (it's now the wrapper)
        assert transformed_task.get_marginal_curve is not original_method_obj
        # Check that the returned object is indeed the same instance (in-place)
        assert transformed_task is task

    def test_multiple_calls_consistency(self, task_setup):
        """Ensure that calling the curve function multiple times remains consistent."""
        task, transform = task_setup
        transformed_task = transform(task)

        hp_configs = torch.rand((2, task.num_inputs))
        fidelities = np.array([0.5])

        curve_fn = transformed_task.get_marginal_curve(hp_configs, )

        res1 = curve_fn(fidelities, cid=0, noise=False)
        res2 = curve_fn(fidelities, cid=0, noise=False)

        assert res1 == res2, "Subsequent calls to the transformed curve returned different results."