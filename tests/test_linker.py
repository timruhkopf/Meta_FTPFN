import pytest
import numpy as np
from scipy.stats import norm, gamma, expon

import time
import pandas as pd

from unittest.mock import patch

from ppfn.dataset.prior.bnn_link_fn import VectorizedParameterLinker
from ppfn.dataset.prior.bnn_link_fn_old import weighted_curve_model, ECDFParameterLinker


# Assuming your classes are imported or defined above
# from your_module import ECDFParameterLinker, VectorizedParameterLinker

class MockBNNPrior:
    def __init__(self, n_samples=1000):
        # Generate some dummy sorted samples for the ECDF
        self.output_samples = np.sort(np.random.normal(0, 1, n_samples))


@pytest.fixture
def bnn_prior():
    return MockBNNPrior()


@pytest.fixture
def input_data():
    batch_size = 10
    num_params = 23  # The number of outputs expected by the logic
    # Raw BNN outputs (unbounded)
    return np.random.uniform(-3, 3, (batch_size, num_params))




def test_parameter_linker_timing_and_equivalence(bnn_prior, input_data):
    y0, ymax = 0.1, 0.9
    batch_size = input_data.shape[0]

    # 1. Initialize both linkers
    orig_linker = ECDFParameterLinker(bnn_prior)
    vect_linker = VectorizedParameterLinker(bnn_prior)

    # Ensure samples are identical for comparison
    sorted_samples = np.sort(bnn_prior.output_samples.flatten())
    ECDFParameterLinker.y_samples = sorted_samples
    vect_linker.y_samples = sorted_samples

    # --- Timing Original (Stateful/Iterative) ---
    start_orig = time.perf_counter()
    # If your original __call__ handles the batch internally via self.u_values = indices.T,
    # we call it once.
    orig_results = orig_linker(input_data, y0, ymax)
    end_orig = time.perf_counter()
    orig_duration = end_orig - start_orig

    # --- Timing Vectorized ---
    start_vect = time.perf_counter()
    vect_results = vect_results = vect_linker(input_data, y0, ymax)
    end_vect = time.perf_counter()
    vect_duration = end_vect - start_vect

    # --- Reporting ---
    speedup = orig_duration / (vect_duration + 1e-15)
    print(f"\n⏱️ Timing Results (Batch Size: {batch_size})")
    print(f"  - Original Linker:   {orig_duration:.6f}s")
    print(f"  - Vectorized Linker: {vect_duration:.6f}s")
    print(f"  - Speedup:           {speedup:.2f}x")

    # --- Equivalence Check ---
    param_names = ["Y0", "Yinf", "w", "alpha", "Xsat", "PREC", "Rpsat", "sigma"]
    for i, name in enumerate(param_names):
        assert np.allclose(orig_results[i], vect_results[i], atol=1e-7), f"Mismatch in: {name}"




@pytest.fixture
def test_setup(bnn_prior):
    batch_size = 5
    bnn_outputs = np.random.uniform(-2, 2, (batch_size, 23))
    y0, ymax = 0.1, 0.9
    x_eval = np.linspace(0.1, 10, 50)
    return bnn_prior, bnn_outputs, y0, ymax, x_eval


def test_curve_factory_equivalence(test_setup):
    prior, bnn_outputs, y0, ymax, x_eval = test_setup

    # Initialize both linkers
    # Note: Ensure both use the same sorted y_samples internally
    orig_linker = ECDFParameterLinker(prior)
    vect_linker = VectorizedParameterLinker(prior)

    # We turn off noise for the comparison to ensure the underlying math is equivalent
    # Or patch np.random.normal to return zeros
    with patch('numpy.random.normal', return_value=np.zeros_like(x_eval)):
        # 1. Create closures from both factories
        # These call the internal __call__ methods we already verified
        orig_curve_fn = orig_linker.curve_factory(bnn_outputs, y0, ymax)
        vect_curve_fn = vect_linker.curve_factory(bnn_outputs, y0, ymax)

        # 2. Test across different configuration IDs (cid)
        for cid in range(bnn_outputs.shape[0]):
            y_orig = orig_curve_fn(x_eval, cid=cid)
            y_vect = vect_curve_fn(x_eval, cid=cid)

            # Verify the curves match
            assert np.allclose(y_orig, y_vect, atol=1e-9), f"Curve mismatch at cid {cid}"


def test_noise_scaling_consistency(test_setup):
    """Verifies that the sigma parameter is being indexed correctly for noise."""
    prior, bnn_outputs, y0, ymax, x_eval = test_setup

    orig_linker = ECDFParameterLinker(prior)
    vect_linker = VectorizedParameterLinker(prior)

    # Extract parameters directly to see if sigma matches
    _, _, _, _, _, _, _, sigma_orig = orig_linker(bnn_outputs, y0, ymax)
    _, _, _, _, _, _, _, sigma_vect = vect_linker(bnn_outputs, y0, ymax)

    assert np.allclose(sigma_orig, sigma_vect), "Sigma indexing mismatch"
