import numpy as np
import torch
import time
from ppfn.dataset.prior import AllocationPrior

import pytest


@pytest.fixture
def benchmark_data():
    """Generates consistent mock data for testing."""
    np.random.seed(42)
    seq_len = 10000  # Increased for more stable timing
    n_levels = 10
    single_eval_pos = 4000

    # Random weights
    p = np.random.dirichlet(np.ones(seq_len), size=1).flatten()

    # Generate common ordering
    ids = np.arange(seq_len)
    all_levels = np.repeat(ids, n_levels)
    all_p = np.repeat(p, n_levels) / n_levels
    ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

    return {
        "seq_len": seq_len,
        "single_eval_pos": single_eval_pos,
        "ordering": ordering,
    }


def test_bincount_performance_and_logic(benchmark_data, capsys):
    """Verifies correctness and prints execution timing."""
    seq_len = benchmark_data["seq_len"]
    ordering = benchmark_data["ordering"]
    single_eval_pos = benchmark_data["single_eval_pos"]

    # --- METHOD 1: ORIGINAL LOOP ---
    start_loop = time.perf_counter()
    epochs_loop = np.zeros((seq_len,), dtype=int)
    cutoff_loop = np.zeros((seq_len,), dtype=int)

    for i in range(seq_len):
        cid = ordering[i]
        epochs_loop[cid] += 1
        if i < single_eval_pos:
            cutoff_loop[cid] += 1
    end_loop = time.perf_counter()
    loop_time = (end_loop - start_loop) * 1000

    # --- METHOD 2: VECTORIZED BINCOUNT ---
    start_vec = time.perf_counter()
    epochs_vec = np.bincount(ordering, minlength=seq_len)
    cutoff_vec = np.bincount(ordering[:single_eval_pos], minlength=seq_len)
    end_vec = time.perf_counter()
    vec_time = (end_vec - start_vec) * 1000

    # --- TIMING OUTPUT ---
    with capsys.disabled():
        print(f"\n\n{' BENCHMARK RESULTS ':=^40}")
        print(f"Original Loop:     {loop_time:.4f} ms")
        print(f"Vectorized:        {vec_time:.4f} ms")
        print(f"Speedup:           {loop_time / vec_time:.1f}x")
        print(f"{'':=^40}\n")

    # --- VALIDATION ---
    np.testing.assert_array_equal(
        epochs_loop, epochs_vec, err_msg="Total epochs mismatch"
    )
    np.testing.assert_array_equal(
        cutoff_loop, cutoff_vec, err_msg="Cutoff counts mismatch"
    )


def test_allocation_equivalence_and_bench(capsys):
    """Combines functional check and performance benchmarking."""
    seq_len, n_levels, num_params, single_eval_pos = 1024, 100, 8, 512

    def mock_curves(x, cid):
        return (x * (cid + 1)).astype(np.float64)

    prior = AllocationPrior(seq_len, n_levels)
    curve_configs = np.random.rand(seq_len, num_params).astype(np.float64)

    # 1. Shared Allocation
    np.random.seed(42)
    allocation = prior.sample_abstract_allocation(single_eval_pos)

    # 2. Timing Slow Version
    np.random.seed(123)
    t0 = time.time()
    x_slow, y_slow = prior.parse_allocation_into_sequence_slow(
        curve_configs, mock_curves, num_params, single_eval_pos, allocation
    )
    slow_time = time.time() - t0

    # 3. Timing Fast Version
    np.random.seed(123)
    t1 = time.time()
    x_fast, y_fast = prior.parse_allocation_into_sequence(
        curve_configs, mock_curves, num_params, single_eval_pos, allocation
    )
    fast_time = time.time() - t1

    # 4. Equivalence Assertions (Force float32)
    torch.testing.assert_close(x_slow.float(), x_fast.float(), rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(y_slow.float(), y_fast.float(), rtol=1e-5, atol=1e-8)

    # 5. Report Benchmark
    speedup = slow_time / fast_time
    with capsys.disabled():  # Allows output to show even during pytest
        print(f"\n🚀 Benchmark (seq_len={seq_len}):")
        print(f"   Slow: {slow_time:.4f}s | Fast: {fast_time:.4f}s")
        print(f"   Speedup: {speedup:.2f}x")
