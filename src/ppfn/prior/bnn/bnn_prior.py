import threading
from pathlib import Path

import numpy as np
import torch

from ppfn.prior.bnn.mlp import MLP


class BNNPrior(torch.nn.Module):
    output_samples = None  # Global cache for BNN output samples for ECDF fitting
    CACHE_DIR = Path(__file__).parent / "prior_ecdf"

    _lock = threading.Lock()  # Prevents race conditions during generation

    N_datasets = 10000  # Number of datasets to sample for ECDF approximation
    N_per_dataset = 1

    @classmethod
    def ensure_ecdf_loaded(cls, num_inputs, num_outputs=23):
        """
        Thread-safe method to ensure data is loaded/generated exactly once.
        """
        # Double-checked locking pattern for efficiency
        # FIXME: move to datadir!
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file = Path(cls.CACHE_DIR / "bnn_prior_ecdf.npy")
        if cls.output_samples is None:
            with cls._lock:
                if cls.output_samples is None:
                    if file.exists():
                        print(f"Loading BNN prior ECDF cache from {file}")
                        cls.output_samples = np.load(file)
                    else:
                        print(
                            f"Generating BNN prior ECDF samples for inputs of size {num_inputs}..."
                        )
                        # Note: We call a class-level generator here
                        raw_samples = cls._generate_ecdf_samples(
                            cls.N_datasets, cls.N_per_dataset, num_inputs, num_outputs
                        )
                        cls.output_samples = np.sort(raw_samples.numpy())
                        np.save(file, cls.output_samples)
                        print(f"CDF approximation saved to {file}")

    def __init__(self, num_inputs, num_outputs, nn_cls=MLP):
        super(BNNPrior, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nn_cls = nn_cls

        self.ensure_ecdf_loaded(num_inputs, num_outputs)

    # def load_ecdf_cache(self):
    #     if BNNPrior.CACHE_FILE.exists():
    #         print(f"Loading BNN prior ECDF cache from {BNNPrior.CACHE_FILE}")
    #         BNNPrior.output_samples = np.load(BNNPrior.CACHE_FILE)

    #     else:
    #         print("Generating BNN prior ECDF samples...")
    #         raw_samples = self._generate_ecdf_samples(self.N_datasets, self.N_per_dataset, self.num_outputs)
    #         sorted_samples = np.sort(raw_samples.numpy())
    #         np.save(self.CACHE_FILE, sorted_samples)
    #         print(f"CDF approximation saved to {self.CACHE_FILE}")

    def sample(self):
        return BNNPrior.sample_mlp(self.num_inputs, self.num_outputs, self.nn_cls)

    @classmethod
    def sample_mlp(cls, num_inputs, num_outputs, nn_cls=MLP):

        num_layers = np.random.randint(8, 16)
        num_hidden = np.random.randint(36, 150)

        # init_std used to be drawn independently of num_hidden (width), so
        # the effective per-layer variance-scaling factor
        # crit = init_std**2 * width (Xavier/mean-field criticality for a
        # tanh net -- crit ~ 1 roughly preserves variance layer to layer;
        # well below 1, signal collapses toward a constant a few layers in,
        # and with 8-16 layers that collapse compounds fast) ended up
        # scattered ~Uniform(0.25, 4.7) across instances by sheer chance,
        # since width and init_std were two independent draws. Measured
        # empirically (see docs/labbook/): corr(output complexity, crit) =
        # 0.68 across 300 sampled instances, vs corr(., depth) = 0.16 -- crit
        # is the real driver, not depth. About a quarter of instances landed
        # sub-critical and came out visually flat.
        #
        # Sample crit directly instead of backing into it via two independent
        # draws. Range chosen empirically against the real MLP class (not
        # assumed) -- Uniform(1.0, 5.5) keeps the old sampling's upper spread
        # (its own effective crit reached ~5.5) but floors it at the critical
        # value, which is what actually mattered: on a matched 300-sample
        # comparison against the old (width, init_std)-independent sampling,
        # median output range over the domain went 0.17 -> 0.83 and the
        # fraction of near-flat draws (range < 0.05) went 0.24 -> 0.03.
        # Uniform(0.5, 2.0) was tried first and made things WORSE (median
        # 0.07, flat fraction 0.41) -- the old sampling's median crit was
        # already ~1.6, so centering the new range at 1.25 was a regression,
        # not a fix. Re-run this comparison if you change the range again;
        # "sounds reasonable" was wrong here on the first attempt.
        crit = np.random.uniform(1.0, 5.5)
        init_std = float(np.sqrt(crit / num_hidden))

        sparseness = 0.145
        preactivation_noise_std = np.random.uniform(
            0.0003, 0.0014
        )  # TODO: check value for this!
        output_noise = np.random.uniform(0.0004, 0.0013)

        return nn_cls(
            num_inputs,
            num_outputs,
            num_layers,
            num_hidden,
            preactivation_noise_std,
            output_noise,
            init_std,
            sparseness,
        )

    # FIXME: use class attributes?
    @classmethod
    def _generate_ecdf_samples(
        cls, N_datasets, N_per_dataset, num_inputs, num_outputs, nn_cls=MLP
    ):
        """
        Generate and cache a ECDF on the BNN output over the BNN prior.

        This method generates samples from a Bayesian Neural Network (BNN) to approximate
        the Empirical Cumulative Distribution Function (ECDF) of its output distribution. The samples
        are collected across multiple datasets and stored globally for subsequent use.

        It is done once during init of training and serves for any subsequent BNN instantiation as y-quantile function.

        Args:
            N_datasets (int): Number of datasets to sample from.
            N_per_dataset (int): Number of samples to generate per dataset.
            num_outputs (int): Dimensionality of the BNN output space.

        Returns:
            None: The method stores the sorted output samples in the class variable
            `DatasetPrior.output_sorted` for later retrieval.

        Notes:
            - This method is called only once to initialize the CDF approximation cache.
            - A total of N_datasets times N_per_dataset samples are generated (default 1M).
            - Each sample is generated by:
                1. Sampling random uniform input vectors.
                2. Generating a new dataset via `self.new_dataset()`.
                3. Computing BNN output via `self._sample_curve_params()`.
            - The outputs are flattened and sorted to create an empirical CDF.
            - Progress is printed every 100 datasets.
            - The cached CDF is used for quantile estimation and prior sampling.
        """
        output = torch.zeros((N_datasets, N_per_dataset, num_outputs))
        inputs = torch.from_numpy(
            np.random.uniform(size=(N_datasets, N_per_dataset, num_inputs))
        ).to(torch.float32)

        with torch.no_grad():
            for i in range(N_datasets):
                if i % 100 == 99:
                    print(f"{i + 1}/{N_datasets}")

                mlp = cls.sample_mlp(
                    num_inputs, num_outputs, nn_cls
                )  # Sample a new BNN state
                for j in range(N_per_dataset):
                    output[i, j, :] = mlp(inputs[i, j])

        return torch.flatten(output)

    # FIXME: the Link function still has the uniform method, which is basically just looking at the quantile of the cached samples!
    # def y_quantile(self, u):
    #     """Get the quantile function value for given uniform samples u in [0,1]."""
    #         if BNNPrior.output_samples is None:
    #             self.load_ecdf_cache()

    #     n_samples = BNNPrior.output_samples.shape[0]
    #     indices = (u * (n_samples - 1)).astype(int)
    #     return BNNPrior.output_samples[indices]

    #  def uniform(self, a=0.0, b=1.0): # FIXME: during the call, we could just once apply this to all outputs and store the u_values matrix!. Then we just need to apply the respective ppfs for the respective parameters!
    # u = (b - a) * self.u_values[self.counter] + a
    # self.counter += 1
    # return u


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md. BNNPrior has no A/B
    pairing yet (that's M4's job -- see docs/milestones/M4-bnn-prior-relatedness.md);
    right now it's just a sampler over ground-truth functions, so "how the
    prior looks" means: what does the diversity of sampled draws look like?
    Several independent draws, evaluated on a dense grid, overlaid in one
    panel -- same "no query-position artifacts" principle as the harmonics
    demo (src/ppfn/prior/harmonics/stream_dataset.py), just without a second
    domain to subplot against yet."""
    import matplotlib.pyplot as plt

    # BNNPrior.sample_mlp draws from numpy's global RNG, not torch's -- seed
    # both, or this demo silently isn't reproducible (found while verifying
    # the crit fix above: two "identical" runs gave different plots).
    np.random.seed(3)
    torch.manual_seed(1)
    num_inputs, num_outputs = 1, 1
    n_draws = 8

    grid = torch.linspace(0.0, 1.0, 300).unsqueeze(-1)  # [300, 1], matches the
    # normalizer's implicit assumption of inputs ~ Uniform(0, 1) (mean 0.5,
    # std sqrt(1/12) -- see TorchStandardScaler usage in mlp.py's Normalize).

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_draws):
        prior = BNNPrior(num_inputs=num_inputs, num_outputs=num_outputs)
        mlp = prior.sample()
        mlp.eval()
        with torch.no_grad():
            y = mlp(grid)
        ax.plot(grid.squeeze(-1).numpy(), y.squeeze(-1).numpy(), alpha=0.7, lw=1.2)

    ax.set_title(f"BNNPrior — {n_draws} independent sampled ground-truth functions")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()
