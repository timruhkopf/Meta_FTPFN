import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

import math
import threading

from pfns4hpo.encoders import Normalize


class MLP(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_layers,
        num_hidden,
        preactivation_noise_std,
        output_noise,
        init_std,
        sparseness,
    ):
        super(MLP, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.preactivation_noise_std = preactivation_noise_std
        self.output_noise = output_noise
        self.init_std = init_std
        self.sparseness = sparseness

        activation = "tanh"

        # (x - m) / sigma to normalize BNN inputs
        self.normalizer = Normalize(0.5, math.sqrt(1 / 12))

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(num_inputs, num_hidden)]
            + [torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers - 2)]
            + [torch.nn.Linear(num_hidden, num_outputs)]
        )

        self.reset_parameters()

        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "elu": torch.nn.ELU(),
            "identity": torch.nn.Identity(),
        }[activation]

    def reset_parameters(self, init_std=None, sparseness=None):
        init_std = init_std if init_std is not None else self.init_std
        sparseness = sparseness if sparseness is not None else self.sparseness
        for linear in self.linears:
            linear.reset_parameters()

        with torch.no_grad():
            if init_std is not None:
                for linear in self.linears:
                    linear.weight.normal_(0, init_std)
                    linear.bias.normal_(0, init_std)

            if sparseness > 0.0:
                for linear in self.linears[1:-1]:
                    linear.weight /= (1.0 - sparseness) ** (1 / 2)
                    linear.weight *= torch.bernoulli(
                        torch.ones_like(linear.weight) * (1.0 - sparseness)
                    )

    def forward(self, x):
        self.normalizer(x)
        for linear in self.linears[:-1]:
            x = linear(x)
            x = x + torch.randn_like(x) * self.preactivation_noise_std
            x = torch.tanh(x)
        x = self.linears[-1](x)
        return x + torch.randn_like(x) * self.output_noise


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
        init_std = np.random.uniform(0.089, 0.193)
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

if __name__ == '__main__':
    # Runtime comparison between the BNNPrior in loops or a vectorized super net with sparse masking and skipping.
    # while the BNNPrior will use 100% of the weights for every instance, it is highly sequential.
    # The vectorized super net will because of the masks use only a fraction of the weights and layers (!) for each
    # instance, making it extremely wasteful, if we have one max-layer instance and many smaller networks.
    # Also the number of batch items will factor greatly into the time needed for the loop

    import torch
    import torch.nn.functional as F
    import torch.utils.benchmark as benchmark


    # --- 1. Vectorized Implementation ---
    class ParametricVectorizedBNN:
        def __init__(self,
                     num_inputs: int,
                     num_outputs: int,
                     layer_bounds: tuple = (8, 16),
                     hidden_bounds: tuple = (36, 150),
                     device: str = "cpu"):
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.min_L, self.max_L = layer_bounds
            self.min_H, self.max_H = hidden_bounds
            self.device = device

        def forward(self, x: torch.Tensor):
            """
            x shape: (batch_size, T, num_inputs)
            """
            B = x.size(0)
            D_in = self.num_inputs
            D_out = self.num_outputs
            device = self.device

            # 1. Sample shapes
            num_layers = torch.randint(self.min_L, self.max_L + 1, (B,), device=device)
            num_hidden = torch.randint(self.min_H, self.max_H + 1, (B,), device=device)
            init_std = torch.empty(B, 1, 1, device=device).uniform_(0.089, 0.193)

            # 2. Width Mask (B, max_H, 1)
            indices = torch.arange(self.max_H, device=device).view(1, self.max_H)
            width_mask = (indices < num_hidden.view(B, 1)).float().unsqueeze(-1)

            # 3. Vectorized Weights
            W_in = torch.randn(B, self.max_H, D_in, device=device) * init_std
            b_in = torch.zeros(B, self.max_H, 1, device=device)

            W_hid = torch.randn(B, self.max_L - 1, self.max_H, self.max_H, device=device)
            W_hid *= init_std.unsqueeze(1)
            b_hid = torch.zeros(B, self.max_L - 1, self.max_H, 1, device=device)

            W_out = torch.randn(B, D_out, self.max_H, device=device) * init_std
            b_out = torch.zeros(B, D_out, 1, device=device)

            # 4. Forward Pass
            # x is (B, T, D_in) -> permute to (B, D_in, T) for bmm
            h = torch.bmm(W_in, x.transpose(1, 2)) + b_in
            h = F.relu(h) * width_mask

            for i in range(self.max_L - 1):
                active = (i < (num_layers - 2)).float().view(B, 1, 1)
                h_next = torch.bmm(W_hid[:, i], h) + b_hid[:, i]
                h_next = F.relu(h_next) * width_mask
                h = (active * h_next) + ((1.0 - active) * h)

            out = torch.bmm(W_out, h) + b_out

            # out is (B, D_out, T) -> transpose back to (B, T, D_out)
            return out.transpose(1, 2)


    # --- 2. Looped Implementation ---
    class ParametricLoopedBNN:
        def __init__(self,
                     num_inputs: int,
                     num_outputs: int,
                     layer_bounds: tuple = (8, 16),
                     hidden_bounds: tuple = (36, 150),
                     device: str = "cpu"):
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.min_L, self.max_L = layer_bounds
            self.min_H, self.max_H = hidden_bounds
            self.device = device

        def forward(self, x: torch.Tensor):
            """
            x shape: (batch_size, T, num_inputs)
            """
            B = x.size(0)
            results = []

            for i in range(B):
                # Sample parameters for THIS instance
                L = torch.randint(self.min_L, self.max_L + 1, (1,)).item()
                H = torch.randint(self.min_H, self.max_H + 1, (1,)).item()
                std = torch.empty(1).uniform_(0.089, 0.193).item()

                # Weights
                W_in = torch.randn(H, self.num_inputs, device=self.device) * std
                b_in = torch.zeros(H, 1, device=self.device)
                W_hid = [torch.randn(H, H, device=self.device) * std for _ in range(L - 2)]
                b_hid = [torch.zeros(H, 1, device=self.device) for _ in range(L - 2)]
                W_out = torch.randn(self.num_outputs, H, device=self.device) * std
                b_out = torch.zeros(self.num_outputs, 1, device=self.device)

                # Evaluate
                # x[i] is (T, num_inputs) -> transposed to (num_inputs, T)
                curr_x = x[i].t()

                h = F.relu(torch.matmul(W_in, curr_x) + b_in)
                for W, b in zip(W_hid, b_hid):
                    h = F.relu(torch.matmul(W, h) + b)

                out = torch.matmul(W_out, h) + b_out

                # out is (num_outputs, T) -> transpose to (T, num_outputs)
                results.append(out.t())

            # Stack to (B, T, num_outputs)
            return torch.stack(results)


    # --- Benchmark Execution ---
    def run_benchmark():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Benchmarking on: {device}")

        # Your specifications: Batch size, Sequence Length (T), Input features
        B, T, in_dim, out_dim = 1024, 1000, 10, 1
        x = torch.randn(B, T, in_dim, device=device)

        # Initialize with proper kwargs
        vec_bnn = ParametricVectorizedBNN(num_inputs=in_dim, num_outputs=out_dim, device=device)
        loop_bnn = ParametricLoopedBNN(num_inputs=in_dim, num_outputs=out_dim, device=device)

        # Warmup
        _ = vec_bnn.forward(x)
        _ = loop_bnn.forward(x)

        t0 = benchmark.Timer(
            stmt='vec_bnn.forward(x)',
            globals={'vec_bnn': vec_bnn, 'x': x},
            num_threads=1,
            label='Vectorized BNN'
        )

        t1 = benchmark.Timer(
            stmt='loop_bnn.forward(x)',
            globals={'loop_bnn': loop_bnn, 'x': x},
            num_threads=1,
            label='Looped BNN'
        )

        print("\n--- Benchmark Results ---")
        # Timeit runs multiple times. We'll use a smaller number since T=1000 makes the payload quite large.
        print(t0.timeit(100))
        print(t1.timeit(100))


    run_benchmark()