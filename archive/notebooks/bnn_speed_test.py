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