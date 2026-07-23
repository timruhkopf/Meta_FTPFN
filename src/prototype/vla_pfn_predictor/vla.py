import torch
import torch.nn as nn
import math



from tabpfn.architectures.shared.bar_distribution import BarDistribution, FullSupportBarDistribution

class VLALayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, B, A_k, A_v):
        # 1. Cross-Attention: B queries A's KV cache to find feature alignments
        B_cross, _ = self.cross_attn(query=self.norm1(B), key=A_k, value=A_v)
        B = B + B_cross

        # 2. Self-Attention: B resolves the manifold dynamically based on new mappings
        B_self, _ = self.self_attn(query=self.norm2(B), key=self.norm2(B), value=self.norm2(B))
        B = B + B_self

        # 3. Feed Forward
        B = B + self.ffn(self.norm3(B))
        return B


class VLAUnwarper(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=4, num_bars=1000):
        super().__init__()
        # Learnable task tokens can be appended prior to passing into this module
        self.layers = nn.ModuleList([
            VLALayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Project back to the number of bars required by BarDistribution
        # We assume 1D target prediction (e.g., the y-value of the token) for simplicity
        self.to_bars = nn.Linear(embed_dim, num_bars)

    def forward(self, B, A_k, A_v):
        raise NotImplementedError('Tabpfn fwd on A must be spliced in layer by layer')
        for layer in self.layers:
            B = layer(B, A_k, A_v)

        logits = self.to_bars(B)
        return logits


def generate_harmonic_data(batch_size=16, seq_A=50, seq_B=200):
    """
    Generates sparse A, dense B_in_A, and distorted B using random harmonics.
    """
    # Random frequencies and phases per batch
    w1 = torch.randn(batch_size, 1, 1) * 2 + 1
    w2 = torch.randn(batch_size, 1, 1) * 2 + 1
    phase = torch.rand(batch_size, 1, 1) * 2 * math.pi

    def harmonic(x):
        return torch.sin(w1 * x + phase) + 0.5 * torch.cos(w2 * x)

    # 1. Generate Target Task A (Sparse)
    x_A = torch.rand(batch_size, seq_A, 1) * 5
    y_A = harmonic(x_A)

    # 2. Generate Dense B_in_A (The unwarped ground truth)
    x_B_in_A = torch.rand(batch_size, seq_B, 1) * 5
    y_B_in_A = harmonic(x_B_in_A)

    # 3. Apply distortions to create B
    # Warp: x -> x ^ 1.3 (Invertible warp)
    # Scale & Shift: y -> y * 3.5 - 2.0
    x_B = torch.pow(x_B_in_A, 1.3)
    y_B = y_B_in_A * 3.5 - 2.0

    return (x_A, y_A), (x_B_in_A, y_B_in_A), (x_B, y_B)


def train_vla_step():
    # Hyperparameters
    batch_size = 8
    embed_dim = 256
    num_bars = 1000  # Number of buckets for TabPFN BarDistribution

    # 1. Instantiate Models
    vla_module = VLAUnwarper(embed_dim=embed_dim, num_bars=num_bars)
    optimizer = torch.optim.AdamW(vla_module.parameters(), lr=1e-4)

    # Mocking a basic embedding layer for the inputs (in reality, TabPFN's encoder)
    token_embedder = nn.Linear(2, embed_dim)

    # Instantiate BarDistribution (Requires Borders)
    # Using dummy borders representing the y-value range [-5, 5]
    borders = torch.linspace(-5.0, 5.0, num_bars)

    # In TabPFN codebase, BarDistribution handles the NLL calculation
    bar_dist = FullSupportBarDistribution(borders=borders)

    vla_module.train()
    optimizer.zero_grad()

    # 2. Get Synthetic Data
    (x_A, y_A), (x_B_in_A, y_B_in_A), (x_B, y_B) = generate_harmonic_data(batch_size)

    # Create tokens [batch, seq_len, 2] -> Embed to [batch, seq_len, embed_dim]
    tokens_A = token_embedder(torch.cat([x_A, y_A], dim=-1))
    tokens_B = token_embedder(torch.cat([x_B, y_B], dim=-1))

    # 3. Mock A's KV Cache (In reality, this comes from the frozen PFN on A_train)
    raise NotImplementedError('Tabpfn fwd on A')
    A_k = tokens_A  # Mocking Key
    A_v = tokens_A  # Mocking Value

    # 4. Forward Pass through VLA
    # Output is the logits for the probability distribution of B_in_A
    logits = vla_module(tokens_B, A_k, A_v)  # Shape: [batch, seq_B, num_bars]

    # 5. Compute Probabilistic NLL Loss via BarDistribution
    # We want to predict the true unwarped 'y' values of B_in_A
    target_y = y_B_in_A.squeeze(-1)  # Shape: [batch, seq_B]

    # TabPFN BarDistribution NLL Loss Calculation (Abstracted based on linked source)
    loss = bar_dist(logits, target_y)

    # Proxy loss for purely runnable demonstration without TabPFN package:
    # We map continuous targets to the nearest bar index to compute Cross Entropy
    # bucketized_target = torch.bucketize(target_y, borders).clamp(0, num_bars - 1)
    # loss = nn.functional.cross_entropy(logits.view(-1, num_bars), bucketized_target.view(-1))

    loss.backward()
    optimizer.step()

    print(f"VLA Representation Alignment Loss: {loss.item():.4f}")


# Run a training step
train_vla_step()