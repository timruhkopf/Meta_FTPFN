import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5, TabPFNBlock, _batched_scaled_dot_product_attention, \
    AlongColumnAttention
from tabpfn.architectures.kv_cache import KVCache, KVCacheEntry

from typing_extensions import override

from pfns4hpo.bar_distribution import BarDistribution
from typing import TYPE_CHECKING, Any, Literal, cast


class TabPFNV2p5Config:
    """Configuration for the single-file TabPFN v2.5 architecture."""
    num_buckets: int = -1
    max_num_classes: int = -1
    name: str = "TabPFN-v2.5"
    emsize: int = 192
    nlayers: int = 3
    nhead: int = 3
    """Number of key/value heads to use for per-column-inter-row attention."""

    features_per_group: int = 3
    """If > 1, the features will be grouped into groups of this size and the attention
    is across groups."""

    num_thinking_rows: int = 0
    """Number of "thinking rows" to prepend to each dataset, see AddThinkingRows."""

    encoder_type: Literal["linear", "mlp"] = "linear"
    """Whether to use a linear or MLP encoder in the input encoder."""

    encoder_mlp_hidden_dim: int = 128
    """Hidden dimension for the MLP embedder."""


@dataclasses.dataclass
class TabPFNV2p5Cache(KVCache):
    """Explicit KV cache for the TabPFN v2.5 architecture.

    Stores everything derived from the training data that is needed to make predictions
    for test rows without the training (or thinking) rows being present:

    Attributes:
        kv: Per-block key/value projections for the between-cells attention
            (thinking + train rows, only the first multi-query-attention head is
            stored).
        scaler_cache: Fitted standard-scaler statistics (``mean``, ``std``). Reused for
            both imputation and standardisation of test-only data.
        feature_state: The remaining train-derived feature preprocessing parameters
            that depend on the full train+test input: the constant-feature column mask
            and the feature-group normalisation parameters.
        test_y_embedding: The embedded all-NaN target column for a single test row, of
            shape ``(batch_size, emsize)``. Broadcast across all test rows to form the
            target column of the transformer input.
        train_shape: ``(batch_size, num_train_labels)``.
    """

    scaler_cache: dict[str, torch.Tensor] | None = None
    feature_state: dict[str, torch.Tensor] | None = None
    test_y_embedding: torch.Tensor | None = None
    train_shape: tuple[int, int] = (0, 0)

    @override
    def to(self, device: torch.device | str):
        """Move all cached tensors to the given device. Returns a new cache."""
        return TabPFNV2p5Cache(
            kv=self._kv_to(device),
            scaler_cache=self._dict_of_tensors_to(self.scaler_cache, device),
            feature_state=self._dict_of_tensors_to(self.feature_state, device),
            test_y_embedding=(
                None
                if self.test_y_embedding is None
                else self.test_y_embedding.to(device)
            ),
            train_shape=self.train_shape,
        )


class TabPFN_Predictor(nn.Module):
    def __init__(self, emsize, nhead, dim_feedforward):
        super().__init__()
        # A single-block PFN serves as a highly capable relational predictor
        self.block = TabPFNBlock(
            emsize=emsize,
            nhead=nhead,
            dim_feedforward=dim_feedforward
        )
        self.norm = nn.LayerNorm(emsize)

    def forward(self, z):
        # Predictor block takes the projected context (z)
        # and processes it with its own weights
        out, _ = self.block(z, single_eval_pos=0, save_peak_memory_factor=None)
        return self.norm(out)


# MONKEY PATCH ------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import tabpfn.architectures.tabpfn_v2_5 as tabpfn_module
from tabpfn.architectures.tabpfn_v2_5 import AlongColumnAttention, KVCacheEntry
# Ensure you import your dot product function:
from tabpfn.architectures.tabpfn_v2_5 import _batched_scaled_dot_product_attention


# =====================================================================
# 1. DEFINE THE PATCH FUNCTIONS
# =====================================================================

def inject_predictor_weights(attention_layer: AlongColumnAttention):
    """Dynamically adds non-shared Predictor weights to an existing layer instance."""
    # Read properties directly from the base linear layer to avoid shape guesswork
    embedding_size = attention_layer.q_projection.in_features
    out_features = attention_layer.q_projection.out_features
    device = attention_layer.q_projection.weight.device
    dtype = attention_layer.q_projection.weight.dtype

    # Attach new, completely distinct linear projections
    attention_layer.q_projection_test = nn.Linear(embedding_size, out_features, bias=False, device=device, dtype=dtype)
    attention_layer.k_projection_test = nn.Linear(embedding_size, out_features, bias=False, device=device, dtype=dtype)
    attention_layer.v_projection_test = nn.Linear(embedding_size, out_features, bias=False, device=device, dtype=dtype)
    attention_layer.out_projection_test = nn.Linear(out_features, embedding_size, bias=False, device=device,
                                                    dtype=dtype)

    # Initialize them exactly like standard PFN attention
    torch.nn.init.xavier_uniform_(attention_layer.q_projection_test.weight)
    torch.nn.init.xavier_uniform_(attention_layer.k_projection_test.weight)
    torch.nn.init.xavier_uniform_(attention_layer.v_projection_test.weight)
    torch.nn.init.zeros_(attention_layer.out_projection_test.weight)


def patched_column_forward(self, x_BcRE: torch.Tensor, single_eval_pos: int | None = None, *, cached_kv=None,
                           return_kv: bool = False):
    """The replacement forward pass that splits the computational graph."""
    Bc, R, _ = x_BcRE.shape
    N = R if single_eval_pos is None else single_eval_pos

    # Path A: Inference Mode using Cache (All rows are test queries)
    if cached_kv is not None:
        q_BcRHD_test = self.q_projection_test(x_BcRE).view(Bc, R, -1, self.head_dim)
        k_Bc1 = cached_kv.key
        v_Bc1 = cached_kv.value

        if k_Bc1.dtype != q_BcRHD_test.dtype:
            k_Bc1 = k_Bc1.to(q_BcRHD_test.dtype)
            v_Bc1 = v_Bc1.to(q_BcRHD_test.dtype)

        output_BcSHD = _batched_scaled_dot_product_attention(q_BcRHD_test, k_Bc1, v_Bc1)
        output_BcSF = output_BcSHD.reshape(Bc, R, self.head_dim * self.num_heads)
        return self.out_projection_test(output_BcSF), None

    # Path B: Standard Training Pass (Asymmetric Graph)
    x_train = x_BcRE[:, :N]

    # --- ENCODER PATH (Train rows self-attend via Base Weights) ---
    q_BcNHD_train = self.q_projection(x_train).view(Bc, N, -1, self.head_dim)
    k_BcNHD_train = self.k_projection(x_train).view(Bc, N, -1, self.head_dim)
    v_BcNHD_train = self.v_projection(x_train).view(Bc, N, -1, self.head_dim)

    out_train_BcNHD = _batched_scaled_dot_product_attention(q_BcNHD_train, k_BcNHD_train, v_BcNHD_train)
    out_train_BcNF = out_train_BcNHD.reshape(Bc, N, self.head_dim * self.num_heads)
    out_train_final = self.out_projection(out_train_BcNF)

    if single_eval_pos == R:
        return out_train_final, None

    # --- PREDICTOR PATH (Test rows cross-attend via Non-Shared Predictor Weights) ---
    x_test = x_BcRE[:, N:]
    q_BcMHD_test = self.q_projection_test(x_test).view(Bc, R - N, -1, self.head_dim)

    # Predictor projects context tokens through its isolated weights
    k_BcNHD_test = self.k_projection_test(x_train).view(Bc, N, -1, self.head_dim)
    v_BcNHD_test = self.v_projection_test(x_train).view(Bc, N, -1, self.head_dim)

    out_test_BcMHD = _batched_scaled_dot_product_attention(
        q_BcMHD_test,
        k_BcNHD_test[:, :, :1],  # Maintain Multi-Query Attention
        v_BcNHD_test[:, :, :1]
    )
    out_test_BcMF = out_test_BcMHD.reshape(Bc, R - N, self.head_dim * self.num_heads)
    out_test_final = self.out_projection_test(out_test_BcMF)

    # Merge outputs back into the row sequence dimension
    output_final = torch.cat([out_train_final, out_test_final], dim=1)

    kv_entry = None
    if return_kv:
        kv_entry = KVCacheEntry(
            key=k_BcNHD_test[:, :, :1].contiguous().detach(),
            value=v_BcNHD_test[:, :, :1].contiguous().detach(),
        )

    return output_final, kv_entry


# -------------------------------------------------------------------

class TabPFN_JEPA_with_Probe(nn.Module):
    def __init__(self, base_config, num_bars: int, weight_shared_predictor=False):
        super().__init__()
        self.ema_decay = 0.996
        self.weight_shared_predictor = weight_shared_predictor


        # FIXME: This is a global class overwrite and should not be placed inside an init, but on the module level!
        if not weight_shared_predictor:
            # Overwrite the class method directly. Every instance ever made will now run this.
            AlongColumnAttention.forward = patched_column_forward

        # 1. Backbones (Assume TabPFNV2p5 is defined as in your codebase)
        self.student = TabPFNV2p5(config=base_config, task_type="regression", n_out=1)

        if not weight_shared_predictor:
            # 2. Inject the custom Predictor weights into the Student's blocks
            for module in self.student.modules():
                if isinstance(module, AlongColumnAttention):
                    inject_predictor_weights(module)

        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters(): p.requires_grad = False

        # 3. The Predictor
        # This breaks the symmetry and prevents dimensional collapse:
        # If we share the weights in the PFN, this becomes strictly necessary
        emsize = base_config.emsize
        if weight_shared_predictor:
            """
            Because the PFN shares the same MHA for both train and test, the teacher EMA has access to these 
            weights and can trivially collapse, simply because there is no distinct predictor anymore.
            We might get around this, if we undo the parameter sharing for the MHA between train and test in the column
            """
            self.student_predictor = nn.Sequential(
                nn.Linear(emsize, emsize),
                nn.LayerNorm(emsize),
                nn.GELU(),
                nn.Linear(emsize, emsize)
            )

        # 3. The Downstream Probe (Maps JEPA latents to BarDistribution bins)
        self.probe = nn.Linear(emsize, num_bars)
        self.probe_raw = nn.Linear(emsize, num_bars)

    @torch.no_grad()
    def update_ema(self, current_step: int, total_steps: int, base_ema: float = 0.996):
        # Linearly anneal momentum from base_ema to 1.0
        current_ema = base_ema + current_step * (1.0 - base_ema) / total_steps

        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data.mul_(current_ema).add_(s_p.data, alpha=1 - current_ema)

    def forward(self, x_train, y_train, x_test, y_test):

        @torch.no_grad()
        def teacher_fwd(x_train, x_test, y_train, y_test):
            x_full = torch.cat([x_train, x_test], dim=0)
            y_full = torch.cat([y_train, y_test], dim=0)

            # Because len(y_full) == len(x_full), single_eval_pos covers all rows.
            # Every row self-attends to every other row (High Semantic Context).
            t_out = self.teacher(x_full, y_full, only_return_standard_out=False)

            # Extract the teacher's latent targets for the test rows
            y = t_out["train_embeddings"]

            y = F.layer_norm(y, (y.size(-1),))
            y_test_hat = y[x_train.shape[0]:] # just return the test section

            return y_test_hat


        def student_fwd(x_train, x_test, y_train):
            x_full = torch.cat([x_train, x_test], dim=0)

            s_out = self.student(x_full, y_train, only_return_standard_out=False)

            if self.weight_shared_predictor:
                raise NotImplementedError('predictor is not self & cross attending yet!')
                # Notice: this is basically only the output of the train-related MHA, which
                # corresponds to the "student" part of the PFN
                s_train = s_out["train_embeddings"]
                # FIXME: basically, the student needs to be a smaller TABPFN instance here!
                # If the predictor is vertically decoupled:
                y_test_hat = self.student_predictor(s_train, x_test)
            else:
                # if the predictor is vertically integrated with the student:
                y_test_hat = s_out["test_embeddings"]

            return y_test_hat

        s_y_test_hat = student_fwd(x_train, x_test, y_train)
        # --- DETACHED PROBE --- "Downstream Task": in PFN: predict the
        # Detach so the NLL objective doesn't pollute the self-supervised manifold
        # this is just checking if the predictor removes info interesting to nll
        logits = self.probe_raw(s_y_test_hat.detach())

        if self.training:
            t_y_test_hat = teacher_fwd(x_train, x_test, y_train, y_test)
            jepa_loss = F.smooth_l1_loss(s_y_test_hat, t_y_test_hat)
            return jepa_loss, logits
        else:
            return logits


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    def sample_prior_functions(batch_size: int, num_train: int, num_test: int, input_dim: int, device: str = 'cuda'):
        """Generates random sine functions: Y = sin(freq * X + phase) + noise."""
        R = num_train + num_test
        # X shape: (Rows, Batch, Features) - matching TabPFN expected input
        X = torch.randn(R, batch_size, input_dim, device=device)

        # Random frequency and phase per batch
        freq = (torch.rand(1, batch_size, input_dim, device=device) * 3) + 0.5
        phase = torch.rand(1, batch_size, input_dim, device=device) * 2 * 3.1415

        # Generate Y and sum across feature dimension for a single target
        Y_full = torch.sin(X * freq + phase).sum(dim=-1)
        Y_full += torch.randn(R, batch_size, device=device) * 0.1  # Add observation noise

        # Clamp Y to strictly fit within standard borders (e.g., -5 to 5) for the BarDistribution
        Y_full = torch.clamp(Y_full, min=-4.9, max=4.9)

        # Split into train (context) and test (queries)
        return (
            X[:num_train], Y_full[:num_train],
            X[num_train:], Y_full[num_train:]
        )


    def train_jepa_pfn(steps):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Configuration
        # (Assuming TabPFNV2p5Config and BarDistribution are initialized here)
        config = TabPFNV2p5Config(

        )
        borders = torch.linspace(-5, 5, steps=101, device=device)  # 100 bars
        bar_dist = BarDistribution(borders=borders).to(device)

        model = TabPFN_JEPA_with_Probe(base_config=config, num_bars=100).to(device)

        from torch.nn.init import trunc_normal_

        def init_jepa_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Apply to projectors
        model.probe.apply(init_jepa_weights)

        # 1. Separate parameters for Weight Decay
        def get_parameter_groups(model, weight_decay=1e-5):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # Skip the frozen Teacher

                # 1D tensors (LayerNorm weights/biases) and any bias terms get NO weight decay
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            return [
                {'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}
            ]

        # 2. Use the groups in your optimizer
        param_groups = get_parameter_groups(model, weight_decay=1e-4)

        # We can give the probe its own parameter group with a higher LR if we want,
        # but for now, let's just use the decoupled weight decay
        optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

        # 3. Add the OneCycleLR scheduler (simulates the Warmup + Cosine Decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=steps,
            pct_start=0.05,
            anneal_strategy='cos'
        )
        scaler = torch.amp.GradScaler(device)

        model.train()

        for step in range(steps):
            optimizer.zero_grad()

            # 1. Sample Data
            x_train, y_train, x_test, y_test = sample_prior_functions(
                batch_size=32, num_train=100, num_test=50, input_dim=3, device=device
            )

            # 2. Forward Pass with AMP
            with torch.amp.autocast(device):
                jepa_loss, logits = model(x_train, y_train, x_test, y_test)

            # 3. BarDistribution in Full Precision
            # Move logits to float32 BEFORE computing the NLL loss
            logits_fp32 = logits.float()

            # bar_dist returns shape (T, B) - we average it
            probe_loss = bar_dist(logits_fp32, y_test).mean()

            # 4. Backward & Step
            # Because we used .detach() inside the model, summing these losses is perfectly safe.
            # probe_loss gradients will stop at self.probe. jepa_loss gradients go to the backbone.
            total_loss = jepa_loss + probe_loss

            # 2. Scale the loss and compute gradients
            scaler.scale(total_loss).backward()

            # 3. Save the scale factor BEFORE the update
            scale_before = scaler.get_scale()

            # 4. Unscale and Step the optimizer
            # (If gradients are inf/nan, this skips the optimizer step inside)
            scaler.step(optimizer)

            # 5. Update the scale factor
            # (If the step was skipped, the scale factor is reduced here)
            scaler.update()

            # 6. The Guarded Scheduler Step
            scale_after = scaler.get_scale()

            # If the scale didn't drop, we know the optimizer successfully stepped!
            if scale_before <= scale_after:
                scheduler.step()

            # 5. EMA Update
            model.update_ema(current_step=step, total_steps=steps)

            if step % 50 == 0:
                print(f"Step {step} | JEPA Loss: {jepa_loss.item():.4f} | Probe NLL: {probe_loss.item():.4f}")


    def train_baseline_pfn(steps):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = TabPFNV2p5Config()
        borders = torch.linspace(-5, 5, steps=101, device=device)
        bar_dist = BarDistribution(borders=borders).to(device)

        # Supervised baseline: output 100 logits directly
        model = TabPFNV2p5(config=config, task_type="regression", n_out=100).to(device)

        # We can use a slightly higher LR for supervised learning
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler(device)

        model.train()

        for step in range(steps):
            optimizer.zero_grad()

            x_train, y_train, x_test, y_test = sample_prior_functions(
                batch_size=32, num_train=100, num_test=50, input_dim=3, device=device
            )

            with torch.amp.autocast(device):
                # Native TabPFN split: pass x_full, but ONLY y_train.
                # It automatically sets single_eval_pos and restricts attention.
                x_full = torch.cat([x_train, x_test], dim=0)

                # out shape: (Test_Rows, Batch, 100)
                out = model(x_full, y_train, only_return_standard_out=True)

            logits_fp32 = out.float()
            loss = bar_dist(logits_fp32, y_test).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % 50 == 0:
                print(f"Step {step} | Baseline NLL: {loss.item():.4f}")


    # train_baseline_pfn(steps=10_000)
    train_jepa_pfn(steps=30_000)
