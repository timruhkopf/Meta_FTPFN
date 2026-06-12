import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5, TabPFNBlock
from tabpfn.architectures.kv_cache import KVCache

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
    def to(self, device: torch.device | str) :
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

class TabPFN_JEPA_with_Probe(nn.Module):
    def __init__(self, base_config, num_bars: int, projector_dim: int = 512):
        super().__init__()
        self.ema_decay = 0.996

        # 1. Backbones (Assume TabPFNV2p5 is defined as in your codebase)
        self.student = TabPFNV2p5(config=base_config, task_type="regression", n_out=1)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters(): p.requires_grad = False

        # 2. JEPA Projectors
        emsize = base_config.emsize
        self.student_projector = nn.Sequential(
            nn.Linear(emsize, projector_dim), nn.LayerNorm(projector_dim), nn.GELU(),
            nn.Linear(projector_dim, projector_dim)
        )
        self.teacher_projector = copy.deepcopy(self.student_projector)
        for p in self.teacher_projector.parameters(): p.requires_grad = False

        """
        Because the PFN shares the same MHA for both train and test, the teacher EMA has access to these 
        weights and can trivially collapse, simply because there is no distinct predictor anymore 
        """
        # 3. The Predictor (STUDENT ONLY - NO EMA)
        # This breaks the symmetry and prevents dimensional collapse
        self.student_predictor =         nn.Sequential(
            nn.Linear(projector_dim, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.GELU(),
            nn.Linear(projector_dim, projector_dim)
        )

        # 3. The Downstream Probe (Maps JEPA latents to BarDistribution bins)
        self.probe = nn.Linear(projector_dim, num_bars)
        self.probe_raw = nn.Linear(base_config.emsize, num_bars)

    @torch.no_grad()
    def update_ema(self, current_step: int, total_steps: int, base_ema: float = 0.996):
        # Linearly anneal momentum from base_ema to 1.0
        current_ema = base_ema + current_step * (1.0 - base_ema) / total_steps

        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data.mul_(current_ema).add_(s_p.data, alpha=1 - current_ema)
        for s_p, t_p in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            t_p.data.mul_(current_ema).add_(s_p.data, alpha=1 - current_ema)

    def forward(self, x_train, y_train, x_test, y_test):
        # --- TEACHER PATH (Global Context) ---
        with torch.no_grad():
            x_full = torch.cat([x_train, x_test], dim=0)
            y_full = torch.cat([y_train, y_test], dim=0)

            # Because len(y_full) == len(x_full), single_eval_pos covers all rows.
            # Every row self-attends to every other row (High Semantic Context).
            t_out = self.teacher(x_full, y_full, only_return_standard_out=False)

            # Extract the teacher's latent targets for the test rows
            t_latents = t_out["train_embeddings"][x_train.shape[0]:]
            s_y = self.teacher_projector(t_latents)
            s_y = F.layer_norm(s_y, (s_y.size(-1),))

        # --- STUDENT PATH (Context-Restricted) ---
        s_out = self.student(x_full, y_train, only_return_standard_out=False)
        s_test_latents = s_out["test_embeddings"]

        # Project, THEN Predict
        z = self.student_projector(s_test_latents)
        s_y_hat = self.student_predictor(z)

        # --- JEPA LOSS (Latent Space) ---
        jepa_loss = F.smooth_l1_loss(s_y_hat, s_y)

        # --- DETACHED PROBE ---
        # Detach so the NLL objective doesn't pollute the self-supervised manifold
        logits = self.probe(s_y_hat.detach())
        # logits = self.probe(s_out["test_embeddings"].detach())
        # fixme, this is just checking if the predictor removes info interesting to nll
        logits = self.probe_raw(s_out["test_embeddings"].detach())

        return jepa_loss, logits

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
        model.student_projector.apply(init_jepa_weights)
        model.teacher_projector.apply(init_jepa_weights)
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

    # train_baseline_pfn(steps=10000)
    train_jepa_pfn(steps=10000)