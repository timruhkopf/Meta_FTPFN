"""
POC: Cross-Table Relational Attention Network (CT-RAN) for Multi-Table TabPFN

Extends TabPFN to simultaneously process two tabular datasets, A (Target) and B (Related),
with mismatched row counts (R_A != R_B) and feature counts (F_A != F_B). Enables zero-shot
cross-table data augmentation at inference time while preserving row permutation
equivariance, feature-count invariance, and strict test-set isolation.

ASSUMPTIONS & PIPELINE STAGES:
---------------------------------------------------------------------------------
1. Dedicated Target Encoding (Authentic TabPFN Format):
    - Input values and targets y are embedded into a uniform hidden dimension D.
    - Targets y are concatenated as a dedicated column along the feature axis:
        A in R^[B x (F_A + 1) x R_A x D]  and  B in R^[B x (F_B + 1) x R_B x D]
    - For test rows, the target column slice (A_test[:, -1, :, :]) is initialized
      with a learned query placeholder token.

2. Native Intra-Table Context Phase:
    - Independent, batch-stacked Row and Feature Self-Attentions on A and B.
    - Row Attention: Subsumes feature dimension (F + 1) into batch: [(B * (F + 1)), R, D].
    - Feature Attention: Subsumes row dimension R into batch: [(B * R), (F + 1), D].
    - Allows the dedicated target column to attend natively across all feature tokens.

3. Perceiver-IO Bottleneck & Zero-Shot Cross-Table Imputation:
    3.1 Encoder (Row Compression to Shared Latent Basis):
        - A learnable latent seed L in R^[R_L x D] queries rows of A and B across
          all features independently, compressing variable rows:
            A_lat in R^[B x (F_A + 1) x R_L x D]
            B_lat in R^[B x (F_B + 1) x R_L x D]
        - Shared weights of L force "latent slot k" to specialize consistently across tables.

    3.2 Cross-Table Feature Attention:
        - Align latent rows in batch [(B * R_L)] and concatenate along the feature axis:
            H_lat in R^[(B * R_L) x ((F_A + 1) + (F_B + 1)) x D]
        - MHSA across H_lat allows every feature (and target) in A to attend to every
          feature (and target) in B across the shared R_L basis.

    3.3 Per-Target-Column Query Extraction (Preserving Column Modalities):
        - For each target column slot f in (F_target + 1), a learned query q_f
          attends across source table feature tokens for row r:
            s[r, f] = CrossAttn(Q = q_f, K = V = Source_tokens[r, :, :])
        - Yields specialized queries S in R^[B x (F_target + 1) x R_source x D],
          avoiding lossy feature-mean pooling.

    3.4 Conditioned Decoding (Off-Diagonal Block Construction):
        - Decodes updated latents conditioned on s[r, f] to generate cross-imputed blocks:
            B_in_A in R^[B x (F_A + 1) x R_B x D]  (Table B rows in Table A schema)
            A_in_B in R^[B x (F_B + 1) x R_A x D]  (Table A rows in Table B schema)

4. Support Matrix Assembly & Source Conditioning:
    - Inject learnable source indicator tokens (e_real, e_synth) to prevent
      cardinality imbalance during row attention.
    - Assemble unified 2D Block Matrix M_support in R^[B x (F_A + F_B + 2) x (R_A + R_B) x D]:
            [ A_hat    | A_in_B_hat ]
            [ B_in_A_hat | B_hat    ]
    - Execute 2D Row & Feature Self-Attention across M_support.

5. Asymmetric Vertical Split for Test-Set Inference:
    - Test rows NEVER attend horizontally across tables to prevent zero-padding
      contamination and maintain strict test-set isolation.
    - Test set A (A_test) queries its native vertical column block vertically across rows:
        A_out = CrossAttn(Q = A_test + e_real, K = V = [A_hat || B_in_A_hat])
    - Test set B (B_test) queries its native vertical column block vertically across rows:
        B_out = CrossAttn(Q = B_test + e_real, K = V = [A_in_B_hat || B_hat])

6. Target Extraction & Decoupled Prediction:
    - Extract strictly the final dedicated target column slice (-1) from test outputs:
        target_repr_A = A_out[:, -1, :, :]  # [B, R_test_A, D]
        target_repr_B = B_out[:, -1, :, :]  # [B, R_test_B, D]
    - Decode via linear layer to class logits for independent NLL loss computation:
        L_total = NLL(Logits_A, Y_A_test) + lambda * NLL(Logits_B, Y_B_test)
"""

import torch
from torch import nn
import pydantic
import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override
from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5, TabPFNV2p5Cache
from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution


from tabpfn.architectures.interface import (
    Architecture,
    ArchitectureConfig,
    PerformanceOptions,
)
from tabpfn.architectures.kv_cache import (
    KVCache,
    KVCacheEntry,
)

import torch
import torch.nn as nn
class SoftColumnCorrespondenceBridge(nn.Module):
    """
    Cross-Table Bridge operating natively on TabPFN [B, R, C, D] tensors.
    Computes the row-stochastic column-correspondence matrix A_{A <- B} across
    the shared Thinking Rows (N_r) and applies cross-table feature updates.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        num_thinking_rows: int = 64,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_r = num_thinking_rows
        self.temperature = temperature

        # Multi-head projections for Column Correspondence (Q_A, K_B, V_B)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj_A = nn.Linear(embed_dim, embed_dim)
        self.out_proj_B = nn.Linear(embed_dim, embed_dim)

        self.norm_A = nn.LayerNorm(embed_dim)
        self.norm_B = nn.LayerNorm(embed_dim)

        self.mlp_A = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.mlp_B = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def _compute_consensus_correspondence(
        self, T_A: torch.Tensor, T_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes row-stochastic correspondence matrices A_{A <- B} and A_{B <- A}
        by averaging dot-product attention across all N_r thinking-row scenarios.

        T_A shape: [B, N_r, C_A + 1, D]
        T_B shape: [B, N_r, C_B + 1, D]
        Returns:
            A_A_from_B: [B, C_A + 1, C_B + 1]
            A_B_from_A: [B, C_B + 1, C_A + 1]
        """
        #FIXME:exclude y!
        T_A_feat, T_B_feat = T_A[:, :, :-1, :], T_B[:, :, :-1, :]  # exclude target column

        B, N_r, C_A, D = T_A.shape
        _, _, C_B, _ = T_B.shape

        # Fold thinking rows into batch: [(B * N_r), C, D]
        Q_A = self.q_proj(T_A.reshape(B * N_r, C_A, D))
        K_B = self.k_proj(T_B.reshape(B * N_r, C_B, D))

        # Scaled dot-product across column spaces: [(B * N_r), C_A, C_B]
        scale = (D ** -0.5) / self.temperature
        scores = torch.bmm(Q_A, K_B.transpose(1, 2)) * scale

        # Row-stochastic soft permutation weights
        attn_A_from_B = torch.softmax(scores, dim=-1).reshape(B, N_r, C_A, C_B)
        attn_B_from_A = torch.softmax(scores.transpose(1, 2), dim=-1).reshape(B, N_r, C_B, C_A)

        # Consensus average across thinking rows: [B, C_A, C_B]
        return attn_A_from_B.mean(dim=1), attn_B_from_A.mean(dim=1)

    @staticmethod
    def translate_rows(x_source: torch.Tensor, correspondence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Zero-shot LUPI translation of physical rows into the target column schema.

        x_source:              [B, R_source, **C_source + 1**, D]
        correspondence_matrix: [B, C_target + 1, C_source + 1]
        Returns:               [B, R_source, **C_target + 1**, D]
        """
        # Einsum matrix multiplication along the column axis, after excluding y
        x_source_feat = x_source[:, :, :-1, :]
        x_source_y = x_source[:, :, -1:, :]  # carried through untouched
        translated_feat = torch.einsum("b c s, b r s d -> b r c d", correspondence_matrix, x_source_feat)
        return torch.cat([translated_feat, x_source_y], dim=2)
    def forward(
        self,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates thinking rows of Table A and Table B and returns correspondence matrices.
        """
        B, R_tot_A, C_A, D = x_A.shape
        _, R_tot_B, C_B, _ = x_B.shape

        # 1. Slice shared Thinking Rows (N_r)
        T_A = x_A[:, : self.n_r, :, :]  # [B, N_r, C_A, D]
        T_B = x_B[:, : self.n_r, :, :]  # [B, N_r, C_B, D]

        # 2. Compute soft column correspondence (F_A^T F_B)
        A_A_from_B, A_B_from_A = self._compute_consensus_correspondence(T_A, T_B)

        # 3. Cross-table feature update on Thinking Rows
        V_B = self.v_proj(T_B)  # [B, N_r, C_B, D]
        V_A = self.v_proj(T_A)  # [B, N_r, C_A, D]

        # Apply correspondence weights across columns: [B, N_r, C_target, D]
        up_A = torch.einsum("b i j, b r j d -> b r i d", A_A_from_B, V_B)
        up_B = torch.einsum("b i j, b r j d -> b r i d", A_B_from_A, V_A)

        # Residual + MLP + LayerNorm on Thinking Rows
        T_A_out = self.norm_A(T_A + self.out_proj_A(up_A))
        T_A_out = T_A_out + self.mlp_A(T_A_out)

        T_B_out = self.norm_B(T_B + self.out_proj_B(up_B))
        T_B_out = T_B_out + self.mlp_B(T_B_out)

        # 4. Autograd-safe override: join updated thinking rows with untouched real rows
        out_A = torch.cat([T_A_out, x_A[:, self.n_r :, :, :]], dim=1)
        out_B = torch.cat([T_B_out, x_B[:, self.n_r :, :, :]], dim=1)

        return out_A, out_B, A_A_from_B, A_B_from_A


@pydantic.dataclasses.dataclass
class AnamorphicTabPFNV2p5Config(ArchitectureConfig):
    """Configuration for the CT-RAN Anamorphic TabPFN v2.5 architecture."""

    name: str = "AnamorphicTabPFN-v2.5"
    emsize: int = 192
    nlayers: int = 4
    nhead: int = 3
    features_per_group: int = 1
    num_thinking_rows: int = 64
    encoder_type: Literal["linear", "mlp"] = "linear"
    encoder_mlp_hidden_dim: int = 1024
    temperature: float = 1.0

@pydantic.dataclasses.dataclass
class AnamorphicTabPFNV2p5Config(ArchitectureConfig):
    """Configuration for the CT-RAN Anamorphic TabPFN v2.5 architecture."""

    name: str = "AnamorphicTabPFN-v2.5"
    emsize: int = 192
    nlayers: int = 4
    nhead: int = 3
    features_per_group: int = 1
    num_thinking_rows: int = 64
    encoder_type: Literal["linear", "mlp"] = "linear"
    encoder_mlp_hidden_dim: int = 1024
    temperature: float = 1.0

    # Probabilistic Regression BarDistribution setup
    n_out: int = 200
    """Number of histogram bins (n_out) for FullSupportBarDistribution."""
    min_target_val: float = -7.0
    max_target_val: float = 7.0

class AnamorphicTabPFN(TabPFNV2p5):
    def __init__(self, config: AnamorphicTabPFNV2p5Config, *args, **kwargs):
        super().__init__(config=config, n_out=config.n_out, *args, **kwargs)

        self.num_thinking_rows = config.num_thinking_rows


        # 201 border edges linearly spaced from -7.0 to 7.0
        borders = torch.linspace(
            config.min_target_val,
            config.max_target_val,
            config.n_out + 1,
        )
        self.criterion = FullSupportBarDistribution(borders=borders)

        self.perceiver_blocks = nn.ModuleList(
            SoftColumnCorrespondenceBridge(
                embed_dim=self.emsize,
                n_heads=config.nhead,
                num_thinking_rows=config.num_thinking_rows,
                temperature=config.temperature,
            )
            for _ in range(config.nlayers)
        )

    def _prepare_single_table(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, int, int]:
        """Preprocesses, embeds features/targets, and prepends thinking rows."""
        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        x_RiBC = x
        num_input_rows, batch_size, *_ = x_RiBC.shape
        num_train_labels = y.shape[0]

        # 1. Embed features independently for this table's schema
        embedded_x_BRiGX, _, _ = self._preprocess_and_embed_features(
            x_RiBC=x_RiBC,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
            return_state=False,
        )

        # 2. Embed targets independently for this table's label distribution
        embedded_y_BRiX = self._preprocess_and_embed_targets(
            y=y,
            num_train_rows=num_input_rows,
            num_train_labels=num_train_labels,
            batch_size=batch_size,
        )

        # 3. Append target as dedicated `-1` column: [B, R_input, C + 1, D]
        x_BRiCD = torch.cat((embedded_x_BRiGX, embedded_y_BRiX[:, :, None]), dim=2)
        del embedded_x_BRiGX, embedded_y_BRiX

        if self._do_encoder_nan_check:
            if torch.isnan(x_BRiCD).any():
                raise ValueError("Found NaNs in encoded x/y. Ensure a NaN-handling encoder is used.")

        # 4. Prepend native thinking rows: [B, R_total, C + 1, D]
        x_BRCD, block_single_eval_pos = self.add_thinking_rows(
            x_BRiCD, single_eval_pos=num_train_labels
        )

        return x_BRCD, block_single_eval_pos, num_train_labels

    def backbone_fwd(
        self,
        x_A_BRCD: torch.Tensor,
        x_B_BRCD: torch.Tensor,
        block_single_eval_pos_A: int,
        block_single_eval_pos_B: int,
        save_peak_memory_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A_A_from_B, A_B_from_A = None, None

        for tabblock, bridge_block in zip(self.blocks, self.perceiver_blocks):
            # 1. Independent TabPFN Row & Feature Attention per table
            container_A = [x_A_BRCD]
            del x_A_BRCD
            x_A_BRCD, _ = tabblock(container_A, block_single_eval_pos_A, save_peak_memory_factor)

            container_B = [x_B_BRCD]
            del x_B_BRCD
            x_B_BRCD, _ = tabblock(container_B, block_single_eval_pos_B, save_peak_memory_factor)

            # 2. Consensus Column Correspondence & Thinking Row Update
            x_A_BRCD, x_B_BRCD, A_A_from_B, A_B_from_A = bridge_block(x_A_BRCD, x_B_BRCD)

        return x_A_BRCD, x_B_BRCD, A_A_from_B, A_B_from_A

    @override
    def forward(  # noqa: C901, PLR0912
            self,
            x_A: torch.Tensor | dict[str, torch.Tensor],
            y_A: torch.Tensor | dict[str, torch.Tensor] | None,
            x_B: torch.Tensor | dict[str, torch.Tensor],
            y_B: torch.Tensor | dict[str, torch.Tensor] | None,
            *,
            only_return_standard_out: bool = False,
            performance_options: Any | None = None,
            return_lupi_imputations: bool = False,
    ) -> dict[str, Any]:
        """
        Perform a dual-table forward pass across Table A (Target) and Table B (Related).
        If `return_lupi_imputations=True`, translates real physical rows zero-shot
        into the respective other schema for Variant (B) LUPI auxiliary supervision.
        """
        if performance_options is None:
            performance_options = self.get_default_performance_options()
        save_peak_memory_factor = performance_options.save_peak_memory_factor

        # 1. Independently embed and prepare both tables
        x_A_BRCD, eval_pos_A, num_train_A = self._prepare_single_table(x_A, y_A)
        x_B_BRCD, eval_pos_B, num_train_B = self._prepare_single_table(x_B, y_B)

        # 2. Run dual-stream backbone
        x_A_BRCD, x_B_BRCD, A_A_from_B, A_B_from_A = self.backbone_fwd(
            x_A_BRCD=x_A_BRCD,
            x_B_BRCD=x_B_BRCD,
            block_single_eval_pos_A=eval_pos_A,
            block_single_eval_pos_B=eval_pos_B,
            save_peak_memory_factor=save_peak_memory_factor,
        )

        # 3. Decode standard test bar logits independently (-1 target column slice)
        output_A = self._decode(
            x_A_BRCD,
            test_start=eval_pos_A,
            train_start=self.add_thinking_rows.num_thinking_rows,
            train_end=eval_pos_A,
            only_return_standard_out=only_return_standard_out,
        )
        output_B = self._decode(
            x_B_BRCD,
            test_start=eval_pos_B,
            train_start=self.add_thinking_rows.num_thinking_rows,
            train_end=eval_pos_B,
            only_return_standard_out=only_return_standard_out,
        )

        results = {
            "logits_A": output_A,
            "logits_B": output_B,
            "A_A_from_B": A_A_from_B,
            "A_B_from_A": A_B_from_A,
        }

        # 4. Zero-Shot LUPI Physical Row Imputation (Variant B)
        if return_lupi_imputations and A_A_from_B is not None and A_B_from_A is not None:
            # Extract real physical rows (skipping thinking rows)
            real_x_A = x_A_BRCD[:, self.num_thinking_rows:, :, :]
            real_x_B = x_B_BRCD[:, self.num_thinking_rows:, :, :]

            # Translate physical rows into respective other column schemas
            results["X_B_in_A"] = SoftColumnCorrespondenceBridge.translate_rows(
                real_x_B, A_A_from_B
            )
            results["X_A_in_B"] = SoftColumnCorrespondenceBridge.translate_rows(
                real_x_A, A_B_from_A
            )

        return results

def compute_bar_distribution_nll(
    criterion,
    logits_MBK: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Computes FullSupportBarDistribution regression NLL over (-7.0, 7.0) with 200 bins.

    logits_MBK: Output from model._decode(), shape [M_test_rows, B_batch_size, 200]
    y_true:     Ground-truth target floats, shape [M_test_rows, B_batch_size]
                (or [B, M] which will be transposed automatically).
    Returns:    Scalar mean negative log-likelihood loss.
    """
    M_test, B_batch, n_bins = logits_MBK.shape

    # 1. Ensure y_true matches [M, B] orientation
    if y_true.shape == (B_batch, M_test) and B_batch != M_test:
        y_true = y_true.transpose(0, 1)
    elif y_true.shape != (M_test, B_batch):
        raise ValueError(f"y_true shape {y_true.shape} does not match logits shape [M={M_test}, B={B_batch}]")

    # 2. Flatten sequence and batch dimensions:
    #    logits  -> [(M * B), 200]
    #    targets -> [(M * B)]
    flat_logits = logits_MBK.reshape(-1, n_bins)
    flat_targets = y_true.reshape(-1)

    # 3. FullSupportBarDistribution computes log-densities across the 200 bins
    losses = criterion(flat_logits, flat_targets)

    # 4. Return scalar average NLL
    return losses.mean()

if __name__ == '__main__':
    from tqdm import tqdm
    from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream

    BATCH_SIZE = 32
    N_A, N_B = 10, 50
    device = 'cuda'
    prior = HarmonicMixturePrior(warp=False)
    dataset = InfiniteHarmonicsStream(prior, batch_size=BATCH_SIZE, n_A=N_A, n_B=N_B)
    batch = next(iter(dataset))

    train, test = batch['train'], batch['test']

    # target data
    X_A_train, y_A_train = train['X_A'], train['Y_A']
    X_A_test, y_A_test = test['X_A'], test['Y_A']

    # related
    X_B_train, y_B_train = train['X_B'], train['Y_B']
    X_B_test, y_B_test = test['X_B'], test['Y_B']

    # projected in resp. other domain
    Y_A_in_B = train['Y_A_in_B']
    Y_B_in_A = train['Y_B_in_A']

    model = AnamorphicTabPFN(
        config=AnamorphicTabPFNV2p5Config(
            n_latents=10
        ),
        task_type='regression'
    )

    # 1. Forward pass requesting zero-shot LUPI translations
    out = model.forward(
        x_A=torch.cat([X_A_train, X_A_test], dim=0), y_A=y_A_train,
        x_B=torch.cat([X_B_train, X_B_test], dim=0), y_B=y_B_train,
        return_lupi_imputations=True
    )
    criterion = model.criterion

    # 2. Standard Task NLL Loss on native test sets
    loss_standard = compute_bar_distribution_nll(criterion, out["logits_A"]['standard'], y_A_test.squeeze(-1)) \
                    + compute_bar_distribution_nll(criterion, out["logits_B"]['standard'], y_B_test.squeeze(-1))

    # 3. Variant (B) Direct LUPI Loss
    # Feed translated physical rows X_B_in_A into Table A's standard decoder head
    logits_LUPI_B_in_A = model._decode(
        out["X_B_in_A"],
        test_start=0,
        train_start=0,
        train_end=0,
        only_return_standard_out=True
    )
    loss_lupi = compute_bar_distribution_nll(criterion, logits_LUPI_B_in_A, Y_B_in_A.squeeze(-1))

    # Total Curriculum Objective
    total_loss = loss_standard + (gamma * loss_lupi)


    #
    # model.forward(
    #     x_A=torch.cat([X_A_train, X_A_test], dim=0),y_A=y_A_train,
    #     x_B=torch.cat([X_B_train, X_B_test], dim=0),y_B=y_B_train,
    #     only_return_standard_out=False
    # )
    #
    # # TODO project to the number of bins !
    # print(model)


