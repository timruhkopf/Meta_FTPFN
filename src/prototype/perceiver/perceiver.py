
from tabpfn.architectures.shared.column_embeddings import load_column_embeddings
from tabpfn.preprocessing.torch import TorchStandardScaler

from typing_extensions import override

from tabpfn.architectures.kv_cache import KVCacheEntry
from tabpfn.architectures.shared.chunked_evaluate import chunked_evaluate_maybe_inplace
from tabpfn.architectures.tabpfn_v2_5 import Attention, _batched_scaled_dot_product_attention, LowerPrecisionLayerNorm, \
    AlongColumnAttention, AlongRowAttention, AddThinkingRows, ENCODING_SIZE_MULTIPLIER

from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2p5 as TabPFNV2p5_super

from tqdm import tqdm

import torch
import torch.nn as nn

from typing import Literal, cast

from pfns4hpo.bar_distribution import BarDistribution
from prototype.harmonic_restart import InfiniteHarmonicsStream

"""
Contender Paper: 
https://openreview.net/pdf?id=Ge3wbgb2Vi

They will suffer with multi-fidelity tasks due to quadratic scaling

we can also go for in context domain alignment!

related paper, but instead of disjoint datasets, model vector valued y https://arxiv.org/html/2605.20234v1
"""

# --- Updated Batched SDPA to support attn_mask ---
def _batched_scaled_dot_product_attention(
        q_BSHD: torch.Tensor, k_BSJD: torch.Tensor, v_BSJD: torch.Tensor, attn_mask: torch.Tensor | None = None
) -> torch.Tensor:
    q_BHSD = q_BSHD.permute(0, 2, 1, 3)
    k_BJSD = k_BSJD.permute(0, 2, 1, 3)
    v_BJSD = v_BSJD.permute(0, 2, 1, 3)

    dtype_supports_gqa = q_BHSD.dtype in {torch.float16, torch.bfloat16}
    # gqa_is_supported() is assumed to be defined in your environment
    if True and dtype_supports_gqa:  # Replace True with gqa_is_supported()
        keys = k_BJSD
        values = v_BJSD
        enable_gqa = {"enable_gqa": True}
    else:
        keys = k_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        values = v_BJSD.expand(-1, q_BHSD.shape[-3], -1, -1)
        enable_gqa = {}

    backends = [
        torch.backends.cuda.sdp_kernel,  # Assuming standard PyTorch backends
    ]
    num_parallel_calls = q_BHSD.shape[:2].numel()
    CUDA_MAX_GRID = 65536
    num_iterations = (num_parallel_calls + CUDA_MAX_GRID - 1) // CUDA_MAX_GRID
    sub_batch = (q_BHSD.shape[0] + num_iterations - 1) // num_iterations

    # Optional mask slicing
    def get_mask_slice(mask, idx, size):
        return mask[idx * size: (idx + 1) * size] if mask is not None else None

    outputs = []
    for i in range(num_iterations):
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                q_BHSD[i * sub_batch: (i + 1) * sub_batch],
                keys[i * sub_batch: (i + 1) * sub_batch],
                values[i * sub_batch: (i + 1) * sub_batch],
                attn_mask=get_mask_slice(attn_mask, i, sub_batch),
                **enable_gqa,
            )
        )
    output_BHSD = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
    return output_BHSD.permute(0, 2, 1, 3)


class PerceiverDomainTransfer(nn.Module):
    def __init__(self, embedding_size: int, num_heads: int, head_dim: int, num_latents: int = 1, device=None,
                 dtype=None):
        super().__init__()
        self.num_latents = num_latents
        device_and_dtype = {"device": device, "dtype": dtype}

        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, 1, embedding_size, **device_and_dtype))

        # Using the base Attention class from your codebase
        self.bottleneck_attn = Attention(embedding_size, num_heads, head_dim, **device_and_dtype)
        self.translate_attn = Attention(embedding_size, num_heads, head_dim, **device_and_dtype)

        if num_latents > 1:
            self.kv_proj = nn.Linear(num_latents * embedding_size, embedding_size, bias=False, **device_and_dtype)

        layer_norm_args = {"device": device, "dtype": dtype, "elementwise_affine": False}
        self.ln_bottleneck_A = LowerPrecisionLayerNorm(embedding_size, **layer_norm_args)
        self.ln_translate_B = LowerPrecisionLayerNorm(embedding_size, **layer_norm_args)

    def encode_target(self, A_train_BRCE: torch.Tensor, padding_mask_A: torch.Tensor | None = None) -> torch.Tensor:
        """Step 1: Compress domain A into latent summaries (Calculated ONCE)."""
        B_batch, R_A, C, E = A_train_BRCE.shape
        head_dim = self.bottleneck_attn.head_dim
        num_heads = self.bottleneck_attn.num_heads

        A_flat_BcRE = A_train_BRCE.transpose(1, 2).reshape(B_batch * C, R_A, E)

        # Apply Pre-LN to A
        A_flat_BcRE_norm = self.ln_bottleneck_A(A_flat_BcRE)

        q_latents = self.latent_queries.expand(B_batch, self.num_latents, C, E).transpose(1, 2).reshape(
            B_batch * C, self.num_latents, E)

        q_BcHNK = self.bottleneck_attn.q_projection(q_latents).view(B_batch * C, self.num_latents, -1, head_dim)
        k_BcRNK = self.bottleneck_attn.k_projection(A_flat_BcRE_norm).view(B_batch * C, R_A, -1, head_dim)
        v_BcRNK = self.bottleneck_attn.v_projection(A_flat_BcRE_norm).view(B_batch * C, R_A, -1, head_dim)

        # Handle Padding Mask for A
        attn_mask = None
        if padding_mask_A is not None:
            mask = torch.zeros_like(padding_mask_A, dtype=q_BcHNK.dtype)
            mask.masked_fill_(padding_mask_A, float('-inf'))
            mask = mask.unsqueeze(1)
            mask_Bc = mask.expand(B_batch, C, R_A).reshape(B_batch * C, R_A)
            # This expansion is mathematically safe for F.scaled_dot_product_attention
            attn_mask = mask_Bc.view(B_batch * C, 1, 1, R_A).expand(-1, num_heads, self.num_latents, -1)

        A_bottleneck_BcKHD = _batched_scaled_dot_product_attention(q_BcHNK, k_BcRNK, v_BcRNK, attn_mask=attn_mask)
        A_bottleneck_BcKF = A_bottleneck_BcKHD.reshape(B_batch * C, self.num_latents, num_heads * head_dim)
        A_bottleneck_BcK = self.bottleneck_attn.out_projection(A_bottleneck_BcKF)
        A_bottleneck_BCKE = A_bottleneck_BcK.view(B_batch, C, self.num_latents, E)

        if self.num_latents > 1:
            return self.kv_proj(A_bottleneck_BCKE.reshape(B_batch, C, self.num_latents * E))
        else:
            return A_bottleneck_BCKE.squeeze(2)

    def translate_source(self, B_train_BRCE: torch.Tensor, A_bottleneck_BCE: torch.Tensor) -> torch.Tensor:
        """Step 2: Translate domain B using A's latent summaries (Can be looped)."""
        B_batch, R_B, C, E = B_train_BRCE.shape
        head_dim = self.translate_attn.head_dim
        num_heads = self.translate_attn.num_heads

        B_flat_BrCE = B_train_BRCE.reshape(B_batch * R_B, C, E)

        # Apply Pre-LN to B
        B_flat_BrCE_norm = self.ln_translate_B(B_flat_BrCE)

        A_bottleneck_BrCE = A_bottleneck_BCE.unsqueeze(1).expand(B_batch, R_B, C, E).reshape(B_batch * R_B, C, E)

        # Cross-feature attention: Sequence dim is C
        q_BrCHF = self.translate_attn.q_projection(B_flat_BrCE_norm).view(B_batch * R_B, C, -1, head_dim)
        k_BrCHF = self.translate_attn.k_projection(A_bottleneck_BrCE).view(B_batch * R_B, C, -1, head_dim)
        v_BrCHF = self.translate_attn.v_projection(A_bottleneck_BrCE).view(B_batch * R_B, C, -1, head_dim)

        translated_BrCHD = _batched_scaled_dot_product_attention(q_BrCHF, k_BrCHF, v_BrCHF)
        translated_BrCF = translated_BrCHD.reshape(B_batch * R_B, C, num_heads * head_dim)
        translated_BrCE = self.translate_attn.out_projection(translated_BrCF)

        return translated_BrCE.view(B_batch, R_B, C, E)

    def forward(self, A_train_BRCE: torch.Tensor, B_train_BRCE: torch.Tensor,
                padding_mask_A: torch.Tensor | None = None) -> torch.Tensor:
        """
        Standard forward pass for a single step of domain transfer.
        Compresses domain A, then uses it to translate domain B.
        """
        A_bottleneck_BCE = self.encode_target(A_train_BRCE, padding_mask_A)
        return self.translate_source(B_train_BRCE, A_bottleneck_BCE)


class TabPFNBlock(nn.Module):
    """A block of one column-wise, one row-wise attention layer, perceiver transfer, and an MLP."""

    def __init__(
            self,
            *,
            emsize: int,
            nhead: int,
            dim_feedforward: int,
            num_latents: int = 1,  # <--- NEW for Perceiver
            device: torch.device | str | None = None,
            dtype: torch.dtype | str | None = None,
            use_task_emb_valve: bool = True,  # <--- NEW for optional task embedding valve
            num_recursion: int = 2,  # <--- NEW for recursive alignment steps
    ) -> None:
        super().__init__()
        device_and_dtype = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0

        self.per_sample_attention_between_features = AlongRowAttention(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead, **device_and_dtype,
        )

        # --- NEW: Domain Transfer Block ---
        self.alignment_num_recursion = num_recursion
        self.perceiver_domain_transfer = PerceiverDomainTransfer(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead, num_latents=num_latents,
            **device_and_dtype,
        )

        # To avoid contaminating the Query too early, when the domains A and B aren't aligned yet, we
        # allow the model to opt out and have A_test only attend to A
        # FIXME: rather than a static valve, that is allowing the model to opt out over the depth,
        #  we could also try a dynamic valve based on the perceiver alignment; i.e. computing both A's and B's
        #  feature vector space, then take those embeddings and use their angle as task embedding added to B's
        #  representations. alternatively, we can use [log(n_A), log(n_B), entropy_attn_weights in feature cross attn]
        #  and pass it through an MLP to obtain the two vectors E_A, E_B that are being added to [A_train, A_test(!)]
        #  and B_train. The idea is, that
        #  with this we measure in-context what the alignment quality is to not contaminate the rowwise-cross attn
        #  from A_test to A_train and B_test
        self.use_task_emb_valve = use_task_emb_valve
        if self.use_task_emb_valve:
            self.task_emb_A = nn.Parameter(torch.zeros(1, 1, 1, emsize, **device_and_dtype))
            self.task_emb_B = nn.Parameter(torch.zeros(1, 1, 1, emsize, **device_and_dtype))
            torch.nn.init.normal_(self.task_emb_A, std=0.02)
            torch.nn.init.normal_(self.task_emb_B, std=0.02)

        self.per_column_attention_between_cells = AlongColumnAttention(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead, **device_and_dtype,
        )

        layer_norm_args = {**device_and_dtype, "elementwise_affine": False}
        self.layernorm_mha1 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_domain = LowerPrecisionLayerNorm(emsize, **layer_norm_args)  # <--- NEW
        self.layernorm_mha2 = LowerPrecisionLayerNorm(emsize, **layer_norm_args)
        self.layernorm_mlp = LowerPrecisionLayerNorm(emsize, **layer_norm_args)

        self.mlp = nn.Sequential(
            torch.nn.Linear(emsize, dim_feedforward, bias=False, **device_and_dtype),
            torch.nn.GELU(),
            torch.nn.Linear(dim_feedforward, emsize, bias=False, **device_and_dtype),
        )
        torch.nn.init.zeros_(cast("torch.nn.Linear", self.mlp[2]).weight)

    @override
    def forward(
            self,
            x_BRCE: torch.Tensor,
            single_eval_pos: int,
            save_peak_memory_factor: int | None,
            *,
            n_train_A: int | None = None,
            padding_mask_A: torch.Tensor | None = None,
            cached_kv: KVCacheEntry | None = None,
            return_kv: bool = False,
    ) -> tuple[torch.Tensor, KVCacheEntry | None, torch.Tensor | None]:

        # -- 1. Feature Self-Attention
        x_BRCE_residual = chunked_evaluate_maybe_inplace(
            self.per_sample_attention_between_features, x_BRCE, save_peak_memory_factor, residual=False, batch_dims=2,
        )
        x_BRCE = x_BRCE + x_BRCE_residual
        x_BRCE = chunked_evaluate_maybe_inplace(
            self.layernorm_mha1, x_BRCE, save_peak_memory_factor, residual=False, batch_dims=3
        )

        B_translated = None
        # -- 2. Perceiver Domain Transfer
        if n_train_A is not None and cached_kv is None:
            A_train = x_BRCE[:, :n_train_A]
            B_train = x_BRCE[:, n_train_A: single_eval_pos]
            A_test = x_BRCE[:, single_eval_pos:]

            # FIX: Calculate A's bottleneck exactly once
            A_bottleneck = self.perceiver_domain_transfer.encode_target(A_train, padding_mask_A)

            # FIX: Only the translation of B occurs in the recursive loop
            for step in range(self.alignment_num_recursion):
                B_translated = self.perceiver_domain_transfer.translate_source(B_train, A_bottleneck)
                # The crucial residual connection
                B_train = B_train + B_translated

            # This allows A_test to zero-out its attention to B_train via the dot product
            if self.use_task_emb_valve:
                A_train = A_train + self.task_emb_A
                A_test = A_test + self.task_emb_A
                B_train = B_train + self.task_emb_B

            x_BRCE = torch.cat([A_train, B_train, A_test], dim=1)

        # -- 3. Row Cross-Attention
        x_BCRE = x_BRCE.transpose(1, 2).contiguous()
        del x_BRCE
        kv_entry: KVCacheEntry | None = None

        if return_kv or cached_kv is not None:
            B, C = x_BCRE.shape[:2]
            attn_out, kv_entry = self.per_column_attention_between_cells(
                x_BCRE.flatten(0, 1), single_eval_pos=single_eval_pos, cached_kv=cached_kv, return_kv=return_kv,
            )
            x_BCRE = x_BCRE + attn_out.unflatten(0, (B, C))
        else:
            x_BCRE_residual = chunked_evaluate_maybe_inplace(
                lambda x, single_eval_pos=None:
                self.per_column_attention_between_cells(x, single_eval_pos=single_eval_pos)[0],
                x_BCRE, save_peak_memory_factor, residual=False, batch_dims=2, single_eval_pos=single_eval_pos,
            )
            x_BCRE = x_BCRE + x_BCRE_residual

        x_BCRE = chunked_evaluate_maybe_inplace(self.layernorm_mha2, x_BCRE, save_peak_memory_factor, residual=False,
                                                batch_dims=3)
        x_BRCE = x_BCRE.transpose(1, 2).contiguous()
        del x_BCRE

        # -- 4. MLP
        x_BRCE_residual = chunked_evaluate_maybe_inplace(self.mlp, x_BRCE, save_peak_memory_factor, residual=False,
                                                         batch_dims=3)
        x_BRCE = x_BRCE + x_BRCE_residual
        x_BRCE = chunked_evaluate_maybe_inplace(self.layernorm_mlp, x_BRCE, save_peak_memory_factor, residual=False,
                                                batch_dims=3)

        return x_BRCE, kv_entry, B_translated

class TabPFNV2p5(TabPFNV2p5_super):  # Assuming Architecture base class is defined
    def __init__(
            self,
            *,
            config,
            task_type,
            n_out: int = 1,
            feature_positional_embedding: Literal["subspace"] | None = "subspace",
            device: torch.device | str | None = None,
            dtype: torch.dtype | str | None = None,
    ):
        """Initializes the PerFeatureTransformer module.

        Args:
            config: The model hyperparameters.
            encoder: An InputEncoder, which takes a dictionary with tensors of shape
                [num_rows, batch_size, num_cols, features] and returns a single tensor
                of shape [num_rows, batch_size, input_size].
            task_type: The type of task the model should perform.
            n_out: The number of outputs the model should produce.
            feature_positional_embedding: The positional embedding type to use.
                The  positional embedding is added to the features to help the model
                distinguish them. Currently, only "subspace" is supported.
            device: The device to use for the layer parameters.
            dtype: The data type to use for the layer parameters.
        """
        torch.nn.Module.__init__(self)
        if feature_positional_embedding != "subspace":
            raise ValueError("Currently only 'subspace' is supported.")
        self.input_size = config.emsize
        self.hidden_size = self.input_size * 2
        self.features_per_group = config.features_per_group
        self.n_out = n_out
        self.task_type = task_type

        self.feature_group_embedder = self._get_feature_group_embedder(config)
        self.target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, config.emsize)
        self.add_thinking_rows = AddThinkingRows(
            num_thinking_rows=config.num_thinking_rows,
            embedding_size=config.emsize,
        )
        if config.num_thinking_rows > 0:
            raise NotImplementedError('There may be a hidden index error')
            """
            # In the code: 
            A_train = x_BRCE[:, :n_train_A]
            B_train = x_BRCE[:, n_train_A: single_eval_pos]
            
            # If thinking rows are present, A_train will accidentally ingest the thinking rows, truncating the actual A_train data and shifting B_train completely out of alignment.

            # The untested Fix:
            # Account for the thinking rows offset when slicing.
            # In TabPFNBlock.forward
            if n_train_A is not None and cached_kv is None:
                # Safely calculate where the actual training data starts
                start_idx = x_BRCE.shape[1] - single_eval_pos # Or pass num_thinking_rows explicitly
                
                A_train = x_BRCE[:, start_idx : start_idx + n_train_A]
                B_train = x_BRCE[:, start_idx + n_train_A : single_eval_pos]
                A_test  = x_BRCE[:, single_eval_pos:]
                
                B_translated = self.perceiver_domain_transfer(A_train, B_train, padding_mask_A)
                # ...
                x_BRCE = torch.cat([x_BRCE[:, :start_idx], A_train, B_train_updated, A_test], dim=1)
            """

        self.blocks = nn.ModuleList(
            TabPFNBlock(
                emsize=config.emsize,
                nhead=config.nhead,
                dim_feedforward=self.hidden_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.nlayers)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, n_out),
        )
        self.standard_scaler = TorchStandardScaler()

        self.pre_generated_column_embeddings = load_column_embeddings()
        self.feature_positional_embedding_embeddings = nn.Linear(
            self.input_size // 4, self.input_size
        )
        self._do_encoder_nan_check = True
        self.emsize = config.emsize


    def embed_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Helper to compute Z_target for the auxiliary domain transfer loss."""
        num_rows, batch_size, _ = x.shape
        x_emb, _, _ = self._preprocess_and_embed_features(x, num_rows, batch_size)
        y_emb = self._preprocess_and_embed_targets(y, num_rows, num_rows, batch_size)
        return torch.cat((x_emb, y_emb[:, :, None]), dim=2)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None, *,
                n_train_A: int | None = None, padding_mask_A: torch.Tensor | None = None,
                task_type: str | None = None,
                only_return_standard_out: bool = True, kv_cache=None, return_kv_cache: bool = False,
                x_is_test_only: bool = False):

        del task_type
        # if performance_options is None:
        #     performance_options = self.get_default_performance_options()
        # force_recompute_layer = performance_options.force_recompute_layer
        # save_peak_memory_factor = performance_options.save_peak_memory_factor
        # del categorical_inds

        using_cache = kv_cache is not None and not kv_cache.is_empty()
        if x_is_test_only and not using_cache:
            raise ValueError(
                "x_is_test_only=True requires a populated kv_cache; the standard "
                "forward needs the full train+test tensor."
            )

        if isinstance(x, dict):
            x = x["main"]
        if isinstance(y, dict):
            y = y["main"]
        if y is None:
            y = torch.zeros(0, device=x.device, dtype=x.dtype)

        if (
                not self.training
                and self.task_type == "multiclass"
                and (y > self.n_out - 1).any()
        ):
            raise ValueError(
                "Target is out of range. Make sure to use an ordinal encoded target. "
                f"Expected target values between 0 and {self.n_out - 1}, but got values"
                f" greater than {self.n_out - 1}."
            )

        # Ri = number of input rows. In the standard / build paths these are the
        # train (+ optionally test) rows; in the cache path they are test-only rows.
        # B = batch size, C = number of columns before grouping.
        x_RiBC = x

        num_input_rows, batch_size, *_ = x.shape
        num_train_labels = y.shape[0]

        embedded_x_BRiGX, _, _ = self._preprocess_and_embed_features(x, num_train_labels, batch_size)
        embedded_y_BRiX = self._preprocess_and_embed_targets(y, num_input_rows, num_train_labels, batch_size)

        x_BRiCD = torch.cat((embedded_x_BRiGX, embedded_y_BRiX[:, :, None]), dim=2)
        x_BRCD, block_single_eval_pos = self.add_thinking_rows(x_BRiCD, single_eval_pos=num_train_labels)

        kv_out = {}
        all_B_translated = []

        # Iterate Blocks
        for layer_idx, block in enumerate(self.blocks):
            x_BRCD, kv_entry, b_trans = block(
                x_BRCD, block_single_eval_pos, None,
                n_train_A=n_train_A, padding_mask_A=padding_mask_A,
                cached_kv=kv_cache.kv[layer_idx] if kv_cache else None, return_kv=return_kv_cache
            )
            if b_trans is not None:
                all_B_translated.append(b_trans)
            if return_kv_cache:
                kv_out[layer_idx] = kv_entry

        output = self._decode(x_BRCD, test_start=block_single_eval_pos,
                              train_start=self.add_thinking_rows.num_thinking_rows, train_end=block_single_eval_pos,
                              only_return_standard_out=only_return_standard_out)

        return {"standard": output, "b_translated": all_B_translated[-1] if all_B_translated else None}

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    from torch.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import CosineAnnealingLR


    from tqdm import tqdm
    from typing import Literal


    def train_baseline_pfn(
            model,  # Should be the base TabPFNV2p5_super
            dataloader,
            nll_criterion,
            epochs: int = 100,
            steps_per_epoch: int = 500,
            lr: float = 1e-4,

            device: str = 'cuda'
    ):
        """
        Trains the baseline TabPFN model to establish NLL bounds.
        """
        model.to(device)
        nll_criterion.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6)
        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        model.train()
        pbar = tqdm(range(epochs), desc=f"Baseline PFN")

        for epoch in pbar:
            epoch_loss = 0.0

            batch_iter = iter(dataloader)

            for step in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                batch = next(batch_iter)

                # 1. Extract Target Task A and Test points
                X_A = batch['train']['X_A'].to(device)  # (n_A, Batch, C)
                Y_A = batch['train']['Y_A'].to(device)  # (n_A, Batch, 1)

                X_test = batch['test']['X_A'].to(device)  # (n_test, Batch, C)
                Y_test = batch['test']['Y_A'].to(device)  # (n_test, Batch, 1)

                X_train = X_A
                Y_train = Y_A.squeeze(-1)


                # TabPFN expects Train points followed by Test points
                X_input = torch.cat([X_train, X_test], dim=0)

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):

                    # 3. Main Forward Pass (Standard TabPFN signature)
                    # Note: The original TabPFN does not take n_train_A or padding masks,
                    # and directly returns the standard predictions (or a tuple depending on your exact super class wrapper).
                    out = model(x=X_input, y=Y_train)

                    # Handle output based on whether the superclass returns a dict or raw tensor
                    logits = out["standard"] if isinstance(out, dict) else out

                    # 4. Compute NLL (Must be FP32 for numerical stability)
                    # Ensure we only calculate loss on the test set predictions
                    loss_nll = nll_criterion(logits.float(), Y_test.squeeze(-1).float()).mean()

                # 5. Backward and Optimize
                scaler.scale(loss_nll).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss_nll.item()

            pbar.set_description(
                f"Epoch {epoch} | NLL: {epoch_loss / steps_per_epoch:.3f}"
            )

        return model

    # TODO: integrate mlflow, run baseline marginal models (TABPFN with different amount of training data!)
    # TODO: split the file
    # TODO: integrate with trainer and callbacks!
    def train_domain_transfer_pfn(
            model: TabPFNV2p5,
            dataloader: InfiniteHarmonicsStream,
            nll_criterion: BarDistribution,
            epochs: int = 100,
            steps_per_epoch: int = 500,
            lr: float = 1e-4,
            aux_weight: float = 1.0,
            device: str = 'cuda'
    ):
        model.to(device)
        nll_criterion.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6)
        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

        model.train()
        pbar = tqdm(range(epochs), desc="Training Perceiver Domain Transfer")  # type: ignore
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_aux = 0.0

            batch_iter = iter(dataloader)

            for step in range(steps_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                batch = next(batch_iter)

                # 1. Prepare Inputs
                X_A = batch['train']['X_A'].to(device)  # (n_B, Batch, 1)
                Y_A = batch['train']['Y_A'].to(device)  # (n_B, Batch, 1)
                X_B = batch['train']['X_B'].to(device)  # (n_B, Batch, 1)
                Y_B = batch['train']['Y_B'].to(device)  # (n_B, Batch, 1)

                X_test = batch['test']['X_A'].to(device)  # (n_test, Batch, 1)
                Y_test = batch['test']['Y_A'].to(device)  # (n_test, Batch, 1)

                pad_mask_A = batch['train']['padding_mask_A'].to(device)  # (Batch, n_B)

                n_train_A = X_A.shape[0]
                n_train_B = X_B.shape[0]

                # TabPFN expects Train then Test.
                X_input = torch.cat([X_A, X_B, X_test], dim=0)
                Y_train = torch.cat([Y_A, Y_B], dim=0).squeeze(-1)  # Squeeze target feature dim

                # 2. Get Ground Truth SSL Targets (B mapped into A)
                X_B_in_A = batch['train']['X_B_in_A'].to(device)
                Y_B_in_A = batch['train']['Y_B_in_A'].to(device).squeeze(-1)

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    if aux_weight > 0.0:
                        with torch.no_grad():
                            # The total number of rows for this secondary pass
                            num_rows_B = X_B_in_A.shape[1]

                            # NATIVE TABPFN TRAIN-ONLY MODE:
                            Z_target_dict = model(
                                x=X_B_in_A,
                                y=Y_B_in_A,
                                # Setting eval_pos to the total length means NO test set
                                single_eval_pos=num_rows_B,
                                # Setting this to None natively bypasses all Perceiver blocks
                                n_train_A=None,
                                padding_mask_A=None
                            )

                            # Extract the detached targets
                            Z_target = Z_target_dict['standard'].detach()

                    # 3. Main Forward Pass
                    out = model(
                        x=X_input,
                        y=Y_train,
                        n_train_A=n_train_A,
                        padding_mask_A=pad_mask_A
                    )

                    logits = out["standard"]  # (n_test, Batch, num_bars)
                    Z_pred = out["b_translated"]  # (Batch, R_B, C, E)

                    # 4. Compute NLL (Must be FP32 for numerical stability in bucket assignments)
                    loss_nll = nll_criterion(logits.float(), Y_test.squeeze(-1).float()).mean()

                    # 5. Compute Auxiliary SSL Domain Transfer Loss
                    # Force B's representation to align geometrically with A's coordinate reality
                    loss_aux = F.mse_loss(Z_pred, Z_target) if aux_weight > 0.0 else torch.tensor(0.).to(device)

                    # Total Loss
                    loss = loss_nll + (aux_weight * loss_aux)

                # 6. Backward and Optimize
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_nll += loss_nll.item()
                epoch_aux += loss_aux.item()


            pbar.set_description(
                f"Epoch {epoch} | NLL: {loss_nll.item():.3f} | Aux MSE: {loss_aux.item():.3f}"
            )

        return model



    # 1. Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Instantiate the "Hidden Harmonic Mixture" Prior
    # We sample 10 points for target A, and 50 points for source B.
    n_A, n_B = 10, 50
    prior_stream = InfiniteHarmonicsStream(
        batch_size=32,
        n_A=n_A,
        n_B=n_B,
        n_test=200,
        x_range=(-5, 5),
        num_components=4, # todo ablate complexity
        noise_std=0.05,
        share_unrelated=0.0,
        scale=True,
        shift=True,
        warp=True
    )


    # 3. Create Dummy TabPFN Configurations
    # Match these parameters to your specific TabPFN configuration schema
    class DummyConfig:
        emsize = 128
        nhead = 4
        nlayers = 4
        num_thinking_rows = 0
        features_per_group = 1
        encoder_type = "mlp"
        encoder_mlp_hidden_dim = 256


    config = DummyConfig()
    num_bars = 200

    # 4. Instantiate the Modified TabPFN Model
    # TaskType is assumed to be an Enum or string matching your package setup
    model = TabPFNV2p5(
        config=config,
        task_type="regression",
        n_out=num_bars,
        feature_positional_embedding="subspace", # FIXME?
        device=device
    )

    # 5. Define Borders for the BarDistribution Criterion
    # TabPFN approaches regression by predicting probabilities across continuous bins.
    # We map the typical output range of your harmonics prior into discrete buckets.
    min_y, max_y = -15.0, 15.0
    borders = torch.linspace(min_y, max_y, steps=num_bars + 1, device=device)

    nll_criterion = BarDistribution(
        borders=borders,
        smoothing=0.01,
        ignore_nan_targets=True
    )

    # 6. Kick off Training
    # Adjust epochs and steps_per_epoch based on your computational budget
    model = train_domain_transfer_pfn(
        model=model,
        dataloader=prior_stream,
        nll_criterion=nll_criterion,
        epochs=1000,
        steps_per_epoch=100,
        lr=1e-4,
        aux_weight=0.0, # fixme: add weight here to encourage similar representations
        device=device
    )

    # Baseline training -------------------------------------------------------
    # (1) NLL lower bound: the full model with complete information n_A + n_B trained directly
    # in the domain of A (no domain transfer necessary).
    baseline = TabPFNV2p5(
        config=config,
        task_type="regression",
        n_out=num_bars,
        feature_positional_embedding="subspace",  # FIXME?
        device=device
    )
    prior_stream = InfiniteHarmonicsStream(
        batch_size=32,
        n_A=n_A + n_B,
        n_B=n_A + n_B, # will be ignored anyways
        n_test=200,
        x_range=(-5, 5),
        num_components=4,  # todo ablate complexity
        noise_std=0.05,
        share_unrelated=0.0,
        scale=True,
        shift=True,
        warp=True
    )
    baseline = train_baseline_pfn(
        baseline,
        dataloader=prior_stream,
        nll_criterion=nll_criterion,
        epochs=1000,
        steps_per_epoch=100,
        lr=1e-4,
        device=device
    )

    # (2) NLL "upper bound" that we need to undercut: the unconditional model A ------------------
    # Stratify this baseline over different n_A from n_A to n_A+n_B to see how much information
    # we were able to extract from this single related task B!
    baseline = TabPFNV2p5(
        config=config,
        task_type="regression",
        n_out=num_bars,
        feature_positional_embedding="subspace",  # FIXME?
        device=device
    )
    prior_stream = InfiniteHarmonicsStream(
        batch_size=32,
        n_A=n_A,
        n_B=n_A + n_B,  # will be ignored anyways
        n_test=200,
        x_range=(-5, 5),
        num_components=4,  # todo ablate complexity
        noise_std=0.05,
        share_unrelated=0.0,
        scale=True,
        shift=True,
        warp=True
    )
    baseline = train_baseline_pfn(
        baseline,
        dataloader=prior_stream,
        nll_criterion=nll_criterion,
        epochs=1000,
        steps_per_epoch=100,
        lr=1e-4,
        device=device
    )

    # visualize -------------------------------------
    from prototype.harmonic_restart.harmonic_prior import HeatmapVisualizer

    with torch.no_grad():
        # Get a batch
        batch_data = next(prior_stream.__iter__())

        # Extract tensors and move to the model's device
        device = next(model.parameters()).device

        X_A = batch_data['train']['X_A'].to(device)
        Y_A = batch_data['train']['Y_A'].to(device)
        X_B = batch_data['train']['X_B'].to(device)
        Y_B = batch_data['train']['Y_B'].to(device)

        X_test = batch_data['test']['X_A'].to(device)
        pad_mask_A = batch_data['train'].get('padding_mask_A', None)
        if pad_mask_A is not None:
            pad_mask_A = pad_mask_A.to(device)

        n_train_A = X_A.shape[0]

        # TabPFN expects Train (A + B) then Test (A) stacked along the sequence dimension
        X_input = torch.cat([X_A, X_B, X_test], dim=0)
        Y_train = torch.cat([Y_A, Y_B], dim=0).squeeze(-1)  # Squeeze feature dim for targets

        # 1. Run the marginal model (if you uncomment it later)
        # out_marginals = model_marginals(x=..., y=...)
        logits_A, logits_B = None, None

        # 2. Run the perceiver model for Stream C
        out_perceiver = model(
            x=X_input,
            y=Y_train,
            n_train_A=n_train_A,
            padding_mask_A=pad_mask_A
        )

        # TabPFNV2p5 returns a dict containing "standard" and "b_translated"
        logits_C = out_perceiver['standard']