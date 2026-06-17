
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


# --- Perceiver Domain Transfer ---
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

    def forward(self, A_train_BRCE: torch.Tensor, B_train_BRCE: torch.Tensor,
                padding_mask_A: torch.Tensor | None = None) -> torch.Tensor:
        B_batch, R_A, C, E = A_train_BRCE.shape
        _, R_B, _, _ = B_train_BRCE.shape
        head_dim = self.bottleneck_attn.head_dim
        num_heads = self.bottleneck_attn.num_heads

        # 1. Bottleneck: Compress A
        A_flat_BcRE = A_train_BRCE.transpose(1, 2).reshape(B_batch * C, R_A, E)
        q_latents = self.latent_queries.expand(B_batch, self.num_latents, C, E).transpose(1, 2).reshape(
            B_batch * C, self.num_latents, E)

        q_BcHNK = self.bottleneck_attn.q_projection(q_latents).view(B_batch * C, self.num_latents, -1, head_dim)
        k_BcRNK = self.bottleneck_attn.k_projection(A_flat_BcRE).view(B_batch * C, R_A, -1, head_dim)
        v_BcRNK = self.bottleneck_attn.v_projection(A_flat_BcRE).view(B_batch * C, R_A, -1, head_dim)

        # Handle Padding Mask for A
        attn_mask = None
        if padding_mask_A is not None:
            mask = torch.zeros_like(padding_mask_A, dtype=q_BcHNK.dtype)
            mask.masked_fill_(padding_mask_A, float('-inf'))

            # 1. Add the feature dimension C
            mask = mask.unsqueeze(1)  # Shape: (B_batch, 1, R_A)

            # 2. Expand across C and flatten to match A_flat_BcRE's memory layout
            mask_Bc = mask.expand(B_batch, C, R_A).reshape(B_batch * C, R_A)

            # 3. Expand for heads and latents
            attn_mask = mask_Bc.view(B_batch * C, 1, 1, R_A).expand(-1, num_heads, self.num_latents, -1)

        A_bottleneck_BcKHD = _batched_scaled_dot_product_attention(q_BcHNK, k_BcRNK, v_BcRNK, attn_mask=attn_mask)
        A_bottleneck_BcKF = A_bottleneck_BcKHD.reshape(B_batch * C, self.num_latents, num_heads * head_dim)
        A_bottleneck_BcK = self.bottleneck_attn.out_projection(A_bottleneck_BcKF)
        A_bottleneck_BCKE = A_bottleneck_BcK.view(B_batch, C, self.num_latents, E)

        if self.num_latents > 1:
            A_bottleneck_BCE = self.kv_proj(A_bottleneck_BCKE.reshape(B_batch, C, self.num_latents * E))
        else:
            A_bottleneck_BCE = A_bottleneck_BCKE.squeeze(2)

        # 2. Translate B
        B_flat_BrCE = B_train_BRCE.reshape(B_batch * R_B, C, E)
        A_bottleneck_BrCE = A_bottleneck_BCE.unsqueeze(1).expand(B_batch, R_B, C, E).reshape(B_batch * R_B, C, E)

        q_BrCHF = self.translate_attn.q_projection(B_flat_BrCE).view(B_batch * R_B, C, -1, head_dim)
        k_BrCHF = self.translate_attn.k_projection(A_bottleneck_BrCE).view(B_batch * R_B, C, -1, head_dim)
        v_BrCHF = self.translate_attn.v_projection(A_bottleneck_BrCE).view(B_batch * R_B, C, -1, head_dim)

        translated_BrCHD = _batched_scaled_dot_product_attention(q_BrCHF, k_BrCHF, v_BrCHF)
        translated_BrCF = translated_BrCHD.reshape(B_batch * R_B, C, num_heads * head_dim)
        translated_BrCE = self.translate_attn.out_projection(translated_BrCF)

        return translated_BrCE.view(B_batch, R_B, C, E)


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
    ) -> None:
        super().__init__()
        device_and_dtype = {"device": device, "dtype": dtype}
        assert emsize % nhead == 0

        self.per_sample_attention_between_features = AlongRowAttention(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead, **device_and_dtype,
        )

        # --- NEW: Domain Transfer Block ---
        self.perceiver_domain_transfer = PerceiverDomainTransfer(
            embedding_size=emsize, num_heads=nhead, head_dim=emsize // nhead, num_latents=num_latents,
            **device_and_dtype,
        )

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
            n_train_A: int | None = None,  # <--- NEW
            padding_mask_A: torch.Tensor | None = None,  # <--- NEW
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

            B_translated = self.perceiver_domain_transfer(A_train, B_train, padding_mask_A)

            B_train = B_train + B_translated
            B_train = chunked_evaluate_maybe_inplace(
                self.layernorm_domain, B_train, save_peak_memory_factor, residual=False, batch_dims=3
            )
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
                    # Extract Z_target dynamically without a full forward pass
                    if aux_weight > 0.0:
                        with torch.no_grad():
                            raise NotImplementedError(
                                'Auxiliary loss should not look at the embedding layer!'
                                ' instead it should look at the PFN\'s final output!'
                                'But to do this, we would need to pass X & Y for B_in_A as training set without test set!'
                            )

                            Z_target = model.embed_target(X_B_in_A, Y_B_in_A).detach()
                            # Z_target shape: (Batch, R_B, C, E)

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
    prior_stream = InfiniteHarmonicsStream(
        batch_size=32,
        n_A=10,
        n_B=50,
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

    # visualize -------------------------------------
    from prototype.harmonic_restart.harmonic_prior import HeatmapVisualizer
    import matplotlib.pyplot as plt
    # TODO model_marginals will be TabPFN trained without perceiver & feature cross-attn
    # 1. Put both models in eval mode
    # model_margi