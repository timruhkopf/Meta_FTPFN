# ruff: noqa # fixme: current issue is right_key_mask to be fixed before removing this
import torch
from torch import Tensor
from typing import Optional
from torch.utils.checkpoint import checkpoint

import os
from pathlib import Path

from dotenv import load_dotenv
from ifbo.surrogate import FTPFN
import types

from pfns4hpo.transformer import TransformerModel


class PaddableTransformerModel(TransformerModel):
    def forward(self, *args, **kwargs):
        # Add our new valid key to the allowed list dynamically
        allowed_keys = {
            "src_mask",
            "style",
            "only_return_standard_out",
            "single_eval_pos",
            "src_key_padding_mask",
        }

        unrecognized = set(kwargs.keys()) - allowed_keys
        if unrecognized:
            raise ValueError(f"Unrecognized keyword argument: {unrecognized}")

        if len(args) == 3:
            # Case: model(train_x, train_y, test_x, ...)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            style = kwargs.pop("style", None)
            # Pass everything to _forward
            return self._forward(
                (style, x, args[1]), single_eval_pos=len(args[0]), **kwargs
            )

        elif len(args) == 1 and isinstance(args[0], tuple):
            # Case: model((x,y), ...) or model((style,x,y), ...)
            return self._forward(*args, **kwargs)

        # Fallback for other potential signatures
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        src,
        src_mask=None,
        single_eval_pos=None,
        only_return_standard_out=True,
        src_key_padding_mask: Optional[Tensor] = None,
    ):
        assert isinstance(src, tuple), (
            "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        )

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src

        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]

        x_src = self.encoder(x_src)

        if self.decoder_dict_once is not None:
            x_src = torch.cat(
                [x_src, self.decoder_dict_once_embeddings.repeat(1, x_src.shape[1], 1)],
                dim=0,
            )

        y_src = (
            self.y_encoder(
                y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src
            )
            if y_src is not None
            else None
        )
        if self.style_encoder:
            assert style_src is not None, (
                "style_src must be given if style_encoder is used"
            )
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_src.device)
        global_src = (
            torch.tensor([], device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(
                1, x_src.shape[1], 1
            )
        )

        if src_mask is not None:
            assert self.global_att_embeddings is None or isinstance(src_mask, tuple)

        if src_mask is None:
            if self.global_att_embeddings is None:
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(
                        torch.ones((full_len, full_len), dtype=torch.bool)
                    ).to(x_src.device)
                elif self.efficient_eval_masking:
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(
                        full_len, len(x_src) - single_eval_pos
                    ).to(x_src.device)
            else:
                src_mask_args = (
                    self.global_att_embeddings.num_embeddings,
                    len(x_src) + len(style_src),
                    len(x_src) + len(style_src) - single_eval_pos,
                )
                src_mask = (
                    self.generate_global_att_globaltokens_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_trainset_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                    self.generate_global_att_query_matrix(*src_mask_args).to(
                        x_src.device
                    ),
                )

        train_x = x_src[:single_eval_pos]
        if y_src is not None:
            train_x = train_x + y_src[:single_eval_pos]
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)

        num_prefix_positions = len(style_src) + (
            self.global_att_embeddings.num_embeddings
            if self.global_att_embeddings
            else 0
        )
        if self.return_all_outputs:
            out_range_start = num_prefix_positions
        else:
            out_range_start = single_eval_pos + num_prefix_positions

        # In the line below, we use the indexing feature, that we have `x[i:None] == x[i:]`
        out_range_end = (
            -len(self.decoder_dict_once_embeddings)
            if self.decoder_dict_once is not None
            else None
        )

        # take care the output once are counted from the end
        output_once = (
            {
                k: v(output[-(i + 1)])
                for i, (k, v) in enumerate(self.decoder_dict_once.items())
            }
            if self.decoder_dict_once is not None
            else {}
        )

        output = (
            {
                k: v(output[out_range_start:out_range_end])
                for k, v in self.decoder_dict.items()
            }
            if self.decoder_dict is not None
            else {}
        )

        if only_return_standard_out:
            return output["standard"]

        if output_once:
            return output, output_once
        return output


# Monkeypatch for TransformerEncoderLayer's forward method
def fixed_forward(
    self,
    src: Tensor,
    src_mask: Tensor | Optional[int] | Optional[tuple] = None,
    src_key_padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """This is a monkeypatch for the forward method of TransformerEncoderLayer to allow padding masks for the train set."""
    # 1. Pre-processing Norm
    if self.pre_norm:
        src_ = self.norm1(src)
    else:
        src_ = src

    # --- BRANCH: Custom Integer Logic (Your use case) ---
    if isinstance(src_mask, int):
        single_eval_position = src_mask
        src_to_attend_to = src_[:single_eval_position]

        # Handle Mask Slicing
        # Standard shape for key_padding_mask is [Batch, Seq_Len]
        left_key_mask = None
        right_key_mask = None
        if src_key_padding_mask is not None:
            left_key_mask = src_key_padding_mask[:, :single_eval_position]
            # For src_right, we attend to the left side (the 'training' data)
            right_key_mask = src_key_padding_mask[:, :single_eval_position]

        if self.save_trainingset_representations:
            # ... (keep your existing save logic here) ...
            if single_eval_position == src_.shape[0] or single_eval_position is None:
                self.saved_src_to_attend_to = src_to_attend_to
            elif single_eval_position == 0:
                src_to_attend_to = self.saved_src_to_attend_to
                # If using saved reps, we'd need to have saved the mask too
                # or assume no padding in the saved training set.

        # Attention for the 'Training' part
        src_left = self.self_attn(
            src_[:single_eval_position],
            src_[:single_eval_position],
            src_[:single_eval_position],
            key_padding_mask=left_key_mask,  # <----- THIS IS THE FIX
        )[0]

        # Attention for the 'Test' part (attending to training part)
        src_right = self.self_attn(
            src_[single_eval_position:],
            src_to_attend_to,
            src_to_attend_to,
            # FIXME: check that we won't need this!
            # key_padding_mask=right_key_mask #   we don't need the mask here!
        )[0]

        src2 = torch.cat([src_left, src_right], dim=0)

    # --- BRANCH: Standard or Tuple Logic ---
    elif isinstance(src_mask, tuple):
        # If you ever use the tuple mode, you'd need to slice masks here too.
        # For now, we'll keep it simple to fix your primary issue.
        raise NotImplementedError("Padding mask not yet implemented for Tuple src_mask")
    else:
        # Standard Path
        if self.recompute_attn:
            src2 = checkpoint(
                self.self_attn, src_, src_, src_, src_key_padding_mask, True, src_mask
            )[0]
        else:
            src2 = self.self_attn(
                src_,
                src_,
                src_,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]

    # 2. Residual and MLP Blocks
    src = src + self.dropout1(src2)
    if not self.pre_norm:
        src = self.norm1(src)

    if self.pre_norm:
        src_ = self.norm2(src)
    else:
        src_ = src

    src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
    src = src + self.dropout2(src2)

    if not self.pre_norm:
        src = self.norm2(src)
    return src


def ft_pfn_padding():
    """load the frozen PFN model and patch it to support padding masks"""
    load_dotenv(dotenv_path=Path(__file__).parents[4] / ".env")

    model_path = os.getenv("MODELDIR") + "pfn_ckpt"
    assert Path(model_path).exists(), f"Model path {model_path} does not exist."

    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device="cpu"
    ).model

    frozen_model.__class__ = PaddableTransformerModel

    # Inject the fixed forward into every encoder layer
    for layer in frozen_model.transformer_encoder.layers:
        # We use types.MethodType to bind the function to the specific instance
        layer.forward = types.MethodType(fixed_forward, layer)

    return frozen_model


if __name__ == "__main__":
    # FIXME: check that the CrossFusion layer can deal with padding properly!

    frozen_model = ft_pfn_padding()
    print("Model loaded and patched successfully.")

    import torch

    def generate_dummy_data(batch_size=4, seq_len=10, feature_dim=32, eval_pos=7):
        # 1. Create src: [S, B, D]
        # First dim of D: integers 0-999 (scaled or raw, usually PFNs expect raw or normalized)
        # Subsequent dims: 0 or 1
        X = torch.zeros((seq_len, batch_size, feature_dim))

        # Fill first feature dimension with ints 0-999
        # (using random ints for the dummy example)
        X[:, :, 0] = torch.randint(0, 1000, (seq_len, batch_size)).float()

        # Fill remaining feature dimensions with 0 or 1
        X[:, :, 1:] = torch.randint(
            0, 2, (seq_len, batch_size, feature_dim - 1)
        ).float()

        y = torch.rand(seq_len, batch_size)  # Dummy target values if needed

        # 2. Create src_key_padding_mask: [B, S]
        # Let's mask the last 2 elements of every batch to see if it works
        # True = Masked (padded), False = Not masked
        padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        padding_mask[:, int(seq_len / 2) : eval_pos] = True

        return (X, y), padding_mask

    # Parameters
    S, B, D = 100, 4, 4
    eval_pos = 80  # The 'int' src_mask you use

    src, padding_mask = generate_dummy_data(B, S, D, eval_pos)

    # Assuming 'frozen_model' is the model returned by your ft_pfn()
    frozen_model.eval()

    try:
        with torch.no_grad():
            output = frozen_model(
                src, src_mask=eval_pos, src_key_padding_mask=padding_mask
            )
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")  # Should be [S, B, D]
    except Exception as e:
        print(f"Forward pass failed: {e}")
