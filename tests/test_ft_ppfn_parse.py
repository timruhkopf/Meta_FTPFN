import torch
import pytest
from types import SimpleNamespace

from ppfn.model.mymodel.ft_ppfn import FT_PPFN
import ppfn.model.mymodel.ft_ppfn as ft_ppfn_mod
from ppfn.utils.mybatch import MyBatch


class DummyFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_src_key_padding_mask = None

    def forward(self, src, **kwargs):
        # record mask if provided
        self.last_src_key_padding_mask = kwargs.get("src_key_padding_mask", None)
        # return a trivial tensor shaped [test_len, batch]
        if len(src) == 3:
            _, x_src, _ = src
        else:
            x_src, _ = src
        S, B, D = x_src.shape
        single_eval_pos = kwargs.get("single_eval_pos", S)
        test_len = max(0, S - single_eval_pos)
        return torch.zeros((test_len, B))


def make_mybatch(S=20, B=6, D=4, eval_pos=12, device=torch.device("cpu")):
    x = torch.randn((S, B, D), device=device)
    y = torch.randn((S, B), device=device)
    return MyBatch(x=x, y=y, target_y=y, style=None, src_key_padding_mask=None, single_eval_pos=eval_pos)


def test_parse_eval_batch_masks_shape_and_batch_joining():
    S, B, D = 24, 5, 3
    eval_pos = 16
    batch = make_mybatch(S=S, B=B, D=D, eval_pos=eval_pos)

    model = FT_PPFN(frozen_model=DummyFrozen(), interleaved_layers={})
    model.eval()

    # create mask shape [B, S]
    mask = torch.zeros((B, S), dtype=torch.bool)
    mask[:, :3] = True

    parsed_batch, parsed_mask = model.parse_eval_batch(batch, eval_pos, src_key_padding_mask=mask)

    # parsed batch should have 3 streams concatenated along batch dim: R + (B-1) + R => 3*R
    R = B - 1
    assert parsed_batch.x.shape[1] == 3 * R
    # parsed mask rows should match that concatenation
    assert parsed_mask.shape == (3 * R, S)


def test_parse_train_batch_masks_shape_and_batch_joining():
    S, B, D = 20, 6, 3  # B even to allow ::2 slicing
    eval_pos = 10
    batch = make_mybatch(S=S, B=B, D=D, eval_pos=eval_pos)

    model = FT_PPFN(frozen_model=DummyFrozen(), interleaved_layers={})
    model.train()  # ensure training branch

    mask = torch.zeros((B, S), dtype=torch.bool)
    mask[:, :2] = True

    parsed_batch, parsed_mask = model.parse_train_batch(batch, eval_pos, src_key_padding_mask=mask)

    # For train parsing, batch is split into two streams by ::2 and 1::2, then concatenated as three streams
    # original even-indexed count = ceil(B/2) and odd-indexed count = floor(B/2)
    even_count = (B + 1) // 2
    odd_count = B // 2
    expected_cols = even_count + odd_count + even_count
    assert parsed_batch.x.shape[1] == expected_cols
    assert parsed_mask.shape == (expected_cols, S)


def test_forward_passes_mask_to_paddable(monkeypatch):
    # Make FT_PPFN treat DummyFrozen as the PaddableTransformerModel
    monkeypatch.setattr(ft_ppfn_mod, "PaddableTransformerModel", DummyFrozen)

    S, B, D = 18, 4, 5
    eval_pos = 12
    dummy = DummyFrozen()
    model = FT_PPFN(frozen_model=dummy, interleaved_layers={})
    model.eval()

    batch = make_mybatch(S=S, B=B, D=D, eval_pos=eval_pos)
    mask = torch.zeros((B, S), dtype=torch.bool)
    mask[:, :4] = True

    _ = model(batch, src_key_padding_mask=mask)

    # Dummy should have received a mask (after parse_eval_batch concatenation)
    assert dummy.last_src_key_padding_mask is not None


def test_forward_ignores_mask_for_non_paddable(monkeypatch):
    # Ensure the PaddableTransformerModel symbol refers to some other class so isinstance check fails
    monkeypatch.setattr(ft_ppfn_mod, "PaddableTransformerModel", torch.nn.Linear)

    S, B, D = 18, 4, 5
    eval_pos = 12
    dummy = DummyFrozen()
    model = FT_PPFN(frozen_model=dummy, interleaved_layers={})
    model.eval()

    batch = make_mybatch(S=S, B=B, D=D, eval_pos=eval_pos)
    mask = torch.zeros((B, S), dtype=torch.bool)
    mask[:, :4] = True

    _ = model(batch, src_key_padding_mask=mask)

    # Dummy should NOT receive a parsed mask because FT_PPFN avoids passing it when frozen_model isn't Paddable
    assert dummy.last_src_key_padding_mask is None

