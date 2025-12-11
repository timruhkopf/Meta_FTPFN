import pytest
import torch

from ppfn.model.ppfn.ppfn import PPFN
from ppfn.model.ppfn.crossfusion import CrossFusion


def test_PPFN_CrossFusion_single_interleaved(ft_pfn, dummy_ft_batch):
    """Test cross fusion functionality in ft_pfn model"""
    x, T_split = dummy_ft_batch

    pfn, _ = ft_pfn

    # make sure to get the output before registering the hooks to the model!
    naive_fwd = pfn(x, single_eval_pos=T_split)

    # the structure of the pfn is presented here for clarity
    print(pfn)

    # the available layers to interleave into are:
    for name, module in pfn.named_modules():
        print(name, "->", module.__class__.__name__)

    meta_model = PPFN(
        frozen_model=pfn,
        interleaved_layers={
            "transformer_encoder.layers.0": CrossFusion(
                d_model=pfn.ninp, num_heads=2, dropout=0.0
            )
        },
        pre_hook=True,
    )

    output = meta_model(x, single_eval_pos=T_split)
    # FIXME: this is not the correct shape to be expected!
    T, B, D = x[0].shape
    assert output.shape == (T - T_split, 2 * B, 1000), (
        f"Unexpected output shape: {output.shape}, expected {(T - T_split, 2 * B, 1000)} due to test output and batch doubling in cross fusion."
    )

    assert torch.allclose(output[:, :B, :], naive_fwd, rtol=1e-6, atol=1e-6), (
        'The "old" part of the output does not match the frozen model output.'
    )

    assert not torch.allclose(
        output[:, :B, :], output[:, B:, :].roll(-1, dims=1), rtol=1e-6, atol=1e-6
    ), (
        "The 'new' part of the output should differ from the 'old' part due to cross attention."
    )


def test_PFN_CrosssFusion_multiple_interleaved(ft_pfn, dummy_ft_batch):
    """Test multiple cross fusion layers in ft_pfn model"""
    x, T_split = dummy_ft_batch

    pfn, _ = ft_pfn

    # make sure to get the output before registering the hooks to the model!
    naive_fwd = pfn(x, single_eval_pos=T_split)

    meta_model = PPFN(
        frozen_model=pfn,
        interleaved_layers={
            "transformer_encoder.layers.0": CrossFusion(
                d_model=pfn.ninp, num_heads=2, dropout=0.0
            ),
            "transformer_encoder.layers.2": CrossFusion(
                d_model=pfn.ninp, num_heads=2, dropout=0.0
            ),
        },
        pre_hook=True,
    )

    output = meta_model(x, single_eval_pos=T_split)
    T, B, D = x[0].shape
    assert output.shape == (T - T_split, 2 * B, 1000), (
        f"Unexpected output shape: {output.shape}, expected {(T - T_split, 2 * B, 1000)} due to test output and batch doubling in cross fusion."
    )

    assert torch.allclose(output[:, :B, :], naive_fwd, rtol=1e-6, atol=1e-6), (
        'The "old" part of the output does not match the frozen model output.'
    )
