import pytest
import torch

from ppfn.model.mymodel.interleavedmodel import PPFN
from ppfn.model.mymodel.crossfusion import CrossFusion


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


def test_eval_mode(ft_pfn, dummy_ft_batch):
    """This test demonstrates the way evaluation is supposed to happen."""

    x, T_split = dummy_ft_batch

    T, B, D = x[0].shape

    # Notice: the first task in the batch is the target task during evaluation; 
    # all others are assumed to be the support tasks.

    pfn, _ = ft_pfn

    # collect unhooked execution for comparison with eval mode for same dropout behaviour
    pfn.eval()
    with torch.no_grad():
        pfn_output_eval = pfn(x, single_eval_pos=T_split)
    pfn.train()

    cross_fusion_layer = CrossFusion(d_model=pfn.ninp, num_heads=2, dropout=0.5)
    meta_model = PPFN(
        frozen_model=pfn,
        interleaved_layers={"transformer_encoder.layers.0": cross_fusion_layer},
        pre_hook=True,
    )

    meta_model.eval()

    # communicating state to connector worked (important for target in batch) 
    connector = cross_fusion_layer.connector
    assert cross_fusion_layer.training is False
    assert connector.training is False

    # for cross attention in eval mode, the target task is the first one in the batch and needs to be expanded
    # notice, that for this assert, we use the [xtrain, xtest] --> regular fwd would have ytrain in that tensor as well
    assert torch.equal(
        connector.create_target_in_batch(x[0]), x[0][:, :1, :].repeat(1, B , 1)
        # TODO Notice: the first pairing is with itself, so we can igonre it during inference!!
    )

    with torch.no_grad():
        output_eval = meta_model(x, single_eval_pos=T_split)
        # ouput: (T - T_split, B * 2, 1000)
        # notice: the first B batch items are the old ones and should correspond to the frozen model output
        # the last B batch items are basically \tau^1 | \tau^i pairings and we should drop (1,1) during inference!
    
    # The first B items are identical to the frozen model output over the batch (we can compare against especially the first item, which is the marginal/unconditional of the the target!)
    assert torch.allclose(
        output_eval[:, :B, :], pfn_output_eval, rtol=1e-6, atol=1e-6
    ), "The 'old' part of the output does not match the frozen model output in eval mode."
    
    
    
