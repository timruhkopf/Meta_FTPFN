import pytest
import torch
import torch.nn as nn
from ppfn.model.mymodel.cross_fusion import CrossFusion

from ppfn.model.mymodel.interleaved_model import HierarchicalPFN



@pytest.fixture
def model_wrapped(get_ft_pfn):
    """
    Fixture that wraps the real frozen model with CrossFusion layers.
    Note: 'ft_pfn' is assumed to be provided by your existing test suite.
    """
    ft_pfn = get_ft_pfn()
    interleaved_layers = {
        "transformer_encoder.layers.0.linear1": CrossFusion(d_model=512, num_heads=8),
        "transformer_encoder.layers.2.linear1": CrossFusion(d_model=512, num_heads=8),
    }
    return HierarchicalPFN(frozen_model=ft_pfn, interleaved_layers=interleaved_layers)

def test_integration_constancy_eval(model_wrapped,get_ft_pfn, ft_batch_factory):
    """
    Checks if Stream A and B survive the ENTIRE HierarchicalPFN pass 
    completely unchanged during evaluation.
    """
    
    (x, y), single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
    # In eval mode: B=9 -> R = (9-1)//2 = 4
    # Stream A: [:, 0, :]
    # Stream B: [:, 1:5, :]
    # Stream C: [:, 5:, :]
    ft_pfn = get_ft_pfn()
    ft_pfn.eval()
    with torch.no_grad():
        frozen_out = ft_pfn((x, y), single_eval_pos=single_eval_pos)
    
    model_wrapped.eval()
    with torch.no_grad():

        
        # Run full Hierarchical PFN pass
        # The output of the PFN is usually a tensor (logits or embeddings)
        output = model_wrapped((x, y), single_eval_pos=single_eval_pos)
        
        # Check if output is a tuple or tensor depending on your PFN head
        out_x = output[0] if isinstance(output, tuple) else output

    # CRITICAL TEST: 
    # Even after many layers of PFN, the first 1+R indices of the batch 
    # should be identical to the input because CrossFusion shields them.
    R = (x.shape[1] - 1) // 2
    
    # Check Stream A
    torch.testing.assert_close(out_x[:, :1, :], frozen_out[:, :1, :], 
                               msg="Target Task (Stream A) was corrupted by the PFN pass.")
    
    # Check Stream B
    torch.testing.assert_close(out_x[:, 1:R+1, :], frozen_out[:, 1:R+1, :], 
                               msg="Related Marginals (Stream B) were corrupted by the PFN pass.")
    
    # Check Stream C (Should be different)
    assert not torch.allclose(out_x[:, R+1:, :], frozen_out[:, R+1:, :]), \
        "Workspace (Stream C) was not updated by the PFN pass."

def test_integration_constancy_train(model_wrapped, get_ft_pfn, ft_batch_factory):
    """Checks for constancy in 3R training mode."""
    (x, y), single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
    # Batch is 9, so R=3 for training [A:3, B:3, C:3]
    R = x.shape[1] // 3

    ft_pfn = get_ft_pfn()
    ft_pfn.train()
    output_frozen = ft_pfn((x, y), single_eval_pos=single_eval_pos)
    
    model_wrapped.train()
   
    
    output = model_wrapped((x, y), single_eval_pos=single_eval_pos)
    out_x = output[0] if isinstance(output, tuple) else output

    # Check Streams A and B (indices 0 to 2*R)
    torch.testing.assert_close(out_x[:, :2*R, :], output_frozen[:, :2*R, :],
                               msg="Streams A or B corrupted during training forward pass.")

def test_single_eval_pos_reset(model_wrapped, ft_batch_factory):
    """Ensures that single_eval_pos is cleaned up to prevent side effects."""
    batch_data, single_eval_pos = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)
    
    model_wrapped.eval()
    _ = model_wrapped(batch_data, single_eval_pos=single_eval_pos)
    
    assert model_wrapped.single_eval_pos is None
    for layer in model_wrapped.interleaved_layers.values():
        assert layer.single_eval_pos is None

def test_pfn_param_freezing(model_wrapped):
    """Verify that only CrossFusion layers have gradients."""

    # Check an arbitrary frozen weight
    intercepted = model_wrapped.interleaved_layers.keys()
    for name, param in model_wrapped.frozen_model.named_parameters():
        if any(name.startswith(n) for n in intercepted): 
            continue

        assert not param.requires_grad, f"Parameter {name} was not frozen!"
    
    # Check CrossFusion weight
    for layer in model_wrapped.interleaved_layers.values():
        for param in layer.parameters():
            assert param.requires_grad, "CrossFusion layer was accidentally frozen!"