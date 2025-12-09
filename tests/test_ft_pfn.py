import pytest

def test_ft_pfn_initialization(ft_pfn):
    """Test if the ft_pfn fixture initializes correctly"""
    assert ft_pfn is not None

def test_ft_pfn_forward_pass(ft_pfn, dummy_ft_batch):
    """Test a forward pass through the ft_pfn model"""
    x, T_split = dummy_ft_batch

    pfn, _ = ft_pfn
    output = pfn(x, single_eval_pos=T_split)
    assert output is not None
    assert output.shape[0] == x[0].shape[0] - T_split