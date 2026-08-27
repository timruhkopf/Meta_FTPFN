def test_ft_pfn_initialization(get_ft_pfn):
    """Test if the ft_pfn fixture initializes correctly"""
    ft_pfn = get_ft_pfn
    assert ft_pfn is not None


def test_ft_pfn_forward_pass(get_ft_pfn, ft_batch_factory):
    """Test a forward pass through the ft_pfn model"""
    (x, y), T_split, _ = ft_batch_factory(T=32, B=9, D=5, Tsplit=25)

    pfn = get_ft_pfn
    output = pfn((x, y), single_eval_pos=T_split)
    assert output is not None
    assert output.shape[0] == x.shape[0] - T_split
