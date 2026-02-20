import pytest
import torch
import types

from ppfn.model.baselines.ft_pfn_padding import ft_pfn_padding, PaddableTransformerModel


def generate_dummy_data(batch_size=4, seq_len=10, feature_dim=32, eval_pos=7):
    """Utility to generate PFN-compatible tensors."""
    X = torch.zeros((seq_len, batch_size, feature_dim))
    X[:, :, 0] = torch.randint(0, 1000, (seq_len, batch_size)).float()
    X[:, :, 1:] = torch.randint(0, 2, (seq_len, batch_size, feature_dim - 1)).float()
    y = torch.rand(seq_len, batch_size)

    # Simple zero-mask for the return value
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    return (X, y), padding_mask


def test_model_loading_and_class_injection():
    """Checks if class inheritance and monkeypatching were applied successfully."""
    model = ft_pfn_padding()
    assert isinstance(model, PaddableTransformerModel)

    # Ensure every layer in the encoder has the bound monkeypatched method
    for layer in model.transformer_encoder.layers:
        assert isinstance(layer.forward, types.MethodType), (
            "Forward method not monkeypatched"
        )


def test_forward_pass_with_padding():
    """Verifies the model returns test predictions for a single sequence split by eval_pos."""
    model = ft_pfn_padding()
    model.eval()

    # Define a single sequence where first 40 are train, last 10 are test
    S, B, D = 50, 4, 4
    eval_pos = 40

    # Generate one single block of data
    (X, y), _ = generate_dummy_data(B, S, D, eval_pos)

    # Create a mask for the first 5 elements of the sequence
    padding_mask = torch.zeros((B, S), dtype=torch.bool)
    padding_mask[:, :5] = True

    with torch.no_grad():
        # Passing (X, y) as a tuple requires single_eval_pos to define the split
        output = model(
            (X, y), single_eval_pos=eval_pos, src_key_padding_mask=padding_mask
        )

    # Expected output length is S - eval_pos (50 - 40 = 10)
    expected_test_len = S - eval_pos
    assert output.shape[0] == expected_test_len, (
        f"Expected {expected_test_len} outputs, got {output.shape[0]}"
    )
    assert output.shape[1] == B, "Batch size mismatch"


def test_padding_mask_impact_on_output():
    """Ensures that changing the padding mask on the training prefix changes the test predictions."""
    model = ft_pfn_padding()
    model.eval()

    # Total length 20: 15 train, 5 test
    S, B, D = 20, 2, 4
    eval_pos = 15
    (X, y), _ = generate_dummy_data(B, S, D, eval_pos)

    # Case 1: No masking at all
    # Case 2: Mask the first 10 training points
    mask_padded = torch.zeros((B, S), dtype=torch.bool)
    mask_padded[:, :10] = True

    with torch.no_grad():
        # Both calls use the exact same X and y, only the mask differs
        output_no_mask = model(
            (X, y),
            single_eval_pos=eval_pos,
        )
        output_with_mask = model(
            (X, y), single_eval_pos=eval_pos, src_key_padding_mask=mask_padded
        )

    # If the monkeypatch is working, the attention mechanism ignores the masked 10 points,
    # leading to different hidden states for the remaining 5 test points.
    assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-7), (
        "The padding mask was ignored! Outputs were identical despite masking training data."
    )


def test_shape_with_tuple_input():
    """Tests the 1-arg tuple signature: model((x, y), single_eval_pos=...)."""
    model = ft_pfn_padding()
    model.eval()

    S, B, D = 20, 2, 4
    eval_pos = 15  # First 15 are 'train', last 5 are 'test'
    (X, y), _ = generate_dummy_data(B, S, D)

    with torch.no_grad():
        output = model((X, y), single_eval_pos=eval_pos)

    assert output.shape[0] == (S - eval_pos)


def test_invalid_kwargs_rejection():
    """Verify standard PFN error handling for unknown kwargs."""
    model = ft_pfn_padding()
    (X, y), _ = generate_dummy_data(2, 10, 4)

    with pytest.raises(ValueError, match="Unrecognized keyword argument"):
        model((X, y), invalid_param="should_fail")
