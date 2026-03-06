import pytest
import torch

from ppfn.model.mymodel.stream_mutations import AppendATrainToBTestMutation, ForceSameQueryMutation


# Assuming the mutations are imported from your module
# from your_module import ForceSameQueryMutation, AppendATrainToBTestMutation

# --- Mock Dependencies / Fixtures ---

@pytest.fixture
def sample_streams():
    """
    Creates distinctly identifiable A, B, and C streams.
    Sequence Length = 10, Batch Size = 2, Feature Dim = 4.
    A values = 1.0
    B values = 2.0
    C values = 3.0
    """
    seq_len, b_dim, d_model = 10, 2, 4

    # We set initial mask to True (padded) to verify the appended mask is False (unpadded)
    def create_stream(val):
        x = torch.full((seq_len, b_dim, d_model), val)
        y = torch.full((seq_len, b_dim), val)
        mask = torch.ones((seq_len, b_dim), dtype=torch.bool)
        hp = torch.full((seq_len, b_dim, 3), val)
        return (x, y, mask, hp)

    return {
        "A": create_stream(1.0),
        "B": create_stream(2.0),
        "C": create_stream(3.0),
    }


# --- Test Cases ---

def test_force_same_query_mutation(sample_streams):
    """
    Intent: Verify that the mutation copies Task A's query/test features and
    labels into Task B's stream starting from `sep`. Crucially, ensures the
    cloning logic prevents in-place mutation of the original tensors.
    """
    mutation = ForceSameQueryMutation()
    sep = 4

    # Store a reference to original B to verify it isn't modified in place
    orig_B_x = sample_streams["B"][0]

    new_streams = mutation(sample_streams, sep)

    A_x, A_y, _, _ = new_streams["A"]
    B_x, B_y, _, _ = new_streams["B"]
    C_x, _, _, _ = new_streams["C"]

    # 1. Check B's train part remains unchanged (should still be 2.0)
    assert torch.all(B_x[:sep] == 2.0)
    assert torch.all(B_y[:sep] == 2.0)

    # 2. Check B's test part now perfectly matches A's test part (should be 1.0)
    assert torch.all(B_x[sep:] == 1.0)
    assert torch.all(B_y[sep:] == 1.0)

    # 3. Ensure C is completely untouched (should still be 3.0)
    assert torch.all(C_x == 3.0)

    # 4. Verify original tensor was NOT mutated (the `.clone()` fix works)
    assert torch.all(orig_B_x == 2.0)


def test_append_a_train_mutation_forward(sample_streams):
    """
    Intent: Verify that A_train (the first `sep` elements of Stream A) is
    correctly appended to the sequence dimension of all streams. Also checks
    that the padding mask is extended with `False` (unpadded) for these new tokens.
    """
    mutation = AppendATrainToBTestMutation()
    sep = 4
    orig_seq_len = 10

    new_streams = mutation(sample_streams, sep)

    for key in ["A", "B", "C"]:
        x, y, mask, hp = new_streams[key]

        # 1. Verify new sequence length
        assert x.shape[0] == orig_seq_len + sep
        assert y.shape[0] == orig_seq_len + sep
        assert mask.shape[0] == orig_seq_len + sep
        assert hp.shape[0] == orig_seq_len + sep

        # 2. Verify the appended data is strictly A_train (which has value 1.0)
        assert torch.all(x[orig_seq_len:] == 1.0)
        assert torch.all(y[orig_seq_len:] == 1.0)
        assert torch.all(hp[orig_seq_len:] == 1.0)

        # 3. Verify mask extension logic
        # Original mask was initialized to True. Appended mask must be False.
        assert torch.all(mask[:orig_seq_len] == True)
        assert torch.all(mask[orig_seq_len:] == False)


def test_append_a_train_mutation_handles_nones():
    """
    Intent: Ensure the mutation does not crash if the padding mask or HP
    coordinates are None (which can happen during certain evaluation or unpadded phases).
    """
    mutation = AppendATrainToBTestMutation()
    sep = 3

    # Create streams with None for mask and hp
    streams = {
        "A": (torch.zeros((10, 2, 4)), torch.zeros((10, 2)), None, None),
        "B": (torch.zeros((10, 2, 4)), torch.zeros((10, 2)), None, None),
        "C": (torch.zeros((10, 2, 4)), torch.zeros((10, 2)), None, None),
    }

    new_streams = mutation(streams, sep)

    for key, (x, y, mask, hp) in new_streams.items():
        assert x.shape[0] == 13
        assert mask is None
        assert hp is None


def test_append_a_train_mutation_splice_end():
    """
    Intent: Verify that `splice_at_fwd_end` successfully removes the exact appended
    `A_train` segment from the output logits, returning the tensor to its original
    sequence length so the objective function receives aligned dimensions.
    """
    mutation = AppendATrainToBTestMutation()
    sep = 4
    orig_seq_len = 10
    mutated_seq_len = orig_seq_len + sep

    # Mock model outputs (sequence length already extended to 14)
    output_streams = {
        "A": torch.zeros((mutated_seq_len, 2, 8)),
        "B": torch.ones((mutated_seq_len, 2, 8)),
        "C": torch.full((mutated_seq_len, 2, 8), 2.0)
    }

    spliced_outputs = mutation.splice_at_fwd_end(output_streams, sep)

    for key, logits in spliced_outputs.items():
        # Check that the last `sep` elements were removed
        assert logits.shape[0] == orig_seq_len