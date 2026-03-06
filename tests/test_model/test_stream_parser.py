import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from ppfn.model.mymodel.stream_parser import StreamParser


# --- Mock Dependencies ---

@dataclass
class MyBatch:
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor = None
    single_eval_pos: int = 5


# Assuming StreamParser is imported from your module
# from your_module import StreamParser

# --- Test Setup ---

@pytest.fixture
def parser():
    """Provides a fresh StreamParser instance for each test."""
    return StreamParser()


@pytest.fixture
def train_batch():
    """
    Creates an interleaved batch of 4 examples (2 targets, 2 related).
    Sequence Length = 10, Batch Size = 4, Features = 8
    Values are distinctly initialized to track their routing:
    Batch 0 (Target 0): All 0s
    Batch 1 (Related 0): All 1s
    Batch 2 (Target 1): All 2s
    Batch 3 (Related 1): All 3s
    """
    seq_len, b_dim, d_model = 10, 4, 8
    x = torch.zeros((seq_len, b_dim, d_model))
    x[:, 1, :] = 1.0
    x[:, 2, :] = 2.0
    x[:, 3, :] = 3.0

    y = torch.zeros((seq_len, b_dim))
    y[:, 1] = 1.0
    y[:, 2] = 2.0
    y[:, 3] = 3.0

    return MyBatch(x=x, y=y, single_eval_pos=5)


@pytest.fixture
def eval_batch():
    """
    Creates an eval batch of 4 examples (1 target, 3 related).
    Batch 0 (Target): All 0s
    Batch 1 (Related 1): All 1s
    Batch 2 (Related 2): All 2s
    Batch 3 (Related 3): All 3s
    """
    seq_len, b_dim, d_model = 10, 4, 8
    x = torch.zeros((seq_len, b_dim, d_model))
    for i in range(1, b_dim):
        x[:, i, :] = float(i)

    y = torch.zeros((seq_len, b_dim))
    for i in range(1, b_dim):
        y[:, i] = float(i)

    return MyBatch(x=x, y=y, single_eval_pos=5)


# --- Test Cases ---

def test_get_raw_streams_train(parser, train_batch):
    """
    Intent: Verify that during training mode, the parser correctly steps through
    the interleaved format (target0/related0/target1/related1), assigning even
    indices to A & C, and odd indices to B.
    """
    parser.train()
    streams = parser.get_raw_streams(train_batch)

    A_x, A_y, A_mask, A_hp = streams["A"]
    B_x, B_y, B_mask, B_hp = streams["B"]
    C_x, C_y, C_mask, C_hp = streams["C"]

    # Check shapes (Batch dim should now be 2)
    assert A_x.shape == (10, 2, 8)
    assert B_x.shape == (10, 2, 8)

    # Verify data routing based on our distinct fixture values
    assert torch.all(A_x[:, 0, :] == 0.0)  # Target 0
    assert torch.all(A_x[:, 1, :] == 2.0)  # Target 1

    assert torch.all(B_x[:, 0, :] == 1.0)  # Related 0
    assert torch.all(B_x[:, 1, :] == 3.0)  # Related 1

    # Stream C must be a perfect copy of Stream A
    assert torch.equal(A_x, C_x)
    assert torch.equal(A_y, C_y)


def test_get_raw_streams_eval(parser, eval_batch):
    """
    Intent: Verify that during eval mode, the single target at index 0 is
    correctly expanded to match the dimension R of the related tasks, and
    related tasks are sliced from index 1 to the end.
    """
    parser.eval()
    streams = parser.get_raw_streams(eval_batch)

    A_x, A_y, _, _ = streams["A"]
    B_x, B_y, _, _ = streams["B"]

    R = eval_batch.x.shape[1] - 1  # Should be 3

    # Check expanded shapes
    assert A_x.shape == (10, R, 8)
    assert B_x.shape == (10, R, 8)

    # A should be all 0s (expanded target)
    assert torch.all(A_x == 0.0)

    # B should contain the related examples [1.0, 2.0, 3.0]
    assert torch.all(B_x[:, 0, :] == 1.0)
    assert torch.all(B_x[:, 2, :] == 3.0)


def test_assemble_batch_concatenation(parser, train_batch):
    """
    Intent: Verify that `assemble_batch` successfully concatenates A, B, and C
    along the batch dimension (dim=1) and produces the correct output dataclass
    and HP tuple format.
    """
    parser.train()

    # Inject a dummy HP tensor to test HP tuple routing
    dummy_hp = torch.rand(10, 4, 3)
    streams = parser.get_raw_streams(train_batch, hp=dummy_hp)

    new_batch, final_mask, hp_tuple = parser.assemble_batch(streams, train_batch.single_eval_pos)

    # Original batch size was 4 -> R=2. Assembled batch should be 3 * R = 6
    assert new_batch.x.shape[1] == 6
    assert new_batch.y.shape[1] == 6

    # Ensure correct concatenated ordering [A, B, C]
    # Indices 0,1 are A (0s, 2s). Indices 2,3 are B (1s, 3s). Indices 4,5 are C (0s, 2s).
    assert torch.all(new_batch.x[:, 0, :] == 0.0)
    assert torch.all(new_batch.x[:, 2, :] == 1.0)
    assert torch.all(new_batch.x[:, 4, :] == 0.0)

    # Verify HP tuple unpacking
    assert hp_tuple is not None
    assert len(hp_tuple) == 3
    assert hp_tuple[0].shape == (10, 2, 3)  # A_hp


def test_output_parsing_and_assembly(parser):
    """
    Intent: Verify that the parser correctly splits an output tensor back into
    A, B, C streams by dividing the batch dimension by 3, and can safely reassemble it.
    """
    seq_len, total_b, d_model = 10, 9, 8  # total_b must be a multiple of 3 (R=3)

    # Create distinct dummy outputs
    output = torch.zeros((seq_len, total_b, d_model))
    output[:, 0:3, :] = 1.0  # A
    output[:, 3:6, :] = 2.0  # B
    output[:, 6:9, :] = 3.0  # C

    # Parse
    o_streams = parser.parse_output_streams(output, sep=5)

    assert torch.all(o_streams["A"] == 1.0)
    assert torch.all(o_streams["B"] == 2.0)
    assert torch.all(o_streams["C"] == 3.0)

    # Assemble
    reassembled_output = parser.assemble_output_streams(o_streams)
    assert torch.equal(output, reassembled_output)


def test_forward_pass_with_masks(parser, train_batch):
    """
    Intent: Ensure the full `forward` method safely passes through padding masks
    without dimension mismatch errors.
    """
    parser.train()

    # Create a dummy mask of shape (Batch, Seq) -> (4, 10)
    # Target 0 padded at end, Related 0 not padded
    dummy_mask = torch.zeros((4, 10), dtype=torch.bool)
    dummy_mask[0, 8:] = True

    new_batch, final_mask, hp_tuple = parser(train_batch, hp=None, src_key_padding_mask=dummy_mask)

    # Assembled mask should stack A, B, C along dim 0. Shape: (6, 10)
    assert final_mask is not None
    assert final_mask.shape == (6, 10)

    # Check if the mask routed correctly to Stream A (Index 0 in concatenated mask)
    assert torch.all(final_mask[0, 8:] == True)
    # Check if mask routed correctly to Stream B (Index 2 in concatenated mask)
    assert torch.all(final_mask[2, 8:] == False)


def test_hp_routing_train_values(parser, train_batch):
    """
    Intent: Verify that the hyperparameter (HP) tensor values are correctly
    sliced and routed during training. This ensures that the target coordinates
    never accidentally leak into the related stream's coordinates.
    """
    parser.train()

    # Create a distinct HP tensor: Shape (seq_len=10, batch_size=4, hp_dim=3)
    hp = torch.zeros((10, 4, 3))
    hp[:, 0, :] = 0.0  # Target 0
    hp[:, 1, :] = 1.0  # Related 0
    hp[:, 2, :] = 2.0  # Target 1
    hp[:, 3, :] = 3.0  # Related 1

    # Parse streams
    streams = parser.get_raw_streams(train_batch, hp=hp)
    A_hp, B_hp, C_hp = streams["A"][3], streams["B"][3], streams["C"][3]

    # Verify A_hp contains Target 0 and Target 1
    assert torch.all(A_hp[:, 0, :] == 0.0)
    assert torch.all(A_hp[:, 1, :] == 2.0)

    # Verify B_hp contains Related 0 and Related 1
    assert torch.all(B_hp[:, 0, :] == 1.0)
    assert torch.all(B_hp[:, 1, :] == 3.0)

    # Verify C_hp perfectly matches A_hp
    assert torch.equal(A_hp, C_hp)

    # Verify assembly packs them correctly into the tuple
    _, _, hp_tuple = parser.assemble_batch(streams, train_batch.single_eval_pos)
    assert torch.equal(hp_tuple[0], A_hp)
    assert torch.equal(hp_tuple[1], B_hp)
    assert torch.equal(hp_tuple[2], C_hp)


def test_hp_routing_eval_values(parser, eval_batch):
    """
    Intent: Verify that during evaluation, the HP tensor for the single target task
    is correctly expanded to match the number of related tasks (R), and the related
    HPs are properly sliced.
    """
    parser.eval()

    # Create a distinct HP tensor for Eval: Shape (seq_len=10, batch_size=4, hp_dim=3)
    hp = torch.zeros((10, 4, 3))
    hp[:, 0, :] = 0.0  # Target
    hp[:, 1, :] = 1.0  # Related 1
    hp[:, 2, :] = 2.0  # Related 2
    hp[:, 3, :] = 3.0  # Related 3

    streams = parser.get_raw_streams(eval_batch, hp=hp)
    A_hp, B_hp, C_hp = streams["A"][3], streams["B"][3], streams["C"][3]

    # In eval, R = 3. Stream A's HP should be the Target (0.0) expanded 3 times.
    assert A_hp.shape == (10, 3, 3)
    assert torch.all(A_hp[:, 0, :] == 0.0)
    assert torch.all(A_hp[:, 1, :] == 0.0)
    assert torch.all(A_hp[:, 2, :] == 0.0)

    # Stream B's HP should strictly contain the related tasks
    assert B_hp.shape == (10, 3, 3)
    assert torch.all(B_hp[:, 0, :] == 1.0)
    assert torch.all(B_hp[:, 1, :] == 2.0)
    assert torch.all(B_hp[:, 2, :] == 3.0)

    # Verify C_hp perfectly matches the expanded A_hp
    assert torch.equal(A_hp, C_hp)

    # Verify assembly packs them correctly into