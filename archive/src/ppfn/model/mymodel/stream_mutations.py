import torch
from torch import nn

class AbstractStreamMutation(nn.Module):
    """
    Base class for stream mutations. Each mutation should implement the `__call__` method,
    which takes the current streams and the separation index, and returns the mutated streams.

    nn.Module inheritance is used to allow these mutations to be registered as submodules of the containing model,
    enabling proper (training) state tracking and device management.
    """
    def __init__(self):
        super().__init__()

    def forward(self, streams, sep):
        raise NotImplementedError("Stream mutations must implement the __call__ method.")

    def splice_at_fwd_end(self, output_streams, batch):
        """ Optional method to apply mutations at the end of the forward pass instead of the beginning.
        Notice, that any change to the batch (incl. e.g. appending A_train also changed the labels contained in the batch object)"""
        return output_streams  # By default, do nothing. Override if needed.



class ForceSameQueryMutation(AbstractStreamMutation):
    """
    Mutation to force the Related stream (B) to have the same query positions as the Target stream (A).
    This is done by copying the features from A to B for the positions after the separation index.
    """

    def forward(self, streams, sep):
        # Now unpacks 4 elements because of HP
        A_x, A_y, A_mask, A_hp = streams["A"]
        B_x, B_y, B_mask, B_hp = streams["B"]
        C_x, C_y, C_mask, C_hp = streams["C"]

        # CLONE to detach the view from the original global batch
        B_x = B_x.clone()
        B_y = B_y.clone()

        # Safely mutate the clones
        B_x[sep:, ...] = A_x[sep:, ...]
        B_y[sep:, ...] = A_y[sep:, ...]

        return {
            "A": (A_x, A_y, A_mask, A_hp),
            "B": (B_x, B_y, B_mask, B_hp),
            "C": (C_x, C_y, C_mask, C_hp),
        }


class AppendATrainToBTestMutation(AbstractStreamMutation):
    """
    Appends the 'A_train' context (the support set of the target task)
    to the test/query section of all streams.
    This allows the model to reason about what Stream B (Related)
    'thinks' about the support points of Stream A (Target).
    """

    def __init__(self):
        super().__init__()
        # hotfix!
        self._len_a_train = None  # Will be set on the first forward pass when we know the shape of A_train

    def forward(self, streams, sep):
        # 1. Unpack all 4 elements
        A_x, A_y, A_mask, A_hp = streams["A"]
        device = A_x.device

        # 2. Extract A_train portion (now including HP)
        a_train_x = A_x[:sep, ...].clone()
        a_train_y = A_y[:sep, ...].clone()
        a_train_hp = A_hp[:sep, ...].clone() if A_hp is not None else None

        new_streams = {}
        # 3. Unpack 4 elements in the loop
        for key, (x, y, mask, hp) in streams.items():

            # Concatenate features and targets
            new_x = torch.cat([x, a_train_x], dim=0)
            new_y = torch.cat([y, a_train_y], dim=0)

            # Concatenate HP coordinates if they exist
            new_hp = None
            if hp is not None and a_train_hp is not None:
                new_hp = torch.cat([hp, a_train_hp], dim=0)

            # Handle Padding Mask Extension
            new_mask = None
            if mask is not None:
                is_seq_first = (mask.shape[0] == x.shape[0])
                append_mask_shape = (sep, x.shape[1]) if is_seq_first else (x.shape[1], sep)
                append_mask = torch.zeros(
                    append_mask_shape,
                    dtype=torch.bool,  # False means NOT padded
                    device=device
                )
                new_mask = torch.cat([mask, append_mask], dim=0 if is_seq_first else 1)

            new_streams[key] = (new_x, new_y, new_mask, new_hp)

        self._len_a_train = len(a_train_x) # hotfix
        return new_streams

    def splice_at_fwd_end(self, output_streams, batch):
        """
        Undoes the mutation on the model outputs. No batch cleanup needed.
        """
        out_streams = {}

        T_test = batch.x.shape[0] - batch.single_eval_pos - self._len_a_train  # Original test length before appending A_train
        for key, logits in output_streams.items():
            # Slice off the appended A_train portion from the end of the sequence dimension
            out_streams[key] = logits[:T_test, ...]

        self._len_a_train = None
        return out_streams