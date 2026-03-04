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

    def splice_at_fwd_end(self, output_streams, batch_streams, sep):
        """ Optional method to apply mutations at the end of the forward pass instead of the beginning.
        Notice, that any change to the batch (incl. e.g. appending A_train also changed the labels contained in the batch object)"""
        return output_streams, batch_streams  # By default, do nothing. Override if needed.



class ForceSameQueryMutation(AbstractStreamMutation):
    """
    Mutation to force the Related stream (B) to have the same query positions as the Target stream (A).
    This is done by copying the features from A to B for the positions after the separation index.
    """

    def forward(self, streams, sep):
        A_x, A_y, A_mask = streams["A"]
        B_x, B_y, B_mask = streams["B"]
        C_x, C_y, C_mask = streams["C"]

        # Force B to have the same query positions as A after the separation index
        # Note that C is initally a copy of A, so it already has the same queries as A.
        B_x[sep:, ...] = A_x[sep:, ...]
        B_y[sep:, ...] = A_y[sep:, ...]

        return {
            "A": (A_x, A_y, A_mask),
            "B": (B_x, B_y, B_mask),
            "C": (C_x, C_y, C_mask),
        }


class AppendATrainToBTestMutation(AbstractStreamMutation):
    """
    Appends the 'A_train' context (the support set of the target task)
    to the test/query section of all streams.
    This allows the model to reason about what Stream B (Related)
    'thinks' about the support points of Stream A (Target).
    """

    def forward(self, streams, sep):
        # We extract Stream A to get the reference training data
        A_x, A_y, A_mask = streams["A"]
        device = A_x.device

        # 1. Extract A_train portion
        # Shape: (sep, R, d_model)
        a_train_x = A_x[:sep, ...].clone()
        a_train_y = A_y[:sep, ...].clone()

        # 2. Modify features: Copy A_train features into B_test slots
        # (Assuming your specific use case requires this feature swap)
        # Usually, this means making the 'test' features identical to 'train'
        # features for the appended segment.
        a_train_x[:, 1::2, ...] = a_train_x[:, ::2, ...]
        a_train_y[:, 1::2] = a_train_y[:, ::2]

        new_streams = {}
        for key, (x, y, mask) in streams.items():
            # 3. Append to sequence dimension (dim=0)
            # Resulting seq_len: original_seq_len + sep
            new_x = torch.cat([x, a_train_x], dim=0)
            new_y = torch.cat([y, a_train_y], dim=0)

            # 4. Handle Padding Mask Extension
            # If a mask exists, we must append 'False' (meaning NOT padded)
            # for the new indices so the transformer actually attends to them.
            new_mask = None
            if mask is not None:
                # Assuming mask shape is (Batch, Seq) or (Seq, Batch)
                # Let's handle the common (Batch, Seq) based on your cat in assemble_batch
                # If your mask is (Seq, Batch), dim=0. If (Batch, Seq), dim=1.

                # Check current mask orientation (assuming Seq is the dim we just extended)
                is_seq_first = (mask.shape[0] == x.shape[0])

                append_mask_shape = (sep, x.shape[1]) if is_seq_first else (x.shape[1], sep)
                append_mask = torch.zeros(
                    append_mask_shape,
                    dtype=torch.bool,
                    device=device
                )

                new_mask = torch.cat([mask, append_mask], dim=0 if is_seq_first else 1)

            new_streams[key] = (new_x, new_y, new_mask)

        return new_streams

    def cleanup(self, output_streams, batch_streams, sep):
        """
        We need to undo the mutation at the end of the forward pass to ensure that the loss will not see the appended
         A_train data in B_test
        """
        # Essentially, we just need to slice off the appended A_train portion from the end of the sequence dimension.
        out_streams = {}
        for key, logits in output_streams.items():
            # Assuming the appended portion is at the end of the sequence dimension (dim=0)
            out_streams[key] = logits[:-sep, ...]

        # FIXME: we might not need to clean up the batch, because the trainer step holds the original batch still, that is passed to the
        #  objective function -- no side effects occurred same for the mask
        b_streams = {}
        for key, (x, y, mask) in batch_streams.items():
            # Similarly, we need to remove the appended portion from the batch streams to ensure correct loss computation.
            b_streams[key] = (x[:-sep, ...], y[:-sep, ...], mask[:-sep, ...] if mask is not None else None)

        return out_streams, b_streams
