import torch
from torch import nn

from torch.nn import MultiheadAttention


class Connector(nn.Module):
    """
    Simple connector module to adapt dimensions between frozen model and interleaved layers.
    E.g., linear projection to match d_model sizes.
    """

    def __init__(
        self,
        is_first=False,
    ):
        super().__init__()
        self.is_first = is_first

    def create_target_in_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Create a target context within the batch by shifting inputs."""

        if self.training:
            return x.roll(1, dims=1)  # shift by one along sequence dimension
        else:
            T, B, D = x.shape
            # during eval, the first example is the target task, all other tasks in the batch need to be paired with it
            return x[:, :1, :].repeat(1, B, 1)

    def forward_in(self, x, single_eval_pos=None):
        """
        Splits input x into old and new parts for cross-attention.
        If dual_path is enabled, assumes x contains concatenated old and new parts.
        Otherwise, creates new part by shifting x.


        NOTE:
        To connect to the PFN model, we need to have a look at the two (!) attention calls in layer.py/TransformerEncoderLayer.forward l170-178:

            # (0) standard attention over the whole training set sequence
            src_left = self.self_attn(
                src_[:single_eval_position],
                src_[:single_eval_position],
                src_[:single_eval_position],
            )[0]

            # (1) attention over from the test set sequence to the training set
            # FIXME: their code does not support a key_padding_mask yet that would need to be passed here.
            src_right = self.self_attn(
                src_[single_eval_position:], src_to_attend_to, src_to_attend_to
            )[0]

            src2 = torch.cat([src_left, src_right], dim=0)


        """
        # e.g. pre linear layer inputs are given as single tensor
        if isinstance(x, torch.Tensor):
            # single tensor input
            if self.is_first or not self.dual_path:
                old_x = x
                new_x = self.create_target_in_batch(x)
            else:
                old_x = x[:, : x.shape[1] // 2, :]
                new_x = x[:, x.shape[1] // 2 :, :]

            train_old_x = old_x[:single_eval_pos]
            test_old_x = old_x[single_eval_pos:]

            train_new_x = new_x[:single_eval_pos]
            test_new_x = new_x[single_eval_pos:]

            return (train_old_x, test_old_x), (train_new_x, test_new_x)

        else:
            raise ValueError("Unsupported input type for Connector.forward_in")

        # # pre attention inputs are given as copied (q,k,v) tuples
        # if isinstance(x, tuple) and torch.equal(x[0], x[1]):
        #     # This case corresonds to Note (0) above
        #     self.return_type = tuple

        #     # split into old and new parts
        #     if self.is_first or not self.dual_path:
        #         # there is no new part yet, so copy for cross attention
        #         old_x = x[0]
        #         new_x = self.create_target_in_batch(x[0])
        #     else:
        #         # assuming that there are two parts of equal length just concatenated
        #         old_x = x[0][:, : x[0].shape[1] // 2, :]
        #         new_x = x[0][:, x[0].shape[1] // 2 :, :]

        #     return (old_x, old_x, old_x), (new_x, new_x, new_x)

        # elif isinstance(x, tuple) and not torch.equal(x[0], x[1]):
        #     # This case corresonds to Note (1) above; we need to split in batch dim for each tensor respectively here
        #     self.return_type = tuple

        #     if self.is_first or not self.dual_path:
        #         old_q, old_k, old_v = x
        #         new_q = self.create_target_in_batch(old_q)
        #         new_k = self.create_target_in_batch(old_k)
        #         new_v = self.create_target_in_batch(old_v)

        #     else:
        #         old_q, old_k, old_v = (
        #             x[0][:, : x[0].shape[1] // 2, :],
        #             x[1][:, : x[1].shape[1] // 2, :],
        #             x[2][:, : x[2].shape[2] // 2, :],
        #         )
        #         new_q = x[0][:, x[0].shape[1] // 2 :, :]
        #         new_k = x[1][:, x[1].shape[1] // 2 :, :]
        #         new_v = x[2][:, x[2].shape[2] // 2 :, :]

        #     return (new_q, new_k, new_v), (old_q, old_k, old_v)

    def forward_out(self, train_old_x, test_old_x, train_new_x, test_new_x):
        """Merge old and new parts back into original structure."""
        old = torch.cat([train_old_x, test_old_x], dim=0)  # old part
        new = torch.cat([train_new_x, test_new_x], dim=0)  # new part

        return torch.cat([old, new], dim=1)  # concatenate along sequence dimension


class CrossFusion(nn.Module):
    """
    Skip-connected cross attention:
        output = x + CrossAttention(x, support, support) with LayerNorm
    """

    def __init__(self, d_model, num_heads, dropout=0.0, is_first=False):
        super().__init__()

        self.connector = Connector(is_first=is_first)
        self.cross_train = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_test = MultiheadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)  # optional but recommended
        self._is_first = is_first

        # batch will communicate the single_eval_pos for train test splitting
        self.single_eval_pos = None

    @property
    def is_first(self):
        return self._is_first

    @is_first.setter
    def is_first(self, value):
        # FIXME: we should probably be explicit in the definition of the interleaved layer instead!
        self._is_first = value
        self.connector.is_first = value

    def forward(self, input):
        """
        input: either
            (q,k,v) tuples of shape ((B, L, D), (B, L, D), (B, L, D)) for cross attention
            or single tensor of shape (B, L, D) for linear layers

        The CrossFusion module intercepts attention calls and injects pairwise batch awareness:

        **Case 1: Train-to-train self-attention**
            Input: (src_train, src_train, src_train)  [all identical]
            We inject: attend to other batch examples' training sets
            Output: src_train_augmented with cross-batch context

        **Case 2: Test-to-train cross-attention**
            Input: (src_test, src_train, src_train)  [query differs from key/value]
            We inject: attend to other batch examples' training sets
            Output: src_test_augmented with cross-batch context

        returns: same type as input (tuple or tensor)
        """

        (train_old_x, test_old_x), (train_new_x, test_new_x) = (
            self.connector.forward_in(input[0], single_eval_pos=self.single_eval_pos)
        )

        # Attention to allow batch attention from new to old train set (in pairs)
        train_new_x = self.norm(
            self.cross_train(train_new_x, train_old_x, train_old_x)[0] + train_new_x
        )

        # Attention to allow batch attention from new test set to old train set (in pairs)
        test_new_x = self.norm(
            self.cross_test(test_new_x, train_old_x, train_old_x)[0] + test_new_x
        )

        return self.connector.forward_out(
            train_old_x, test_old_x, train_new_x, test_new_x
        )
