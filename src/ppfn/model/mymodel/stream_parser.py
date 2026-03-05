import torch
import torch.nn as nn
from ppfn.utils.mybatch import MyBatch


class StreamParser(nn.Module):
    """
    Utility class to handle the parsing and reassembly of the interleaved batch format.
    This encapsulates the logic for slicing the input batch into A/B/C streams, applying any necessary mutations,
    and reassembling it back into the format expected by the frozen model.

    nn.Module inheritance is used for training state tracking of the containing model
    """

    def __init__(self, stream_mutations=tuple()):
        super().__init__()
        self.stream_mutations = stream_mutations

    def forward(self, batch, hp=None, src_key_padding_mask=None) -> tuple[MyBatch, torch.Tensor]:
        """ Main entry point for parsing the batch. It extracts the raw streams, applies mutations, and reassembles. """
        sep = batch.single_eval_pos
        streams = self.get_raw_streams(batch, mask=src_key_padding_mask, hp=hp)

        for mutation in self.stream_mutations or []:
            streams = mutation(streams, sep)

        return self.assemble_batch(streams, sep)

    def splice_at_fwd_end(self, output, batch):
        """ Strictly parses and reassembles outputs. Leaves the batch completely alone. """
        sep = batch.single_eval_pos
        o_streams = self.parse_output_streams(output, sep)

        for mutation in self.stream_mutations or []:
            # Only apply if the mutation actually defines a backward/output splice
            if hasattr(mutation, 'splice_at_fwd_end'):
                o_streams = mutation.splice_at_fwd_end(o_streams, sep)

        output = self.assemble_output_streams(o_streams)
        return output



    def get_raw_streams(self, batch, mask=None, hp=None):
        """Extracts raw A/B/C streams from the interleaved batch format."""
        x, y = batch.x, batch.y

        if self.training: # from nn.Module

            B_dim = batch.x.shape[1]
            assert B_dim % 2 == 0, ("Batch size must be multiple of 2 during training,"
                                    " because of the batch format (target0/related0/target1/related1/...)")
            # Train: Interleaved (Step size 2)
            return {
                "A": (
                    x[:, ::2, :], y[:, ::2],
                    mask[::2, :] if mask is not None else None,
                    hp[:, ::2, :] if hp is not None else None
                ),
                "B": (
                    x[:, 1::2, :], y[:, 1::2],
                    mask[1::2, :] if mask is not None else None,
                    hp[:, 1::2, :] if hp is not None else None
                ),
                "C": (
                    x[:, ::2, :], y[:, ::2],
                    mask[::2, :] if mask is not None else None,
                    hp[:, ::2, :] if hp is not None else None
                ),
            }
        else:
            # Eval: 1 Target, R Related.
            R = x.shape[1] - 1
            return {
                "A": (
                    x[:, :1, :].expand(-1, R, -1), y[:, :1].expand(-1, R),
                    mask[:1, :].expand(R, -1) if mask is not None else None,
                    hp[:, :1, :].expand(-1, R, -1) if hp is not None else None
                ),
                "B": (
                    x[:, 1:, :], y[:, 1:],
                    mask[1:, :] if mask is not None else None,
                    hp[:, 1:, :] if hp is not None else None
                ),
                "C": (
                    x[:, :1, :].expand(-1, R, -1), y[:, :1].expand(-1, R),
                    mask[:1, :].expand(R, -1) if mask is not None else None,
                    hp[:, :1, :].expand(-1, R, -1) if hp is not None else None
                ),
            }

    def assemble_batch(self, streams, sep):
        A_x, A_y, A_mask, A_hp = streams["A"]
        B_x, B_y, B_mask, B_hp = streams["B"]
        C_x, C_y, C_mask, C_hp = streams["C"]

        device = A_x.device

        batch_x = torch.cat([A_x, B_x, C_x], dim=1)
        batch_y = torch.cat([A_y, B_y, C_y], dim=1)

        final_mask = None
        if A_mask is not None and B_mask is not None and C_mask is not None:
            final_mask = torch.cat([A_mask, B_mask, C_mask], dim=0).to(device)

        hp_tuple = (A_hp, B_hp, C_hp) if A_hp is not None else None

        return MyBatch(
            x=batch_x.to(device),
            y=batch_y.to(device),
            target_y=batch_y.to(device),
            single_eval_pos=sep
        ), final_mask, hp_tuple

    # --- Output Parsing (if needed) ---
    def parse_output_streams(self, output, sep, src_key_padding_mask=None):
        """ Utility to parse the model's output back into A/B/C streams for loss computation. """
        R = output.shape[1] // 3
        return {
            "A": output[:, :R, ...],
            "B": output[:, R: 2 * R, ...],
            "C": output[:, 2 * R:, ...],
        }

    def assemble_output_streams(self, o_streams):
        return torch.cat([o_streams["A"], o_streams["B"], o_streams["C"]], dim=1)
