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

    def forward(self, batch, src_key_padding_mask=None) -> tuple[MyBatch, torch.Tensor]:
        """ Main entry point for parsing the batch. It extracts the raw streams, applies mutations, and reassembles. """
        sep = batch.single_eval_pos
        streams = self.get_raw_streams(batch, mask=src_key_padding_mask)

        for mutation in self.stream_mutations or []:
            streams = mutation(streams, sep)

        return self.assemble_batch(streams, sep)

    def splice_at_fwd_end(self, output, batch, src_key_padding_mask=None):
        """ Alternative entry point for splicing at the end of the forward pass, if needed. """
        sep = batch.single_eval_pos
        o_streams = self.parse_output_streams(output, sep, src_key_padding_mask=src_key_padding_mask)
        b_streams = self.get_raw_streams(batch, mask=src_key_padding_mask)

        for mutation in self.stream_mutations or []:
            o_streams, b_streams = mutation.splice_at_fwd_end(o_streams, b_streams, sep)
        batch, mask = self.assemble_batch(b_streams, sep)
        output = self.assemble_output_streams(o_streams)
        return batch, mask, output


    # --- Batch Parsing and Assembly ---

    def get_raw_streams(self, batch, mask=None):
        """Extracts raw A/B/C streams from the interleaved batch format."""
        x, y = batch.x, batch.y

        if self.training: # from nn.Module

            B_dim = batch.x.shape[1]
            assert B_dim % 2 == 0, ("Batch size must be multiple of 2 during training,"
                                    " because of the batch format (target0/related0/target1/related1/...)")
            # Train: Interleaved (Step size 2)
            return {
                "A": (
                    x[:, ::2, :],
                    y[:, ::2],
                    mask[::2, :] if mask is not None else None
                ),

                "B": (
                    x[:, 1::2, :],
                    y[:, 1::2],
                    mask[1::2, :] if mask is not None else None
                ),
                "C": (
                    x[:, ::2, :],
                    y[:, ::2],
                    mask[::2, :] if mask is not None else None
                ),
            }
        else:
            # Eval: 1 Target, R Related.
            R = x.shape[1] - 1
            return {
                "A": (
                    x[:, :1, :].expand(-1, R, -1),
                    y[:, :1].expand(-1, R),
                    mask[:1, :].expand(R, -1) if mask is not None else None
                ),
                "B": (
                    x[:, 1:, :],
                    y[:, 1:],
                    mask[1:, :] if mask is not None else None
                ),
                "C": (
                    x[:, :1, :].expand(-1, R, -1),
                    y[:, :1].expand(-1, R),
                    mask[:1, :].expand(R, -1) if mask is not None else None
                ),
            }

    def assemble_batch(self, streams, sep) -> tuple[MyBatch, torch.Tensor]:
        A_x, A_y, A_mask = streams["A"]
        B_x, B_y, B_mask = streams["B"]
        C_x, C_y, C_mask = streams["C"]

        device = A_x.device

        batch_x = torch.cat([A_x, B_x, C_x], dim=1)
        batch_y = torch.cat([A_y, B_y, C_y], dim=1)

        final_mask = None
        if A_mask is not None and B_mask is not None and C_mask is not None:
            final_mask = torch.cat([A_mask, B_mask, C_mask], dim=0).to(device)

        return MyBatch(
            x=batch_x.to(device),
            y=batch_y.to(device),
            target_y=batch_y.to(device),
            single_eval_pos=sep
        ), final_mask

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
        A_x = o_streams["A"]
        B_x = o_streams["B"]
        C_x = o_streams["C"]

        output = torch.cat([A_x, B_x, C_x], dim=1)

        return output
