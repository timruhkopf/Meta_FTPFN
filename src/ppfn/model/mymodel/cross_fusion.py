from typing import Tuple

import torch
from torch import nn
from torch.nn import MultiheadAttention


class CrossFusion(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, use_prenorm=True, add_linear=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_prenorm = use_prenorm

        self.cross_train = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_test = MultiheadAttention(d_model, num_heads, dropout)
        self.linear = (nn.Linear(d_model, d_model) if add_linear else None)

        # PRE-NORM: Normalize inputs, not the output delta
        if self.use_prenorm:
            self.norm_Q = nn.LayerNorm(d_model)
            self.norm_K = nn.LayerNorm(d_model)  # Shared for V usually if K=V

        self.norm = nn.LayerNorm(d_model)  # optional but recommended

        self.initialize_as_identity()
        self.single_eval_pos = None  # placeholder

    def initialize_as_identity(self):
        # 1. Zero out the output projection weights and biases
        nn.init.zeros_(self.cross_train.out_proj.weight)
        nn.init.zeros_(self.cross_train.out_proj.bias)
        nn.init.zeros_(self.cross_test.out_proj.weight)
        nn.init.zeros_(self.cross_test.out_proj.bias)

        # 2. Ensure LayerNorm starts as identity (weight=1, bias=0)
        # PyTorch does this by default, but it's good to be explicit
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

    def validate_forward_args(self, x, *args, **kwargs) -> Tuple[int, int]:
        if "single_eval_pos" in kwargs:
            single_eval_pos = kwargs["single_eval_pos"]
        else:
            # if the pfn does not communicate this argument (e.g. to a linear layer),
            # then the model should communicate it to the interleaved layers directly!
            single_eval_pos = self.single_eval_pos

        assert single_eval_pos is not None, (
            "single_eval_pos must be provided during training"
        )

        # we expect that we get pairs of tasks, i.e. A (target task tensor untainted),
        # B (related tasks untainted), C (related tasks conditional predictions to be updated)
        B = x.shape[1]
        assert B % 3 == 0, (
            "In training mode, batch size must be multiple of 3 (A,B,C task triplets)"
        )
        return B, single_eval_pos

    def forward(self, x, *args, **kwargs):
        """
        Core Motivation:
        Since the PFN has already learned useful representations for the marginal predictions,
        we want to learn a kernel that asserts the similarity between the target task marginal predictions
        and the related tasks marginal predictions, and use that to extract useful information from the related tasks
        marginal predictions to update the target tasks conditional predictions.

        """

        B, sep = self.validate_forward_args(x, *args, **kwargs)

        R = B // 3  # number related tasks

        # Stream (A) query target task marginal predictions (untainted)
        A = x[:, : R, :]

        # Stream (B.1) key related tasks' marginal predictions (untainted)
        B = x[:, R: 2 * R, :]

        # Stream (B.2) Value: what we want to extract from the related tasks based on
        # the learned kernel(A, B) * B, which will give us \Delta C, that we can add to C.
        B = x[:, R: 2 * R, :]  # Key Difference to CrossFusion

        # Stream (C): last' layer's conditional predictions (to be updated)
        C = x[:, 2 * R:, :]

        if self.use_prenorm:
            A = self.norm_Q(A)
            B = self.norm_K(B)
            # V needs no norm, as we want the raw values from the related tasks

        # Handle the train/test split
        A_train, A_test = A[:sep, :, :], A[sep:, :, :]
        B_train, B_test = B[:sep, :, :], B[sep:, :, :]
        # B_train, B_test = B[:sep, :, :], B[sep:, :, :]
        C_train, C_test = C[:sep, :, :], C[sep:, :, :]

        # only the learned delta
        train_delta = self.cross_train(A_train, B_train, B_train)[0]
        test_delta = self.cross_test(A_test, B_train, B_train)[0]

        # FIXME: V_test should probably be altered and added?

        if not self.use_prenorm:
            train_delta = self.norm(train_delta)
            test_delta = self.norm(test_delta)

        train_update = train_delta + C_train
        test_update = test_delta + C_test

        conditional = torch.cat(
            [train_update, test_update], dim=0
        )  # train + test updated conditionals

        if self.linear is not None:
            conditional = self.linear(conditional)

        # combine the untainted streams A, B with the updated conditional stream C
        output = torch.cat([x[:, : 2 * R, :], conditional], dim=1)

        # FIXME: add an mlp layer?

        return output


# THE FOLLOWING CODE IS KEPT HERE, BECAUSE IT IS SIGNIFICANTLY
# RELATED TO THE MULTI-STREAM TRAINING LOGIC
# IT MAY BE MOVED TO A MORE APPROPRIATE LOCATION LATER, ONCE AN AGGREGATION SCHEME IS
# IN PLACE.
import torch
import torch.nn as nn
from typing import Dict, Tuple


class MultiStreamObjective(nn.Module):
    """
    Encapsulates the 'Batch Trick' logic.
    It takes model outputs, computes the specific losses for streams A and C,
    and returns both the loss to optimize and the metrics to log.
    """

    def __init__(self, criterion: nn.Module, model=None, verbose=False):
        super().__init__()
        self.criterion = criterion
        self.verbose = verbose
        # FIXME: deprecaite this; it is needed only for batch parsing in forward (depending on
        #  train/eval state)
        self.model = model  # Optional reference to the model if needed for batch parsing

    def forward(self, output: torch.Tensor, targets: torch.Tensor, single_eval_pos, batch=None) -> \
            Tuple[
                torch.Tensor, Dict[str, float]]:
        # 1. Compute raw loss for all streams
        # Assuming output/targets are shaped correctly for the criterion

        if self.model is not None:
            # FIXME: depreciate this with the model reference
            parser = self.model.parse_batch
            b = parser(batch, single_eval_pos)
            targets = b.y[single_eval_pos:, ...]

        raw_loss = self.criterion(output, targets)  # [T, B]

        B = output.shape[1]
        R = B // 3

        # 2. Separate Streams
        # Ensure we detach the part we don't want gradients flowing back into if necessary
        # (Though usually we just slice the loss tensor we want to optimize)
        loss_stream_A = raw_loss[:, :R, ...]  # Unconditional
        loss_stream_C = raw_loss[:, -R:, ...]  # Conditional / Workspace

        # 3. Define the Optimization Target
        # You only want to optimize Stream C
        optimization_loss, nan_share = torch_nanmean(
            loss_stream_C.mean(0), # T dim avg -- to see which examples failed
            axis=0, # final avg
            return_nanshare=True
        )

        # 4. Compute Metrics (The logic previously in TrainMetricsCallback)
        with torch.no_grad():
            nll_diff = (loss_stream_C - loss_stream_A).detach()

            metrics = {
                "nll/C-A": nll_diff.mean().item(),
                "nll/A": loss_stream_A.mean().item(),
                "nll/C": loss_stream_C.mean().item(),
            }

            # Handle style-based grouping if batch is provided
            if batch is not None and hasattr(batch, 'style') and batch.style is not None:
                style = batch.style[::2].squeeze()  # Extract style for stream A tasks
                metrics.update({
                    'nll/similar_task': nll_diff[:, style != 1].mean().item(),
                    'nll/unrelated_task': nll_diff[:, style == 1].mean().item()
                })

        if self.verbose:
            metrics['nan_share'] = nan_share.item()

        return optimization_loss, metrics
