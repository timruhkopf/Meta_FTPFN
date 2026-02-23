from typing import Tuple

import torch
from torch import nn
from torch.nn import MultiheadAttention


class CrossFusion(nn.Module):
    def __init__(
        self, d_model, num_heads, dropout=0.0, use_prenorm=True, add_linear=True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_prenorm = use_prenorm

        self.cross_train = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_test = MultiheadAttention(d_model, num_heads, dropout)
        self.linear = nn.Linear(d_model, d_model) if add_linear else None

        # PRE-NORM: Normalize inputs, not the output delta
        if self.use_prenorm:
            self.norm_Q = nn.LayerNorm(d_model)
            self.norm_K = nn.LayerNorm(d_model)  # Shared for V usually if K=V

        self.norm = nn.LayerNorm(d_model)  # optional but recommended

        self.initialize_as_identity()
        self.single_eval_pos = None  # placeholder

    def initialize_as_identity(self):
        # 1. Zero out the output projection weights and biases
        nn.init.normal_(self.cross_train.out_proj.weight, std=1e-4)
        nn.init.zeros_(self.cross_train.out_proj.bias)
        nn.init.normal_(self.cross_test.out_proj.weight, std=1e-4)
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
        A = x[:, :R, :]

        # Stream (B.1) key related tasks' marginal predictions (untainted)
        B = x[:, R : 2 * R, :]

        # Stream (B.2) Value: what we want to extract from the related tasks based on
        # the learned kernel(A, B) * B, which will give us \Delta C, that we can add to C.
        B = x[:, R : 2 * R, :]  # Key Difference to CrossFusion

        # Stream (C): last' layer's conditional predictions (to be updated)
        C = x[:, 2 * R :, :]

        if self.use_prenorm:
            A = self.norm_Q(A)
            B = self.norm_K(B)
            # V needs no norm, as we want the raw values from the related tasks

        # Handle the train/test split
        A_train, A_test = A[:sep, :, :], A[sep:, :, :]
        B_train, _ = B[:sep, :, :], B[sep:, :, :]
        # B_train, B_test = B[:sep, :, :], B[sep:, :, :]
        C_train, C_test = C[:sep, :, :], C[sep:, :, :]

        # only the learned delta
        train_delta = self.cross_train(A_train, B_train, B_train)[0]
        test_delta = self.cross_test(A_test, B_train, B_train)[0]

        # FIXME: V_test should probably be altered and added?

        if not self.use_prenorm:
            train_delta = self.norm(train_delta)
            test_delta = self.norm(test_delta)

        if self.linear is not None:
            train_delta = self.linear(train_delta)
            test_delta = self.linear(test_delta)

        train_update = train_delta + C_train
        test_update = test_delta + C_test

        conditional = torch.cat(
            [train_update, test_update], dim=0
        )  # train + test updated conditionals


        # combine the untainted streams A, B with the updated conditional stream C
        output = torch.cat([x[:, : 2 * R, :], conditional], dim=1)

        return output
