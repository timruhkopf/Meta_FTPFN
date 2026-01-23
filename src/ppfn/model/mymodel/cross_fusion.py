import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import mlflow

from typing import Dict, Tuple

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback


class CrossFusion(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, use_prenorm=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        # self.use_gain = use_gain
        self.use_prenorm = use_prenorm

        self.cross_train = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_test = MultiheadAttention(d_model, num_heads, dropout)

        # PRE-NORM: Normalize inputs, not the output delta
        if self.use_prenorm:
            self.norm_Q = nn.LayerNorm(d_model)
            self.norm_K = nn.LayerNorm(d_model)  # Shared for V usually if K=V

        self.norm = nn.LayerNorm(d_model)  # optional but recommended

        # Fading in this layer either using gain parameter or identity init
      #  if use_gain:
      #       self.gain = nn.Parameter(torch.zeros(1))

        self.initialize_as_identity()
        self.single_eval_pos = None  # placeholder

    def initialize_as_identity(self):
        # 1. Zero out the output projection weights and biases
        # This makes the output of the attention mechanism 0 before the residual connection
        # nn.init.constant_(self.cross_train.out_proj.weight, 0.0001)
        # nn.init.constant_(self.cross_train.out_proj.bias, 0.0001)
        #
        # nn.init.constant_(self.cross_test.out_proj.weight, 0.0001)
        # nn.init.constant_(self.cross_test.out_proj.bias, 0.0001)

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
        B, single_eval_pos = self.validate_forward_args(x, *args, **kwargs)
        
        R = B // 3  # number related tasks
        Q = x[ :, :R, : ]
        # (stream A) key: target task marginal predictions (untainted)
        K = x[ :, R : 2 * R, : ]
        # (stream B) value: related tasks' marginal predictions (untainted)
        V = x[ :, 2 * R :, : ]
        # (stream C) query: related tasks' conditional predictions (to be updated)

        # during evaluation we have only one target task and R related tasks (|A|=1), B, C as before
        #     R = (B - 1) // 2  # number related

        #     # (stream A) key: target task marginal predictions (untainted)
        #     Q = x[:, :1, :].expand( -1, R, -1)
        #     # (stream B) value: related tasks' marginal predictions (untainted)
        #     K = x[ :, 1 : R + 1, : ]
        #     # (stream C) query: related tasks' conditional predictions (to be updated)
        #     V = x[ :, R + 1 :, : ]

        # Handle the train/test split
        # we only want to attend to the train set of the target task
        # when updating the conditional predictions of the related tasks
        # the test tokens refer to the same positions and are skipped later.
        Q_train, Q_test = Q[:single_eval_pos, :, :], Q[single_eval_pos:, :, :]
        K_train = K[:single_eval_pos, :, :]  # , K[single_eval_pos:, :, :]
        V_train = V[:single_eval_pos, :, :]  # , V[single_eval_pos:, :, :]

        # Attention to allow batch attention from target to related train set (in pairs)
        train_update = (
            self.norm(self.cross_train(Q_train, K_train, V_train)[0]) + Q_train
        )

        # Attention to allow batch attention from target test set to related train set (in pairs)
        test_update = self.norm(self.cross_test(Q_test, K_train, V_train)[0]) + Q_test

        if self.use_gain:
            gain = torch.sigmoid(self.gain)
            train_update = train_update * gain
            test_update = test_update * gain


        conditional = torch.cat(
            [train_update, test_update], dim=0
        )  # train + test updated conditionals

        # reconstruct the full (partially) updated output
        output = torch.cat([x[:, : 2 * R, :], conditional], dim=1)
        # eval:
        #     output = torch.cat([x[:, : R + 1, :], conditional], dim=1)

        return output
    
class CrossFusionV2(CrossFusion):
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
        A = x[ :, : R , : ]
        
        # Stream (B.1) key related tasks' marginal predictions (untainted)
        B = x[ :, R : 2 * R, : ]

        # Stream (B.2) Value: what we want to extract from the related tasks based on
        # the learned kernel(A, B) * B, which will give us \Delta C, that we can add to C.
        B = x[ :, R: 2 * R , : ] # Key Difference to CrossFusion

        # Stream (C): last' layer's conditional predictions (to be updated)
        C = x[ :, 2 * R :, : ]

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

        # if self.use_gain:
        #     gain = torch.sigmoid(self.gain)
        #     train_delta = train_delta * gain
        #     test_delta = test_delta * gain

        train_update = train_delta + C_train
        test_update = test_delta + C_test

        conditional = torch.cat(
            [train_update, test_update], dim=0
        )  # train + test updated conditionals


        # combine the untainted streams A, B with the updated conditional stream C
        output = torch.cat([x[:, : 2 * R, :], conditional], dim=1)

        return output


class TrainMetricsCallback(AbstractCallback):
    # FIXME: move to model/callbacks to intercept intermediate outputs on the streams prior to
    #  aggregation!
    """A callback to compute loss only on the workspace (Stream C) outputs."""
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose

    def _calc_nll(self, batch, single_eval_pos, output, targets, loss) -> Dict:
        B = output.shape[1]
        R = B // 3

        assert loss.shape[1] == B, (
            "Loss shape does not match output shape; expected loss.shape[1] == output.shape[1]"
        )

        unconditional_loss = loss[:, :R, ...].detach()  # Stream A
        conditional_loss = loss[:, -R:, ...].detach()  # Stream C


        nll_diff = (conditional_loss - unconditional_loss).detach()

        metrics = {'nll/C-A': nll_diff.mean()}

        if self.verbose:
            metrics.update({
                "nll/A": unconditional_loss.mean(),
                "nll/C": conditional_loss.mean(),
            })

        # Handle style-based grouping
        if hasattr(batch, 'style') and batch.style is not None:
            style = batch.style[::2]
            # Ensure we only compute if indices are within bounds
            metrics.update({
                'nll/same_task': nll_diff[:, style != 1].mean(),
                'nll/unrelated_task': nll_diff[:, style == 1].mean()
            })

        metrics = {k: v.cpu().item() for k, v in metrics.items()}

        return metrics

    def _calc_kl(self, batch, single_eval_pos, output, targets, loss) -> Dict:
        B = output.shape[1]
        R = B // 3

        # 1. Target (Stream A): Probabilities
        A_probs = F.softmax(output[:, :R, :], dim=-1)

        # 2. Input (Stream C): Log-Probabilities
        C_log_probs = F.log_softmax(output[:, -R:, :], dim=-1)

        # 3. Compute KL (Sum over classes, mean over batch/time)
        kl_tensor = F.kl_div(
            C_log_probs,
            A_probs,
            reduction="none"
        ).sum(dim=-1).detach()

        kl = {"kl/A-C": kl_tensor.mean().item()}

        if self.verbose and batch.style is not None:
            style = batch.style[::2]  # because the pairs get the same value
            kl.update({
                'kl/same_task': kl_tensor[:, style != 1].mean().detach().cpu().item(),
                'kl/unrelated_task': kl_tensor[:, style == 1].mean().detach().cpu().item()
            })


        return kl

    def _calc_dist_properties(self, batch, single_eval_pos, output, targets, loss) -> Dict:
        B = output.shape[1]
        R = B // 3

        A = output[:, :R, :]
        C = output[:, -R:, :]

        # Calculate variance of the probability distributions
        var_A = torch.var(F.softmax(A, dim=-1), dim=-1)
        var_C = torch.var(F.softmax(C, dim=-1), dim=-1)

        return {
            # High positive value suggests conditioning reduced uncertainty/spread
            'dist/var:A-C': (var_A - var_C).mean().detach().cpu().item(),
        }

    def on_loss_end(self, batch, single_eval_pos, output, targets, loss) -> Dict:
        """
        Modify the loss, since we only care about the conditional (Stream C) outputs.
        We do however want to track the difference between the unconditional (A) and conditional (C) outputs.

        """
        B = output.shape[1]
        R = B // 3

        nll = self._calc_nll(batch, single_eval_pos, output, targets, loss)
        kl = self._calc_kl(batch, single_eval_pos, output, targets, loss)
        # distrib_properties = self._calc_dist_properties(batch, single_eval_pos, output, targets, loss)
        return {
            # **distrib_properties,
            **nll,
            **kl,

        }


class CrossFusionLossCallback(AbstractCallback):
    # FIXME: This callback is temporary, until we found a good aggregation/pooling strategy,
    #  so that we will finally have a different output head for the conditional predictions only.
    """A callback to compute loss only on the workspace (Stream C) outputs."""

    def on_forward_end(self, batch, single_eval_pos, output, targets) -> Dict:
        """
        Modify the loss to only consider the workspace (Stream C) outputs.

        we need to replicate the targets to meet the model's output shape.

        This is a temporary fix: the original batch presented to the model has
        only the independent batch items. the model however parses the batch into
        three streams (A,B,C), so we need to replicate the targets accordingly.

        :param batch: The input batch containing single_eval_pos
        :param output: The model output tensor
        :param targets: The target tensor

        """
        parser = self.trainer.model.parse_batch
        b = parser(batch, single_eval_pos)
        return {"targets": b.y[single_eval_pos:, ...]}

    def on_loss_end(self, batch, single_eval_pos, output, targets, loss) -> Dict:
        """
        Modify the loss, since we only care about the conditional (Stream C) outputs.

        """
        B = output.shape[1]
        R = B // 3

        return {
            "loss": loss[ :, -R: ]
            # loss on Stream C only (the others are frozen anyways!)
        }