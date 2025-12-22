import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class CrossFusion(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.cross_train = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_test = MultiheadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)  # optional but recommended

        self.single_eval_pos = None  # placeholder

    def forward(self, x, *args, **kwargs):
        if "single_eval_pos" in kwargs:
            single_eval_pos = kwargs["single_eval_pos"]
        else:
            # if the pfn does not communicate this argument (e.g. to a linear layer)
            single_eval_pos = self.single_eval_pos

        B = x.shape[1]
        if self.training:
            # we expect that we get pairs of tasks, i.e. A (target task tensor untainted),
            # B (related tasks untainted), C (related tasks conditional predictions to be updated)
            assert B % 3 == 0, "In training mode, batch size must be multiple of 3 (A,B,C task triplets)"
            assert single_eval_pos is not None, "single_eval_pos must be provided during training"
            R = B // 3  # number related tasks
            Q = x[ :, :R, : ]  # (stream A) key: target task marginal predictions (untainted)
            K = x[ :, R : 2 * R, : ]  # (stream B) value: related tasks' marginal predictions (untainted)
            V = x[ :, 2 * R :, : ]  # (stream C) query: related tasks' conditional predictions (to be updated)

        else:
            # during evaluation we have only one target task and R related tasks (|A|=1), B, C as before
            R = (B - 1) // 2  # number related

            Q = x[:, :1, :].expand( -1, R, -1 )  # (stream A) key: target task marginal predictions (untainted)
            K = x[ :, 1 : R + 1, :]  # (stream B) value: related tasks' marginal predictions (untainted)
            V = x[ :, R + 1 :, : ]  # (stream C) query: related tasks' conditional predictions (to be updated)

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

        conditional = torch.cat(
            [train_update, test_update], dim=0
        )  # train + test updated conditionals

        # reconstruct the full (partially) updated output
        if self.training:
            y = torch.cat( [x[:, : 2 * R, :], conditional], dim=1 )  
        else:
            y = torch.cat( [x[:, : R + 1, :], conditional], dim=1 )  

        return y
