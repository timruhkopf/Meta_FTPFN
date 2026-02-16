import torch
import numpy as np
from typing import Dict, List, Any
from torch.utils.data import DataLoader

from ppfn.trainer import AbstractCallback


class HeftyContextEvaluationCallback(AbstractCallback):
    """
    Evaluates model performance across various context levels for the target task,
    with explicit control over the single_eval_pos.
    """

    def __init__(
            self,
            dataset,
            context_levels: List[int],  # Number of points to reveal: [0, 1, 5, 20, 100]
            single_eval_pos: int,  # Explicitly set the boundary for evaluation
            device: str = 'cpu',
    ):
        self.dataset = dataset
        self.context_levels = context_levels
        self.single_eval_pos = single_eval_pos
        self.device = device

    def on_train_end(self, **kwargs):

        if self.dataset.target_first: # to change the way the batch and padding are parsed
            self.trainer.model.eval()

        hefty_stats = {}

        for ctx_size in self.context_levels:
            # Ensure we don't try to show more context than exists before the eval pos
            actual_ctx = min(ctx_size, self.single_eval_pos)

            print(f"Hefty Eval: Target Context Size = {actual_ctx} | Eval Pos = {self.single_eval_pos}")

            results = self._evaluate_at_context_level(actual_ctx)

            # Aggregate metrics with context-specific keys
            for key, val in results.items():
                # Format: metric:ctx_10:evalpos_80:dataset_name
                stat_key = f"{key}:ctx_{actual_ctx}:sep_{self.single_eval_pos}:{self.dataset.name}"
                hefty_stats[stat_key] = val

                # FIXME: rather than 33 metrics, we will want to store a single vector metric for all context sizes at once
                # FIXME: we will want to see the variance per context size as well !

        if self.dataset.target_first:
            self.trainer.model.train()

        return hefty_stats

    def _evaluate_at_context_level(self, visible_ctx: int):

        assert visible_ctx <= self.single_eval_pos, "Visible context cannot exceed the single_eval_pos boundary."
        assert visible_ctx > 0, "Visible context must be greater than 0 for evaluation, otherwise padding will cause nan in attention!"

        loader = DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=lambda x: x[0],
        )

        batch_results = []

        with torch.no_grad():
            for i, batch in enumerate(loader):


                # batch[0] is X: [S, B, D]
                seq_len, batch_size, _ = batch.x.shape

                # Create the mask: True = Masked/Hidden, False = Visible
                # Start with everything masked (True)
                pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)

                # 1. Related Tasks (odd indices 1, 3, 5...):
                # Always reveal full context up to single_eval_pos
                if self.dataset.target_first:
                    pad_mask[0,:self.single_eval_pos] = False
                    pad_mask[0, visible_ctx:self.single_eval_pos] = True

                else:
                    pad_mask[1::2, :self.single_eval_pos] = False
                    pad_mask[1::2, visible_ctx:self.single_eval_pos] = True


                # Forward pass through the trainer's logic
                _, step_metrics = self.trainer._forward_pass(
                    batch,
                    self.single_eval_pos,
                    src_key_padding_mask=pad_mask
                )
                batch_results.append(step_metrics)

        # Average the results across the samples
        aggregated = {}
        if batch_results:
            for key in batch_results[0].keys():
                aggregated[key] = np.mean([r[key] for r in batch_results])

        return aggregated


