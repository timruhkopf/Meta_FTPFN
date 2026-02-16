from typing import Optional
from dataclasses import dataclass

import torch

from pfns4hpo.priors import Batch

@dataclass
class MyBatch(Batch):
    src_key_padding_mask: Optional[torch.Tensor] = None

    def __add__(self, other) -> 'MyBatch':


        # Concatenate core tensors along the batch dimension (dim=1)
        # Assuming shape: [seq_len, batch_size, n_features]
        new_x = torch.cat([self.x, other.x], dim=1)
        new_y = torch.cat([self.y, other.y], dim=1)
        new_target_y = torch.cat([self.target_y, other.target_y], dim=1)

        if self.style is None or other.style is None:
            new_style = None
        else:
            new_style = torch.cat([self.style, other.style], dim=0)

        if self.src_key_padding_mask is None or other.src_key_padding_mask is None:
            new_src_key_padding_mask = None
        else:
            new_src_key_padding_mask = torch.cat(
                [self.src_key_padding_mask, other.src_key_padding_mask], dim=0
            )

        # Create the new instance
        return MyBatch(
            x=new_x,
            y=new_y,
            target_y=new_target_y,
            style=new_style
        )

    def to(self, device: torch.device) -> 'MyBatch':
        """Move all tensors in the batch to the specified device."""
        return MyBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            target_y=self.target_y.to(device),
            style=self.style.to(device) if self.style is not None else None
        )