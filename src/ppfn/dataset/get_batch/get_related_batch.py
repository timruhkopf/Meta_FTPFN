from pfns4hpo.utils import default_device
from ppfn.dataset.prior import MultiFidelityTask, DimensionPrior, FidelityPrior, AllocationPrior
import torch
import numpy as np

from ppfn.utils.mybatch import MyBatch


@torch.no_grad()
def get_batch(
        batch_size,
        seq_len,
        num_features,
        single_eval_pos,
        device=default_device,
        transform=None,
        **kwargs,
):
    num_params = kwargs.get("num_params") or DimensionPrior(num_features).sample()
    target_task = MultiFidelityTask(num_params, 23)

    x_list, y_list = [], []

    for i in range(batch_size // 2):

        # 1. Initialize the base state
        target_task.sample_task()
        target_task.sample_y0_ymax()

        # 2. Apply transform to get two related functional states
        # target_task and related_m are callables (BNNs)
        if transform:
            related_m = transform(target_task)
        else:
            # independent sampling of related task, no relation to target task
            related_m = target_task.clone()
            related_m.sample_task()

        # 3. Generate data for both models in the pair
        for current_task in [target_task, related_m]:
            # Inject the specific model into the task container for evaluation

            # Standard sequence generation logic
            n_levels = FidelityPrior().sample()
            allocation_prior = AllocationPrior(seq_len, n_levels)
            curve_configs = np.random.uniform(size=(seq_len, num_params))

            allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)
            # notice, how the current_task can e.g. be wrapped!
            curves = current_task.get_marginal_curve(
                torch.from_numpy(curve_configs).float()
            )

            x_i, y_i = allocation_prior.parse_allocation_into_sequence(
                curve_configs, curves, num_params, single_eval_pos, allocation
            )

            x_list.append(x_i)
            y_list.append(y_i)

    # Final batch size will be (batch_size * 2)
    x = torch.stack(x_list, dim=1).to(device).float()
    y = torch.stack(y_list, dim=1).to(device).float()

    # fixme: other attributes like single_eval_pos and style
    return MyBatch(x=x, y=y, target_y=y)

if __name__ == '__main__':


    # Example usage
    batch = get_batch(
        batch_size=4,
        seq_len=32,
        num_features=3,
        single_eval_pos=16,
        device="cpu",
        transform=None
    )
