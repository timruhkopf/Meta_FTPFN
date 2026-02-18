from collections.abc import Callable
import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch

from ppfn.dataset.prior import (
    AllocationPrior,
    DimensionPrior,
    FidelityPrior,
    MultiFidelityTask,
)

from ppfn.dataset.get_batch.ftpfn import get_batch as ftpfn_get_batch
from ppfn.model.mymodel.ft_ppfn import MyBatch


# TODO split into multiple files?
def get_batch_eval(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    device=default_device,
    hyperparameters=None,
    **kwargs,
) -> Batch:
    """
    This is the synthetic eval scenario, in which we have a single target task
    and all other tasks are related to it by interpolating between their BNN outputs.
    Notably, all tasks have their own HP samples.
    """
    num_params = DimensionPrior(num_features).sample()

    # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
    n_levels = FidelityPrior().sample()

    dataset_prior = MultiFidelityTask(num_params, 23)
    dataset_prior.sample_task()  # sample initial task

    all_configs = np.random.uniform(
        size=(seq_len, batch_size, num_params)
    )  # sample all configs for the task at once

    target_curves = dataset_prior.get_marginal_curve(
        torch.from_numpy(all_configs[:, 0, :]).float()
    )

    # get all the related curves from the target task's perspective
    target_task_outputs = []
    for i in range(1, batch_size):
        hyperparams = torch.from_numpy(all_configs[:, i, :]).float()
        with torch.no_grad():
            bnn_outputs = dataset_prior.model(hyperparams)
        target_task_outputs.append(bnn_outputs.numpy())

    relatedness = np.random.beta(0.1, 20, size=(batch_size,))
    related_task_curves = []
    for i in range(1, batch_size):
        # sample a new task
        dataset_prior.model = dataset_prior.bnn_prior.sample()

        hyperparams = torch.from_numpy(all_configs[:, i, :]).float()
        with torch.no_grad():  # unbounded Bnn outputs are need to be bounded to the parameter ranges (and looked up in the y ecdf)
            bnn_outputs = dataset_prior.model(hyperparams)

        # interpolate between both tasks' bnn outputs
        alpha = relatedness[i - 1]
        mixed_bnn_outputs = (1 - alpha) * target_task_outputs[
            i - 1
        ] + alpha * bnn_outputs.numpy()

        dataset_prior.sample_y0_ymax()  # move the curves
        parametrized_curve_model = dataset_prior.linker.curve_factory(
            mixed_bnn_outputs, dataset_prior.y0, dataset_prior.ymax, noise=True
        )
        related_task_curves.append(parametrized_curve_model)

    # now we get all the allocations and map them to (x,y) values
    x = []
    y = []
    indicator = [0, *relatedness]
    related_task_curves.insert(0, target_curves)  # first is the target task

    for i in range(batch_size):
        # determine # observations/queries per curve
        allocation_prior = AllocationPrior(seq_len, n_levels)
        allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)

        x_i, y_i = allocation_prior.parse_allocation_into_sequence(
            all_configs[:, i, :],
            related_task_curves[i],
            num_params,
            single_eval_pos,
            allocation,
        )
        x.append(x_i)
        y.append(y_i)

    return MyBatch(
        x=torch.cat(x, dim=1).to(device).float(),
        y=torch.cat(y, dim=1).to(device).float(),
        target_y=torch.cat(y, dim=1).to(device).float(),
        single_eval_pos=single_eval_pos,
        style=torch.tensor(indicator, device=device),
    )


@torch.no_grad()
def get_batch_train(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    num_params: int = None,
    n_levels: int = None,
    device=default_device,
    hyperparameters=None,
    **kwargs,
) -> Batch:
    """
    In this synthetic training scenario, we create a batch where every other example is
    the related task to the previous target task (i.e. related and targets are paired up and
    spliced in the batch).
    """
    assert batch_size % 2 == 0, (
        "Batch size must be even for paired related/target tasks."
    )

    if num_params is None:
        num_params = DimensionPrior(num_features).sample()

    if n_levels is None:
        n_levels = FidelityPrior().sample()

    relatedness = kwargs.get("relatedness", None)
    if relatedness is not None:
        assert len(relatedness) == batch_size // 2, (
            "Relatedness array length must match half the batch size."
        )
        alpha = kwargs.get("alpha", 0.1)
        beta = kwargs.get("beta", 20)

        relatedness = np.random.beta(alpha, beta, size=int(batch_size / 2))

    indicator = relatedness.repeat(2)

    # both the target and related task in one go
    # sample all configs for the task at once
    all_configs = np.random.uniform(size=(seq_len, batch_size, num_params))
    parametrized_curves = []
    for i in range(batch_size // 2):
        # Target task
        target_task_idx = i * 2
        related_task_idx = target_task_idx + 1
        dataset_prior = MultiFidelityTask(num_params, 23)
        dataset_prior.sample_task()  # sample initial task

        # fixing the performance range if provided (for plotting mainly)
        y0, ymax = kwargs.get("y0", None), kwargs.get("ymax", None)
        if y0 is not None and ymax is not None:
            dataset_prior.y0 = y0
            dataset_prior.ymax = ymax

        target_curves = dataset_prior.get_marginal_curve(
            torch.from_numpy(all_configs[:, target_task_idx, :]).float()
        )

        # get all the curves' parameters from the target task's perspective
        hyperparams = torch.from_numpy(all_configs[:, target_task_idx, :]).float()
        with torch.no_grad():
            bnn_outputs_target = dataset_prior.model(hyperparams).numpy()

        # get all the related curves' parameters from the target task's perspective
        hyperparams = torch.from_numpy(all_configs[:, related_task_idx, :]).float()
        with torch.no_grad():
            bnn_outputs_related = dataset_prior.model(hyperparams).numpy()

        # interpolate between both tasks' bnn outputs
        alpha = relatedness[i]
        mixed_bnn_outputs = (
            1 - alpha
        ) * bnn_outputs_target + alpha * bnn_outputs_related

        # fixing the performance range if provided (for plotting mainly)
        y0, ymax = kwargs.get("y0", None), kwargs.get("ymax", None)
        if y0 is not None and ymax is not None:
            dataset_prior.y0 = y0
            dataset_prior.ymax = ymax
        else:
            dataset_prior.sample_y0_ymax()  # move the curves

        parametrized_curve_model = dataset_prior.linker.curve_factory(
            mixed_bnn_outputs, dataset_prior.y0, dataset_prior.ymax, noise=True
        )

        parametrized_curves.extend((target_curves, parametrized_curve_model))

    # now we get all the allocations and map them to (x,y) values
    x = []
    y = []

    for i in range(batch_size):
        # determine # observations/queries per curve
        allocation_prior = AllocationPrior(seq_len, n_levels)
        allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)

        x_i, y_i = allocation_prior.parse_allocation_into_sequence(
            all_configs[:, i, :],
            parametrized_curves[i],
            num_params,
            single_eval_pos,
            allocation,
        )
        x.append(x_i)
        y.append(y_i)

    y = torch.stack(y, dim=1)

    return MyBatch(
        x=torch.stack(x, dim=1).to(device).float(),
        y=y.to(device).float(),
        target_y=y.to(device).float(),
        single_eval_pos=single_eval_pos,
        style=torch.tensor(indicator, device=device),
    )


@torch.no_grad()
def get_batch_mixed(
    batch_size: int,
    seq_len: int,
    num_features: int,
    single_eval_pos: int,
    share_unrelated_tasks: float = 0.0,
    device=default_device,
    hyperparameters=None,
    num_params: int = None,
    **kwargs,
):
    """
    This prior will sample batches,
    - where a fraction of the batch (1 - share_unrelated_tasks) are related tasks
      to each other by interpolating between their BNN outputs (as in get_batch_eval)
    - and the remaining fraction (share_unrelated_tasks) are unrelated tasks
      generated by the ftpfn_get_batch prior.

    It assumes an implicit pairing in the batch dim of (A1, A2, B1, B2, C1, C2, ...) in the
    related tasks, where A1 is related to A2, B1 to B2, etc.

    returns: Batch object with:
        x: (seq_len, batch_size, num_features) tensor of hyperparameter configurations
        y: (seq_len, batch_size) tensor of observed performances
        style: (batch_size,) tensor indicating whether the task is related (0) or unrelated (1)
        seq_len: int, the length of the sequences
    """
    share_related = 1.0 - share_unrelated_tasks
    n_related = int(batch_size * share_related)
    n_related = n_related if n_related % 2 == 0 else n_related + 1  # make even

    n_unrelated = int(batch_size * share_unrelated_tasks)
    n_unrelated = n_unrelated if n_unrelated % 2 == 0 else n_unrelated + 1  # make even

    if num_params is None:
        num_params = DimensionPrior(num_features).sample()

    # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
    # n_levels = FidelityPrior().sample()
    n_levels = None

    related_batch = get_batch_train(
        batch_size=n_related,
        seq_len=seq_len,
        num_features=num_features,
        single_eval_pos=single_eval_pos,
        device=device,
        hyperparameters=hyperparameters,
        num_params=num_params,
        n_levels=n_levels,
        **kwargs,
    )
    if n_unrelated == 0:
        return related_batch
    else:
        unrelated_batch = ftpfn_get_batch(
            batch_size=n_unrelated,
            seq_len=seq_len,
            num_features=num_features,
            single_eval_pos=single_eval_pos,
            device=device,
            hyperparameters=hyperparameters,
            num_params=num_params,
            n_levels=n_levels,
        )
        unrelated_batch = MyBatch(
            x=unrelated_batch.x,
            y=unrelated_batch.y,
            target_y=unrelated_batch.target_y,
            single_eval_pos=unrelated_batch.single_eval_pos,
            style=torch.ones(n_unrelated, device=device),  # that is what we need to add
        )

        return related_batch + unrelated_batch


# TODO move to utils
class Prior:
    def __init__(self, get_batch_fn: Callable):
        self.get_batch = get_batch_fn

#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#     from scipy.stats import beta as beta_dist
#
#     import torch
#     import os
#     from pathlib import Path
#
#     def plot_beta_pdf(alpha, beta):
#         """Plots the probability density of your relatedness factor."""
#         x = np.linspace(0, 1, 100)
#         y = beta_dist.pdf(x, alpha, beta)
#         plt.figure(figsize=(6, 3))
#         plt.plot(x, y, "r-", lw=2)
#         plt.fill_between(x, y, alpha=0.2, color="red")
#         plt.title(f"Relatedness Distribution: Beta(α={alpha}, β={beta})")
#         plt.xlabel("Relatedness Factor (0=Identical, 1=Independent)")
#         plt.ylabel("Density")
#         plt.grid(True, alpha=0.3)
#         plt.show()
#
#     def plot_relatedness_static(batch, pair_idx=0):
#         """
#         Standard Matplotlib 3D plot.
#         Renders directly in PyCharm's SciView or a popup window.
#         """
#         # 1. Extraction (T, B, D structure)
#         target_idx = pair_idx * 2
#         related_idx = target_idx + 1
#
#         # x: [T, B, D], y: [T, B]
#         # We take the first two dimensions of X for the ground plane
#         x_target = batch.x[:, target_idx, 1:3].detach().cpu().numpy()
#         y_target = batch.y[:, target_idx].detach().cpu().numpy()
#
#         x_related = batch.x[:, related_idx, 1:3].detach().cpu().numpy()
#         y_related = batch.y[:, related_idx].detach().cpu().numpy()
#
#         rel_val = batch.style[related_idx].item()
#
#         # 2. Setup Figure
#         fig = plt.figure(figsize=(12, 5))
#
#         # Subplot 1: The Beta Distribution (To see the "Why")
#         ax1 = fig.add_subplot(1, 2, 1)
#         x_beta = np.linspace(0, 1, 500)
#         y_beta = beta_dist.pdf(x_beta, 0.1, 20)
#         ax1.plot(x_beta, y_beta, color="red", lw=2)
#         ax1.fill_between(x_beta, y_beta, color="red", alpha=0.1)
#         ax1.axvline(
#             rel_val,
#             color="black",
#             linestyle="--",
#             label=f"Current Alpha: {rel_val:.4f}",
#         )
#         ax1.set_title("Beta(0.1, 20) Prior")
#         ax1.set_xlabel("Relatedness Factor")
#         ax1.legend()
#
#         # Subplot 2: The 3D Task Surface
#         ax2 = fig.add_subplot(1, 2, 2, projection="3d")
#
#         # Plot Target points
#         ax2.scatter(
#             x_target[:, 0],
#             x_target[:, 1],
#             y_target,
#             c="blue",
#             label="Target Task",
#             s=10,
#             alpha=0.6,
#         )
#
#         # Plot Related points
#         ax2.scatter(
#             x_related[:, 0],
#             x_related[:, 1],
#             y_related,
#             c="red",
#             label="Related Task",
#             s=10,
#             alpha=0.6,
#         )
#
#         ax2.set_title(f"Task Comparison (α={rel_val:.6f})")
#         ax2.set_xlabel("HP 1")
#         ax2.set_ylabel("HP 2")
#         ax2.set_zlabel("Y Value")
#         ax2.legend()
#
#         plt.tight_layout()
#         plt.show()
#
#     # 1. Setup Parameters
#     batch_size = 4
#     seq_len = 50
#     num_features = 5
#     single_eval_pos = 40
#
#     # Controlled relatedness for plotting:
#     # [Very Similar, Somewhat Similar, Different, Very Different]
#     alpha = 0.1
#     beta = 20
#     # Concentration and Spread
#     # * Both > 1 (Unimodal): The distribution has a "hump" (mode). If $\alpha = \beta$, the mass
#     # is perfectly centered at $0.5$.
#     # * Both < 1 (U-Shaped): The mass pushes toward the boundaries ($0$ and $1$),
#     # making extreme values more likely than middle values.
#     # * One < 1 and One > 1: The mass accumulates aggressively
#     # at one of the boundaries. In your specific case ($\alpha=0.1, \beta=20$), you have an L-shaped distribution
#     # where the density spikes at 0 and decays rapidly.
#     relatedness = np.random.beta(alpha, beta, size=(batch_size // 2,))
#
#     # 2. Generate Batch
#     # Note: Modify your get_batch_train to accept 'fixed_relatedness' if desired,
#     # or just use the logic below to visualize a standard sample.
#     batch = get_batch_train(
#         batch_size=batch_size,
#         seq_len=seq_len,
#         num_features=num_features,
#         single_eval_pos=single_eval_pos,
#         # alpha=0.1,
#         # beta=20
#         relatedness=relatedness,
#         y0=0.5,
#         ymax=1,
#     )
#
#     plot_relatedness_static(batch, pair_idx=1)
#     # 4. Plot the Distribution PDF for Context
#     x = np.linspace(0, 1, 100)
#     y = beta_dist.pdf(x, 0.1, 20)
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, y, color="green")
#     plt.fill_between(x, y, alpha=0.2, color="green")
#     plt.title("Beta(0.1, 20) Distribution PDF")
#     plt.xlabel("Relatedness Factor (alpha)")
#     plt.ylabel("Density")
#     plt.show()
#
#     get_batch_mixed(
#         batch_size=12,
#         seq_len=100,
#         num_features=5,
#         single_eval_pos=50,
#         share_unrelated_tasks=0.3,
#     )
#
#     import time
#     import cloudpickle
#     from pfns4hpo.priors.utils import PriorDataLoader, get_expon_sep_sampler
#     from tqdm import tqdm
#
#     from dotenv import load_dotenv
#
#     def store_batch(
#         path,
#         chunk_id,
#         chunk_size,
#         batch_size,
#         seq_len,
#         n_features,
#         partition,
#         prior_hyperparameters,
#     ):
#         if partition:
#             partition_id = chunk_id // 1000
#             chunk_dir = os.path.join(path, f"partition_{partition_id}")
#             chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pkl")
#             os.makedirs(chunk_dir, exist_ok=True)
#         else:
#             chunk_file = os.path.join(path, f"chunk_{chunk_id}.pkl")
#
#         if not os.path.exists(chunk_file):
#             np.random.seed((os.getpid() * int(time.time())) % 123456789)
#             chunk_data = []
#             for bid in tqdm(range(chunk_size // batch_size)):
#                 if eval_pos_sampler is None:
#                     # sample single eval pos log-uniformly ({1, ..., seq_len} log-uniformly - 1)
#                     single_eval_pos = int(
#                         np.floor(np.exp(np.random.uniform(0, np.log(seq_len + 1)))) - 1
#                     )
#                 else:
#                     single_eval_pos = eval_pos_sampler()
#                 assert single_eval_pos < seq_len
#                 b = prior.get_batch(
#                     batch_size=batch_size,
#                     single_eval_pos=single_eval_pos,
#                     seq_len=seq_len,
#                     num_features=n_features,
#                     hyperparameters=prior_hyperparameters,
#                 )
#                 chunk_data.append((single_eval_pos, b))
#             with open(chunk_file, "wb") as file:
#                 cloudpickle.dump(chunk_data, file)
#         else:
#             print("Already done.")
#
#     load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")
#
#     path = os.getenv("DATADIR") + "priors/same_task_new_sample_prior/"
#     os.makedirs(path, exist_ok=True)
#
#     # dl = get_batch_to_dataloader(sampler.get_batch)
#     # prior = sampler
#     eval_pos_sampler = get_expon_sep_sampler(seq_len=1000, base=2.0, min_eval_pos=1)
#
#     # Test the raw store batch call  -------------
#     store_batch(
#         path=path,
#         chunk_id=0,
#         chunk_size=1000,
#         batch_size=25,
#         seq_len=1000,
#         n_features=5,
#         partition=True,
#         prior_hyperparameters={},
#     )
#
#     # test storing via the PriorDataLoader interface ------------
#     pdl = PriorDataLoader(load_path=path, subsample=1, n_chunks=10)
#
    # # fixme: here we will need to call multiple times to get different numbers of features!
    # pdl.store_prior(
    #     prior,
    #     local=True,
    #     chunk_size=20,
    #     batch_size=10,
    #     seq_len=1000,
    #     n_features=5,
    #     partition=True,
    #     prior_hyperparameters={},
    #     eval_pos_sampler=eval_pos_sampler,
    # )
    #
    # print()

    # # taken from pfns4hpo.main ----------------------------------------

    # if configs["load_path"] is None:
    #     get_batch_func = prior.get_batch  # !!!!!!!!!!!!!
    # else:
    #     assert (
    #         configs["batch_size"] == 25
    #     )  # priors are assumed to be stored with batch size 25
    #     if configs["num_gpus"] == 1:
    #         prior_data = priors.utils.PriorDataLoader(
    #             configs["load_path"],
    #             subsample=configs["subsample"],
    #             n_chunks=configs["n_chunks"],
    #         )
    #     else:
    #         prior_data = priors.utils.DistributedPriorDataLoader(
    #             configs["load_path"],
    #             subsample=configs["subsample"],
    #             n_chunks=configs["n_chunks"],
    #             n_gpus=configs["num_gpus"],
    #         )
    #     get_batch_func = lambda *args, **kwargs: prior_data.get_batch(
    #         kwargs.get("device", args[4] if len(args) >= 5 else "cpu")
    #     )
    #     )

    #  while offset < configs["border_batch_size"]:
    #     print(offset)
    #     ys = get_batch_func(  # !!!!!!!!!
    #         configs["batch_size"],
    #         configs["bptt"],
    #         num_features,
    #         hyperparameters=hps,
    #         single_eval_pos=configs["bptt"],
    #     )
    #     _, eff_batch_size = ys.target_y.shape
    #     ys_bucket[
    #         :, offset : min(offset + eff_batch_size, configs["border_batch_size"])
    #     ] = ys.target_y[
    #         :, : min(eff_batch_size, configs["border_batch_size"] - offset)
    #     ]
    #     offset += eff_batch_size

    # bucket_limits = bar_distribution.get_bucket_limits(
    #     configs["num_borders"], ys=ys_bucket
    # )

    # if configs["load_path"] is None:
    #     single_eval_pos_gen = utils.get_weighted_single_eval_pos_sampler(
    #         max_len=configs["bptt"],
    #         min_len=0,
    #         p=configs["power_single_eval_pos_sampler"],
    #     )
    # else:
    #     single_eval_pos_gen = lambda *args, **kwargs: prior_data.get_single_eval_pos()

    # configs_train.update(
    #     dict(
    #         priordataloader_class=priors.get_batch_to_dataloader(get_batch_func),
    #         criterion=criterion,
    #         encoder_generator=prior.get_encoder(),
    #         y_encoder_generator=encoders.get_normalized_uniform_encoder(
    #             encoders.Linear
    #         ),
    #         scheduler=utils.get_cosine_schedule_with_warmup,
    #         extra_prior_kwargs_dict={
    #             # "num_workers": 10,
    #             "num_features": num_features,
    #             "hyperparameters": {
    #                 **hps,
    #             },
    #         },
    #         single_eval_pos_gen=single_eval_pos_gen,
    #         **configs["model_extra_args"],
    #     )
    # )

    # _, _, model, _ = train.train(**configs_train)

    # torch.save(model, os.path.join("final_models", configs["output_file"]))

    # # parsed default config from main.py  --------------------
    # default_config = {
    # "warmup_epochs": -1,
    # "nlayers": 12,
    # "emsize": 512,
    # "batch_size": 8,
    # "epochs": None,  # REQUIRED: No default provided
    # "num_borders": 10000,
    # "ncurves_per_example": 50,
    # "max_epochs_per_curve": 50,
    # "lr": 0.0001,
    # "seq_len": None,
    # "aggregate_k_gradients": 1,
    # "steps_per_epoch": 100,
    # "train_mixed_precision": False,
    # "prior": None,  # REQUIRED: No default provided
    # "run_on_submitit": False,
    # "num_gpus": 1,
    # "time": 1435,  # Calculated from 23 * 60 + 55
    # "partition": "alldlc_gpu-rtx2080",
    # "output_file": None,
    # "num_features": 20,
    # "border_batch_size": 10,
    # "prior_hps": {},  # Parsed from "{}"
    # "load_path": None,
    # "power_single_eval_pos_sampler": -2,
    # "model_extra_args": {},  # Parsed from "{}"
    # "nhead": 4,
    # "subsample": 1,
    # "n_chunks": 2000,
    # "full_support": True,
    # "linspace_borders": False,
    # }
