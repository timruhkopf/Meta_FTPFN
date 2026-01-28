from collections.abc import Callable
from pathlib import Path
import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch


from ppfn.dataset.prior import AllocationPrior, DimensionPrior, FidelityPrior, MultiFidelityTask

from ppfn.dataset.get_batch.ftpfn import get_batch as ftpfn_get_batch
from ppfn.model.mymodel.ft_ppfn import MyBatch

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

    target_curves = dataset_prior.get_marginal_curve(torch.from_numpy(all_configs[:, 0, :]).float())

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
        mixed_bnn_outputs = ((1 - alpha) * target_task_outputs[i - 1] + alpha * bnn_outputs.numpy())

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
            all_configs[:, i, :], related_task_curves[i], num_params, single_eval_pos, allocation
        )
        x.append(x_i)
        y.append(y_i)

    return  MyBatch(
        x=torch.cat(x, dim=1).to(device).float(), y=torch.cat(y, dim=1).to(device).float(),
        target_y=torch.cat(y, dim=1).to(device).float(),
        single_eval_pos=single_eval_pos,
        style=torch.tensor(indicator, device=device)
    )



@torch.no_grad()
def get_batch_train(
        batch_size: int,
        seq_len: int,
        num_features: int,

        single_eval_pos: int,
        num_params: int=None,
        n_levels: int=None,
        device=default_device,
        hyperparameters=None,
        **kwargs,
) -> Batch:
    """
    In this synthetic training scenario, we create a batch where every other example is
    the related task to the previous target task (i.e. related and targets are paired up and
    spliced in the batch).
    """
    assert batch_size % 2 == 0, "Batch size must be even for paired related/target tasks."

    if num_params is None:
        num_params = DimensionPrior(num_features).sample()


    if n_levels is None:
        n_levels = FidelityPrior().sample()


    relatedness = np.random.beta(0.1, 20, size=int(batch_size / 2))
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

        target_curves = dataset_prior.get_marginal_curve(
            torch.from_numpy(all_configs[:, target_task_idx, :]).float())

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
        mixed_bnn_outputs = (1 - alpha) * bnn_outputs_target \
                            + alpha * bnn_outputs_related

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
            all_configs[:, i, :], parametrized_curves[i], num_params, single_eval_pos, allocation
        )
        x.append(x_i)
        y.append(y_i)


    y = torch.stack(y, dim=1)
    return MyBatch(
        x=torch.stack(x, dim=1).to(device).float(), y=y.to(device).float(),
        target_y=y.to(device).float(),
        single_eval_pos=single_eval_pos,
        style=torch.tensor(indicator, device=device)
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
        num_params =num_params,
        n_levels = n_levels
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
            num_params =num_params,
            n_levels= n_levels,
        )

        return related_batch + unrelated_batch

class Prior:
    def __init__(self, get_batch_fn: Callable):
        self.get_batch = get_batch_fn


if __name__ == "__main__":

    get_batch_mixed(
        batch_size=12,
        seq_len=100,
        num_features=5,
        single_eval_pos=50,
        share_unrelated_tasks=0.3,
    )


    import os
    import time
    import cloudpickle
    from pfns4hpo.priors.utils import PriorDataLoader, DistributedPriorDataLoader, \
        get_batch_to_dataloader, get_expon_sep_sampler
    from tqdm import tqdm

    from dotenv import load_dotenv


    def store_batch(path, chunk_id, chunk_size, batch_size, seq_len, n_features, partition,
                    prior_hyperparameters):
        if partition:
            partition_id = chunk_id // 1000
            chunk_dir = os.path.join(path, f"partition_{partition_id}")
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_id}.pkl")
            os.makedirs(chunk_dir, exist_ok=True)
        else:
            chunk_file = os.path.join(path, f"chunk_{chunk_id}.pkl")

        if not os.path.exists(chunk_file):
            np.random.seed((os.getpid() * int(time.time())) % 123456789)
            chunk_data = []
            for bid in tqdm(range(chunk_size // batch_size)):
                if eval_pos_sampler is None:
                    # sample single eval pos log-uniformly ({1, ..., seq_len} log-uniformly - 1)
                    single_eval_pos = int(
                        np.floor(np.exp(np.random.uniform(0, np.log(seq_len + 1)))) - 1)
                else:
                    single_eval_pos = eval_pos_sampler()
                assert single_eval_pos < seq_len
                b = prior.get_batch(batch_size=batch_size,
                                    single_eval_pos=single_eval_pos,
                                    seq_len=seq_len,
                                    num_features=n_features,
                                    hyperparameters=prior_hyperparameters)
                chunk_data.append((single_eval_pos, b))
            with open(chunk_file, 'wb') as file:
                cloudpickle.dump(chunk_data, file)
        else:
            print("Already done.")


    load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

    path = os.getenv("DATADIR") + "priors/same_task_new_sample_prior/"
    os.makedirs(path, exist_ok=True)

    # dl = get_batch_to_dataloader(sampler.get_batch)
    prior = sampler
    eval_pos_sampler = get_expon_sep_sampler(seq_len=1000, base=2.0, min_eval_pos=1)

    # Test the raw store batch call  -------------
    store_batch(
        path=path,
        chunk_id=0,
        chunk_size=1000,
        batch_size=25,
        seq_len=1000,
        n_features=5,
        partition=True,
        prior_hyperparameters={},
    )

    # test storing via the PriorDataLoader interface ------------
    pdl = PriorDataLoader(load_path=path, subsample=1, n_chunks=10)

    # fixme: here we will need to call multiple times to get different numbers of features!
    pdl.store_prior(prior, local=True, chunk_size=20, batch_size=10, seq_len=1000, n_features=5,
                    partition=True, prior_hyperparameters={}, eval_pos_sampler=eval_pos_sampler)

    print()

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