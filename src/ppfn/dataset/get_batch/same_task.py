from collections.abc import Callable
from pathlib import Path
import torch
import numpy as np

from pfns4hpo.utils import default_device
from pfns4hpo.priors.utils import Batch

from ppfn.dataset.prior import AllocationPrior, DimensionPrior, FidelityPrior, MultiFidelityTask


@torch.no_grad()
def get_batch(
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
    This variant is proving the point, that cross attention will work. 
    This is a sanity check, because if we always are in the same task, 
    and we have twice as much datapoints, we should be more certain!
    
    For every batch, we sample a new dataset, that is the same for all tasks in the batch.
    Main difference: The sampled Trajectory (hp and budget allocation) differs across tasks
    
    :param batch_size: Number of tasks in the batch
    :param seq_len: Number of total datapoints per task
    :param num_features: Number of hyperparameters
    :param single_eval_pos: Position of the train test split in the sequence
    :param share_unrelated_tasks: Fraction of tasks in the batch that should come from different datasets
    :param device: Device to put the tensors on
    """

    num_params = DimensionPrior(num_features).sample()

    dataset_prior = MultiFidelityTask(num_params, 23)
    dataset_prior.sample_task()

    # determine the number of fidelity levels (ranging from 1: BB, up to seq_len)
    n_levels = FidelityPrior().sample()

    x = []
    y = []
    indicator = []

    # FIXME: (low-prio) efficiency: since all is the same task, we could just do one single fwd (get_marginal curve)
    #  and collect all sequences at once. No looping requried.

    for i in range(int(batch_size * (1 - share_unrelated_tasks))):
        # determine # observations/queries per curve
        # TODO: also make this a dirichlet thing
        allocation_prior = AllocationPrior(seq_len, n_levels)

        # determine config, x, y for every curve -----
        # (1) sample "available" hyperparameter configurations, these will later be subselected and
        # determined to be either observation or query points
        # FIXME: move this into the allocation prior, since it is basically an internal representation!
        curve_configs = np.random.uniform(size=(seq_len, num_params))

        # (2) get the curves for these configurations
        allocation = allocation_prior.sample_abstract_allocation(single_eval_pos)
        # get callable to evaluate (hp, t) --> y
        curves = dataset_prior.get_marginal_curve(torch.from_numpy(curve_configs).float())

        # (3) map the allocation to actual (x,y) values
        x_i, y_i = allocation_prior.parse_allocation_into_sequence(
            curve_configs, curves, num_params, single_eval_pos, allocation
        )
        x.append(x_i)
        y.append(y_i)
        indicator.append(0)  # same task indicator

    x = torch.stack(x, dim=1).to(device).float()
    y = torch.stack(y, dim=1).to(device).float()
    indicator = torch.tensor(indicator, device=device).long()

    return Batch(x=x, y=y, target_y=y, single_eval_pos=single_eval_pos, style=indicator)

# TODO move to utils.py
class Prior:
    def __init__(self, get_batch_fn: Callable):
        self.get_batch = get_batch_fn


if __name__ == "__main__":

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
