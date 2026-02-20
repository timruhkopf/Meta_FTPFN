from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
    eta_min: float = 1e-6,
    start_factor: float = 0.001,
):
    # 1. Define the Warmup Scheduler
    warmup_sched = LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup_epochs
    )

    # 2. Define the Cosine Decay Scheduler
    # T_max is the remaining epochs after warmup
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=(max_epochs - warmup_epochs), eta_min=eta_min
    )

    # 3. Combine them
    return SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )
