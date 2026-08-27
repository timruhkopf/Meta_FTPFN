import torch
from ppfn.prior.harmonics_fix.stream_dataset import InfiniteHarmonicsStream
from ppfn.prior.harmonics_fix.harmonic_mixture_prior import HarmonicMixturePrior
from ppfn.model.anamorphic.ideas.enc_dec_tabpfnv3.train import train_student, MetaBatch
from ppfn.model.anamorphic.ideas.enc_dec_tabpfnv3.model import WarpAlignPFN, WarpAlignConfig
from ppfn.model.anamorphic.ideas.enc_dec_tabpfnv3.losses import LossWeights


def create_metabatch_iterator(stream, device="cuda"):
    """
    Adapts the dictionary output of InfiniteHarmonicsStream into
    the MetaBatch dataclass expected by the training loop.
    """
    for batch in stream:
        train_data = batch['train']
        test_data = batch['test']
        params = batch['params']

        # MetaBatch expects x_A and y_A to contain both train and test splits concatenated along the row dimension (T).
        x_A = torch.cat([train_data['X_A'], test_data['X_A']], dim=0).to(device)
        y_A = torch.cat([train_data['Y_A'], test_data['Y_A']], dim=0).to(device).squeeze(-1)
        n_A_train = train_data['X_A'].shape[0]

        # Standardize the A targets using A_train's mean and std so the fixed bar-distribution grid applies uniformly.
        y_A_train = y_A[:n_A_train]
        mean_y = y_A_train.mean(dim=0, keepdim=True)
        std_y = y_A_train.std(dim=0, keepdim=True).clamp_min(1e-6)

        y_A_std = (y_A - mean_y) / std_y

        # Retrieve B's observed data (already strictly drifted by the sampler).
        x_B = train_data['X_B_obs'].to(device)
        y_B = train_data['Y_B_obs'].to(device).squeeze(-1)

        # Retrieve the undrifted B data for the training-only oracle[cite: 2].
        # It must also be standardized using A_train's stats so the translation head matches A's bins.
        y_B_inA = train_data['Y_B_in_A'].to(device).squeeze(-1)
        y_B_inA_std = (y_B_inA - mean_y) / std_y

        # The severity measures how unrelated the B task is.
        # is_unrelated = (n_shared == 0)[cite: 2], which we map to a 1.0 (unrelated) or 0.0 (related) float.
        severity = params['is_unrelated'].float().to(device)

        yield MetaBatch(
            x_A=x_A,
            y_A=y_A_std,
            n_A_train=n_A_train,
            x_B=x_B,
            y_B=y_B,  # Handled internally by the B target embedder
            severity=severity,
            y_B_inA=y_B_inA_std,  # Supervised in A's bar distribution bins[cite: 5]
            x_A_pool=None,  # Leave None unless doing Phase 0 teacher distillation[cite: 3]
            y_A_pool=None
        )


def run_training_pipeline():
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize the Prior and Dataset Stream[cite: 1, 2]
    # We use a max of 2K components for A's context to keep it under-determined[cite: 2].
    prior = HarmonicMixturePrior(
        num_components=4,
        device=device
    )
    stream = InfiniteHarmonicsStream(
        prior=prior,
        batch_size=32,
        n_A_range=(3, 10),
        n_B_range=(40, 80)
    )

    # 2. Setup the Data Adapter[cite: 3]
    metabatch_iter = create_metabatch_iterator(stream, device=device)

    # 3. Initialize Model and Configuration[cite: 5]
    config = WarpAlignConfig(
        emsize=192,
        nhead=3,
        n_bins=256
    )

    print("Initializing Student Model...")
    student = WarpAlignPFN(config=config, device=device)

    # Note: Phase 1 assumes training a student against a frozen Phase 0 teacher[cite: 3].
    # If you want to warm-start with a teacher, you can build it here. We will set it to None
    # to train purely on the task NLL and auxiliary translation losses without distillation.
    teacher = None

    # Define Loss Weights. We set KL to 0.0 since we are skipping the teacher[cite: 6].
    weights = LossWeights(
        kl=0.0,
        translation=0.5,
        variance=0.1,
        severity=0.1,
        mmd=0.0
    )

    print("Starting training loop...")
    # 4. Execute Phase 1 Student Training[cite: 3]
    trained_student = train_student(
        student=student,
        teacher=teacher,
        batches=metabatch_iter,
        total_steps=5000,  # Adjust based on your convergence needs
        lr=3e-4,
        weights=weights,
        anneal_aux_from=0.6,
        log_every=100,
        device=device
    )

    print("Training complete.")
    return trained_student


if __name__ == "__main__":
    run_training_pipeline()