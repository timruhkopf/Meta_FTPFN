from ppfn.dataset.get_batch.transforms.active_subspace import RandomAnchorSubspaceTransform, ActiveSubspaceTransform
from ppfn.dataset.get_batch.transforms.affine_shift import AffineShiftTransform, LogitAffineShiftTransform
from ppfn.dataset.get_batch.transforms.beta_warping import MobiusWarpingTransform, WarpingTransform
from ppfn.dataset.get_batch.transforms.bnn_interpolation import OutputInterpolationTransform
from ppfn.dataset.get_batch.transforms.fidelity_warp import FidelityWarpTransform
from ppfn.dataset.get_batch.transforms.latent_input import LatentInputTransform
from ppfn.dataset.get_batch.transforms.misc import InputWarpingTransform, VectorizedInputWarpingTransform
from ppfn.dataset.get_batch.transforms.rank_consistency import BetaRankTransform
from ppfn.dataset.get_batch.transforms.same_task import SameTaskTransform
from ppfn.dataset.prior import MultiFidelityTask


def debug_plot_transformation(transform, num_features=3):
    """
    Creates a comparison plot for a Target Task and its Related Task.
    """
    import matplotlib.pyplot as plt
    # 1. Initialize and Sample
    target = MultiFidelityTask(num_features, 23)
    target.sample_task()

    # 2. Clone and Transform
    related, relatedness = transform(target)

    # 3. Plotting
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    target.plot_surface(ax=ax1, title="Target Task (Original)")

    ax2 = fig.add_subplot(122, projection='3d')
    related.plot_surface(ax=ax2, title=f"Related Task ({type(transform).__name__}), Relatedness: {relatedness:.2f}")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':



    # for alpha in [0.1, 0.3, 0.7, 1.5, 2.0]:
    alpha = 0.3

    debug_plot_transformation(AffineShiftTransform())
    debug_plot_transformation(LogitAffineShiftTransform())
    debug_plot_transformation(MobiusWarpingTransform())
    debug_plot_transformation(BetaRankTransform())
    debug_plot_transformation(ActiveSubspaceTransform()) # these only make sense to look at in higher dim
    debug_plot_transformation(RandomAnchorSubspaceTransform()) # these only make sense to look at in higher dim
    debug_plot_transformation(WarpingTransform()) # error

    debug_plot_transformation(OutputInterpolationTransform(alpha=alpha, resample_y0_ymax=False))
    debug_plot_transformation(SameTaskTransform(resample_y0_ymax=False))
    debug_plot_transformation(FidelityWarpTransform(alpha=alpha, resample_y0_ymax=False))
    debug_plot_transformation(LatentInputTransform(resample_y0_ymax=False))
    debug_plot_transformation(InputWarpingTransform(resample_y0_ymax=False))
    debug_plot_transformation(VectorizedInputWarpingTransform(resample_y0_ymax=False, strength=0.8))

    OutputInterpolationTransform(alpha=0.5, resample_y0_ymax=True).plot_alpha_distribution()
    FidelityWarpTransform(alpha=None, resample_y0_ymax=True).plot_alpha_distribution()
    LatentInputTransform(sigma=0.3, resample_y0_ymax=True).plot_latent_distribution()
    VectorizedInputWarpingTransform(strength=0.8, resample_y0_ymax=True).plot_warp_distribution(num_inputs=4)
