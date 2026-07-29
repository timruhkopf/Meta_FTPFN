import torch
import numpy as np
import matplotlib.pyplot as plt

from ppfn.prior.harmonics.harmnoic_mixture_prior import HarmonicMixturePrior
from ppfn.prior.harmonics.heatmap_callback import HeatmapVisualizer
from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream



def create_mock_logits(Y_test, borders):
    """Simulates a trained model by generating log-probabilities clustered around the true Y values."""
    centers = (borders[:-1] + borders[1:]) / 2.0
    # Y_test is shape [Seq, Batch, 1], centers is [num_bins]
    # Broadcast subtraction to get distance from true curve
    diff = centers.view(1, 1, -1) - Y_test

    # Create a Gaussian "bump" of probabilities around the true value
    sigma = 0.8  # Spread of the heatmap band
    logits = -0.5 * (diff / sigma) ** 2
    return logits


def run_showcase():
    print("1. Initializing Prior and Data Stream...")
    prior = HarmonicMixturePrior(num_components=3, noise_std=0.2, scale=True, shift=True, warp=True)
    dataset = InfiniteHarmonicsStream(prior=prior, batch_size=4, n_A=25, n_B=75, n_test=150)

    # Define our discrete bins for the heatmap
    borders = torch.linspace(-8, 8, 100)  # 99 bins

    print("2. Generating a batch of data...")
    data_iter = iter(dataset)
    batch = next(data_iter)

    print("3. Simulating neural network predictions (Logits)...")
    # Stream A: Should learn the clean Canonical curve (A)
    logits_A = create_mock_logits(batch['test']['Y_A'], borders)

    # Stream B: Should learn the warped Observation curve (B)
    logits_B = create_mock_logits(batch['test']['Y_B'], borders)

    # Stream C: Flow matching, usually maps B back to A, so it targets A's curve
    logits_C = create_mock_logits(batch['test']['Y_A'], borders)

    print("4. Rendering Visualizations...")
    visualizer = HeatmapVisualizer(borders=borders, x_range=(-5, 5))

    fig = visualizer.generate_figure(
        batch_data=batch,
        logits=(logits_A, logits_B, logits_C)
    )

    save_path = "showcase_heatmaps.png"
    plt.show()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Done! Open '{save_path}' to see the result.")

if __name__ == '__main__':
    run_showcase()