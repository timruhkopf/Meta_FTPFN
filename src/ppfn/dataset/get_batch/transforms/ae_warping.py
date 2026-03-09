import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ScalableResAE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, depth):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        enc_dims = np.linspace(input_dim, bottleneck_dim, depth + 2).astype(int)
        dec_dims = np.linspace(bottleneck_dim, input_dim, depth + 2).astype(int)

        # Build Encoder
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*enc_layers)

        # Build Decoder
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming init with centered uniform bias to prevent severe "drift"
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                nn.init.uniform_(m.bias, -0.1, 0.1)

        # Final layer
        linear_decoders = [m for m in self.decoder if isinstance(m, nn.Linear)]
        nn.init.normal_(linear_decoders[-1].weight, mean=0.0, std=1.0)
        nn.init.zeros_(linear_decoders[-1].bias)

    def folding_op(self, x):
        return torch.abs(((x + 1.0) % 2.0) - 1.0)

    def forward(self, hp, alpha=1.0):
        latent = self.encoder(hp)
        delta = self.decoder(latent)
        return self.folding_op(hp + alpha * delta)


def target_function(hp1, hp2):
    """A synthetic 'performance' metric with distinct peaks and valleys."""
    return np.sin(2 * np.pi * hp1) * np.cos(2 * np.pi * hp2)


def visualize_functional_warp(model, ax1, ax2, target_function, alpha=1.5):
    # 2. Create the base HP search grid
    resolution = 50
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    HP1, HP2 = np.meshgrid(x, y)

    # Base Performance Landscape (Task A)
    Z_base = target_function(HP1, HP2)

    # 3. Apply the Warp Prior
    flat_coords = np.stack([HP1.ravel(), HP2.ravel()], axis=1)
    hp_tensor = torch.tensor(flat_coords, dtype=torch.float32)

    with torch.no_grad():
        warped_tensor = model(hp_tensor, alpha=alpha)

    warped_coords = warped_tensor.numpy()

    # Warped Performance Landscape (Task B)
    # We evaluate the underlying function at the NEW coordinates
    HP1_warped = warped_coords[:, 0].reshape(resolution, resolution)
    HP2_warped = warped_coords[:, 1].reshape(resolution, resolution)
    Z_warped = target_function(HP1_warped, HP2_warped)

    ax1.plot_surface(HP1, HP2, Z_base, cmap='viridis', edgecolor='k', linewidth=0.1)
    ax1.set_title("Base Task (Alpha = 0.0)\nPerformance Landscape")
    ax1.set_xlabel('HP 1');
    ax1.set_ylabel('HP 2');
    ax1.set_zlabel('Performance (y)')
    ax1.set_zlim([-1.5, 1.5])

    # Related Task B (Plotted against the ORIGINAL HP grid!)

    ax2.plot_surface(HP1, HP2, Z_warped, cmap='viridis', edgecolor='k', linewidth=0.1)
    ax2.set_title(f"Related Task (Alpha = {alpha})\nObserved Performance Landscape")
    ax2.set_xlabel('HP 1');
    ax2.set_ylabel('HP 2');
    ax2.set_zlabel('Performance (y)')
    ax2.set_zlim([-1.5, 1.5])

    return ax1, ax2


def visualize_vector_field(model, ax, alpha=1.0):
    # 2. Create the base HP search grid
    # Using a lower resolution (e.g., 20x20) so the arrows aren't too crowded
    resolution = 20
    x = np.linspace(0.02, 0.98, resolution)  # Slightly offset from edges to see folding better
    y = np.linspace(0.02, 0.98, resolution)
    HP1, HP2 = np.meshgrid(x, y)

    flat_coords = np.stack([HP1.ravel(), HP2.ravel()], axis=1)
    hp_tensor = torch.tensor(flat_coords, dtype=torch.float32)

    # 3. Apply the Warp Prior
    with torch.no_grad():
        warped_tensor = model(hp_tensor, alpha=alpha)

    warped_coords = warped_tensor.numpy()

    HP1_warped = warped_coords[:, 0].reshape(resolution, resolution)
    HP2_warped = warped_coords[:, 1].reshape(resolution, resolution)

    # 4. Calculate the Vector Field (Displacement)
    U = HP1_warped - HP1
    V = HP2_warped - HP2

    # Calculate magnitude for background coloring (helps spot dead zones)
    magnitude = np.sqrt(U ** 2 + V ** 2)

    # Draw background contour to show regions of high vs low distortion
    contour = ax.contourf(HP1, HP2, magnitude, cmap='Blues', alpha=0.4, levels=15)
    # cbar = ax.colorbar(contour, ax=ax)
    # cbar.set_label('Warp Magnitude (Distance Moved)', rotation=270, labelpad=15)

    # Quiver plot for the arrows
    # scale_units='xy' and scale=1 ensures the arrow length exactly matches the displacement
    ax.quiver(HP1, HP2, U, V,
              color='black',
              angles='xy', scale_units='xy', scale=1,
              width=0.003, headwidth=4, headlength=6)

    # Add red boundary box to show the valid [0, 1] HP space
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='red', linewidth=2, linestyle='--')

    ax.set_title(
        f"HP Space Vector Field (Alpha = {alpha})\nArrows point from Base Task configs to Related Task configs")
    ax.set_xlabel('Hyperparameter 1')
    ax.set_ylabel('Hyperparameter 2')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, linestyle=':', alpha=0.6)

    return ax


def plot(model, target_function=target_function, alpha=1.0):
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection='3d')

    visualize_functional_warp(model, ax1, ax3, alpha=alpha, target_function=target_function)
    visualize_vector_field(model, ax=ax2, alpha=alpha)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = ScalableResAE(input_dim=2, bottleneck_dim=2, depth=1)
    model.eval()

    plot(model, alpha=1)
    # Run the visualizer!
    # Try running this a few times to see different random initializations.
    visualize_vector_field(model, alpha=1)
    # # Run the visualization
    visualize_functional_warp(model, alpha=1)
