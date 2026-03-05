import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ppfn.model.mymodel.layers.adapter_wrapper import Unified1dValidationWrapper
from ppfn.model.mymodel.layers.delta_surrogate_adapter import  DeltaSurrogateAdapter
# from validated_layers_1d import generate_shared_complex_batch
from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.model.mymodel.layers.nw_adapter import NadarayaWatsonAdapter


def plot_learning_curve(ax, loss_history, model_name):
    ax.plot(loss_history, color='purple', alpha=0.8)
    ax.set_title(f'Learning Curve (MSE) - {model_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True)


def plot_alignment(ax, A_plot, B_plot, C_plot, A_train, B_belief_plot, sep):
    ax.scatter(B_plot[:, 0], B_plot[:, 1], c='gray', alpha=0.5, label='Task B (Distorted)')
    ax.plot(A_plot[:, 0], A_plot[:, 1], 'g--', linewidth=2, label='Task A (GT)')
    ax.scatter(A_train[:, 0], A_train[:, 1], c='green', s=100, marker='*', label='A Train')
    ax.scatter(B_belief_plot[:, 0], B_belief_plot[:, 1], c='red', s=40, marker='x', label='B Belief')
    ax.scatter(C_plot[:, 0], C_plot[:, 1], c='blue', s=30, label='Stream C')

    # Draw error vectors
    for i in range(sep):
        ax.plot([A_train[i, 0], B_belief_plot[i, 0]],
                [A_train[i, 1], B_belief_plot[i, 1]], 'r:', alpha=0.4)
    ax.set_title('Zero-Shot Alignment')
    ax.legend()
    ax.grid(True)


def plot_hidden_warp(ax, debug_info, x_limits=None):
    t_B, distorted_t_B = debug_info['t_B'], debug_info['distorted_t_B']
    base_B, task_B = debug_info['base_B'], debug_info['task_B']

    ax.plot(distorted_t_B, base_B, 'k--', alpha=0.4, linewidth=2, label='Pristine (Hidden)')
    ax.scatter(t_B, task_B, c='gray', s=20, label='Task B (Observed)')
    for i in range(len(t_B)):
        ax.annotate('', xy=(t_B[i], task_B[i]), xytext=(distorted_t_B[i], base_B[i]),
                    arrowprops=dict(arrowstyle="->", color="orange", alpha=0.6))
    if x_limits: ax.set_xlim(x_limits)
    ax.set_title('Ground Truth Displacement')
    ax.legend()
    ax.grid(True)


def plot_attention_heatmap(ax, fig, attn_matrix, b_cutoff_idx=None, title="Attention"):
    im = ax.imshow(attn_matrix.T, cmap='viridis', aspect='auto', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Queries')
    ax.set_ylabel('Keys')
    if b_cutoff_idx:
        ax.axvline(x=b_cutoff_idx, color='red', linestyle='--', label='Extrapolation Boundary')
        ax.legend()
    fig.colorbar(im, ax=ax, label='Weight')


def run_inference_for_plot(wrapper_model, sep, T, batch_size=1):
    wrapper_model.eval()
    with torch.no_grad():
        # Data generation gives us the streams naturally
        (A_train, A_test, B_train, B_test,
         C_train, C_test, B_belief_A_train,
         sep, debug_info) = generate_shared_complex_batch(batch_size, T, sep)

        # 2. Mask the Y-values for testing (Preventing data leakage)
        # We replace the y-values (index 1) with noise or zeros for the extrapolation targets
        noise = torch.randn_like(C_test[:, :, 1]) * 0.1
        C_test[:, :, 1] = noise

        # (Optional) If your A_test also needs masking before hitting the model
        A_test_masked = A_test.clone()
        A_test_masked[:, :, 1] = 0.0

        A_stream = torch.cat([A_train, A_test_masked], dim=0)
        B_stream = torch.cat([B_train, B_test], dim=0)
        C_stream = torch.cat([C_train, C_test], dim=0)

        # 4. Forward Pass
        out_batch, stats = wrapper_model(A_stream, B_stream, C_stream, B_belief_A_train, sep)

        # Extract C stream (Batch size is 1)
        R = A_train.shape[1]
        C_out = out_batch[:, -R:, :]

        # Squeeze out the batch dimension [T, B, F] -> [T, F] for plotting
        A_train_sq = A_train.squeeze(1)
        A_test_sq = A_test.squeeze(1)  # Use original A_test for GT plotting, not masked
        B_stream_sq = B_stream.squeeze(1)
        C_out_sq = C_out.squeeze(1)
        B_belief_sq = B_belief_A_train.squeeze(1)

        return {
            # Use original A_test here so the plot shows the target line, not the 0-mask
            "A_plot": torch.cat([A_train_sq, A_test_sq], dim=0).cpu().numpy(),
            "B_plot": B_stream_sq.cpu().numpy(),
            "C_plot": C_out_sq.cpu().numpy(),
            "A_train": A_train_sq.cpu().numpy(),
            "B_belief": B_belief_sq.cpu().numpy(),
            "debug": debug_info,
            "model_name": wrapper_model.adapter.__class__.__name__,
            "stats": stats
        }


def plot_results(sep, T, adapter, loss_history):
    # 1. Get Data
    data = run_inference_for_plot(adapter, sep, T)

    # 2. Derive logic (Boundary)
    max_x = data["A_train"][-1, 0]
    b_cutoff = np.argmax(data["B_plot"][:, 0] > max_x) or (len(data["B_plot"]) - 1)

    # 3. Build Figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))

    # Top Row
    plot_learning_curve(axs[0, 0], loss_history, data["model_name"])
    plot_alignment(axs[0, 1], data["A_plot"], data["B_plot"], data["C_plot"],
                   data["A_train"], data["B_belief"], sep)

    # Bottom Left
    plot_hidden_warp(axs[1, 0], data["debug"], x_limits=axs[0, 1].get_xlim())

    # Bottom Right: Telemetry Logic
    # Replace ForwardMetaContext with your actual retrieval method
    attn_scores = ForwardMetaContext.get("corrector_att_scores")  # or from a stats dict

    if attn_scores is not None:
        matrix = attn_scores.squeeze().cpu().numpy()
        plot_attention_heatmap(axs[1, 1], fig, matrix, b_cutoff, "Corrector Attention")
    else:
        axs[1, 1].text(0.5, 0.5, "No Telemetry Available", ha='center', va='center')
        axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close('all')
    adapter.train()


def generate_shared_complex_batch(batch_size, T, sep):
    # ==========================================
    # 1. Task A (The Shared Irregular Target)
    # ==========================================
    t_A = torch.linspace(0, 2 * np.pi, T).view(T, 1, 1).expand(T, batch_size, 1)

    f1 = torch.rand(1, batch_size, 1) * 2.0 + 1.0
    f2 = torch.rand(1, batch_size, 1) * 3.0 + 2.0
    bump_center = torch.rand(1, batch_size, 1) * 2.0 + 3.5
    bump_height = torch.rand(1, batch_size, 1) * 2.0 - 1.0
    bump_width = 0.4

    def shared_base_shape(x):
        base = torch.sin(f1 * x) + 0.5 * torch.cos(f2 * x)
        bump = bump_height * torch.exp(-((x - bump_center) ** 2) / (2 * bump_width ** 2))
        return base + bump

    A_data = torch.cat([t_A, shared_base_shape(t_A)], dim=-1)

    # ==========================================
    # 2. Task B (The Distorted Domain)
    # ==========================================
    # B has a different spatial support than A
    t_B = torch.linspace(-0.5, 2 * np.pi + 0.5, T).view(T, 1, 1).expand(T, batch_size, 1)

    warp = torch.rand(1, batch_size, 1) * 0.8 + 0.2
    curve = torch.rand(1, batch_size, 1) * 0.2 - 0.1

    def task_B_func(x):
        distorted_x = x + warp * torch.sin(x)
        return shared_base_shape(distorted_x) + curve * (x ** 2)

    B_data = torch.cat([t_B, task_B_func(t_B)], dim=-1)

    # ==========================================
    # 3. Splits and Belief Calculation
    # ==========================================
    A_train, A_test = A_data[:sep], A_data[sep:]
    B_train, B_test = B_data[:sep], B_data[sep:]

    # C starts as an exact geometrical copy of A
    C_train = A_train.clone()
    C_test = A_test.clone()

    # B's zero-shot belief of A's training points
    t_A_train = t_A[:sep]
    B_belief_A_train = torch.cat([t_A_train, task_B_func(t_A_train)], dim=-1)

    # ==========================================
    # 4. Debug Information
    # ==========================================
    with torch.no_grad():
        distorted_t_B = t_B + warp * torch.sin(t_B)
        base_B = shared_base_shape(distorted_t_B)

        debug_info = {
            't_B': t_B.squeeze().cpu().numpy(),
            'distorted_t_B': distorted_t_B.squeeze().cpu().numpy(),
            'base_B': base_B.squeeze().cpu().numpy(),
            'task_B': B_data[:, :, 1].squeeze().cpu().numpy()
        }

    return A_train, A_test, B_train, B_test, C_train, C_test, B_belief_A_train, sep, debug_info


def main(wrapper_model, steps=20000, batch_size=128, T=60, sep=20):
    optimizer = optim.Adam(wrapper_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper_model = wrapper_model.to(device)

    print(f"Starting meta-training with {wrapper_model.adapter.__class__.__name__}...")
    loss_history = []
    pbar = tqdm(range(steps), desc="Training")

    for step in pbar:
        optimizer.zero_grad()

        # Data generation gives us the streams naturally
        (A_train, A_test, B_train, B_test,
         C_train, C_test, B_belief_A_train,
         sep, debug_info) = generate_shared_complex_batch(batch_size, T, sep)

        # 2. Mask the Y-values for testing (Preventing data leakage)
        # We replace the y-values (index 1) with noise or zeros for the extrapolation targets
        noise = torch.randn_like(C_test[:, :, 1]) * 0.1
        C_test[:, :, 1] = noise

        # (Optional) If your A_test also needs masking before hitting the model
        A_test_masked = A_test.clone()
        A_test_masked[:, :, 1] = 0.0

        A_stream = torch.cat([A_train, A_test_masked], dim=0).to(device)
        B_stream = torch.cat([B_train, B_test], dim=0).to(device)
        C_stream = torch.cat([C_train, C_test], dim=0).to(device)

        # 4. Forward Pass
        out_batch, stats = wrapper_model(A_stream, B_stream, C_stream, B_belief_A_train.to(device), sep)

        # Extract Stream C from the combined output
        # (wrapper returns cat([A, B, C]), so C starts at 2 * batch_size)
        C_out = out_batch[:, 2 * batch_size:, :]

        # Loss calculation: Target is A_data (the ground truth)
        # Notice, that the wrapper up and down projects, so not adding A_train to the loss of C_train
        # will likely give us a distored down projection, leading to artifacts of C on A_train in the plot
        loss = criterion(C_out[sep:T], A_test) + criterion(C_out[:sep], A_train)

        # Auxiliary loss handling
        gate_loss_key = f"gate_train/{getattr(wrapper_model.adapter, 'address', '')}"
        if gate_loss_key in stats:
            loss += stats[gate_loss_key]

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if (step + 1) % 200 == 0:
            pbar.set_postfix({"Loss": f"{current_loss:.6f}"})

        if (step + 1) % 2000 == 0:
            plot_results(sep, T, wrapper_model, loss_history)


if __name__ == '__main__':
    # model = Unified1dValidationWrapper(
    #     adapter_module=NadarayaWatsonAdapter(
    #
    #         d_model=64, seq_len=60, n_heads=4, dropout=0.0
    #     ),
    #     input_dim=2, d_model=64, seq_len=60
    # )
    # model.adapter.address = "NW_Adapter_Test"
    #
    # model = Unified1dValidationWrapper(
    #     adapter_module=CrossFusionAdapter(
    #         # Fixme num_heads vs n_heads naming inconsistency
    #         d_model=64, num_heads=4, use_prenorm=True, use_gate=False, add_linear=True
    #     ),
    #     input_dim=2, d_model=64
    # )
    # model.adapter.address = "CF_Adapter_Test"
    #
    # model = Unified1dValidationWrapper(
    #     adapter_module=MHA_StreamAdapter(
    #         d_model=64, n_heads=4, dropout=0.0
    #     ), input_dim=2, d_model=64
    # )
    # model.adapter.address = "MHA_Adapter_Test"


    model = Unified1dValidationWrapper(
        adapter_module=DeltaSurrogateAdapter(
            d_model=16, d_hp=16, d_k=32,
        ),
        input_dim=2, d_model=16,
    )
    model.adapter.address = "CalibratedSurrogateUpdate"

    main(model, steps=20000, batch_size=128, T=60, sep=20)
