"""Compare what different score models output for different SELE inputs - standalone testing"""
from __future__ import annotations
import torch

from src.regularization.score_model.model_definition import ScoreNetwork
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from src.regularization.score_model.standalones.helpers import load_S_B_G


# --- 1. Plot Physical SELE Vector ---
def plot_sele_profile(S: NDArray, x_axis: NDArray = None, title: str = "SELE Profile (S_phys)") -> None:
    """
    Plots the physical SELE S vector.

    Args:
        S: The physical profile array.
        x_axis: Optional array for physical depth (e.g., in cm or um).
                If None, plots against the array index.
        title: Title of the plot.
    """
    plt.figure(figsize=(8, 5))

    if x_axis is None:
        plt.plot(S, linewidth=2, color='blue', label='S (Physical)')
        plt.xlabel("Index")
    else:
        plt.plot(x_axis, S, linewidth=2, color='blue', label='S (Physical)')
        plt.xlabel("Depth (cm)")

    plt.ylabel("Sp (1/eV)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- 2. Normalize SELE Vector ---
def normalize_sele_vector(S: NDArray, d_min: float, d_max: float) -> NDArray:
    """
    Normalizes a physical SELE profile to the [-1, 1] range using standard min-max scaling.
    This matches the normalization expected by the pre-trained Score Network.

    Args:
        S: The physical SELE vector.
        d_min: The minimum value from the training dataset.
        d_max: The maximum value from the training dataset.

    Returns:
        S_norm: The normalized vector.
    """
    # Formula: 2 * (x - min) / (max - min) - 1
    S_norm = 2.0 * (S - d_min) / (d_max - d_min) - 1.0
    return S_norm


# --- 3. Plot Normalized S and Score Gradient ---
def plot_normalized_s_and_score(S_norm: NDArray, score_grad: NDArray,
                                title: str = "Diffusion Forces: Normalized S vs. Score Gradient") -> None:
    """
    Plots the normalized SELE vector alongside the gradient produced by the Score Network.
    Uses dual y-axes since their magnitudes can differ significantly.

    Args:
        S_norm: The normalized SELE vector [-1, 1].
        score_grad: The gradient output from the Score Network.
        title: Title of the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Normalized S on the left Y-axis
    color_s = 'tab:blue'
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Normalized S [-1, 1]', color=color_s, fontweight='bold')
    ax1.plot(S_norm, color=color_s, linewidth=2, label='Normalized S')
    ax1.tick_params(axis='y', labelcolor=color_s)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot Score Gradient on the right Y-axis
    color_score = 'tab:red'
    ax2.set_ylabel('Score Gradient (Network Output)', color=color_score, fontweight='bold')
    ax2.plot(score_grad, color=color_score, linewidth=2, linestyle='--', label='Score Gradient')
    ax2.tick_params(axis='y', labelcolor=color_score)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

ALON_MODEL_PATH = './Data/alon_sele_score_net_d32.pt'
MY_MODEL_PATH = './Data/sele_score_net_d32_100k.pt'
T0 = 5e-2
device = torch.device('cpu')

# Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

def get_alon_model_grad(S):
    # 1. Load Model
    try:
        score_network = torch.load(ALON_MODEL_PATH, map_location=device, weights_only=False)
        score_network.eval()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load ScoreNet: {e}")

    d_min = 2.7389012e-21
    d_max = 0.03475773
    S_normalized = normalize_sele_vector(S, d_min=d_min, d_max=d_max)
    x_input = np.concatenate([S_normalized, [T0]])
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        model_score_grad = score_network(x_tensor).squeeze().numpy()
    return S_normalized, model_score_grad


def get_my_trained_model_grad(S, model_path: str):
    # 1. Load Model and Configuration
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Retrieve configuration to initialize the correct architecture
        model_config = checkpoint['config']

        score_network = ScoreNetwork(
            input_dim=model_config['target_length'] + 1,
            output_dim=model_config['target_length'],
            hidden_dims=model_config['hidden_dims'],
        )

        # Load the state dictionary into the model
        score_network.load_state_dict(checkpoint['model_state_dict'])
        score_network.to(device)
        score_network.eval()

    except Exception as e:
        raise FileNotFoundError(f"Failed to load ScoreNet checkpoint: {e}")

    d_min = checkpoint['data_min']
    d_max = checkpoint['data_max']
    S_norm = normalize_sele_vector(S, d_min, d_max)
    x_tensor = torch.tensor(S_norm, dtype=torch.float32, device=device).unsqueeze(0)
    t_tensor = torch.tensor(np.array([T0]), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        model_score_grad = score_network(x_tensor, t_tensor).squeeze().numpy()

    return S_norm, model_score_grad


if __name__ == '__main__':
    random_sample = np.random.randint(100, 1000)
    model_size: int = 32  # 32 or 500
    print(f"Random curve number {random_sample}")
    data, G = load_S_B_G(points_amount=model_size, lower_index=random_sample, upper_index=random_sample + 1)

    for item in data:
        S_gt = item['S_gt']
        # 1. Alon model
        plot_sele_profile(S_gt)
        S_normalized, alon_model_grad = get_alon_model_grad(S_gt)
        plot_normalized_s_and_score(S_normalized, alon_model_grad)

        # 2. My model
        plot_sele_profile(S_gt)
        S_norm, my_grad = get_my_trained_model_grad(S_gt, MY_MODEL_PATH)
        plot_normalized_s_and_score(S_norm, my_grad)
