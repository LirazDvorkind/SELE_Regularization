"""Compare what different score models output for different SELE inputs - standalone testing"""
from __future__ import annotations
import torch
from enum import Enum
from pathlib import Path

from src.regularization.score_model.model_definition import ScoreNetwork
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


class TestMode(Enum):
    CORRECT = "correct"
    INCREASING_EXP = "increasing_exp"
    PIECEWISE_LINEAR = "piecewise_linear"
    STRAIGHT_LINE = "straight_line"


_DATA_DIR = Path(__file__).resolve().parents[5] / "Data" / "score_model"
ALON_MODEL_PATH = _DATA_DIR / 'models' / 'alon_sele_score_net_d32.pt'
MY_MODEL_PATH_D32 = _DATA_DIR / 'models' / 'sele_score_net_d32.pt'
MY_MODEL_PATH_D500 = _DATA_DIR / 'models' / 'sele_score_net_d500.pt'
DATASET_PATH = _DATA_DIR / 'datasets' / 'sele_simulated_1000_curves_500_long_more_dip.csv'

T0 = 0.1
device = torch.device('cpu')

# ---- Change this to select what to test ----
TEST_MODE = TestMode.PIECEWISE_LINEAR


def load_correct_curve() -> NDArray:
    dataset = np.loadtxt(DATASET_PATH, delimiter=',')
    idx = np.random.randint(0, len(dataset))
    print(f"Loaded curve index {idx} from dataset")
    return dataset[idx]


def make_increasing_exp(S_ref: NDArray) -> NDArray:
    n = len(S_ref)
    z = np.linspace(0, 1, n)
    a = S_ref.min()
    b_val = S_ref.max()
    # a * exp(k*z) where a=start, a*exp(k)=end => k = log(end/start)
    k = np.log(b_val / max(a, 1e-30))
    return a * np.exp(k * z)


def make_piecewise_linear(S_gt: NDArray) -> NDArray:
    n = len(S_gt)
    knot_indices = np.linspace(0, n - 1, 10, dtype=int)
    knot_values = S_gt[knot_indices]
    return np.interp(np.arange(n), knot_indices, knot_values)


def make_straight_line(S_ref: NDArray) -> NDArray:
    n = len(S_ref)
    span = S_ref.max() - S_ref.min()
    slope = np.random.uniform(-span / n, span / n)
    z = np.arange(n, dtype=float)
    line = slope * z
    line += S_ref.mean() - line.mean()
    return line
    # return np.clip(line, 0, None)


def plot_all(
    S_phys: NDArray,
    score_phys_grad: NDArray,
    S_phys_applied: NDArray,
    S_norm: NDArray,
    score_grad: NDArray,
    x_axis: NDArray | None = None,
    title: str = "Score Model Analysis",
    S_original: NDArray | None = None,
) -> None:
    x = x_axis if x_axis is not None else np.arange(len(S_phys))
    xlabel = "Depth (cm)" if x_axis is not None else "Index"

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(title)

    # --- Top: physical space ---
    ax_top.set_title("Physical Space: Input Curve with Score Gradient", fontweight="bold")
    ax_top.set_xlabel(xlabel)
    ax_top.set_ylabel("Curve value [0, 1]", color="tab:blue", fontweight="bold")
    if S_original is not None:
        ax_top.plot(x, S_original, color="tab:gray", linewidth=1.5, linestyle=":", label="Original curve")
    ax_top.plot(x, S_phys, color="tab:blue", linewidth=2, label="Input S (physical)")
    ax_top.plot(x, S_phys_applied, color="tab:green", linewidth=1.5, linestyle="-.",
                label="S after score step (REG_WEIGHT=1)")
    ax_top.tick_params(axis="y", labelcolor="tab:blue")
    ax_top.grid(True, linestyle="--", alpha=0.7)

    ax_top_r = ax_top.twinx()
    ax_top_r.set_ylabel("Score gradient (magnitude normalized)", color="tab:red", fontweight="bold")
    ax_top_r.plot(x, score_phys_grad, color="tab:red", linewidth=1.5, linestyle="--",
                  label="Score gradient (physical)")
    ax_top_r.tick_params(axis="y", labelcolor="tab:red")

    lines, labels = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_top_r.get_legend_handles_labels()
    ax_top.legend(lines + lines_r, labels + labels_r, loc="best").set_draggable(True)

    # --- Bottom: normalized space ---
    ax_bot.set_title("Normalized Space: S and Score Network Output", fontweight="bold")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Normalized S [-1, 1]", color="tab:blue", fontweight="bold")
    ax_bot.plot(x, S_norm, color="tab:blue", linewidth=2, label="Normalized S")
    ax_bot.tick_params(axis="y", labelcolor="tab:blue")
    ax_bot.grid(True, linestyle="--", alpha=0.7)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.set_ylabel("Score Gradient (Network Output)", color="tab:red", fontweight="bold")
    ax_bot_r.plot(x, score_grad, color="tab:red", linewidth=2, linestyle="--", label="Score Gradient")
    ax_bot_r.tick_params(axis="y", labelcolor="tab:red")

    lines, labels = ax_bot.get_legend_handles_labels()
    lines_r, labels_r = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(lines + lines_r, labels + labels_r, loc="best").set_draggable(True)

    fig.tight_layout()


def get_alon_model_grad(S):
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
    return S_normalized, model_score_grad, d_min, d_max


def normalize_sele_vector(S: NDArray, d_min: float, d_max: float) -> NDArray:
    return 2.0 * (S - d_min) / (d_max - d_min) - 1.0


def get_my_trained_model_grad(S, model_path: str):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_config = checkpoint['config']

        score_network = ScoreNetwork(
            input_dim=model_config['target_length'] + 1,
            output_dim=model_config['target_length'],
            hidden_dims=model_config['hidden_dims'],
            use_layer_norm=model_config.get('use_layer_norm', False),
            use_residual=model_config.get('use_residual', False),
            use_time_embedding=model_config.get('use_time_embedding', False),
            time_embed_dim=model_config.get('time_embed_dim', 128),
        )

        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        score_network.load_state_dict(state_dict)
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

    return S_norm, model_score_grad, d_min, d_max


if __name__ == '__main__':
    my_model_path = MY_MODEL_PATH_D500

    S_ref = load_correct_curve()

    if TEST_MODE == TestMode.CORRECT:
        S_gt = S_ref
    elif TEST_MODE == TestMode.INCREASING_EXP:
        S_gt = make_increasing_exp(S_ref)
    elif TEST_MODE == TestMode.PIECEWISE_LINEAR:
        S_gt = make_piecewise_linear(S_ref)
    elif TEST_MODE == TestMode.STRAIGHT_LINE:
        S_gt = make_straight_line(S_ref)
    else:
        raise ValueError(f"Unknown TEST_MODE: {TEST_MODE}")

    print(f"TEST_MODE: {TEST_MODE.value}")

    S_norm, grad, d_min, d_max = get_my_trained_model_grad(S_gt, my_model_path)

    norm_scale = 2.0 / (d_max - d_min)

    # Normalize score to unit magnitude (adaptive weighting, REG_WEIGHT=1)
    score_mag = np.linalg.norm(grad) + 1e-12
    score_unit = grad / score_mag

    # Score effect in physical coordinates: the actual change to S_phys from applying the normalized score
    score_phys_grad = score_unit / norm_scale

    # Applied curve: apply unit-normalized score in normalized space, then convert back to physical
    S_norm_applied = S_norm + score_unit
    S_phys_applied = (S_norm_applied + 1.0) / norm_scale + d_min

    plot_all(
        S_gt, score_phys_grad, S_phys_applied, S_norm, grad,
        title=f"Score Model Analysis ({TEST_MODE.value})",
        S_original=S_ref if TEST_MODE == TestMode.PIECEWISE_LINEAR else None,
    )

    plt.show()
