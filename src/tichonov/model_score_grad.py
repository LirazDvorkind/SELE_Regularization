"""Gradient descent solver using score-based regularization."""
from __future__ import annotations
import numpy as np
import torch
from numpy.typing import NDArray

from src.__init__ import CONFIG

def solve_gradient_descent(G: NDArray, B: NDArray,
                           learning_rate: float, steps: int, reg_weight: float,
                           force_zero_last: bool = True) -> NDArray:
    """
    Solve for S using gradient descent:
    Minimize ||GS - B||^2 - reg_weight * log p(S_norm)

    where S_norm = S - 1.0 (assuming the network was trained on SELE - 1).

    Update rule:
    S_{t+1} = S_t + lr * (obj_grad + reg_weight * score_reg)

    where score_reg is the gradient of log p(S_norm) provided by the network,
    and obj_grad is the negative gradient of the data fidelity term ||G_norm S - B_norm||^2.
    We normalize G and B to improve numerical conditioning.
    """
    # Load score network
    model_path = CONFIG.data_paths.score_model_curve # This points to csv?
    # Wait, CONFIG.data_paths.score_model_curve points to "Data/sele_score_model_curve.csv".
    # The network is a .pt file.
    # The config does NOT have a path for the .pt file.
    # I should check if I added it? No.
    # I will revert to hardcoded path with fallback but clearer.

    model_path_pt = "Data/sele_score_net_d32.pt"
    try:
        score_network = torch.load(model_path_pt, weights_only=False)
    except Exception as e:
        try:
            score_network = torch.load("../" + model_path_pt, weights_only=False)
        except:
             raise FileNotFoundError(f"Could not load score network from {model_path_pt} or ../{model_path_pt}. Error: {e}")

    score_network.eval()

    N = G.shape[1]

    # Scaling for numerical stability
    mean_G = np.mean(np.abs(G))

    g_scale = 1.0 / (mean_G + 1e-12)
    G_norm = G * g_scale
    B_norm = B * g_scale

    # Initialize S
    S = np.zeros(N)

    # Time parameter for score network (fixed as per instructions)
    t_val = 1e-2

    print(f"Starting Gradient Descent (steps={steps}, lr={learning_rate}, reg={reg_weight})")

    for step in range(steps):
        # 1. Calculate Data Fidelity Gradient (Normalized)

        residual = G_norm @ S - B_norm
        grad_fidelity = 2 * G_norm.T @ residual

        obj_grad = -grad_fidelity

        # 2. Calculate Score Regularization
        # Input to network needs to be (S - 1) concatenated with t_val
        x_input = S - 1.0
        x_with_t = np.concatenate([x_input, [t_val]])
        x_tensor = torch.tensor(x_with_t, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score_reg = score_network(x_tensor).squeeze().numpy()

        # 3. Update S
        delta = learning_rate * (obj_grad + reg_weight * score_reg)
        S += delta

        # Force last element to zero if required
        if force_zero_last:
            S[-1] = 0.0

        if step % 200 == 0:
            print(f"Step {step:04d}: |Grad|={np.linalg.norm(obj_grad):.2e}, |Score|={np.linalg.norm(score_reg):.2e}")

    return S
