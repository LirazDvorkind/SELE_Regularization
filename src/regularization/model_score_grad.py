"""Gradient descent solver using score-based regularization."""
from __future__ import annotations
import numpy as np
import torch
from numpy.typing import NDArray

def solve_gradient_descent(G: NDArray, B: NDArray,
                           learning_rate: float, steps: int, reg_weight: float) -> NDArray:
    """
    Solve for S using gradient descent:
    Minimize ||GS - B||^2 - reg_weight * log p(S_norm)

    where S_norm = 2 * (S - d_min) / (d_max - d_min) - 1

    Update rule:
    S_{t+1} = S_t + lr * (obj_grad + reg_weight * score_reg)

    where score_reg is the gradient of log p(S_norm) provided by the network,
    and obj_grad is the negative gradient of the data fidelity term ||G_norm S - B_norm||^2.
    We normalize G and B to improve numerical conditioning.
    """
    # Load score network
    model_path_pt = "Data/sele_score_net_d32.pt"
    try:
        score_network = torch.load(model_path_pt, weights_only=False)
    except Exception as e:
        raise FileNotFoundError(f"Could not load score network from {model_path_pt}. Error: {e}")

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
        # Map S from [d_min, d_max] -> [-1, 1] (This is the normalization used by Alon).
        d_max, d_min =  0.03475773, 2.7389012e-21
        scale_factor = 2.0 / (d_max - d_min)

        S_norm = 2 * (S - d_min) / (d_max - d_min) - 1
        x_with_t = np.concatenate([S_norm, [t_val]])
        x_tensor = torch.tensor(x_with_t, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score_out = score_network(x_tensor).squeeze().numpy()

        # APPLY CHAIN RULE
        score_reg = score_out * scale_factor

        # 3. Update S
        delta = learning_rate * (obj_grad + reg_weight * score_reg)
        S += delta

        if step % 200 == 0:
            print(f"Step {step:04d}: |Grad|={np.linalg.norm(obj_grad):.2e}, |Score|={np.linalg.norm(score_reg):.2e}")

    return S
