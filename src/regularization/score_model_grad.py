"""
Conditional Generation using Reverse SDE (Euler-Maruyama with Guidance).

METHODOLOGY:
This module generates a SELE profile 'S' by solving the Reverse-Time SDE
from t=1.0 (Pure Noise) to t=0.0 (Clean Signal).

It combines two forces at every time step:
1. The Score Network: Pushes the signal to look like a valid SELE profile.
2. The Data Gradient: Pushes the signal to satisfy the measurement G*S = B.

This is superior to simple optimization because it allows the model to
traverse the "generative path," refining coarse features first (at high t)
and fine details later (at low t), effectively avoiding local minima.
"""
from __future__ import annotations
import numpy as np
import torch
from numpy.typing import NDArray

# --- Solver Implementation ---
def solve_gradient_descent(G: NDArray, B: NDArray, steps: int, reg_weight: float) -> NDArray:
    """
    Solves for S using Conditional Reverse SDE.

    NOTE ON ARGUMENTS:
    - 'steps': Number of diffusion steps (e.g., 1000).
    - 'reg_weight': Now acts as the GUIDANCE SCALE. Controls how strictly we enforce G*S=B.
    """
    model_path_pt = "Data/sele_score_net_d32.pt"
    device = torch.device('cpu')

    # 1. Load Model
    try:
        score_network = torch.load(model_path_pt, map_location=device, weights_only=False)
        score_network.eval()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load ScoreNet: {e}")

    # 2. Hardcoded Constants
    d_min = 2.7389012e-21
    d_max = 0.03475773
    norm_scale_factor = 2.0 / (d_max - d_min) # ~57.54

    # Diffusion Constants (From TrainingConfig)
    BETA_MIN = 0.1
    BETA_MAX = 20.0

    # 3. Setup Physics
    N = G.shape[1]

    # Normalize G and B for numerical stability of the gradient
    # (Important so guidance scale doesn't need to be 1e15)
    mean_G = np.mean(np.abs(G))
    g_scale = 1.0 / (mean_G + 1e-12)
    G_norm = G * g_scale
    B_norm = B * g_scale

    # 4. Initialization (Pure Gaussian Noise)
    # The diffusion process starts from N(0, I) in the NORMALIZED space.
    S_norm = np.random.randn(N)

    print(f"Starting Reverse SDE. Steps={steps}, Guidance (reg_weight)={reg_weight}")

    # 5. Reverse SDE Loop (t = 1.0 -> 0.0)
    dt = 1.0 / steps

    for i in range(steps):
        # Current time t (going backwards)
        t = 1.0 - (i / steps)

        # Calculate Beta(t)
        # Linear schedule: beta(t) = min + t*(max - min)
        beta_t = BETA_MIN + t * (BETA_MAX - BETA_MIN)

        # --- A. Data Guidance Gradient ---
        # 1. Un-normalize current S to physical space to check error
        # S_physical = (S_norm + 1) / 2 * range + min
        S_physical = (S_norm + 1.0) / norm_scale_factor + d_min

        # 2. Calculate Gradient w.r.t Physical S
        residual = G_norm @ S_physical - B_norm
        grad_fidelity = 2 * G_norm.T @ residual

        # 3. Chain Rule: Convert to Gradient w.r.t Normalized S
        # grad_norm = grad_phys * d(Phys)/d(Norm)
        # d(Phys)/d(Norm) = 1 / norm_scale_factor
        grad_fidelity_norm = grad_fidelity * (1.0 / norm_scale_factor)

        # --- B. Score Network Prediction ---
        x_input = np.concatenate([S_norm, [t]])
        x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            score_model = score_network(x_tensor).squeeze().numpy()

        # --- C. Combine Forces (Guided Score) ---
        # Score pushes up log-p. Gradient pushes down Error.
        guided_score = score_model - (reg_weight * grad_fidelity_norm)

        # --- D. Euler-Maruyama Update Step ---
        diffusion_drift = (0.5 * beta_t * S_norm + beta_t * guided_score) * dt

        # Sample noise z (unless it's the last steps, then 0 to denoise fully)
        z = np.random.randn(N) if i < steps - steps/25 else np.zeros(N)
        diffusion_noise = np.sqrt(beta_t * dt) * z

        S_norm = S_norm + diffusion_drift + diffusion_noise

        # Monitoring
        if i % (steps // 10) == 0:
            score_mag = np.linalg.norm(score_model)
            grad_mag = np.linalg.norm(grad_fidelity_norm)
            print(f"t={t:.2f} | Score={score_mag:.2e} | DataGrad={grad_mag:.2e}")
            # import matplotlib.pyplot as plt
            # plt.plot(score_model, label="Score model")
            # plt.plot(-reg_weight*grad_fidelity_norm, label="-reg_weight*grad")
            # plt.plot(S_norm, label="S_norm")
            # plt.plot(diffusion_drift, label="diffusion_drift")
            # plt.plot(diffusion_noise, label="diffusion_noise")
            # plt.legend()
            # plt.title("[debug] - S_norm and the values that push it around")
            # plt.show()

    # 6. Final Un-normalization
    S_final = (S_norm + 1.0) / norm_scale_factor + d_min

    return S_final