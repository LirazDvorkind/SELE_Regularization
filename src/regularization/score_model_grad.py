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
import matplotlib.pyplot as plt # Ensure matplotlib is imported for the debug plots

# --- Solver Implementation ---
def solve_gradient_descent(G: NDArray, B: NDArray, steps: int, reg_weight: float) -> NDArray:
    """
    Solves for S using Nesterov Accelerated Gradient (NAG) with Score-Based Priors.

    Implements Algorithm 1 from "Solving Ill-Conditioned Polynomial Equations...".
    Replaces the Reverse SDE loop with a deterministic momentum-based optimization.
    """

    # --- 0. HYPERPARAMETERS --- TODO - move these to the __init__ config section
    OPT_STEPS = 5000       # "T" in paper: Number of optimization iterations
    LR = 1e-2               # "eta" in paper: Learning rate
    MOMENTUM = 0.95         # "mu" in paper: Nesterov momentum coefficient. Controls the "inertia" (how much past velocity is kept).
    T0 = 1e-3               # Small fixed time step to sample "clean" score
    IS_SHOW_DEBUG_PLOT = False
    reg_weight = 10

    # "reg_weight" passed as argument acts as the "score factor" or regularization strength.
    # reg_weight too high = ignoring data, too low = ignoring physics (score model)
    # MOMENTUM high = plow through noise but may cause overshoot, too low = slower but less overshoot
    # T0: Larger t means the model expects more noise, so it pushes toward coarser, blurrier features.
    #     Smaller t (like 1e-4) assumes the image is nearly clean, enforcing finer details.

    print(f"Starting NAG Solver. Steps={OPT_STEPS}, LR={LR}, Momentum={MOMENTUM}, Reg={reg_weight}")

    model_path_pt = "Data/sele_score_net_d32.pt"
    device = torch.device('cpu')

    # Set random seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

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

    # 3. Setup Physics
    N = G.shape[1]

    # Normalize G and B for numerical stability
    mean_G = np.mean(np.abs(G))
    g_scale = 1.0 / (mean_G + 1e-12)
    G_norm = G * g_scale
    B_norm = B * g_scale

    # 4. Initialization
    # Initialize x (S_norm) and velocity (v)
    S_norm = np.random.randn(N)
    velocity = np.zeros_like(S_norm) # Initialize momentum buffer v^(0) = 0

    # 5. Nesterov Optimization Loop
    for i in range(OPT_STEPS):

        # --- A. Nesterov Lookahead ---
        # Evaluate gradients at the predicted position (x + mu*v)
        S_lookahead = S_norm + MOMENTUM * velocity

        # --- B. Data Gradient (at lookahead) ---
        # 1. Un-normalize lookahead to physical space
        S_phys_lookahead = (S_lookahead + 1.0) / norm_scale_factor + d_min

        # 2. Calculate Gradient w.r.t Physical S
        # Grad(Error) = 2 * G.T @ (G*S - B)
        residual = G_norm @ S_phys_lookahead - B_norm
        grad_fidelity = 2 * G_norm.T @ residual

        # 3. Chain Rule: Convert to Gradient w.r.t Normalized S
        grad_fidelity_norm = grad_fidelity * (1.0 / norm_scale_factor)

        # --- C. Score Network Prediction (at lookahead) ---
        # Use fixed small time T0 to approximate the score of the clean data
        x_input = np.concatenate([S_lookahead, [T0]])
        x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            score_model = score_network(x_tensor).squeeze().numpy()

        # --- D. Adaptive Weighting & Update ---
        # Calculate norms for adaptive scaling
        grad_mag = np.linalg.norm(grad_fidelity_norm)
        score_mag = np.linalg.norm(score_model) + 1e-12 # avoid div by zero

        # Adaptive weight: scales score to match data gradient magnitude
        # alpha = ||g|| / ||s||. We also apply the user's reg_weight here.
        adaptive_factor = (grad_mag / score_mag) * reg_weight

        # The 'force' is: -Gradient + Weighted_Score
        # We subtract the gradient (descent) and add the score (ascent on prior probability)
        # Equivalently: update_direction = - (Gradient - Weighted_Score)
        score_weighted = score_model * adaptive_factor
        total_update = grad_fidelity_norm - score_weighted

        # --- E. Momentum Update ---
        # v^(t+1) = mu * v^(t) - eta * (grad - score_weighted)
        velocity = MOMENTUM * velocity - LR * total_update

        # x^(t+1) = x^(t) + v^(t+1)
        S_norm = S_norm + velocity

        # --- Monitoring & Plotting ---
        if i % (OPT_STEPS // 10) == 0:
            print(f"Iter={i:04d} | ScoreMag={score_mag:.2e} | DataGradMag={grad_mag:.2e} | AdaptFactor={adaptive_factor:.2e}")

            # Debug Plotting
            if IS_SHOW_DEBUG_PLOT:
                plt.figure(figsize=(10, 5))
                plt.plot(score_weighted, label="Weighted Score (Prior)", alpha=0.7)
                plt.plot(-grad_fidelity_norm, label="Negative Data Grad (Likelihood)", alpha=0.7)
                plt.plot(S_norm, label="Current S_norm", color='k', linewidth=1)
                plt.plot(velocity * 10, label="Velocity (x10)", linestyle='--')
                plt.legend()
                plt.title(f"[Debug] Optimization Forces at Iter {i}")
                plt.grid(True, alpha=0.3)
                plt.show()

    # 6. Final Un-normalization
    S_final = (S_norm + 1.0) / norm_scale_factor + d_min

    return S_final