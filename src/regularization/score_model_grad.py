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
def solve_gradient_descent(G: NDArray, B: NDArray, steps: int, reg_weight: float, lr_max=1e-2, momentum=0.95, S_gt: NDArray = None) -> NDArray:
    """
    Solves for S using Nesterov Accelerated Gradient (NAG) with Score-Based Priors.

    Implements Algorithm 1 from "Solving Ill-Conditioned Polynomial Equations...".
    Replaces the Reverse SDE loop with a deterministic momentum-based optimization.
    """

    # --- 0. HYPERPARAMETERS --- TODO - move these to the __init__ config section
    MAX_STEPS = 5000       # "T" in paper: Number of optimization iterations
    LR_MAX = 1e-2               # "eta" in paper: Learning rate
    LR_MIN = 1e-5               # Cosine annealing to LR_MIN
    MOMENTUM = 0.95         # "mu" in paper: Nesterov momentum coefficient. Controls the "inertia" (how much past velocity is kept).
    T0 = 1e-3               # Small fixed time step to sample "clean" score
    IS_SHOW_DEBUG_PLOT = False
    reg_weight = 10
    STOP_CHANGE = 1e-8       # Stop if error changes less than this
    STOP_STEPS = 20          # How many steps the mse diff is less than the threshold
    MIN_STEPS = 50          # Minimum steps to run before checking (to let momentum build)

    # "reg_weight" passed as argument acts as the "score factor" or regularization strength.
    # reg_weight too high = ignoring data, too low = ignoring physics (score model)
    # MOMENTUM high = plow through noise but may cause overshoot, too low = slower but less overshoot
    # T0: Larger t means the model expects more noise, so it pushes toward coarser, blurrier features.
    #     Smaller t (like 1e-4) assumes the image is nearly clean, enforcing finer details.

    print(f"Starting NAG Solver. Steps={MAX_STEPS}, LR={LR_MAX} to {LR_MIN}, Momentum={MOMENTUM}, Reg={reg_weight}")

    model_path_pt = "Data/sele_score_net_d32.pt"
    device = torch.device('cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

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

    # Trackers
    mse_history = []
    small_error_steps_amount = 0

    # 5. Nesterov Optimization Loop
    for i in range(MAX_STEPS):

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
        # alpha = ||g|| / ||s||. We also apply reg_weight here.
        adaptive_factor = (grad_mag / score_mag) * reg_weight

        # The 'force' is: -Gradient + Weighted_Score
        # We subtract the gradient (descent) and add the score (ascent on prior probability)
        # Equivalently: update_direction = - (Gradient - Weighted_Score)
        score_weighted = score_model * adaptive_factor
        total_update = grad_fidelity_norm - score_weighted

        # --- E. Momentum Update ---
        # Smoothly decays LR from LR_MAX to LR_MIN over the course of MAX_STEPS
        current_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(i / MAX_STEPS * np.pi))
        # v^(t+1) = mu * v^(t) - eta * (grad - score_weighted)
        velocity = MOMENTUM * velocity - current_lr * total_update

        # x^(t+1) = x^(t) + v^(t+1)
        S_norm = S_norm + velocity

        # --- F. MSE Tracking ---
        if S_gt is not None:
            # 1. Un-normalize current estimate to physical space
            S_current_phys = (S_norm + 1.0) / norm_scale_factor + d_min

            # 2. Interpolate S_current to match S_gt length if needed
            if len(S_current_phys) != len(S_gt):
                x_curr = np.linspace(0, 1, len(S_current_phys))
                x_gt = np.linspace(0, 1, len(S_gt))
                # Interpolate current estimate onto GT grid
                S_interp = np.interp(x_gt, x_curr, S_current_phys)
                diff = S_interp - S_gt
            else:
                diff = S_current_phys - S_gt

            # 3. Calculate MSE
            current_mse = np.mean(diff ** 2)
            mse_history.append(current_mse)

            # --- G. STOPPING CONDITION CHECK ---
            if i > MIN_STEPS:
                if np.abs(mse_history[-1] - mse_history[-2]) < STOP_CHANGE:
                    if small_error_steps_amount > STOP_STEPS:
                        print(f"Stopping Early: MSE diff < {STOP_CHANGE} at step {i}")
                        break
                    else:
                        small_error_steps_amount += 1
                else:
                    small_error_steps_amount = 0

        # --- Monitoring & Plotting ---
        if i % (MAX_STEPS // 20) == 0:
            mse_str = f" | MSE={current_mse:.2e}" if S_gt is not None else ""
            print(f"Iter={i:04d} | ScoreMag={score_mag:.2e} | DataGradMag={grad_mag:.2e} | AdaptFactor={adaptive_factor:.2e}{mse_str}")

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

    # Plot MSE History
    if S_gt is not None and len(mse_history) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(mse_history, label="SELE Reconstruction error vs GT")
        plt.yscale('log')  # Log scale is usually better for convergence plots
        plt.xlabel("Optimization Step")
        plt.ylabel("Mean Squared Error (Physical Units)")
        plt.title("Convergence of SELE Reconstruction")
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    # 6. Final Un-normalization
    S_final = (S_norm + 1.0) / norm_scale_factor + d_min

    return S_final