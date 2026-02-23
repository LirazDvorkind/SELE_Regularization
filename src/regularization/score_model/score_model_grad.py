from __future__ import annotations
import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt # Ensure matplotlib is imported for the debug plots

from src.types.score_model_params import NesterovHyperparams
from src.utils import match_length_interp

from src.regularization.score_model.model_definition import ScoreNetwork

# --- Solver Implementation ---
def solve_gradient_descent(G: NDArray, B: NDArray, hyperparams: NesterovHyperparams, S_gt: NDArray) -> NDArray:
    """
    Solves for SELE using Nesterov Accelerated Gradient (NAG) with Score-Based Priors.
    :param G: Photogeneration matrix, NxM size
    :param B: ELE vector, B = GS, Nx1 size
    :param hyperparams: NesterovHyperparams dataclass
    :param S_gt: SELE ground truth vector to plot difference and calculate metrics
    :return: S the SELE we found, Mx1 size
    """
    if hyperparams.IS_SHOW_DEBUG_DATA:
        print(f"Starting NAG Solver. LR={hyperparams.LR_MAX} to {hyperparams.LR_MIN}, Momentum={hyperparams.MOMENTUM}, Reg={hyperparams.REG_WEIGHT}")

    device = torch.device('cpu')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Model and Configuration
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(hyperparams.model_path, map_location=device, weights_only=False)

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

    # 2. Normalization Constants
    # Using the exact min/max saved during training ensures perfect data reconstruction
    d_min = checkpoint['data_min']
    d_max = checkpoint['data_max']
    norm_scale_factor = 2.0 / (d_max - d_min)

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
    for i in range(hyperparams.MAX_STEPS):

        # --- A. Nesterov Lookahead ---
        # Evaluate gradients at the predicted position (x + mu*v)
        S_lookahead = S_norm + hyperparams.MOMENTUM * velocity

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
        x_tensor = torch.tensor(S_lookahead, dtype=torch.float32, device=device).unsqueeze(0)
        t_tensor = torch.tensor(np.array([hyperparams.T0]), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            score_model = score_network(x_tensor, t_tensor).squeeze().numpy()

        # --- D. Adaptive Weighting & Update ---
        # Calculate norms for adaptive scaling
        grad_mag = np.linalg.norm(grad_fidelity_norm)
        score_mag = np.linalg.norm(score_model) + 1e-12 # avoid div by zero

        # Adaptive weight: scales score to match data gradient magnitude
        # alpha = ||g|| / ||s||. We also apply REG_WEIGHT here.
        adaptive_factor = (grad_mag / score_mag) * hyperparams.REG_WEIGHT

        # The 'force' is: -Gradient + Weighted_Score
        # We subtract the gradient (descent) and add the score (ascent on prior probability)
        # Equivalently: update_direction = - (Gradient - Weighted_Score)
        score_weighted = score_model * adaptive_factor
        total_update = grad_fidelity_norm - score_weighted

        # --- E. Momentum Update ---
        # Smoothly decays LR from LR_MAX to LR_MIN over the course of MAX_STEPS
        current_lr = hyperparams.LR_MIN + 0.5 * (hyperparams.LR_MAX - hyperparams.LR_MIN) * (1 + np.cos(i / hyperparams.MAX_STEPS * np.pi))
        # v^(t+1) = mu * v^(t) - eta * (grad - score_weighted)
        velocity = hyperparams.MOMENTUM * velocity - current_lr * total_update

        # x^(t+1) = x^(t) + v^(t+1)
        S_norm = S_norm + velocity

        # --- F. MSE Tracking ---
        if S_gt is not None:
            # 1. Un-normalize current estimate to physical space
            S_current_phys = (S_norm + 1.0) / norm_scale_factor + d_min

            # 2. Interpolate S_current to match S_gt length if needed
            if len(S_current_phys) != len(S_gt):
                S_interp = match_length_interp(S_current_phys, len(S_gt))
                diff = S_interp - S_gt
            else:
                diff = S_current_phys - S_gt

            # 3. Calculate MSE
            current_mse = np.mean(diff ** 2)
            mse_history.append(current_mse)

            # --- G. STOPPING CONDITION CHECK ---
            if i > hyperparams.MIN_STEPS:
                if mse_history[-1] > 1:
                    if hyperparams.IS_SHOW_DEBUG_DATA:
                        print(f"Stopping Early: MSE > 1 at step {i}")
                    break
                if np.abs(mse_history[-1] - mse_history[-2]) < hyperparams.STOP_CHANGE:
                    if small_error_steps_amount > hyperparams.STOP_STEPS:
                        if hyperparams.IS_SHOW_DEBUG_DATA:
                            print(f"Stopping Early: MSE diff < {hyperparams.STOP_CHANGE} at step {i}")
                        break
                    else:
                        small_error_steps_amount += 1
                else:
                    small_error_steps_amount = 0

        # --- Monitoring & Plotting ---
        if hyperparams.IS_SHOW_DEBUG_DATA and i % (hyperparams.MAX_STEPS // 20) == 0:
            mse_str = f" | MSE={current_mse:.2e}" if S_gt is not None else ""
            print(f"Iter={i:04d} | ScoreMag={score_mag:.2e} | DataGradMag={grad_mag:.2e} | AdaptFactor={adaptive_factor:.2e}{mse_str}")

            # Debug Plotting
            if hyperparams.IS_SHOW_DEBUG_PLOT:
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
    if hyperparams.IS_SHOW_MSE_PLOT and S_gt is not None and len(mse_history) > 0:
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