from __future__ import annotations
import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt # Ensure matplotlib is imported for the debug plots

from src.types.config import ModelScoreGradConfig
from src.utils import match_length_interp

from src.regularization.score_model.model_definition import ScoreNetwork


# Cosine-annealed diffusion-time schedule for querying the score network.
# At step 0 the network is queried at SCORE_T_MAX (smoother, coarser prior --
# the score net was trained on heavily-corrupted data here); by step MAX_STEPS
# it reaches hyperparams.T0 (sharper, finer prior). The training range was
# T=1 (pure white noise) to T=0 (clean data), so SCORE_T_MAX must lie in
# (hyperparams.T0, 1]. Set SCORE_T_MAX = hyperparams.T0 to disable annealing.
SCORE_T_MAX = 0.5


def load_score_model(model_path: str) -> tuple:
    """Load a score network checkpoint from disk.

    Returns (score_network, d_min, d_max, target_length).
    Pass the returned tuple as ``preloaded_model`` to ``solve_gradient_descent``
    to skip repeated disk I/O across many solver calls.
    """
    device = torch.device('cpu')
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

    return score_network, checkpoint['data_min'], checkpoint['data_max'], model_config['target_length']


def _plot_step_debug(
    step: int,
    S_norm_before: NDArray,
    S_norm_after: NDArray,
    S_phys_before: NDArray,
    S_phys_after: NDArray,
    score_norm: NDArray,
    data_grad_norm: NDArray,
    data_step_contrib: NDArray,
    prior_step_contrib: NDArray,
    momentum_carry: NDArray,
    velocity: NDArray,
    eta_fit_before: NDArray,
    eta_fit_after: NDArray,
    B_target: NDArray,
    data_error: float,
    prior_error: float,
    current_lr: float,
    reg_weight: float,
    cos_sim: float,
    S_gt: NDArray | None = None,
) -> None:
    """Single-step debug plot (3x2):
      - Row 1: physical S before/after (with optional GT) | normalized S before/after
      - Row 2: score + data gradient at lookahead (twin axes) | per-element step decomposition
      - Row 3: measurement-space fit  G·S vs B (before/after) | per-wavelength residual
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"[Debug] Step {step}  |  lr={current_lr:.2e}  |  reg_w={reg_weight:.2e}  |  "
        f"data_err={data_error:.3e}  |  prior_err={prior_error:.3e}  |  "
        f"cos(-∇data, score)={cos_sim:+.3f}",
        fontsize=12,
    )
    x = np.arange(len(S_phys_before))

    # --- (0,0) Physical space S ---
    ax = axes[0, 0]
    ax.set_title("Physical Space: S before vs. after step", fontweight="bold")
    if S_gt is not None and len(S_gt) == len(S_phys_before):
        ax.plot(x, S_gt, color="tab:gray", linewidth=1.5, linestyle=":", label="Ground truth")
    ax.plot(x, S_phys_before, color="tab:blue", linewidth=2, label="S (before step)")
    ax.plot(x, S_phys_after, color="tab:green", linewidth=2, linestyle="--", label="S (after step)")
    ax.set_xlabel("Depth index")
    ax.set_ylabel("S (physical)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best")

    # --- (0,1) Normalized space S ---
    ax = axes[0, 1]
    ax.set_title("Normalized Space: S before vs. after step", fontweight="bold")
    ax.plot(x, S_norm_before, color="tab:blue", linewidth=2, label="S_norm (before step)")
    ax.plot(x, S_norm_after, color="tab:green", linewidth=2, linestyle="--", label="S_norm (after step)")
    ax.axhline(1.0, color="tab:gray", linewidth=0.8, linestyle=":")
    ax.axhline(-1.0, color="tab:gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Depth index")
    ax.set_ylabel("S_norm  (training range = [-1, 1])")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best")

    # --- (1,0) Gradients (score + data) ---
    ax = axes[1, 0]
    ax.set_title("Gradients at lookahead (normalized space)", fontweight="bold")
    ax.plot(x, score_norm, color="tab:red", linewidth=2,
            label=f"Score (||.||={np.linalg.norm(score_norm):.2e})")
    ax.set_xlabel("Depth index")
    ax.set_ylabel("Score", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax.grid(True, linestyle="--", alpha=0.7)

    ax_r = ax.twinx()
    ax_r.plot(x, data_grad_norm, color="tab:blue", linewidth=2, linestyle="--",
              label=f"Data gradient (||.||={np.linalg.norm(data_grad_norm):.2e})")
    ax_r.set_ylabel("Data gradient", color="tab:blue")
    ax_r.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines + lines_r, labels + labels_r, loc="best")

    # --- (1,1) Step contributions ---
    ax = axes[1, 1]
    ax.set_title("Step contributions to ΔS_norm", fontweight="bold")
    ax.plot(x, data_step_contrib, color="tab:blue", linewidth=1.5,
            label=f"Data step  -lr·∇data  (||.||={np.linalg.norm(data_step_contrib):.2e})")
    ax.plot(x, prior_step_contrib, color="tab:red", linewidth=1.5,
            label=f"Prior step  +lr·w·score  (||.||={np.linalg.norm(prior_step_contrib):.2e})")
    ax.plot(x, momentum_carry, color="tab:orange", linewidth=1.5, linestyle=":",
            label=f"Momentum carry  μ·v_prev  (||.||={np.linalg.norm(momentum_carry):.2e})")
    ax.plot(x, velocity, color="k", linewidth=2, linestyle="--",
            label=f"Total velocity (||.||={np.linalg.norm(velocity):.2e})")
    ax.axhline(0.0, color="tab:gray", linewidth=0.5)
    ax.set_xlabel("Depth index")
    ax.set_ylabel("ΔS_norm")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=8)

    # --- (2,0) Measurement-space fit: G·S vs B ---
    ax = axes[2, 0]
    ax.set_title("Measurement Space: G·S before/after vs. target B", fontweight="bold")
    w_idx = np.arange(len(B_target))
    ax.plot(w_idx, B_target, color="k", linewidth=2, marker="o", markersize=4, label="B (target)")
    ax.plot(w_idx, eta_fit_before, color="tab:blue", linewidth=1.8, linestyle="--",
            label="G·S (before step)")
    ax.plot(w_idx, eta_fit_after, color="tab:green", linewidth=1.8, linestyle="-.",
            label="G·S (after step)")
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("eta (normalized units)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best")

    # --- (2,1) Per-wavelength residual ---
    ax = axes[2, 1]
    ax.set_title("Per-wavelength residual  G·S - B", fontweight="bold")
    res_before = eta_fit_before - B_target
    res_after = eta_fit_after - B_target
    ax.plot(w_idx, res_before, color="tab:blue", linewidth=1.8, marker="o", markersize=3,
            label=f"Before step  (||.||={np.linalg.norm(res_before):.2e})")
    ax.plot(w_idx, res_after, color="tab:green", linewidth=1.8, marker="s", markersize=3, linestyle="--",
            label=f"After step  (||.||={np.linalg.norm(res_after):.2e})")
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_xlabel("Wavelength index")
    ax.set_ylabel("Residual")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


# --- Solver Implementation ---
def solve_gradient_descent(
    G: NDArray,
    B: NDArray,
    hyperparams: ModelScoreGradConfig,
    S_gt: NDArray,
    preloaded_model: tuple | None = None,
) -> NDArray:
    """
    Solves for SELE using Nesterov Accelerated Gradient (NAG) with Score-Based Priors.
    :param G: Photogeneration matrix, NxM size
    :param B: ELE vector, B = GS, Nx1 size
    :param hyperparams: ModelScoreGradConfig dataclass
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
    if preloaded_model is not None:
        score_network, d_min, d_max, target_length = preloaded_model
        if G.shape[1] != target_length:
            raise ValueError(
                f"G has {G.shape[1]} spatial elements but model expects {target_length}. "
                f"The mesh must be built with mesh_resolution={target_length} to match the loaded model."
            )
    else:
        try:
            checkpoint = torch.load(hyperparams.model_path, map_location=device, weights_only=False)
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

            if G.shape[1] != model_config['target_length']:
                raise ValueError(
                    f"G has {G.shape[1]} spatial elements but model expects {model_config['target_length']}. "
                    f"The mesh must be built with mesh_resolution={model_config['target_length']} to match the loaded model."
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

    # 2. Normalization Constants
    # Using the exact min/max saved during training ensures perfect data reconstruction
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
    S_norm = np.clip(np.random.randn(N) * 0.5, -1.0, 1.0)
    velocity = np.zeros_like(S_norm) # Initialize momentum buffer v^(0) = 0

    # Trackers
    mse_history = []
    score_mag_history = []
    residual_norm_history = []
    velocity_norm_history = []
    lr_history = []
    adaptive_factor_history = []
    cos_sim_history = []
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
        # Cosine-anneal T from SCORE_T_MAX down to hyperparams.T0 across MAX_STEPS.
        # High T early -> coarse/smooth score (the prior the net learned for noisy data).
        # Low T late   -> sharp score that can resolve fine SELE structure.
        current_T = hyperparams.T0 + 0.5 * (SCORE_T_MAX - hyperparams.T0) \
            * (1 + np.cos(i / hyperparams.MAX_STEPS * np.pi))
        x_tensor = torch.tensor(S_lookahead, dtype=torch.float32, device=device).unsqueeze(0)
        t_tensor = torch.tensor(np.array([current_T]), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            score_model = score_network(x_tensor, t_tensor).squeeze().numpy()

        # --- D. Adaptive Scalar Weighting ---
        # Adaptive scalar weighting: REG_WEIGHT controls ratio of data fidelity vs prior, independent of their absolute magnitudes.
        grad_norm_mag = np.linalg.norm(grad_fidelity_norm)
        score_mag = np.linalg.norm(score_model) + 1e-12
        score_mag_history.append(score_mag)
        residual_norm_history.append(np.linalg.norm(residual))
        adaptive_factor = (grad_norm_mag / score_mag)
        adaptive_factor_history.append(adaptive_factor)
        # Cosine similarity between data-pull direction (-∇data) and prior-pull direction (score).
        # +1 = data and prior reinforce; -1 = they fight; 0 = orthogonal.
        cos_sim = float(-np.dot(grad_fidelity_norm, score_model) / (grad_norm_mag * score_mag + 1e-12))
        cos_sim_history.append(cos_sim)
        score_weighted = score_model * adaptive_factor
        # --- E. Momentum Update ---
        # Smoothly decays LR from LR_MAX to LR_MIN over the course of MAX_STEPS
        current_lr = hyperparams.LR_MIN + 0.5 * (hyperparams.LR_MAX - hyperparams.LR_MIN) * (1 + np.cos(i / hyperparams.MAX_STEPS * np.pi))
        lr_history.append(current_lr)
        # Cosine anneal REG_WEIGHT from its initial value to 0
        current_reg_weight = hyperparams.REG_WEIGHT * 0.5 * (1 + np.cos(i / hyperparams.MAX_STEPS * np.pi))
        total_update = grad_fidelity_norm - (score_model *  hyperparams.REG_WEIGHT )
        # Save pre-update velocity so the debug plot can show the momentum carry-over.
        velocity_prev = velocity.copy()
        # v^(t+1) = mu * v^(t) - eta * (grad - score_weighted)
        velocity = hyperparams.MOMENTUM * velocity - current_lr * total_update
        velocity_norm_history.append(float(np.linalg.norm(velocity)))

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
                if mse_history[-1] > 1 or np.isnan(mse_history[-1]):
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
            print(f"Iter={i:04d} | ScoreMag={score_mag:.2e} | DataGradMag={grad_norm_mag:.2e} | AdaptFactor={adaptive_factor:.2e}{mse_str}")

            # Debug Plotting -- comprehensive single-step view (physical + normalized,
            # gradients, per-element decomposition, and measurement-space fit)
            if hyperparams.IS_SHOW_DEBUG_PLOT:
                # S_norm has just been updated; recover the start-of-iter state.
                S_norm_before = S_norm - velocity
                S_norm_after = S_norm
                S_phys_before = (S_norm_before + 1.0) / norm_scale_factor + d_min
                S_phys_after = (S_norm_after + 1.0) / norm_scale_factor + d_min

                # Decompose the velocity into its three contributions.
                data_step_contrib = -current_lr * grad_fidelity_norm
                prior_step_contrib = current_lr * score_model * hyperparams.REG_WEIGHT
                momentum_carry = hyperparams.MOMENTUM * velocity_prev

                # Measurement-space reconstruction before/after the step.
                eta_fit_before = G_norm @ S_phys_before
                eta_fit_after = G_norm @ S_phys_after

                _plot_step_debug(
                    step=i,
                    S_norm_before=S_norm_before,
                    S_norm_after=S_norm_after,
                    S_phys_before=S_phys_before,
                    S_phys_after=S_phys_after,
                    score_norm=score_model,
                    data_grad_norm=grad_fidelity_norm,
                    data_step_contrib=data_step_contrib,
                    prior_step_contrib=prior_step_contrib,
                    momentum_carry=momentum_carry,
                    velocity=velocity,
                    eta_fit_before=eta_fit_before,
                    eta_fit_after=eta_fit_after,
                    B_target=B_norm,
                    data_error=float(np.dot(residual, residual)),
                    prior_error=float(score_mag ** 2),
                    current_lr=current_lr,
                    reg_weight=hyperparams.REG_WEIGHT,
                    cos_sim=cos_sim,
                    S_gt=S_gt,
                )

    # Plot convergence history -- 2x3 grid of diagnostics
    if hyperparams.IS_SHOW_MSE_PLOT:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle("Optimization Diagnostics", fontsize=13, fontweight="bold")

        # (0,0) MSE vs ground truth ----------------------------------------
        ax = axes[0, 0]
        if S_gt is not None and len(mse_history) > 0:
            ax.plot(mse_history, color="tab:purple", label="MSE(S, S_gt)")
            ax.set_yscale('log')
            ax.set_title("SELE reconstruction error vs. ground truth")
            ax.set_ylabel("MSE (physical units)")
            ax.legend(loc="best")
        else:
            ax.set_title("MSE vs. GT (no ground truth provided)")
            ax.text(0.5, 0.5, "S_gt = None", ha="center", va="center", transform=ax.transAxes,
                    color="tab:gray", fontsize=12)
        ax.set_xlabel("Optimization Step")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)

        # (0,1) Data residual ----------------------------------------------
        ax = axes[0, 1]
        ax.plot(residual_norm_history, color="tab:blue", label="||G·S - B||")
        ax.set_yscale('log')
        ax.set_title("Data residual convergence")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("L2 norm of residual")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend(loc="best")

        # (0,2) Score magnitude --------------------------------------------
        ax = axes[0, 2]
        ax.plot(score_mag_history, color="tab:red", label="||score||")
        ax.set_yscale('log')
        ax.set_title("Score network output magnitude")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("L2 norm of score")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend(loc="best")

        # (1,0) Velocity norm ----------------------------------------------
        ax = axes[1, 0]
        ax.plot(velocity_norm_history, color="k", label="||velocity||")
        ax.set_yscale('log')
        ax.set_title("Step size (velocity norm)")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("||v||")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.legend(loc="best")

        # (1,1) Cosine similarity of -∇data and score ---------------------
        ax = axes[1, 1]
        ax.plot(cos_sim_history, color="tab:green", linewidth=1.2,
                label="cos(-∇data, score)")
        ax.axhline(0.0, color="tab:gray", linewidth=0.7, linestyle="--")
        ax.axhline(1.0, color="tab:gray", linewidth=0.4, linestyle=":")
        ax.axhline(-1.0, color="tab:gray", linewidth=0.4, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Data vs. prior agreement\n(+1 = reinforce, -1 = fight)")
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Cosine similarity")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="best")

        # (1,2) LR schedule + adaptive factor ------------------------------
        ax = axes[1, 2]
        ax.plot(lr_history, color="tab:blue", label="current_lr")
        ax.set_yscale('log')
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel("Learning rate", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title("LR schedule  &  adaptive ratio ||∇data|| / ||score||")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)

        ax_r = ax.twinx()
        ax_r.plot(adaptive_factor_history, color="tab:orange", linestyle="--",
                  label="||∇data|| / ||score||")
        ax_r.set_yscale('log')
        ax_r.set_ylabel("Adaptive factor", color="tab:orange")
        ax_r.tick_params(axis="y", labelcolor="tab:orange")

        lines, labels = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines + lines_r, labels + labels_r, loc="best")

        plt.tight_layout()
        plt.show(block=False)

    # 6. Final Un-normalization
    S_final = (S_norm + 1.0) / norm_scale_factor + d_min

    return S_final