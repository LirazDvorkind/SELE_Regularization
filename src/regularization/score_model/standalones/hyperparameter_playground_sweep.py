"""Test a hyperparameter set on a curve to see how it fares"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.io import load_csv
from src.regularization.score_model.score_model_grad import load_score_model, solve_gradient_descent
from src.regularization.score_model.standalones.helpers import load_S_B_G
from src.types.config import SCORE_MODEL_PRESETS

# Pick a preset, then override only the fields you want to experiment with.
PRESET = "d500"  # "d32" or "d500"
BASE_HYPERPARAMS = replace(
    SCORE_MODEL_PRESETS[PRESET],
    LR_MAX=1e-5,
    LR_MIN=1e-8,
    MOMENTUM=0.9,
    MAX_STEPS=3000,
    T0=5e-2,
    IS_SHOW_DEBUG_DATA=False,
    IS_SHOW_MSE_PLOT=False,
    IS_SHOW_DEBUG_PLOT=False,
)

# --- L-curve REG_WEIGHT sweep ---
REG_WEIGHT_MIN = 1.0
REG_WEIGHT_MAX = 300.0
N_REG_WEIGHT = 6
REG_WEIGHT_VALS = np.logspace(np.log10(REG_WEIGHT_MIN), np.log10(REG_WEIGHT_MAX), N_REG_WEIGHT)


if __name__ == "__main__":
    random_sample = 765 # np.random.randint(100, 1000)
    model_size: int = 32 if PRESET == "d32" else 500
    print(f"Random curve number {random_sample}")
    data, G = load_S_B_G(points_amount=model_size, lower_index=random_sample, upper_index=random_sample + 1)

    # Load model once — used both by the solver and for score magnitude evaluation
    preloaded_model = load_score_model(BASE_HYPERPARAMS.model_path)
    score_network, d_min, d_max, _ = preloaded_model
    norm_scale_factor = 2.0 / (d_max - d_min)

    for item in data:
        B_target = item['B']
        S_gt = item['S_gt']

        # --- L-CURVE SWEEP over REG_WEIGHT ---
        sele_mses = []
        score_mags = []
        S_list = []

        for reg_w in REG_WEIGHT_VALS:
            hp = replace(BASE_HYPERPARAMS, REG_WEIGHT=reg_w)
            S_est = solve_gradient_descent(
                G=G, B=B_target, hyperparams=hp, S_gt=S_gt,
                preloaded_model=preloaded_model,
            )
            S_list.append(S_est)
            sele_mses.append(np.mean((S_est - S_gt) ** 2))

            # Compute score magnitude at the final solution
            S_norm = (S_est - d_min) * norm_scale_factor - 1.0
            x_t = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
            t_t = torch.tensor([[BASE_HYPERPARAMS.T0]], dtype=torch.float32)
            with torch.no_grad():
                score_vec = score_network(x_t, t_t).squeeze().numpy()
            score_mags.append(float(np.linalg.norm(score_vec)))

            print(f"REG_WEIGHT={reg_w:8.2f} | SELE_MSE={sele_mses[-1]:.4e} | ScoreMag={score_mags[-1]:.4e}")

        sele_mses = np.array(sele_mses)
        score_mags = np.array(score_mags)

        # Knee: minimum normalised Euclidean distance to origin in log-log space
        log_x = np.log10(np.maximum(sele_mses, 1e-30))
        log_y = np.log10(np.maximum(score_mags, 1e-30))
        log_x_n = (log_x - log_x.min()) / (log_x.max() - log_x.min() + 1e-30)
        log_y_n = (log_y - log_y.min()) / (log_y.max() - log_y.min() + 1e-30)
        knee_idx = int(np.argmin(np.sqrt(log_x_n ** 2 + log_y_n ** 2)))
        reg_weight_knee = REG_WEIGHT_VALS[knee_idx]
        print(f"\nKnee at REG_WEIGHT={reg_weight_knee:.4f} (index {knee_idx})")

        # --- L-curve plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(sele_mses, score_mags, 'b.-', linewidth=1.5, label='L-curve')
        ax.loglog(sele_mses[knee_idx], score_mags[knee_idx], 'rx',
                  markersize=12, markeredgewidth=2, label=f'Knee  REG_WEIGHT={reg_weight_knee:.2f}')
        for i, rw in enumerate(REG_WEIGHT_VALS):
            ax.annotate(f'{rw:.1f}', (sele_mses[i], score_mags[i]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.7)
        ax.set_xlabel(r'SELE MSE  $\|S - S_{gt}\|^2$')
        ax.set_ylabel(r'Score Magnitude  $\|\mathrm{score}(S)\|$')
        ax.set_title('L-Curve: REG_WEIGHT Sweep')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show(block=False)

        # --- Results at knee ---
        S_knee = S_list[knee_idx]
        B_est = G @ S_knee
        mse_ele = np.mean((B_est - B_target) ** 2)
        mse_sele = np.mean((S_knee - S_gt) ** 2)
        print(f"Knee solution — mse_ele={mse_ele:.4e}  mse_sele={mse_sele:.4e}")

        W = 30e-4  # cm
        x_res = G.shape[1]
        z_centres = np.linspace(0, W, x_res) * 1e4  # µm

        fig2, ax2 = plt.subplots()
        ax2.plot(z_centres, S_knee, label=f'SELE reconstructed  (REG_WEIGHT={reg_weight_knee:.2f})')
        ax2.plot(z_centres, S_gt, 'k--', label='SELE ground truth')
        ax2.set_xlabel('z [µm]')
        ax2.set_ylabel('SELE')
        ax2.set_title('SELE vs Ground Truth (Knee Solution)')
        ax2.legend()
        plt.show(block=False)

        wavelengths = load_csv(
            str(Path(__file__).resolve().parents[4] / "Data" / "wavelength_nm.csv")).ravel()
        fig3, ax3 = plt.subplots()
        ax3.plot(wavelengths, B_target, label='Measured')
        ax3.plot(wavelengths, B_est, '--', label='Reconstructed')
        ax3.set_xlabel('Wavelength [nm]')
        ax3.set_ylabel(r'$\eta_{ext}$')
        ax3.set_title('Reconstructed ELE (Knee Solution)')
        ax3.legend()
        plt.show(block=True)
