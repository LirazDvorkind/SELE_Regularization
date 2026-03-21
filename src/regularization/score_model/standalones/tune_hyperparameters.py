"""Tune hyperparameters over a large set of curves"""
from dataclasses import replace
import numpy as np
import pandas as pd
import itertools
import os

from src.regularization.score_model.standalones.helpers import load_S_B_G
from src.regularization.score_model.score_model_grad import solve_gradient_descent
from src.types.config import SCORE_MODEL_PRESETS

# Preset whose model_path and non-tuned settings are inherited by every trial.
PRESET = "d500"  # "d32" or "d500"


# --- CONFIGURATION ---
# Ranges are tuned for d500: larger model → smaller safe LR, high-dim space → lower momentum,
# sinusoidal time embedding → T0 matters more and should be explored explicitly.
PARAM_GRID = {
    'reg_weight': [1, 5, 15],               # Low / mid / high score trust
    'lr_max':     [5e-4, 2e-3, 5e-3],       # All below d32's 1e-2; d500 needs smaller steps
    'momentum':   [0.75, 0.9],              # Test reduced inertia vs standard
    't0':         [5e-3, 2e-2, 1e-1],       # Fine detail → coarse; d500 sinusoidal embed is sensitive to this
}


def run_tuning_suite(dataset, G_matrix):
    # Create all combinations
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_runs = len(combinations) * len(dataset)
    print(f"--- Starting Grid Search ---")
    print(f"Testing {len(combinations)} configs on {len(dataset)} curves.")
    print(f"Total Simulations: {total_runs}")

    results = []
    run_count = 0

    for config in combinations:
        ele_errors = []  # Data Misfit (G*S - B)
        sele_errors = []  # Reconstruction Error (S - S_gt)

        for item in dataset:
            B_target = item['B']
            S_gt = item['S_gt']

            # --- RUN SOLVER ---
            S_est = solve_gradient_descent(
                G=G_matrix,
                B=B_target,
                hyperparams=replace(
                    SCORE_MODEL_PRESETS[PRESET],
                    REG_WEIGHT=config['reg_weight'],
                    LR_MAX=config['lr_max'],
                    MOMENTUM=config['momentum'],
                    T0=config['t0'],
                    IS_SHOW_DEBUG_PLOT=False,
                    IS_SHOW_DEBUG_DATA=False,
                    IS_SHOW_MSE_PLOT=False,
                ),
                S_gt=S_gt,
            )

            # 1. Calculate ELE Error (Data Misfit) - [USER REQUESTED METRIC]
            B_est = G_matrix @ S_est
            mse_ele = np.mean((B_est - B_target) ** 2)
            ele_errors.append(mse_ele)

            # 2. Calculate SELE Error (Ground Truth Misfit) - [FOR SAFETY CHECK]
            mse_sele = np.mean((S_est - S_gt) ** 2)
            sele_errors.append(mse_sele)

            run_count += 1
            if run_count % (total_runs // 10) == 0:
                print(f"Progress: {100 * run_count / total_runs:.2f}%... ({run_count}/{total_runs})")

        results.append({
            **config,
            'mean_ele_error': np.mean(ele_errors),  # The primary metric
            'mean_sele_error': np.mean(sele_errors),  # The sanity check
            'max_ele_error': np.max(ele_errors)
        })

    return pd.DataFrame(results)


def generate_report(df):
    # Sort by SELE Error
    df_sorted = df.sort_values(by='mean_sele_error')
    best_run = df_sorted.iloc[0]

    print("\n" + "=" * 50)
    print(f"🏆 WINNER (Lowest SELE Data Error)")
    print("=" * 50)
    print(f"Parameters: Reg={best_run['reg_weight']} | LR={best_run['lr_max']} | Mom={best_run['momentum']} | T0={best_run['t0']}")
    print(f"Error (SELE) : {best_run['mean_sele_error']:.2e} (Primary)")
    print(f"Error (ELE): {best_run['mean_ele_error']:.2e} (Sanity Check)")
    print("=" * 50)

    # Check for Overfitting Warning
    # If ELE error is low but SELE error is high compared to other runs, we are overfitting noise.
    if best_run['mean_ele_error'] > df['mean_ele_error'].median():
        print("⚠️ WARNING: The selected parameters have low Reconstruction Error but HIGH Data Error.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    points_amount = 32 if PRESET == "d32" else 500
    data, G = load_S_B_G(points_amount=points_amount)

    # Run Tuning
    results_df = run_tuning_suite(data, G)

    # Generate Report and Plots
    generate_report(results_df)

    # Save Stats
    os.makedirs("Results/tuning", exist_ok=True)
    results_df.to_csv("Results/tuning/hyperparameter_tuning_results_500.csv", index=False)
    print("Saved tuning results to Results/tuning/hyperparameter_tuning_results_500.csv")
