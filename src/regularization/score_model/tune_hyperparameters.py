import numpy as np
import pandas as pd
import itertools
import os

from src.regularization.score_model.helpers import load_S_B_G
from src.regularization.score_model.score_model_grad import solve_gradient_descent
from src.types.score_model_params import NesterovHyperparams


# --- CONFIGURATION ---
PARAM_GRID = {
    'reg_weight': [1, 2.5, 5, 10, 20],
    'lr_max': [0.01, 0.02, 0.04],
    'momentum': [0.8, 0.9, 0.95]
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
                hyperparams=NesterovHyperparams(
                    REG_WEIGHT=config['reg_weight'],
                    LR_MAX=config['lr_max'],
                    MOMENTUM=config['momentum'],
                    model_path="../../../Data/sele_score_net_d32.pt",
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
    print(f"Parameters: Reg={best_run['reg_weight']} | LR={best_run['lr_max']} | Mom={best_run['momentum']}")
    print(f"Error (SELE) : {best_run['mean_sele_error']:.2e} (Primary)")
    print(f"Error (ELE): {best_run['mean_ele_error']:.2e} (Sanity Check)")
    print("=" * 50)

    # Check for Overfitting Warning
    # If ELE error is low but SELE error is high compared to other runs, we are overfitting noise.
    if best_run['mean_ele_error'] > df['mean_ele_error'].median():
        print("⚠️ WARNING: The selected parameters have low Reconstruction Error but HIGH Data Error.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    data, G = load_S_B_G()

    # Run Tuning
    results_df = run_tuning_suite(data, G)

    # Generate Report and Plots
    generate_report(results_df)

    # Save Stats
    os.makedirs("../../../results/tuning", exist_ok=True)
    results_df.to_csv("../../../results/tuning/hyperparameter_tuning_results.csv", index=False)
    print("Saved tuning results to results/tuning/hyperparameter_tuning_results.csv")
