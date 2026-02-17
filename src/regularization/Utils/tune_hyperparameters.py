import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.interpolate import interp1d

from src.regularization.score_model_grad import solve_gradient_descent

# TODO - read this code and understand it, then add the MATLAB curves to it.
#    https://gemini.google.com/u/2/gem/d5c1be0fd6b8/bf2d3f78150a16b4

# --- CONFIGURATION ---
PARAM_GRID = {
    'reg_weight': [1, 5, 10, 20, 50],
    'lr_max': [0.005, 0.01, 0.02],
    'momentum': [0.8, 0.9, 0.95]
}

SOLVER_DIM = 32
GT_DIM = 100

def preprocess_ground_truth(S_gt_100, target_dim=32):
    x_old = np.linspace(0, 30, len(S_gt_100))
    x_new = np.linspace(0, 30, target_dim)
    f = interp1d(x_old, S_gt_100, kind='linear')
    return f(x_new)


def generate_synthetic_data(S_gt_profiles, G_matrix):
    dataset = []
    for S_100 in S_gt_profiles:
        S_32 = preprocess_ground_truth(S_100, target_dim=SOLVER_DIM)
        # B = G * S (Physical measurement)
        B = G_matrix @ S_32
        dataset.append({'S_gt_32': S_32, 'B': B})
    return dataset


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
            S_gt = item['S_gt_32']

            # --- RUN SOLVER ---
            S_est = solve_gradient_descent(
                G=G_matrix,
                B=B_target,
                steps=3000,
                reg_weight=config['reg_weight'],
                lr_max=config['lr_max'],
                momentum=config['momentum'],
                S_gt=S_gt
            )

            # 1. Calculate ELE Error (Data Misfit) - [USER REQUESTED METRIC]
            B_est = G_matrix @ S_est
            mse_ele = np.mean((B_est - B_target) ** 2)
            ele_errors.append(mse_ele)

            # 2. Calculate SELE Error (Ground Truth Misfit) - [FOR SAFETY CHECK]
            mse_sele = np.mean((S_est - S_gt) ** 2)
            sele_errors.append(mse_sele)

            run_count += 1
            if run_count % 50 == 0:
                print(f"Progress: {run_count}/{total_runs}...")

        results.append({
            **config,
            'mean_ele_error': np.mean(ele_errors),  # The primary metric
            'mean_sele_error': np.mean(sele_errors),  # The sanity check
            'max_ele_error': np.max(ele_errors)
        })

    return pd.DataFrame(results)


def generate_report(df):
    # Sort by ELE Error (Data Fidelity) as requested
    df_sorted = df.sort_values(by='mean_ele_error')
    best_run = df_sorted.iloc[0]

    print("\n" + "=" * 50)
    print(f"🏆 WINNER (Lowest ELE Data Error)")
    print("=" * 50)
    print(f"Parameters: Reg={best_run['reg_weight']} | LR={best_run['lr_max']} | Mom={best_run['momentum']}")
    print(f"Error (ELE) : {best_run['mean_ele_error']:.2e} (Primary)")
    print(f"Error (SELE): {best_run['mean_sele_error']:.2e} (Sanity Check)")
    print("=" * 50)

    # Check for Overfitting Warning
    # If ELE error is low but SELE error is high compared to other runs, we are overfitting noise.
    if best_run['mean_sele_error'] > df['mean_sele_error'].median():
        print("⚠️ WARNING: The selected parameters have low Data Error but HIGH Reconstruction Error.")
        print("This suggests the solver is 'overfitting' (ignoring physics to fit the data).")
        print("Consider picking a parameter set with slightly higher ELE error but lower SELE error.")

    # Plotting code (simplified for brevity)
    # We plot ELE error vs Reg Weight to see the trade-off
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='reg_weight', y='mean_ele_error', hue='momentum', marker='o', label='Data Error (ELE)')
    plt.yscale('log')
    plt.title("Tuning Results: Data Misfit vs Regularization")
    plt.ylabel("Mean Squared Error (ELE)")
    plt.grid(True, alpha=0.3)
    plt.show()


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    import os
    from src.io import load_csv

    print("Loading SELE and G...")

    # 1. Load optical parameters from CONFIG
    G = load_csv("../../../Data/G_score_model.csv")
    # 4. Load the dataset of 100-point SELE profiles
    S_profiles_100 = load_csv("../../../Data/sele_dataset.csv")

    # 5. Prepare synthetic measurements (B) for all profiles
    print(f"Preparing synthetic dataset for {len(S_profiles_100)} profiles...")
    data = generate_synthetic_data(S_profiles_100, G)

    # 6. Run Tuning
    results_df = run_tuning_suite(data, G)

    # 7. Generate Report and Plots
    generate_report(results_df)

    # 8. Save Stats
    os.makedirs("results/tuning", exist_ok=True)
    results_df.to_csv("results/tuning/hyperparameter_tuning_results.csv", index=False)
    print("Saved tuning results to results/tuning/hyperparameter_tuning_results.csv")
