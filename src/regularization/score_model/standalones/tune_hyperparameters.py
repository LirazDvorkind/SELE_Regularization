"""Tune hyperparameters over a large set of curves"""
from dataclasses import replace
import numpy as np
import pandas as pd
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.regularization.score_model.standalones.helpers import load_S_B_G
from src.regularization.score_model.score_model_grad import solve_gradient_descent, load_score_model
from src.types.config import SCORE_MODEL_PRESETS

# Preset whose model_path and non-tuned settings are inherited by every trial.
PRESET = "d500"  # "d32" or "d500"

# Number of parallel worker processes.
# -1 = use all available CPU cores; 1 = single-process (no parallelism).
N_WORKERS = -1

# --- CONFIGURATION ---
# Ranges are tuned for d500: larger model → smaller safe LR, high-dim space → lower momentum,
# sinusoidal time embedding → T0 matters more and should be explored explicitly.
PARAM_GRID = {
    'reg_weight': [1, 5, 15],               # Low / mid / high score trust
    'lr_max':     [5e-4, 2e-3, 5e-3],       # All below d32's 1e-2; d500 needs smaller steps
    'momentum':   [0.75, 0.9],              # Test reduced inertia vs standard
    't0':         [5e-3, 2e-2, 1e-1],       # Fine detail → coarse; d500 sinusoidal embed is sensitive to this
}


# --- Module-level globals used inside worker processes ---
_worker_model = None  # Set once per worker via _worker_init


def _worker_init(model_path: str) -> None:
    """Called once per worker process to load the score model into a global."""
    global _worker_model
    _worker_model = load_score_model(model_path)


def _run_single_simulation(args: tuple) -> dict:
    """Run one (config, curve) simulation. Executed inside a worker."""
    config, item, G_matrix = args
    B_target = item['B']
    S_gt = item['S_gt']

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
        preloaded_model=_worker_model,
    )

    B_est = G_matrix @ S_est
    return {
        **config,
        'ele_error': np.mean((B_est - B_target) ** 2),
        'sele_error': np.mean((S_est - S_gt) ** 2),
    }


def run_tuning_suite(dataset, G_matrix):
    # Create all combinations
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    n_workers = os.cpu_count() if N_WORKERS == -1 else N_WORKERS
    model_path = SCORE_MODEL_PRESETS[PRESET].model_path

    total_simulations = len(combinations) * len(dataset)
    print(f"--- Starting Grid Search ---")
    print(f"Testing {len(combinations)} configs on {len(dataset)} curves.")
    print(f"Total Simulations: {total_simulations}")
    print(f"Workers: {n_workers} (model loaded once per worker)")

    args_list = [(config, item, G_matrix) for config in combinations for item in dataset]

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(model_path,),
    ) as executor:
        raw = list(tqdm(executor.map(_run_single_simulation, args_list), total=total_simulations, desc="Grid Search"))

    # Aggregate per-config: mean/max over all curves
    config_keys = list(PARAM_GRID.keys())
    df = pd.DataFrame(raw)
    return (
        df.groupby(config_keys)
        .agg(
            mean_ele_error=('ele_error', 'mean'),
            mean_sele_error=('sele_error', 'mean'),
            max_ele_error=('ele_error', 'max'),
        )
        .reset_index()
    )


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


# TODO: I asked: go over the @src/regularization/score_model/standalones/tune_hyperparameters.py code and tell me if you feel it implemented the tuning correctly
"""
Overall the tuning is **well-structured**, but there are a few real issues worth flagging:

---

## What's Correct

- **Worker efficiency**: model is loaded once per worker via `_worker_init`, not once per simulation — correct use of `ProcessPoolExecutor`.
- **Config inheritance**: `replace(SCORE_MODEL_PRESETS[PRESET], ...)` properly inherits all non-tuned settings (LR_MIN, MAX_STEPS, STOP_CHANGE, etc.) from the preset.
- **Shape safety**: `S_gt` in the dataset is always interpolated to `G_matrix.shape[1]` points in `generate_synthetic_data`, so `np.mean((S_est - S_gt) ** 2)` won't mismatch.
- **Dual metrics**: `sele_error` as primary (direct quality), `ele_error` as sanity check (data fidelity) is the right design.
- **Aggregation**: groupby + mean/max per config is correct.

---

## Real Issues

**1. Overfitting warning is inverted** (`generate_report`, line 125)

```python
if best_run['mean_ele_error'] > df['mean_ele_error'].median():
    print("⚠️ WARNING: ... low Reconstruction Error but HIGH Data Error.")
```

The condition fires when the best-SELE-error run has *above-median* `ele_error` (i.e., **high** reconstruction error). But the warning message says "low Reconstruction Error" — the opposite. The scenario being detected is "prior dominated the solution at the cost of data fidelity," not overfitting. The comment above it describes the opposite case (low ELE error, high SELE error), which isn't what the code checks.

**2. Early stopping creates unequal comparison budgets**

The solver stops when `|MSE[i] - MSE[i-1]| < STOP_CHANGE` for 20 consecutive steps. Configs that converge quickly run far fewer than `MAX_STEPS=5000` iterations. This is fine conceptually (you want the converged value), but configs that are slow to converge might be cut short by `MSE > 1 → break` at step 51, making the comparison unfair between configs that diverge vs. converge slowly.

**3. `LR_MIN` is not co-tuned with `LR_MAX`**

The cosine schedule goes `LR_MAX → LR_MIN = 1e-5`. When `LR_MAX = 5e-4`, the ratio is 50×; when `LR_MAX = 5e-3`, it's 500×. These are structurally different schedules, so comparing them conflates schedule shape with peak LR.

---

## Minor Points

- **No incremental save** — if the run crashes after 90% of simulations, everything is lost.
- **`TODO` in helpers.py (line 14)**: "Something seems off here when we run with the 500 long one" — this unresolved comment suggests the B generation for d500 may have a known uncertainty that could affect all tuning results.

The biggest actionable fix is the overfitting warning logic — it currently detects the wrong scenario and prints a misleading message. The rest are design trade-offs rather than bugs.
"""