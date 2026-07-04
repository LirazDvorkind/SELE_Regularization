"""Fast auto-tuner for the two main MODEL_SCORE_GRAD solver hyperparameters.

Instead of the brute-force grid search in ``tune_hyperparameters.py`` (54 configs
x 100 curves = 5,400 full solves), this recommends ``LR_MAX`` and ``REG_WEIGHT``
(plus a derived ``LR_MIN``) for a given checkpoint in seconds-to-a-minute, using:

  Phase A0 - Analytical seed: read the model's actual gradient/score scale at init
             (via ``compute_init_diagnostics``) and seed a learning rate so the first
             momentum-amplified step moves S_norm by ~TARGET_NORM_STEP of its range.
             This is what makes the LR ladder auto-adapt to any newly trained model.

  Phase A  - LR range test: short solves across a geometric ladder centred on the seed,
             find the largest LR that stays stable (no ground truth needed), and back off
             by LR_SAFETY_FACTOR.

  Phase B  - REG_WEIGHT micro-sweep: with LR fixed, a tiny 1-D sweep over REG_GRID on a
             few validation curves, ranked by mean MSE_SELE (the same metric the grid
             tuner optimizes), with a prior-dominated warning.

Run it directly (it is the sole entrypoint; nothing here runs via score_model_grad.py):

    python -m src.regularization.score_model.standalones.new_model_toolkit.auto_tune_hyperparameters

It only PRINTS a recommendation + a paste-ready snippet; it never edits config.py.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io import load_csv
from src.regularization.score_model.score_model_grad import (
    solve_gradient_descent,
    load_score_model,
    compute_init_diagnostics,
)
from src.regularization.score_model.standalones.helpers import generate_synthetic_data
from src.types.config import SCORE_MODEL_PRESETS

# ===========================================================================
# INPUTS -- EDIT THESE WHEN TUNING A NEW MODEL / NEW TRAINING DATA
# ===========================================================================
# Preset whose non-tuned settings (MOMENTUM, T0, MAX_STEPS shape, model_path) are
# inherited by every trial. Pick the one matching your model architecture.
PRESET = "d500"  # "d32" or "d500"

# Path to the .pt checkpoint to tune. None -> use SCORE_MODEL_PRESETS[PRESET].model_path.
# Point this at your freshly trained model.
MODEL_PATH: str | None = None

_DATA_DIR = Path(__file__).resolve().parents[5] / "Data" / "score_model"

# SELE profiles used to build synthetic validation curves (B = G @ S, noise-free).
# Repoint this at whatever dataset your new model was trained on.
DATASET_PATH: str = str(_DATA_DIR / "datasets" / "sele_simulated_1000_curves_500_long.csv")

# Precomputed photogeneration matrix. Its column count MUST equal the model's
# target_length (32 -> G_score_model.csv, 500 -> G_score_model_500.csv). For an
# arbitrary target_length, build G via calc_mesh_and_G (see module docstring / plan).
G_SOURCE: str | None = None  # None -> auto-pick by target_length read from the checkpoint

# The single curve used for the Phase-A0 analytical seed. Pick a curve whose shape
# is TYPICAL of your dataset (the seed just needs a representative gradient scale).
REPRESENTATIVE_CURVE_INDEX = 0

# Validation curves (row slice of DATASET_PATH) for the REG sweep, and how many of
# them the LR range test uses (a small subset is enough to detect instability).
VAL_LOWER, VAL_UPPER = 0, 8
N_LR_PROBE_CURVES = 3

# ---------------------------------------------------------------------------
# Tuner knobs (sensible defaults; rarely need changing)
# ---------------------------------------------------------------------------
PROBE_STEPS = 300           # short solve budget for the LR range test
REG_PROBE_STEPS = 1000      # slightly longer budget for the REG sweep
REG_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # 0 = pure data-fit baseline
# The step-0 seed is a conservative LOWER bound on the usable peak LR (the gradient at a
# random init is near-maximal, since it is farthest from the solution), so the ladder
# extends mostly UPWARD from the seed to bracket the true stability ceiling.
LR_LADDER_LO_MULT = 1e-1    # ladder starts at LR_seed * this
LR_LADDER_HI_MULT = 1e4     # ladder ends   at LR_seed * this
LR_LADDER_POINTS = 12
LR_SAFETY_FACTOR = 3.0      # LR_MAX = LR_ceiling / this
LR_MIN_RATIO = 1e-3         # LR_MIN = LR_MAX * this
TARGET_NORM_STEP = 0.02     # desired first-step motion in normalized [-1,1] space
# Normalized residual ||G S - B|| / ||B||: the S=0 / random-init baseline is O(1), and a
# stable solve drives it below that. True divergence makes the residual EXPLODE by orders of
# magnitude (e.g. 1e2 -> 1e12), leaving a huge gap between "slow but stable" (~O(1)) and
# "diverged" (>>1). Put the threshold in that gap: high enough that an under-converged-but-
# stable short run (residual a few x the baseline) is not mislabelled, low enough to catch any
# real explosion. Detection is scale-free, so this transfers across models/datasets.
DIVERGE_RESID = 10.0        # normalized residual above this (or non-finite) = diverged/unstable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _solve(G, item, preset, preloaded, *, lr_max, reg_weight, max_steps):
    """Run one short, quiet solve and return (S_est, mse_sele, mse_ele, norm_resid, diverged)."""
    cfg = replace(
        preset,
        LR_MAX=lr_max,
        LR_MIN=lr_max * LR_MIN_RATIO,
        REG_WEIGHT=reg_weight,
        MAX_STEPS=max_steps,
        MIN_STEPS=10,
        IS_SHOW_DEBUG_PLOT=False,
        IS_SHOW_DEBUG_DATA=False,
        IS_SHOW_MSE_PLOT=False,
    )
    S_gt = item['S_gt']
    B = item['B']
    S_est = solve_gradient_descent(G=G, B=B, hyperparams=cfg, S_gt=S_gt, preloaded_model=preloaded)

    B_est = G @ S_est
    finite = np.all(np.isfinite(S_est))
    norm_resid = float(np.linalg.norm(B_est - B) / (np.linalg.norm(B) + 1e-30)) if finite else np.inf
    diverged = (not finite) or norm_resid > DIVERGE_RESID
    mse_sele = float(np.mean((S_est - S_gt) ** 2)) if finite else np.inf
    mse_ele = float(np.mean((B_est - B) ** 2)) if finite else np.inf
    return S_est, mse_sele, mse_ele, norm_resid, diverged


def run(model_path=None, dataset_path=None, preset_name=PRESET):
    preset = SCORE_MODEL_PRESETS[preset_name]
    model_path = model_path or MODEL_PATH or preset.model_path
    dataset_path = dataset_path or DATASET_PATH

    print(f"Loading score model: {model_path}")
    preloaded = load_score_model(model_path)
    _net, d_min, d_max, target_length = preloaded
    print(f"  target_length={target_length}  data_min={d_min:.3e}  data_max={d_max:.3e}")

    # --- Load G (must match target_length) ---
    g_source = G_SOURCE
    if g_source is None:
        suffix = "_500" if target_length == 500 else ""
        g_source = str(_DATA_DIR / f"G_score_model{suffix}.csv")
    G = load_csv(g_source)
    if G.shape[1] != target_length:
        raise ValueError(
            f"G from {g_source} has {G.shape[1]} columns but the model expects "
            f"{target_length}. Point G_SOURCE at a matching matrix (or build one via "
            f"calc_mesh_and_G at mesh_resolution={target_length})."
        )
    print(f"Loaded G {G.shape} from {g_source}")

    # --- Build synthetic validation curves (B = G @ S) ---
    profiles = load_csv(dataset_path)
    val_profiles = profiles[VAL_LOWER:VAL_UPPER, :]
    val_curves = generate_synthetic_data(val_profiles, G)
    rep_curve = generate_synthetic_data(profiles[REPRESENTATIVE_CURVE_INDEX:REPRESENTATIVE_CURVE_INDEX + 1, :], G)[0]
    print(f"Loaded {len(val_curves)} validation curves from {dataset_path} "
          f"(representative index {REPRESENTATIVE_CURVE_INDEX})\n")

    # =====================================================================
    # Phase A0 - Analytical seed
    # =====================================================================
    # Evaluate at the solver's ACTUAL starting point (its deterministic random init,
    # S_init=None) -- that is where the real initial gradient lives. Seeding from the
    # ground-truth profile would give a ~zero data gradient (B = G @ S_gt exactly) and
    # a meaningless LR. The representative curve only supplies the target B.
    diag = compute_init_diagnostics(
        G=G, B=rep_curve['B'], preloaded_model=preloaded, hyperparams=preset,
        S_init=None,
    )
    lr_seed = TARGET_NORM_STEP * (1.0 - preset.MOMENTUM) / (diag['total_update_norm'] + 1e-30)

    print("=" * 62)
    print("Phase A0 - Analytical seed (from representative curve)")
    print("=" * 62)
    print(f"  ||grad_data||     = {diag['grad_norm_mag']:.3e}")
    print(f"  ||score||         = {diag['score_mag']:.3e}")
    print(f"  adaptive_factor   = {diag['adaptive_factor']:.3e}")
    print(f"  cos_sim(data,prior) = {diag['cos_sim']:+.3f}  "
          f"({'reinforce' if diag['cos_sim'] > 0 else 'fight'})")
    print(f"  ||total_update||  = {diag['total_update_norm']:.3e}")
    print(f"  -> LR_seed        = {lr_seed:.3e}  "
          f"(targets ~{TARGET_NORM_STEP:.0%} first-step motion, MOMENTUM={preset.MOMENTUM})\n")

    # =====================================================================
    # Phase A - LR range test
    # =====================================================================
    ladder = lr_seed * np.logspace(
        np.log10(LR_LADDER_LO_MULT), np.log10(LR_LADDER_HI_MULT), LR_LADDER_POINTS
    )
    rows = []
    for lr in tqdm(ladder, desc="Phase A: LR range test"):
        residuals, diverged_flags = [], []
        for item in val_curves[:N_LR_PROBE_CURVES]:
            _, _, _, norm_resid, diverged = _solve(
                G, item, preset, preloaded, lr_max=lr, reg_weight=preset.REG_WEIGHT, max_steps=PROBE_STEPS,
            )
            residuals.append(norm_resid)
            diverged_flags.append(diverged)
        rows.append({
            'lr_max': lr,
            'diverged_frac': float(np.mean(diverged_flags)),
            'median_norm_resid': float(np.median([r for r in residuals if np.isfinite(r)] or [np.inf])),
        })
    lr_df = pd.DataFrame(rows)

    print("\n" + "=" * 62)
    print("Phase A - LR range test")
    print("=" * 62)
    print(lr_df.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    stable = lr_df[lr_df['diverged_frac'] == 0.0]
    if len(stable) == 0:
        lr_ceiling = float(lr_df['lr_max'].min())
        print("\n  WARNING: every ladder LR diverged. Recommending the smallest probed LR / "
              "safety; consider lowering TARGET_NORM_STEP or widening the ladder.")
    else:
        lr_ceiling = float(stable['lr_max'].max())
        if lr_ceiling == float(lr_df['lr_max'].max()):
            print("\n  NOTE: no LR in the ladder diverged - the true ceiling may be higher. "
                  "The recommendation is safe but possibly conservative.")
    rec_lr_max = lr_ceiling / LR_SAFETY_FACTOR
    rec_lr_min = rec_lr_max * LR_MIN_RATIO
    print(f"\n  LR_ceiling (largest stable) = {lr_ceiling:.3e}")
    print(f"  -> recommended LR_MAX = {rec_lr_max:.3e}  (ceiling / {LR_SAFETY_FACTOR:g})")
    print(f"  -> derived     LR_MIN = {rec_lr_min:.3e}  (LR_MAX * {LR_MIN_RATIO:g})\n")

    # =====================================================================
    # Phase B - REG_WEIGHT micro-sweep (validated vs ground truth)
    # =====================================================================
    rows = []
    for reg in tqdm(REG_GRID, desc="Phase B: REG sweep"):
        sele_errs, ele_errs = [], []
        for item in val_curves:
            _, mse_sele, mse_ele, _, _ = _solve(
                G, item, preset, preloaded, lr_max=rec_lr_max, reg_weight=reg, max_steps=REG_PROBE_STEPS,
            )
            sele_errs.append(mse_sele)
            ele_errs.append(mse_ele)
        rows.append({
            'reg_weight': reg,
            'mean_sele_error': float(np.mean(sele_errs)),
            'mean_ele_error': float(np.mean(ele_errs)),
        })
    reg_df = pd.DataFrame(rows)

    print("\n" + "=" * 62)
    print("Phase B - REG_WEIGHT micro-sweep (ranked by mean MSE_SELE)")
    print("=" * 62)
    print(reg_df.sort_values('mean_sele_error').to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    best = reg_df.loc[reg_df['mean_sele_error'].idxmin()]
    rec_reg = float(best['reg_weight'])
    # Prior-dominated warning (same idea as tune_hyperparameters.py): the best-SELE
    # config also fitting the data poorly means the prior is winning over the data.
    prior_dominated = best['mean_ele_error'] > reg_df['mean_ele_error'].median()

    # =====================================================================
    # Report
    # =====================================================================
    print("\n" + "=" * 62)
    print("RECOMMENDATION")
    print("=" * 62)
    print(f"  LR_MAX     = {rec_lr_max:.3e}")
    print(f"  LR_MIN     = {rec_lr_min:.3e}")
    print(f"  REG_WEIGHT = {rec_reg:g}")
    if prior_dominated:
        print("\n  WARNING: the chosen REG_WEIGHT has above-median ELE error - the score "
              "prior may be dominating at the cost of data fidelity. Consider a smaller REG_WEIGHT.")
    print("\n  Paste-ready (in your pipeline / playground config):")
    print(f'  # Recommended for {model_path}')
    print(f'  replace(SCORE_MODEL_PRESETS["{preset_name}"], '
          f'LR_MAX={rec_lr_max:.3e}, LR_MIN={rec_lr_min:.3e}, REG_WEIGHT={rec_reg:g})')
    print("=" * 62)


def main():
    run()


if __name__ == "__main__":
    main()
