# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It contains durable domain knowledge (physics, math, architecture) distilled from the papers and codebase. It intentionally points to source files for values that change often (hyperparameters, dataset paths, exact solver steps) rather than duplicating them.

This principle applies to every README in this repo too: prefer describing concepts and pointing to
where specifics live in code over restating specifics that will drift.

### Comments

Don't overdo comments. Write them only where the WHY is genuinely non-obvious (a physics
convention, a workaround, a non-obvious ordering requirement) -- never to restate WHAT a
self-explanatory line already says. This applies to MATLAB and Python code alike.

### Maintaining this file

Keep CLAUDE.md a generic rulebook. When adding or editing a rule here, don't illustrate it
with an example drawn from the task you were just doing -- write the rule so it stands on
its own for any future task. Task-specific detail belongs in the PR/commit, not here.

---

## Project Overview

This project solves an **inverse problem** in semiconductor photovoltaics. Given wavelength-dependent External Luminescence Efficiency (ELE) measurements of a GaAs wafer, it reconstructs the **Spatial External Luminescence Efficiency (SELE)** -- a depth profile showing how much each layer contributes to photoluminescence.

### The Forward Model

```
eta_ext(lambda_in) = (1 / phi_abs(lambda_in)) * integral_0^d  DeltaG(lambda_in, z) * S(z) dz
```

- `eta_ext`: measured ELE per wavelength
- `DeltaG(lambda, z)`: Beer-Lambert optical generation: `alpha = 4*pi*k/lambda`, `G[lambda,i] = exp(-alpha*z_i) - exp(-alpha*z_{i+1})`
- `S(z)`: the SELE profile to reconstruct (the unknown)
- `phi_abs`: absorbed photon flux

Discretized: `eta_ext = (1/phi_abs) * G @ S`, where G is (L wavelengths × M depth elements), S is (M,). The system is ill-posed -- small noise causes wild oscillations in naive least-squares solutions. Regularization is required.

Because `phi_abs = phi_0 * A` already normalizes by the absorbed fraction, front-surface
reflectance cancels out of the discretized system and must not be applied to G a second time.

### Two Absorption Coefficients

The paper distinguishes `alpha`, which includes free-carrier absorption (the Drude term),
from `alpha_b`, which excludes it. FCA attenuates the beam but is an intra-band transition
that frees no carrier, so attenuation follows `alpha` while generation and emission follow
`alpha_b` -- optical generation carries the prefactor `alpha_b / alpha` (SI eq. S4.1). This is
what makes G a *generation* matrix rather than an absorption matrix; a row of G sums to
`alpha_b/alpha`, not to 1.

`src/optical_constants.py` is the single source of optics for the project — it reads the
paper's own ellipsometry constants from `Data/Tamir_paper_SELE_figs/`, and every G, in the
solver and in the test set alike, is built from it. Pass `k_bulk` alongside `k` when building
G; omitting it silently models absorption instead of generation.

### Physical Intuition for the SELE Profile

Expected shape for a p-type GaAs wafer (paper Figure 2b):

- **Near surface (z ~ 0)**: LOW (~0.05-0.1) due to surface recombination suppressing radiative recombination
- **Rises to peak** (~0.3-0.45) at ~0.5-1 µm; peak position correlates with minority carrier diffusion length
- **Decays toward 0** deeper (self-absorption); the profile approaches ~0 at the back of the 350 µm wafer

| Physical Parameter | Effect on SELE |
|--------------------|----------------|
| Surface recombination velocity (SRV) | Controls dip at z=0. Higher SRV = lower surface SELE. |
| SRH lifetime (tau_SRH) | Controls bulk SELE magnitude. Longer lifetime = higher bulk SELE. |
| Diffusion length | Peak position ~ diffusion length when surface recomb dominates. |
| Photon recycling | Increases quasi-Fermi level separation deep in material; keeps SELE nonzero far from surface. |

---

## Commands

### Run the main pipeline
```bash
python -m src.main
```

### Run standalone scripts (from repo root)
```bash
# Run as scripts:
python src/regularization/score_model/standalones/hyperparameter_playground.py
python src/regularization/score_model/standalones/tune_hyperparameters.py
python src/regularization/score_model/standalones/model_training/sele-score-model-training-script.py
python src/regularization/score_model/standalones/model_training/test-score-models.py

# Or as modules:
python -m src.regularization.score_model.standalones.hyperparameter_playground
python -m src.regularization.score_model.standalones.tune_hyperparameters
```

- **hyperparameter_playground.py**: Quick test of one config on a random synthetic curve. Shows MSE_ELE (data fit) and MSE_SELE (ground truth error). Good for interactive debugging.
- **tune_hyperparameters.py**: Automated grid search with multiprocessing over the hyperparameter space.
- **model_training/sele-score-model-training-script.py**: Trains a score checkpoint from a synthetic SELE dataset.
- **model_training/test-score-models.py**: Visualizes score gradients from trained checkpoints.

### Rebuild the ground-truth test set
```bash
python -m src.test_set.build_test_set   # re-extract profiles from the paper figures
python -m src.test_set.plot_test_set    # overview figure of both parameter sweeps
```

### Install dependencies
```bash
uv sync
```

Key packages: `numpy`, `scipy`, `matplotlib`, `torch`, `cvxpy`, `mplcursors`, `seaborn`, `pandas`

---

## Architecture

### Entry point & flow
`src/main.py` → `src/pipeline.py:run_regularization()` — orchestrates data loading, G-matrix computation, solving, and result export.

### Three regularization modes (set via `CONFIG.regularization_method`)

| Mode | Solver | Key file |
|------|--------|---------|
| `NON_UNIFORM_MESH` | Classical Tikhonov with adaptive mesh | `tikhonov_non_uniform.py` |
| `TOTAL_VARIATION` | Two-parameter Tikhonov (CVXPY) | `total_variation.py` |
| `MODEL_SCORE_GRAD` | Nesterov gradient + score prior | `score_model_grad.py` |

**NON_UNIFORM_MESH**: `min ||G@S - B||² + κ²||L@S||²`, L = second-derivative operator. Sweeps κ, finds L-curve knee, averages solutions in confidence window. Non-uniform mesh: fine linear spacing near surface, logarithmic deeper. This is the original paper's method.

**TOTAL_VARIATION**: Two-parameter Tikhonov solved with CVXPY. Allows sharper features than standard Tikhonov.

**MODEL_SCORE_GRAD**: Replaces L with a learned prior from a score-based diffusion model. Score network learns `grad(log p(S))` from synthetic SELE training data. See Score Model section below.

### Pipeline flow

**NON_UNIFORM_MESH / TOTAL_VARIATION**:
1. Load ELE data and optical parameters (k, λ, z)
2. Build G matrix on chosen mesh (shape: L wavelengths × M-1 elements)
3. Whiten by median row norm (stabilizes ill-conditioned system)
4. Sweep κ → solve `min ||GS - B||² + κ²||LS||²` for each
5. Knee detection: normalized Euclidean distance to origin in log-log (residual vs seminorm)
6. Confidence window around κ_knee → extract S_mean, S_std
7. Save CSV results + generate plots

**MODEL_SCORE_GRAD** (`pipeline.py`, MODEL_SCORE_GRAD branch of `run_regularization()`):
1. Load ELE data (`ELE_sim.csv`), optical params (k, wavelengths, z)
2. Load score model checkpoint → get `target_length` (32 or 500)
3. Build G matrix on linear mesh with `target_length` elements
4. Normalize: multiply G, B by `photon_flux * e_charge` (unit factor)
5. Call `solve_gradient_descent(G, B, hyperparams, S_gt)` (optionally warm-started from a TV solve instead of random init; see `warm_start_with_tv` in `ModelScoreGradConfig`)
6. Upsample SELE from M points to `output_mesh_resolution` via `expand_sele()`
7. Reconstruct: `eta_fit = G_longer @ S_rec / unit_factor`
8. Save CSVs + generate plots

### Key modules

| File | Role |
|------|------|
| `src/main.py` | Entry point |
| `src/pipeline.py` | Orchestrates all regularization methods |
| `src/types/config.py` | Global CONFIG, ModelScoreGradConfig, presets |
| `src/__init__.py` | CONFIG import (avoid circular imports) |
| `src/mesh.py` | Builds G matrix (Beer-Lambert), linear and non-uniform meshes |
| `src/optical_constants.py` | The project's optics: n and k with/without the Drude term |
| `src/matlab_fig.py` | Reads MATLAB `.fig` files without MATLAB |
| `src/operators.py` | L0/L1/L2 regularization matrices |
| `src/regularization/score_model/score_model_grad.py` | NAG solver with score prior |
| `src/regularization/score_model/model_definition.py` | ScoreNetwork PyTorch model |
| `src/regularization/score_model/standalones/hyperparameter_playground.py` | Test single hyperparameter config on one curve |
| `src/regularization/score_model/standalones/tune_hyperparameters.py` | Grid search over hyperparameter space |
| `src/regularization/score_model/standalones/model_training/` | Training scripts and Colab notebook |
| `src/test_set/` | Builds and loads the ground-truth SELE test set extracted from the paper's MATLAB figures |

### Data files

| Category | Location | Contents |
|----------|----------|----------|
| Measurement | `Data/ELE_sim.csv` | Simulated ELE measurement (the "observation") |
| Ground truth | `Data/SELE_ground_truth.csv` | Ground truth SELE for comparison |
| Ground-truth test set | `Data/test_set/` | Multiple ground-truth SELE profiles, each with its G and ELE, for scoring a method across shapes rather than at one operating point (see its README) |
| Paper figures | `Data/Tamir_paper_SELE_figs/` | Source `.fig` files the test set and its optical constants come from (see its README) |
| Optical inputs | `Data/z.csv`, `wavelength_nm.csv` | z-mesh and the measurement wavelengths. k(λ) does **not** live here — it comes from the paper figures via `src/optical_constants.py` |
| Precomputed G matrices | `Data/score_model/G_score_model*.csv` | One per score-model mesh resolution (target_length) |
| Score model checkpoints | `Data/score_model/models/` | `.pt` files, one per trained score net (see checkpoint format below) |
| Score model training datasets | `Data/score_model/datasets/` | Synthetic SELE curve sets used to train checkpoints |

`Data/score_model/models/` and `Data/score_model/datasets/` are gitignored and tracked via DVC
(`.dvc` files) rather than committed directly — pull them with DVC, not git, if missing.

### Ground-truth test set

Built by `src/test_set/` from the paper's `.fig` files; read it through
`src/test_set/loader.py`, never by parsing `index.csv` positionally. Durable points:

- **Every curve is stored on two meshes.** Its own (from the source figure, spanning the full
  wafer) and the linear solver mesh. Each has its own G and ELE, and the two ELEs differ by
  the discretization gap between them — which of the two is "the observation" is a real
  choice, not an implementation detail. `Data/test_set/README.md` quantifies it.
- **A G is data only when storing it beats recomputing it.** It is a pure function of the
  optical constants and the mesh, so an operator too large to be worth committing is left out
  of the set and rebuilt on load. Go through `loader.load_native_G` rather than reaching for
  a file, and never commit a large derived matrix when its inputs are already in the repo.
- **Resample across meshes by depth, never by index.** The profiles span the full 350 µm
  wafer while the solver mesh spans `W`; index-based resampling silently compresses a profile
  onto the wrong depths.
- **Truncating at `W` is free for the forward model but not for SELE-domain metrics.**
  Incident light is fully absorbed within a few µm, so the deep tail contributes nothing to
  the ELE — yet a large fraction of SELE *area* lives out there. An error metric integrated
  over depth and one computed from the residual will disagree for that reason alone.
- **A coarse solver mesh cannot represent the surface value.** Its first element centre sits
  deeper than the figures' first sample, so `sele[0]` there is interpolated, not SELE(z=0) —
  relevant because SRV is read off exactly that point.
- **The set and the solver share one forward model.** Both build G from
  `src/optical_constants.py`, so the test set's `solver_mesh/G_500.csv` is bit-identical to
  what `calc_mesh_and_G` produces. Keep it that way — a benchmark whose G differs from the
  solver's measures discretization error, not reconstruction error.
- **Nothing scores a solver against the set yet, and it holds no noise.** Both are deliberate,
  not gaps to fill unprompted. `results/` predates the optics unification and is not
  comparable to fresh output.

---

## Score Model -- Deep Dive

### Architecture

**ScoreNetwork** (`src/regularization/score_model/model_definition.py`):
- Input: `[S_normalized, t]` where t is diffusion time parameter
- **d32**: Simple 6-layer MLP, Softplus activations, no residual connections
- **d500**: ResNet with skip connections, LayerNorm, sinusoidal time embedding
- Output: score vector (same dimension as S), representing `grad(log p(S|t))`

**Checkpoint format** (`.pt` files):
```python
{
    'config': {'target_length': 32/500, 'hidden_dims': ..., 'use_residual': bool, ...},
    'model_state_dict': {...},
    'data_min': float,
    'data_max': float,
}
```

### Solver Algorithm

`solve_gradient_descent()` in `score_model_grad.py` runs Nesterov Accelerated Gradient
descent in normalized space (`S_norm ∈ [-1, 1]^M`). At each step it evaluates gradients at
the NAG lookahead point, combining a data-fidelity gradient (`2*G_norm.T @ (G_norm@S_phys -
B_norm)`, chain-ruled into normalized space) with an adaptively-weighted score-network
prediction (see Adaptive Weighting below) into one update, applied via momentum. The
diffusion time fed to the score network and the LR/REG_WEIGHT magnitudes are annealed over
the run (schedules are implementation details — see the function for the current form).
Optimization can start from random noise or from an optional warm start (e.g. a TV
solution). Stopping is early-exit on data-residual convergence or divergence, capped by
`MAX_STEPS`. Read the function directly for exact step order and constants — it changes
as the solver is tuned.

### Normalization

The solver works in normalized space where S_norm ∈ [-1, 1]:
- `norm_scale = 2.0 / (d_max - d_min)` (from checkpoint training data range)
- `S_physical = (S_norm + 1) / norm_scale + d_min`
- G and B are also normalized by `1/mean(|G|)` for numerical stability

### Adaptive Weighting

The score network outputs O(1) vectors (normalized data), while the data gradient magnitude depends on G's scale (can be 1e-10 to 1e+5). The adaptive factor ensures REG_WEIGHT controls the *ratio* between data fidelity and prior trust, regardless of absolute magnitudes.

### Hyperparameters

Every field of `ModelScoreGradConfig` (REG_WEIGHT, MOMENTUM, LR_MAX/MIN, MAX_STEPS, T0,
output_mesh_resolution, early-stopping and warm-start controls, ...) is documented inline
with its meaning and tuning direction in `src/types/config.py`. Current tuned values live in
`SCORE_MODEL_PRESETS["d32"]` and `SCORE_MODEL_PRESETS["d500"]` in that same file — read them
there rather than here, since they change as the models are retrained/tuned.

### Training Data Generation

Training SELE curves are generated in MATLAB (`MATLAB SELE Simulation/create_training_set.m`)
rather than in Python, and loaded from a `.mat` file by the training script in
`src/regularization/score_model/standalones/model_training/` (downsampled to the checkpoint's
`target_length`). The model learns the distribution of plausible SELE shapes from this dataset
to serve as a prior during reconstruction. Exact dataset size/path are tuned per training run —
check the training script's config for current values.

---

## Plotting Conventions

All plots (`src/plotting.py` and elsewhere) must be colorblind-accessible: use a colorblind-safe
categorical palette (e.g. Okabe-Ito) instead of default/arbitrary colors for distinct series, and
perceptually-uniform colorblind-safe sequential/diverging colormaps (e.g. `viridis`) for continuous data.
Don't rely on hue alone to distinguish series if avoidable — vary marker/line style too.

---

## Reference Papers

In `Papers/` directory:
- **Main paper**: "Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency" (Yeshurun, Fiegenbaum-Raz, Segev, ACS Appl. Energy Mater. 2024)
- **Supporting Information**: Derivations for PL calibration, optical constants, regularization method (L-curve), photon recycling model, finite element simulation details

Key equations: (2) forward model, (4) ELE from SELE, (5) simulated SELE via perturbation, (8) Tikhonov minimization, (11) photovoltage buildup from SELE.
