# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start for AI

**Read `AI/CONTEXT.md` first** -- it contains full domain knowledge (physics, math, architecture, hyperparameters) distilled from the papers and codebase. This avoids re-reading the papers every session.

## Project Overview

This is a physics-informed inverse problem solver for reconstructing **Spatial External Luminescence Efficiency (SELE)** profiles from External Luminescence Efficiency (ELE) measurements. It combines Tikhonov regularization with a score-based diffusion model prior.

## Commands

### Run the main pipeline
```bash
python -m src.main
```

### Run standalone scripts (from repo root)
```bash
python src/regularization/score_model/standalones/sele-score-model-training-script.py
python src/regularization/score_model/standalones/hyperparameter_playground.py
python src/regularization/score_model/standalones/test-score-models.py
python src/regularization/score_model/standalones/tune_hyperparameters.py
```

### Install dependencies
```bash
uv sync
```

Key packages: `numpy`, `scipy`, `matplotlib`, `torch`, `cvxpy`, `mplcursors`, `seaborn`, `pandas`

## Architecture

### Entry point & flow
`src/main.py` → `src/pipeline.py:run_regularization()` — orchestrates data loading, G-matrix computation, κ sweeping, knee detection, and result export.

### Three regularization modes (set via `CONFIG.regularization_method`)
| Mode | Solver | Key file |
|------|--------|---------|
| `NON_UNIFORM_MESH` | Tikhonov with adaptive mesh | `tikhonov_non_uniform.py` |
| `TOTAL_VARIATION` | Two-parameter Tikhonov (CVXPY) | `tikhonov_total_variation.py` |
| `MODEL_SCORE_GRAD` | Nesterov gradient + score prior | `score_model_grad.py` |

### Key modules
- **`src/types/config.py`** — Global `CONFIG` object (single source of truth for all hyperparameters). Imported via `src/__init__.py` to avoid circular imports.
- **`src/mesh.py`** — Builds optical generation matrix G using Beer-Lambert law: α(λ) = 4πk(λ)/λ [cm⁻¹]. Supports uniform and non-uniform (linear+exponential) meshes.
- **`src/operators.py`** — Builds regularization matrices L (L0=identity, L1=first-diff, L2=second-diff).
- **`src/regularization/score_model/model_definition.py`** — `ScoreNetwork`: 6-layer PyTorch MLP with Softplus activations. Supports d32 and d500 SELE dimensions.
- **`src/regularization/score_model/score_model_grad.py`** — Nesterov Accelerated Gradient solver combining data fidelity with score-based prior. Uses cosine annealing LR and adaptive weighting between gradient and score magnitudes.

### Data files (`Data/` directory)
- `z.csv`, `k.csv`, `n_k_wavelength_nm.csv`, `wavelength_nm.csv` — optical inputs
- `ELE_sim.csv` — simulated ELE measurement (the "observation")
- `SELE_ground_truth.csv`, `z_mesh.csv` — ground truth for comparison
- `sele_score_net_d32.pt`, `sele_score_net_d500.pt` — pre-trained score model checkpoints (PyTorch `.pt` with config + data stats)

### Pipeline internals
1. Load ELE data and optical parameters (k, λ, z)
2. Build G matrix on chosen mesh (shape: L wavelengths × M-1 elements)
3. Whiten by median row norm (stabilizes ill-conditioned system)
4. Sweep κ values → solve `min ||GS - B||² + κ²||LS||²` for each
5. Knee detection: normalized Euclidean distance to origin in log-log (residual vs seminorm)
6. Confidence window around κ_knee → extract S_mean, S_std
7. Save CSV results + generate plots + run report

## Reference Papers
The `Papers/` directory contains the foundational paper and its supporting information. Read them when working on physics or math details (Beer-Lambert law, regularization theory, score model formulation, SELE/ELE definitions).
- `Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency.pdf` — main paper
- `SELE Mapping Supporting Information.pdf` — derivations and supplementary material

### Score model specifics
- Checkpoint format: `{model_state_dict, config, data_min, data_max}` — min/max used for normalization
- `T0` parameter controls the noise level assumed during score evaluation
- Training script generates 1000 synthetic SELE curves via `sele_score_model_curve.csv` template
