# Project Context for AI Assistants

Read this file first when starting a new conversation about this project. It captures domain knowledge, physics, architecture, and debugging context that would otherwise require reading the papers and entire codebase from scratch.

---

## 1. What This Project Does

This project solves an **inverse problem** in semiconductor photovoltaics. Given wavelength-dependent External Luminescence Efficiency (ELE) measurements of a GaAs wafer, it reconstructs the **Spatial External Luminescence Efficiency (SELE)** -- a depth profile showing how much each layer of the device contributes to photoluminescence.

### The Forward Model

```
eta_ext(lambda_in) = (1 / phi_abs(lambda_in)) * integral_0^d  DeltaG(lambda_in, z) * S(z) dz
```

- `eta_ext(lambda_in)`: measured ELE at incident wavelength lambda_in (a scalar per wavelength)
- `DeltaG(lambda_in, z)`: optical generation profile from Beer-Lambert law: `alpha = 4*pi*k/lambda`, `G[lambda,i] = exp(-alpha*z_i) - exp(-alpha*z_{i+1})`
- `S(z)`: the SELE profile we want to extract (the unknown)
- `phi_abs`: absorbed photon flux

Discretized: `eta_ext = (1/phi_abs) * G @ S`, where G is (L wavelengths x M depth elements), S is (M,).

### Why It's Hard

The system `G @ S = B` is ill-posed (ill-conditioned). Small noise in ELE measurements causes wild oscillations in naive least-squares solutions. Regularization is required.

---

## 2. Physical Intuition for the SELE Profile

The expected SELE shape for a p-type GaAs wafer (from the paper, Figure 2b):

- **Near surface (z ~ 0)**: SELE is LOW (~0.05-0.1) because surface recombination through defects suppresses radiative recombination.
- **Rises to a peak** (~0.3-0.45) at roughly 0.5-1 um from the surface. The peak position correlates with the minority carrier diffusion length.
- **Slowly decays** deeper into the material because radiative recombination photons generated deep inside are less likely to escape (self-absorption). However, sub-bandgap photons are hardly reabsorbed, so the decay is more gradual than pure absorption would suggest.
- **Nonzero throughout** the entire 350 um wafer (confirmed by transmission-mode PL measurements).

### Key Physical Parameters Affecting SELE Shape

| Parameter | Effect on SELE |
|-----------|---------------|
| Surface recombination velocity (SRV) | Controls the dip at z=0. Higher SRV = lower surface SELE. Sensitive above 10^4 cm/s. |
| SRH lifetime (tau_SRH) | Controls bulk SELE magnitude. Longer lifetime = higher bulk SELE. |
| Diffusion length | Peak position ~ diffusion length when surface recomb dominates. |
| Photon recycling | Increases quasi-Fermi level separation deep in material, making SELE nonzero far from surface. |

---

## 3. Three Regularization Methods

Set via `CONFIG.regularization_method` in `src/types/config.py`:

### 3a. NON_UNIFORM_MESH (Tikhonov)
- Classical Tikhonov: `min ||G@S - B||^2 + kappa^2 * ||L@S||^2`
- L = second-derivative operator (constrains curvature)
- Sweeps kappa values, finds knee of L-curve, averages solutions in confidence window
- Uses non-uniform mesh: fine linear spacing near surface, logarithmic deeper
- This is the method from the original paper

### 3b. TOTAL_VARIATION
- Two-parameter Tikhonov solved with CVXPY
- Allows sharper features than standard Tikhonov

### 3c. MODEL_SCORE_GRAD (Score Model -- the focus of current work)
- Replaces the explicit L operator with a learned prior from a score-based diffusion model
- Uses Nesterov Accelerated Gradient (NAG) descent
- The score network learns `grad(log p(S))` from synthetic SELE training data
- See Section 4 for full details

---

## 4. Score Model Method -- Deep Dive

### 4a. Architecture

**ScoreNetwork** (`src/regularization/score_model/model_definition.py`):
- Input: `[S_normalized, t]` where t is diffusion time parameter
- Two variants:
  - **d32**: Simple 6-layer MLP, Softplus activations, no residual connections
  - **d500**: ResNet with skip connections, LayerNorm, sinusoidal time embedding
- Output: score vector (same dimension as S), representing `grad(log p(S|t))`

**Checkpoint format** (`.pt` files):
```python
{
    'config': {'target_length': 32/500, 'hidden_dims': ..., 'use_residual': bool, ...},
    'model_state_dict': {...},
    'data_min': float,  # min of training SELE data (for normalization)
    'data_max': float,  # max of training SELE data (for normalization)
}
```

### 4b. The Solver Algorithm

`solve_gradient_descent()` in `src/regularization/score_model/score_model_grad.py`:

```
Initialize S_norm randomly in [-1, 1]^M
velocity = 0

For step i = 0 to MAX_STEPS:
    1. Nesterov lookahead:  S_look = S_norm + MOMENTUM * velocity
    2. Denormalize:         S_phys = (S_look + 1) / norm_scale + d_min
    3. Data gradient:       grad_data = 2 * G_norm.T @ (G_norm @ S_phys - B_norm)
       Chain rule to normalized space: grad_data_norm = grad_data / norm_scale
    4. Score prediction:    score = ScoreNetwork(S_look, T0)
    5. Adaptive weighting:  factor = (||grad_data_norm|| / ||score||) * REG_WEIGHT
       score_weighted = score * factor
    6. Combined update:     total = grad_data_norm - score_weighted
    7. Cosine LR schedule:  lr = LR_MIN + 0.5*(LR_MAX - LR_MIN)*(1 + cos(i/MAX_STEPS * pi))
    8. Momentum update:     velocity = MOMENTUM * velocity - lr * total
                            S_norm += velocity
    9. Early stopping if MSE vs ground truth > 1 or stagnates
```

### 4c. Normalization Details

The solver works in a normalized space where S_norm is in [-1, 1]:
- `norm_scale = 2.0 / (d_max - d_min)` (from checkpoint training data range)
- `S_physical = (S_norm + 1) / norm_scale + d_min`
- G and B are also normalized by `1/mean(|G|)` for numerical stability

### 4d. Adaptive Weighting -- Why It Matters

The score network outputs O(1) vectors (trained on normalized data), while the data gradient magnitude depends on G's scale (can be 1e-10 to 1e+5). Without adaptive weighting, one term would dominate. The formula:

```
adaptive_factor = (||grad_data|| / ||score|| + eps) * REG_WEIGHT
```

ensures REG_WEIGHT controls the *ratio* between data fidelity and prior trust, regardless of absolute magnitudes.

### 4e. Hyperparameters

| Param | d32 preset | d500 preset | What it controls |
|-------|-----------|------------|-----------------|
| REG_WEIGHT | 1.0 | 200 | Prior vs data trust ratio. Higher = trust prior more. |
| MOMENTUM | 0.85 | 0.9 | Velocity inertia. Higher = smoother trajectory but may overshoot. |
| LR_MAX | 1e-2 | 1e-6 | Peak learning rate (start of cosine schedule). |
| LR_MIN | 1e-5 | 1e-8 | Final learning rate (end of cosine schedule). |
| MAX_STEPS | 5000 | 5000 | Hard iteration cap. |
| T0 | 0.1 | 0.05 | Diffusion time for score network. Larger = blurrier/smoother prior. |
| output_mesh_resolution | 500 | 10000 | Upsampled output points. |

Presets are defined in `src/types/config.py` as `SCORE_MODEL_PRESETS["d32"]` and `SCORE_MODEL_PRESETS["d500"]`.

---

## 5. Pipeline Flow

### Entry Point
`src/main.py` -> `src/pipeline.py:run_regularization()`

### MODEL_SCORE_GRAD Branch (pipeline.py ~lines 162-248)
1. Load ELE data (`ELE_sim.csv`), optical params (k, wavelengths, z)
2. Load score model checkpoint -> get `target_length` (32 or 500)
3. Build G matrix on linear mesh with `target_length` elements
4. Normalize: multiply G, B by `photon_flux * e_charge` (unit factor)
5. Call `solve_gradient_descent(G, B, hyperparams, S_gt)`
6. Upsample SELE from M points to `output_mesh_resolution` via `expand_sele()`
7. Reconstruct: `eta_fit = G_longer @ S_rec / unit_factor`
8. Save CSVs + generate plots

### Other Methods
- NON_UNIFORM_MESH: kappa sweep -> L-curve knee -> confidence window -> mean S
- TOTAL_VARIATION: CVXPY solver with two regularization params

---

## 6. Key Files

| File | Role |
|------|------|
| `src/main.py` | Entry point |
| `src/pipeline.py` | Orchestrates all regularization methods |
| `src/types/config.py` | Global CONFIG, ModelScoreGradConfig, presets |
| `src/__init__.py` | CONFIG import (avoid circular imports) |
| `src/mesh.py` | Builds G matrix (Beer-Lambert), linear and non-uniform meshes |
| `src/operators.py` | L0/L1/L2 regularization matrices |
| `src/regularization/score_model/score_model_grad.py` | NAG solver with score prior |
| `src/regularization/score_model/model_definition.py` | ScoreNetwork PyTorch model |
| `src/regularization/score_model/standalones/hyperparameter_playground.py` | Test single hyperparameter config on one curve |
| `src/regularization/score_model/standalones/tune_hyperparameters.py` | Grid search over hyperparameter space |
| `src/regularization/score_model/standalones/model_training/` | Training scripts and Colab notebook |

### Data Files
| File | Contents |
|------|----------|
| `Data/ELE_sim.csv` | Simulated ELE measurement (the "observation") |
| `Data/SELE_ground_truth.csv` | Ground truth SELE for comparison |
| `Data/z.csv`, `k.csv`, `wavelength_nm.csv` | Optical inputs |
| `Data/score_model/G_score_model.csv` | Precomputed G matrix (28 x 32) |
| `Data/score_model/G_score_model_500.csv` | Precomputed G matrix (28 x 500) |
| `Data/score_model/sele_dataset.csv` | Training SELE profiles (32-point) |
| `Data/score_model/sele_dataset_500.csv` | Training SELE profiles (500-point) |
| `Data/score_model/models/sele_score_net_d32.pt` | Trained d32 checkpoint |
| `Data/score_model/models/sele_score_net_d500.pt` | Trained d500 checkpoint |

---

## 7. Training Data Generation

The score model is trained on synthetic SELE curves. The template curve shape is stored in `Data/score_model/sele_score_model_curve.csv`. The training script generates ~1000 (or 100,000 for d500) variations by perturbing this template. The model learns the distribution of plausible SELE shapes so it can serve as a prior during reconstruction.

---

## 8. Standalone Scripts

Located in `src/regularization/score_model/standalones/`. Run as modules from project root:

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground
python -m src.regularization.score_model.standalones.tune_hyperparameters
```

- **hyperparameter_playground.py**: Quick test of one config on a random synthetic curve. Shows MSE_ELE (data fit) and MSE_SELE (ground truth error). Good for interactive debugging.
- **tune_hyperparameters.py**: Automated grid search with multiprocessing. Tests combinations of REG_WEIGHT, LR_MAX, MOMENTUM, T0 across many curves.
- **test-score-models.py**: Visualizes score gradients from trained checkpoints.

---

## 9. Reference Papers

In `Papers/` directory:
- **Main paper**: "Mapping Losses through Empirical Extraction of the Spatial External Luminescence Efficiency" (Yeshurun, Fiegenbaum-Raz, Segev, ACS Appl. Energy Mater. 2024)
- **Supporting Information**: Derivations for PL calibration, optical constants, regularization method (L-curve), photon recycling model, finite element simulation details

Key equations: (2) forward model, (4) ELE from SELE, (5) simulated SELE via perturbation, (8) Tikhonov minimization, (11) photovoltage buildup from SELE.
