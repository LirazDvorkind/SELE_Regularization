# Score model (`MODEL_SCORE_GRAD`)

Replaces the Tikhonov smoothness operator with a learned prior: a score network trained on
synthetic SELE curves learns `grad(log p(S))`, and the solver balances it against data fidelity.

## Pipeline flow

The `MODEL_SCORE_GRAD` branch of `src/pipeline.py:run_regularization()` loads a checkpoint,
builds G on a linear mesh with the checkpoint's `target_length` elements, scales G and B by a
unit factor, calls `solve_gradient_descent()` (optionally warm-started from a TV solve; see
`warm_start_with_tv` in `ModelScoreGradConfig`), then upsamples the result to
`output_mesh_resolution` via `expand_sele()` and reconstructs the ELE fit on the finer mesh.

## Architecture

**ScoreNetwork** (`model_definition.py`):
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

## Solver algorithm

`solve_gradient_descent()` in `score_model_grad.py` runs Nesterov Accelerated Gradient
descent in normalized space (`S_norm ∈ [-1, 1]^M`). At each step it evaluates gradients at
the NAG lookahead point, combining a data-fidelity gradient (`2*G_norm.T @ (G_norm@S_phys -
B_norm)`, chain-ruled into normalized space) with an adaptively-weighted score-network
prediction (see Adaptive weighting below) into one update, applied via momentum. The
diffusion time fed to the score network and the LR/REG_WEIGHT magnitudes are annealed over
the run (schedules are implementation details — see the function for the current form).
Optimization can start from random noise or from an optional warm start (e.g. a TV
solution). Stopping is early-exit on data-residual convergence or divergence, capped by
`MAX_STEPS`. Read the function directly for exact step order and constants — it changes
as the solver is tuned.

## Normalization

The solver works in normalized space where S_norm ∈ [-1, 1]:
- `norm_scale = 2.0 / (d_max - d_min)` (from checkpoint training data range)
- `S_physical = (S_norm + 1) / norm_scale + d_min`
- G and B are also normalized by `1/mean(|G|)` for numerical stability

## Adaptive weighting

The score network outputs O(1) vectors (normalized data), while the data gradient magnitude
depends on G's scale (can be 1e-10 to 1e+5). The adaptive factor ensures REG_WEIGHT controls
the *ratio* between data fidelity and prior trust, regardless of absolute magnitudes.

## Hyperparameters

Every field of `ModelScoreGradConfig` (REG_WEIGHT, MOMENTUM, LR_MAX/MIN, MAX_STEPS, T0,
output_mesh_resolution, early-stopping and warm-start controls, ...) is documented inline
with its meaning and tuning direction in `src/types/config.py`. Current tuned values live in
`SCORE_MODEL_PRESETS["d32"]` and `SCORE_MODEL_PRESETS["d500"]` in that same file — read them
there rather than here, since they change as the models are retrained/tuned.

## Training data

Training SELE curves are generated in MATLAB (`MATLAB SELE Simulation/create_training_set.m`)
rather than in Python, and loaded from a `.mat` file by the training script in
`standalones/model_training/` (downsampled to the checkpoint's `target_length`). Exact
dataset size/path are tuned per training run — check the training script's config for
current values.

## Standalones

Run from the repo root, as modules (`python -m src.regularization.score_model.standalones.<name>`)
or as scripts:

- **`hyperparameter_playground.py`**: one config on a random synthetic curve; reports MSE_ELE
  (data fit) and MSE_SELE (ground-truth error). For interactive debugging.
- **`tune_hyperparameters.py`**: multiprocessing grid search over the hyperparameter space.
- **`model_training/sele-score-model-training-script.py`**: trains a checkpoint from a
  synthetic SELE dataset.
- **`model_training/test-score-models.py`**: visualizes score gradients from trained checkpoints.
