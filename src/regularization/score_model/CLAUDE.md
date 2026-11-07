# Score model (MODEL_SCORE_GRAD)

## Solver Algorithm

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

## Normalization

The solver works in normalized space where S_norm ∈ [-1, 1]:
- `norm_scale = 2.0 / (d_max - d_min)` (from checkpoint training data range)
- `S_physical = (S_norm + 1) / norm_scale + d_min`
- G and B are also normalized by `1/mean(|G|)` for numerical stability

## Adaptive Weighting

The score network outputs O(1) vectors (normalized data), while the data gradient magnitude depends on G's scale (can be 1e-10 to 1e+5). The adaptive factor ensures REG_WEIGHT controls the *ratio* between data fidelity and prior trust, regardless of absolute magnitudes.

## Hyperparameters

Every field of `ModelScoreGradConfig` (REG_WEIGHT, MOMENTUM, LR_MAX/MIN, MAX_STEPS, T0,
output_mesh_resolution, early-stopping and warm-start controls, ...) is documented inline
with its meaning and tuning direction in `src/types/config.py`. Current tuned values live in
`SCORE_MODEL_PRESETS["d32"]` and `SCORE_MODEL_PRESETS["d500"]` in that same file — read them
there rather than here, since they change as the models are retrained/tuned.

## Training Data Generation

Training SELE curves are generated in MATLAB (`MATLAB SELE Simulation/create_training_set.m`)
rather than in Python, and loaded from a `.mat` file by the training script in
`standalones/model_training/` (downsampled to the checkpoint's `target_length`). The model
learns the distribution of plausible SELE shapes from this dataset to serve as a prior during
reconstruction. Exact dataset size/path are tuned per training run — check the training
script's config for current values.
