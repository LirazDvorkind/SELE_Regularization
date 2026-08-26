# Plan — replace the MLP score network with a 1D dilated convolutional one

Rationale, measurements and per-choice justification live in
[`score-network-conv1d-explanation.md`](score-network-conv1d-explanation.md). This file is the
executable plan.

---

## Executive summary

The d500 score model generates curves that are ~3000× rougher than real ones. Measurement
rules out the sampler and rules out undertraining: the network matches an exact Tweedie
denoiser's RMSE at every noise level, so it learned the distribution — its error is simply
small-amplitude and almost entirely high-frequency.

The cause is architectural. `ScoreNetwork` is an MLP; jointly permuting the depth axis and the
weights leaves it functionally identical, so it has no structural notion that depth 200
neighbours depth 201.

**Fix:** add a 1D dilated residual convolutional network alongside the existing one, and
retrain d500. Success is smooth generation with *no* smoothing, filtering or projection at
sampling time.

**Effort:** four file changes, then a Colab retrain. The retrain is not the end — solver
hyperparameters are tuned against the old network's gradient magnitudes and will need redoing.

---

## Scope

Architecture only. The VP-SDE schedule, `var_t` loss weighting, global min–max normalization
and raw-score output convention all stay byte-identical, so any change in sample quality is
attributable to the architecture alone.

Target `target_length = 500` first; d32 is the same code with a shorter dilation list.

**Deliberately not in this change** — each is a real finding and a separate experiment:
per-curve amplitude normalization; ε-prediction instead of raw score; the NAG solver's missing
VP mean-scale factor; removal of the adaptive norm rescaling.

---

## The changes

### 1. `src/regularization/score_model/model_definition.py` — the substance

Add `Conv1dScoreNetwork` **alongside** the existing `ScoreNetwork`, which stays untouched so
all existing checkpoints keep loading.

**Signature.** Identical to `ScoreNetwork`: `forward(x, t)` with `x` shape `(B, L)`, `t` shape
`(B, 1)`, returning `(B, L)`. Internally lift to `(B, 1, L)` channels-first and squeeze on the
way out. Nothing downstream sees a difference.

**Time conditioning.** Reuse the existing `SinusoidalTimeEmbedding` in this file, feed it
through a 2-layer MLP, and inject the result into every residual block as a per-channel
scale-and-shift (FiLM). Do *not* concatenate `t` as an extra input position.

**Body.** Blocks of `Conv1d → GroupNorm → SiLU → FiLM(t) → Conv1d → GroupNorm → SiLU` plus
identity skip, dilation doubling across blocks: `(1, 2, 4, 8, 16, 32, 64)` at kernel size 5,
channels 128. Receptive field `1 + 4·(1+2+…+64) = 509` ≥ 500. At `L = 32` cap dilations at
`(1, 2, 4, 8)` (RF 61).

**Padding must be `replicate`, not zeros** — `dilation·(k−1)/2` per side to hold the length
fixed. Zero padding would teach the network that SELE → 0 at `z = 0`, which is physically
wrong and is exactly the quantity SRV is read from.

**Two details to get right up front:**

- Zero-initialize the final output conv, so the network starts predicting zero score.
- If trained samples come out with correct shapes but collapsed amplitude spread, **drop the
  GroupNorm layers entirely** — it normalizes across the length axis and strips per-sample
  amplitude, which is the dominant variance direction in this dataset. The network is small
  enough not to need norm.

**Also add a `build_score_network(model_config: dict) -> nn.Module` factory** in this file. It
dispatches on `model_config.get('arch', 'mlp')` — an absent key means the old MLP, so every
existing checkpoint loads unchanged. This exists because `ScoreNetwork(...)` is currently
constructed with the same argument block in **five** separate places:

| file | site |
|---|---|
| `score_model_grad.py` | `load_score_model` |
| `score_model_grad.py` | fallback inside `solve_gradient_descent` |
| `standalones/model_training/sele-score-model-training-script.py` | `train()` |
| `standalones/model_training/test-score-models.py` | model loading |
| `standalones/sele_w_score_optimization_example.py` | module level |

Without a factory, every future architecture means touching all five.

### 2. `src/regularization/score_model/score_model_grad.py` — plumbing

Replace both `ScoreNetwork(...)` blocks with factory calls. These two cover the generation
script and the main pipeline, which is everything needed for this round.

`test-score-models.py` and `sele_w_score_optimization_example.py` carry the same block and
will not load a conv checkpoint until they get the same one-line swap. Neither is needed for
generation — leave them until you reach them.

### 3. `standalones/model_training/sele-score-model-training-script.py` — the file that travels to Colab

Add architecture fields to `TrainingConfig`: `arch` (default `'conv1d'`), `channels`,
`n_blocks`, `kernel_size`, `dilations`. Construct via the factory.

`asdict(config)` already goes into the checkpoint's `config` dict, so `arch` propagates to the
saved `.pt` automatically and the factory reads it back on load. **No checkpoint-format change
is needed.**

Everything else in this file — `compute_diffusion_params`, `compute_loss`, the `var_t`
weighting, `load_and_preprocess_data`, the training loop — stays exactly as is.

### 4. `standalones/new_model_toolkit/test_diffusion_generation.py` — the measurement instrument

**Fix this before retraining**, or there is no way to tell whether the new architecture
worked. Two outright bugs plus a metric:

- **Delete the in-loop `x.clamp(-1.0, 1.0)`** in the reverse-SDE loop. It clamps the
  *noise-space* state; at `t ≈ 1` the state is ~N(0, I), so roughly a third of all entries are
  clipped on every step and the reverse process is measurably distorted.
- **Replace the linear `time_grid`** with a power-law grid concentrated near `t → 0`. All shape
  information in this schedule lives at `σ_t < 0.2`, i.e. `t < ~0.06`; the linear grid spends
  ~94% of its steps where the state is indistinguishable from noise.
- **Print a roughness readout** — `rms(diff(x, n=2)) / std(x)` per sample — next to the same
  statistic over real curves from the training CSV, so the comparison is a number rather than
  an eyeball judgement.

No smoothing, filtering or projection is added here. The point of the exercise is that
generation should be clean without one.

---

## Verification

1. **Baseline the instrument first.** Run the fixed generation script against the *current*
   d500 checkpoint, before retraining. Expect roughness ~0.9 against ~0.0003 for real curves.
   This confirms the two bug fixes were not themselves the cause, and banks the number the
   retrain has to beat.
2. **Backward compatibility.** Load all three checkpoints in `MODELS` and confirm they still
   run — `arch` defaults to `'mlp'`, and Alon's bare-`Sequential` path is separate and
   untouched.
3. **Acceptance test, after retraining.** Same script, new checkpoint, no smoothing. Success is
   roughness within ~10× of real curves, i.e. **~3e-3 or below**.
4. **Distribution check, not just smoothness.** Histogram peak height and peak position over
   ~50 generated samples against `sele_simulated_1000_curves_500_long.csv`. A model can be
   smooth and still collapsed onto a single shape; this is what catches that.

---

## What to do next

**Code changes, then a retrain — and the retrain is not the whole job.**

1. Apply the four file changes above.
2. Run Verification step 1 to bank the baseline number.
3. **Port to Colab.** `sele_score_model_training.ipynb` loads `model_definition.py` from the
   Drive folder, so two things travel:
   - the updated `model_definition.py` (new class + factory) → replace it in Drive;
   - the new `TrainingConfig` architecture fields → set in the notebook's config cell.

   The notebook's data loading, loss and training loop need no edits. Train d500 on the
   100k-curve `.mat` corpus as usual.
4. Bring the `.pt` back into `Data/score_model/models/`, `dvc add` / `dvc push` per the root
   README, then run Verification steps 3–4.
5. If the acceptance test passes, point the `d500` preset's `model_path` in
   `src/types/config.py` at the new checkpoint.

**Two things to expect:**

- `LR_MAX` / `LR_MIN` / `REG_WEIGHT` in the `d500` preset are tuned against the old network's
  gradient magnitudes and will need re-tuning via `new_model_toolkit`. That is solver work,
  downstream of this.
- **This fixes generation; it does not by itself fix the NAG solver collapsing to the mean
  curve.** Three separate causes, listed in §7 of the explanation file. Those are the next
  round.
