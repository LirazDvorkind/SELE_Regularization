# Why reverse diffusion returned noise — investigation record

Follow-up to `score-network-conv1d-plan.md` / `score-network-conv1d-explanation.md`.
Those two diagnosed the **MLP** score net and proposed the conv1d rewrite. This
document covers what happened after that rewrite was trained.

---

## 1. The task as given

> We updated `model_definition.py` to implement the conv1d plan. I trained it on Colab
> via `sele_score_model_training.ipynb` using the dataset from `create_training_set.m`,
> placed the checkpoint in `Data/score_model/models/`, then ran
> `test_diffusion_generation.py` — and got noise.
>
> **(a)** Without running or modifying code, understand the context.
> **(b)** Suggest ways to debug. Idea offered: feed a dataset curve to the model, see
> what it outputs; then add noise to that curve and look again.

Later superseded by: *"run your debug plan and come back with answers why the
test_diffusion code returns noisy SELE curves instead of clean curves indistinguishable
from the dataset curves"*, plus *"you may search online to verify hypotheses."*

---

## 2. Debug directions proposed before running anything

Ordered cheapest-first, each designed to *eliminate a layer* rather than confirm a hunch.

| # | Probe | Layer it clears |
|---|-------|-----------------|
| 1 | Verify the checkpoint loads strictly and `head_conv` moved off its zero-init | Serialization / arch mismatch |
| 2 | Confirm the training dataset is current, not a stale artifact | Data provenance |
| 3 | Read the training loss history for a real convergence signal | Undertraining |
| 4 | **The user's own idea** — denoise a known dataset curve at several noise levels, compare `x0_hat` to the original | Is the network a working denoiser at all? |
| 5 | Compare `‖score‖` against the analytically-required magnitude across `t` | Score calibration / saturation |
| 6 | Track mean, std, roughness along the reverse trajectory vs the true marginals | Sampler correctness |
| 7 | Compare the network against an **exact empirical Tweedie denoiser** built from held-out curves | Separates "model is wrong" from "target is unrepresentable" |
| 8 | Compute the roughness a **perfect** model would produce under the current readout | Instrument bias |
| 9 | Frequency-decompose the `x0_hat` error against the signal spectrum | Where the error actually lives |

Probes 7–9 turned out to be decisive. Probe 4 — the user's suggestion — is what made 7
possible: it establishes the *denoising* view of the model, which is the frame the whole
diagnosis ends up living in.

---

## 3. What was checked, and what it ruled out

All diagnostics ran in the session scratchpad. No repo file was modified during the
investigation; the only code change is the fix in §6.

### 3.1 Cleared — not the cause

**Checkpoint integrity.** Loads strictly after stripping the `_orig_mod.` prefix (a
`torch.compile` artifact). 1,482,369 parameters. Receptive field 509 ≥ 500, so every
output point sees the whole curve. `head_conv` weights are far from their zero
initialization — the head trained.

**Data provenance.** `create_training_set.m` still carries `rng(12)`; the only
uncommitted change is `n_samples 1e3 → 1e4` and `.csv → .mat`. The first 1000 rows of the
new `.mat` are bit-identical to the committed June CSV (max relative difference
$1.352 \times 10^{-15}$). The dataset is not stale.

**Undertraining.** 300 epochs, 4661 s (78 min). Loss 0.428 → 0.00309, minimum 0.00278 at
epoch 254. Epochs 20/60/100/200 → 0.01445 / 0.00689 / 0.00503 / 0.00381. Converged.

**Sampler math.** Reverse drift sign, `dt < 0` handling, and the `sqrt(beta*h)` diffusion
coefficient are all correct. The clamp bug and linear-grid problem noted in the conv1d
plan were already fixed in this file.

**GroupNorm amplitude collapse.** Flagged as a risk in the conv1d plan (GroupNorm
normalizes across length and could strip per-sample amplitude, the dataset's dominant
variance direction). It did **not** happen: generated median max 1.88–2.15e-3 against a
real median max of 2.204e-3.

**Trajectory statistics.** Reverse-process mean, std and roughness track the true VP-SDE
marginals down to $t \approx 0.005$.

### 3.2 Cause 1 — the readout (largest single contributor)

`reverse_diffusion_sample()` returned the raw SDE state $x$ at $t = \texttt{time\_eps}$.
That state is a sample from $p_\epsilon$, which by construction still carries
$\sigma_\epsilon$ of white noise.

Roughness a **perfect** score model would produce returning the raw state:

| $t_{\text{stop}}$ | $\sigma_t$ | roughness of $a_t x_0 + \sigma_t z$ |
|---|---|---|
| 0.02 | 0.0772 | 1.930 |
| 0.005 | 0.0274 | 1.027 |
| $10^{-4}$ | 0.0032 | **0.129** |
| real curves | — | **0.000142** |

A ~900× penalty independent of model quality. Applying Tweedie's formula
$\hat{x}_0 = (x_t + \sigma_t^2 s_\theta)/a_t$ to the *identical* trajectory dropped
roughness ~20× and produced visually clean, plausible SELE profiles.

### 3.3 Cause 2 — score saturation below $t \approx 0.005$

| $t$ | $\sigma_t$ | $\|s_\theta\|$ actual | $\|s\|$ ideal | ratio | $\epsilon$-RMSE |
|---|---|---|---|---|---|
| 0.100 | 0.322 | 69.2 | 69.5 | 0.996 | 0.064 |
| 0.020 | 0.077 | 287.2 | 289.9 | 0.991 | 0.086 |
| 0.005 | 0.027 | 762.9 | 818.1 | 0.933 | 0.126 |
| 0.003 | 0.020 | 770.4 | 1134.1 | 0.679 | 0.336 |
| $10^{-4}$ | 0.0032 | 765.0 | 7042.9 | **0.109** | 0.895 |

Well calibrated down to $t = 0.005$, then hard-capped at $\|s_\theta\| \approx 770$.
This is the textbook score blow-up that motivates early stopping.

The cap is architectural: the head is `GroupNorm → SiLU → 1x1 conv` and is
**t-independent by construction** — every FiLM sits *before* a `norm2`, and `head_norm`
renormalizes after the last block, so nothing in the output path can scale like
$1/\sigma_t$. Measured ceiling ≈ 770 (≈34 per element) against a weight-norm estimate of
$128 \times 0.128 \times 1.33 \approx 22$ per element.

> **Correction to an intermediate hypothesis of mine.** I initially claimed this cap bites
> from $t \approx 0.05$ and was the primary cause. Measurement showed calibration is
> 0.99–1.00 all the way to $t = 0.005$. The saturation is real but *secondary*.

### 3.4 Cause 3 — root cause: the raw-score output parameterization

With a score/$\epsilon$ output, the smoothness of $\hat{x}_0$ is not produced by the
architecture. It is produced by a **cancellation**: since
$\hat{x}_0 = (x_t + \sigma_t^2 s_\theta)/a_t$, the network must reproduce the
high-frequency content of its own input with gain exactly $1/\sigma_t$. Any shortfall
lands directly in the sample as white noise, amplified by $\sigma_t/a_t$.

Verified quantitatively at $t = 0.02$:

$$\underbrace{0.0357}_{\text{white part of }\epsilon\text{-error}} \times \frac{\sigma_t}{a_t} = 2.77\times10^{-3} \;\Rightarrow\; \text{predicted roughness } 0.0738$$

Measured $\hat{x}_0$ roughness: **0.0735**. The mechanism accounts for the observed
roughness essentially exactly.

**Required precision.** To reach real-curve smoothness the $\epsilon$ prediction's
high-frequency component would need accuracy $6.9\times10^{-5}$. The network achieves
0.0357 — a **520× shortfall**. Not a capacity or training-budget problem.

**The network is accurate and rough at the same time.** Against an exact empirical
Tweedie denoiser built by softmax-kernel averaging over 800 held-out curves:

| $t$ | net $\hat{x}_0$ roughness | exact $\hat{x}_0$ roughness | net rel-err | exact rel-err |
|---|---|---|---|---|
| 0.05 | 0.0981 | 0.00015 | 0.023 | 0.037 |
| 0.01 | 0.0587 | 0.00014 | **0.0078** | 0.0342 |
| 0.005 | 0.0561 | 0.00014 | **0.0061** | 0.0343 |

The net is *more accurate* than the exact denoiser at low $t$, yet 400× rougher. Same
signature the conv1d plan measured for the MLP — which is why the MLP → conv rewrite only
bought ~10×. **Architecture was never the binding constraint.**

**Where the error lives.** Spectrum at $t = 0.02$: the signal holds 89% of its energy at
$k < 5$ and only 0.29% above $k = 125$; the $\hat{x}_0$ error puts **7.4%** above
$k = 125$ — 25× the signal's share there.

**Low-passing is not a fix.** Filtering the output worsens relative error 0.0123 → 0.0328,
because real SELE curves genuinely carry ~0.9% of their energy above $k = 25$.

**Why the conv prior cannot help.** The quantity the network emits is the score, dominated
by $-z/\sigma_t$ — a white-noise field. A smoothness-biased architecture is the wrong
instrument for emitting white noise. Compounding it, `Conv1dScoreNetwork.forward` has **no
global input→output skip**: residuals live inside blocks in 128-channel space, while
`stem` and `head_conv` are plain convs, so any high-gain identity path must be learned and
preserved through 6 blocks, 12 GroupNorms and 13 SiLUs.

---

## 4. Online material consulted

- **[EDM — Karras et al. 2022](https://arxiv.org/abs/2206.00364)** — exists precisely for
  this failure. $D(x;\sigma) = c_{\text{skip}}(\sigma)x + c_{\text{out}}(\sigma)F(c_{\text{in}}(\sigma)x, c_{\text{noise}}(\sigma))$
  gives an explicit $\sigma$-dependent identity path so the network never reproduces its
  own input, with $c_{\text{out}}$ minimized specifically to *bound the amplification of
  $F$'s error into the denoiser output*. Directly targets §3.4.
- **[Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)**
  — clean data lies on a low-dimensional manifold while noise does not, so predicting noise
  "can fail catastrophically"; predict the clean signal instead.
- **[PDE-Refiner](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)**
  — same failure mode on smooth scientific data: MSE-trained models systematically neglect
  low-amplitude high-frequency components.
- **[Score blow-up / early stopping](https://arxiv.org/abs/2402.15602)** — the standard
  justification for stopping the reverse process at $t > 0$ rather than at `time_eps`.

---

## 5. Stopping-time sweep (32 samples, d500)

Run after the readout fix to pick `T_STOP` empirically rather than by argument. Reverse
process run once to $t = 0.005$, snapshotted at each candidate, Tweedie applied at each.

| $t_{\text{stop}}$ | $\sigma_t$ | raw (SDE) | denoised (SDE) | raw (ODE) | denoised (ODE) |
|---|---|---|---|---|---|
| 0.100 | 0.322 | 2.383 | 0.149 | 2.418 | 0.182 |
| 0.060 | 0.202 | 2.345 | **0.119** | 2.372 | 0.174 |
| 0.040 | 0.140 | 2.265 | 0.182 | 2.296 | 0.194 |
| 0.030 | 0.109 | 2.140 | 0.140 | 2.200 | 0.218 |
| 0.020 | 0.077 | 1.929 | 0.144 | 2.059 | 0.244 |
| 0.015 | 0.061 | 1.703 | 0.129 | 1.911 | 0.261 |
| 0.010 | 0.045 | 1.466 | 0.142 | 1.675 | 0.339 |
| 0.005 | 0.027 | 1.053 | 0.171 | 1.294 | 0.439 |
| real | — | — | **0.000142** | — | — |

Two readings:

1. The optimum is **broad and flat** (0.12–0.18 anywhere in 0.005–0.1) and the SDE beats
   the probability-flow ODE at low $t$. `T_STOP = 0.02` sits comfortably inside it and is
   where score calibration is independently verified at 0.99.
2. **The floor is ~0.13, still ~900× real.** The readout fix is worth ~14× and no more.
   That is §3.4 measured end-to-end: no choice of stopping time rescues a
   score-parameterized model here.

---

## 6. What was changed

`src/regularization/score_model/standalones/new_model_toolkit/test_diffusion_generation.py`
— see `diffusion-readout-fix-walkthrough.md` for the guided version.

- Added `_vp_coefficients(t, beta_min, beta_max)` returning $(a_t, \sigma_t)$.
- `_power_law_time_grid` now takes `(t_stop, t_start, ...)` instead of assuming `time_eps`
  at both ends.
- Added module constant `T_STOP = 0.02`; the reverse SDE stops there.
- `reverse_diffusion_sample` applies the Tweedie readout at `t_stop` and returns
  `(denoised, raw)` so both can be compared in one run.
- `run()` prints both roughness figures. Previously it compared a deliberately-noisy
  $t = \epsilon$ state against clean training data — unfair by ~900× on its own.

Result (5 samples):

| model | raw SDE state | denoised | real reference |
|---|---|---|---|
| Alon's d32 | 0.647 | 0.180 | 0.0276 |
| My d32 | 1.167 | 0.533 | 0.0276 |
| My d500 | 1.727 | **0.118** | 0.000142 |

(5-sample medians are noisy; the 32-sample d500 figure is 0.144.)

---

## 7. Where we are now

**Settled.** The model trained correctly and learned the SELE distribution — generated
shapes and amplitudes are right. The visible noise was, in order: a missing final
denoising step (~900×), integration past the score saturation point, and underneath both,
the score/$\epsilon$ output parameterization.

**Done.** The readout fix, in the repo, worth ~14×. Curves are now recognizably SELE-shaped
and smooth to the eye, though not yet dataset-indistinguishable.

**Not done — the actual fix.** Retrain with **$x_0$-prediction under EDM preconditioning**:
an explicit $c_{\text{skip}}(\sigma)x$ identity path so the network is never asked to
reproduce its own input. ~78 min on the current Colab setup. Open choice: $\sigma_{\text{data}}$
is 0.31 under the current global min–max normalization but 0.055 per-curve — per-curve
amplitude normalization would make the preconditioning much better-conditioned and is worth
deciding at the same time.

**Recalibrate the acceptance test.** `score-network-conv1d-plan.md` targets roughness
$\le 3\times10^{-3}$ with no smoothing. That is unreachable for *any* score/$\epsilon$-parameterized
model on this data; with the best possible readout the current checkpoint sits at 0.13.
The target belongs to the $x_0$-prediction rewrite, not to this checkpoint.

**One correction to the earlier plan.** It concluded "it is not the sampler." That held for
the MLP under the old linear grid and clamp bug. For this checkpoint the sampler's final
readout is the single largest contributor.
