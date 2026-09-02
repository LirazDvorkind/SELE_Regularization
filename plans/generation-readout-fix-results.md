# Clean SELE generation — the readout fix and what it measured

Record of the change to
`src/regularization/score_model/standalones/new_model_toolkit/test_diffusion_generation.py`
and the numbers it produced for the **d500 MLP** checkpoint on this branch.

Prior investigation lives on `origin/features/conv-score-model` under `plans/` (six docs).
Those were written against the **conv1d** checkpoint; this file is the same exercise redone
against the MLP we actually use, which is why the numbers differ.

---

## Why the change was needed

The reverse SDE lands you on a sample from `p_t`, and such a sample carries `σ_t` of white
noise **by construction** — that is the definition of the distribution, not a model failure.
The sampler returned that raw state, with no final denoising step. The loop *looks* finished,
which is why the omission went unnoticed.

The metric hid it too: it compared a deliberately-noisy state against clean training data,
unfair by ~900x before the model is even involved. **Part of the "noise" was in the measuring
stick.**

## What changed

Four things, one file, +65/−17.

1. **Tweedie readout** — `x̂₀ = (x_t + σ_t²·s_θ(x_t, t)) / a_t`, applied once after the loop,
   via a new `_vp_coefficients(t, beta_min, beta_max) → (a_t, σ_t)`. Uses `math.expm1` (the
   naive `1 - exp(...)` loses most of its significant digits at small `t`); re-evaluates the
   score at the final state and `t_stop` rather than reusing the loop's last one; keeps the
   `/a_t`.
2. **`T_STOP = 0.02`** — the reverse integration stops there instead of at `time_eps`. Not a
   separate feature: at `t = 1e-4`, `σ_t² = 1.02e-5` and the network's score is capped at ~765
   against ~7043 ideal, so the Tweedie correction would be `≈ +0.008` — a no-op, and the fix
   would have appeared to fail.
3. **Deleted the in-loop `x.clamp(-1, 1)`** — it clamped the *noise-space* state, where at
   high `t` the state is ~N(0,I) and a large fraction of entries were being clipped every step.
4. **`compute_roughness` = rms(2nd difference)/std**, scale-free. `reverse_diffusion_sample`
   now returns `(denoised, raw)` and `run()` prints both against a real-curve reference from
   `sele_simulated_1000_curves_500_long.csv`, resampled to the model's resolution.

Deliberately **not** changed: power-law step spacing (motivated but untested — a hypothesis,
not a bug), readout ensembling, PCA projection, higher-order solvers, anything requiring a
retrain. Nothing outside this file was touched.

## Instrument check (run before trusting any model number)

Forward process applied to real curves, no network involved — this is what a *perfect* model
would score under the raw readout:

| t | 0.1 | 0.02 | 0.005 | 1e-4 |
|---|---|---|---|---|
| σ_t | 0.3221 | 0.0772 | 0.0274 | 0.0032 |
| roughness of `a_t·x₀ + σ_t·z` | 2.40 | 2.00 | 1.11 | 0.14 |

Real curves: **0.000142**. All match the values documented on the conv branch, so
`_vp_coefficients` is correct.

## Results — d500 MLP (`sele_score_net_d500.pt`)

| | roughness |
|---|---|
| raw SDE state (what the old code returned) | 2.19 |
| denoised (Tweedie readout) | **0.28** |
| real curves | 0.000142 |

`T_STOP` sweep, 32 samples / 2000 steps, sweep-once-snapshot-many (one trajectory, all
answers):

| t_stop | 0.1 | 0.06 | 0.04 | 0.03 | 0.02 | 0.015 | 0.01 | 0.005 |
|---|---|---|---|---|---|---|---|---|
| raw | 2.40 | 2.31 | 2.30 | 2.22 | 2.12 | 1.88 | 1.68 | 1.34 |
| denoised | 0.55 | 0.34 | 0.31 | **0.25** | 0.27 | 0.31 | 0.30 | 0.36 |

Raw falls steadily as σ_t shrinks; denoised does not, because the denoised curve's roughness
is dominated by *model error*, which does not shrink with `t`. The optimum is broad and flat
across 0.02–0.04 — a flat optimum is a result: the constant does not matter, so do not tune it.

Backward compatibility confirmed — all checkpoints still load and run, including Alon's bare
`Sequential` d32 (0.79 → 0.17).

## Three findings

1. **The MLP's floor is ~0.25, roughly 2x worse than the conv checkpoint's 0.13.** So
   architecture was not irrelevant — it was just nowhere near sufficient on its own. Still
   ~1800x above real curves.
2. **The fix bought ~8x here, against ~14x reported for the conv model.** Consistent with the
   MLP being the rougher network; the readout recovered the same sampler-side error either way.
3. **Shapes are only partly right, and this is a distribution problem, not a smoothness one.**
   One sample in five is a proper SELE profile (surface dip, peak ~4 µm, decay). Others rise
   monotonically with no peak, or decay monotonically from the surface, and some start at
   **negative SELE** (min −2.6e-3, at every `t_stop`). The old clamp was masking this — that
   comment about "physically implausible (e.g. negative) SELE" described a real symptom, but
   clamped the wrong variable in the wrong space. Removing it did not cause the problem; it
   revealed it. **No amount of readout work touches this.**

## Known root cause, not yet addressed

From the conv-branch investigation, unchanged by anything here: under a score/ε output
parameterization, output smoothness arises from a **cancellation** — the network must reproduce
the high-frequency content of its own noisy input with gain exactly `1/σ_t`, and any shortfall
lands in the sample as white noise. Measured there at 520x short of what real-curve smoothness
requires. Not a capacity or training-budget problem.

The researched candidates, neither attempted:

- **x₀-prediction with EDM preconditioning** — [Karras 2022](https://arxiv.org/abs/2206.00364),
  [Li & He 2025](https://arxiv.org/abs/2511.13720v1). Gives the network an explicit
  σ-dependent identity path so smoothness never has to come from cancellation.
- **PCA-space diffusion** — 4 components carry 99.99% of this dataset's variance, so diffusing
  in coefficient space makes smoothness exact by construction. Cheapest of the two. A PCA
  projection applied at readout only would preview whether it works, at zero training cost.

Also noted but untested: **ensembling the Tweedie readout** (re-noise and re-denoise `N` times,
average — the white error component should fall ~`1/√N`). Post-hoc low-pass filtering is ruled
out by measurement on the conv branch (worsens relative error 0.0123 → 0.0328; real curves
genuinely carry ~0.9% of their energy above `k = 25`).

---

## TODO — next session

1. **Retrain the model.**
2. **Retry** — rerun this generation test against the new checkpoint (`T_STOP` sweep included;
   the instrument is now trustworthy, so the numbers are directly comparable to the table above).
3. **Read the AI output and decide what next.**

Reproduce the numbers above with:

```bash
python -m src.regularization.score_model.standalones.new_model_toolkit.test_diffusion_generation
```
