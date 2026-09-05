# Score model (diffusion prior) — where things stand

Consolidates and replaces `score-network-conv1d-explanation.md`,
`score-network-conv1d-plan.md`, `score-model-noise-diagnosis.md`,
`diffusion-readout-fix-walkthrough.md`, `tweedie-formula-explained.md` and
`why-tweedie-and-x0-prediction.md`, which are deleted in favor of this file. See the git
history for that commit range if the step-by-step derivations are ever needed again — the
math (Tweedie's formula, the VP-SDE coefficients, the FiLM/dilated-conv architecture
rationale) was correct and is not repeated in full here, only its conclusions.

Companion document: `feasibility-assessment.md` covers a separate, more fundamental
question raised after this work — whether curve-space diffusion is even the right level of
abstraction for this prior in the first place. Read that alongside item 1 below.

---

## TL;DR

The d500 score network **did learn the SELE distribution correctly** — its accuracy matches
an exact empirical Tweedie denoiser at every noise level. Generated samples still looked like
noise because of two sampler bugs (now fixed) sitting on top of one architectural limitation
(not yet fixed):

1. **Fixed** — the reverse SDE was returning a raw noisy sample instead of denoising it as a
   final step. Worth ~14x in roughness.
2. **Fixed** — the reverse process was integrating past the point where the score network's
   output saturates (its learned score is well-calibrated only down to `t ≈ 0.005`).
3. **Not fixed, understood** — the network outputs a raw score/noise prediction, which means
   every clean sample it produces is the result of near-perfect cancellation of the noise it
   was handed. It falls ~520x short of the precision that cancellation needs, and that
   shortfall is exactly the residual roughness measured. Fixing this needs a retrain with
   **x0-prediction under EDM preconditioning**, not a better sampler or a bigger network.

Rewriting the network from an MLP to a dilated 1D convolution (done, see below) was a
reasonable, independently-motivated change but **was not the fix** — it bought ~10x against
a ~3000x problem, and the same signature (accurate but rough) persisted after the rewrite.

## Timeline

1. **Diagnosis → conv1d plan.** The MLP score network has no structural notion of depth
   adjacency (permuting the depth axis and its weights together leaves it functionally
   identical), so a dilated residual 1D-conv architecture with FiLM time-conditioning was
   proposed and built as `Conv1dScoreNetwork` in `model_definition.py`, selected via an
   `arch` field / factory so old MLP checkpoints keep loading unchanged.
2. **Retrained, still noisy.** The conv rewrite was trained on Colab and still produced
   visibly noisy curves — ruling out "architecture was the whole story."
3. **Root-cause investigation** (9 targeted probes, cheapest first) found, in order of
   contribution: the missing final denoising step (~900x penalty, independent of model
   quality — verified by measuring the roughness a *perfect* model would produce under the
   old readout), score saturation below `t ≈ 0.005` (calibration ratio drops from 0.99 to
   0.11), and underneath both, the score-parameterization cancellation problem (network
   error is small in RMSE but almost entirely high-frequency; verified against an exact
   empirical Tweedie denoiser built from held-out curves, and the predicted-vs-measured
   residual roughness matched to three significant figures).
4. **Fix implemented**: added the Tweedie readout (`x̂₀ = (x_t + σ_t² s_θ)/a_t`, using
   `expm1` for numerical stability at small `t`, evaluated at the final state not a stale
   loop value) and stopped the reverse integration at `T_STOP = 0.02` instead of `t → 0`
   (chosen from an empirical sweep — the optimum is broad and flat over `t ∈ [0.005, 0.1]`,
   so this is not a fragile constant). Roughness improved 0.647 → 0.118 for d500 against a
   real-curve reference of 0.000142 — better, but still ~900x off, and that residual gap is
   the structural cancellation problem, not a stopping-time or architecture problem.
5. **Retrospective**: confirmed Tweedie's formula is not a workaround for a weak network —
   the noise it removes was put there by the sampler's own forward process by construction,
   and every standard diffusion codebase (EDM, DDPM) has an explicit equivalent step. Also
   confirmed the conv1d rewrite, while not the fix, is still worth keeping (1.7M params vs
   15.6M, same or better accuracy, no sign of the flagged GroupNorm amplitude-collapse risk)
   and is the right substrate for the actual fix.

## Current code state

- `model_definition.py` — both `ScoreNetwork` (legacy MLP, all pre-existing checkpoints) and
  `Conv1dScoreNetwork` (dilated residual, FiLM-conditioned, ~1.7M params, kernel size 5,
  dilations `(1,2,4,8,16,32,64)` giving receptive field 509 ≥ 500) selected via a
  `build_score_network(model_config)` factory. `arch` defaults to `'mlp'` so nothing old
  breaks.
- `standalones/new_model_toolkit/test_diffusion_generation.py` — the measurement harness.
  Fixed: removed an in-loop `clamp(-1, 1)` that was distorting the reverse process, switched
  to a power-law time grid concentrated near `t → 0` (where all the shape information is),
  added the Tweedie readout at `T_STOP = 0.02`, and now reports roughness for both the raw
  and denoised output next to the real-curve reference number, so the comparison is never
  accidentally apples-to-oranges again.
- `score_model_grad.py` and the training script were updated to construct networks through
  the factory. `arch` propagates through the checkpoint's existing `config` dict, so no
  checkpoint-format change was needed.

## Open items, ranked

1. **Try diffusion in PCA-coefficient space first (4–20 dims), not curve space.** This is
   the same conclusion `feasibility-assessment.md` reaches independently: 4 principal
   components already carry 99.99% of the variance in the training set, real curves have
   roughness ~1.4e-4 (essentially analytic), and the network is currently being asked to
   emit a 500-dimensional white-noise-shaped correction to reconstruct an object that lives
   in ~4–5 real dimensions. This is cheap to try and tests whether curve-space diffusion is
   the right tool at all before investing in a retrain.
2. **If curve-space diffusion is still wanted: retrain with x0-prediction + EDM
   preconditioning.** `D(x;σ) = c_skip(σ)x + c_out(σ)F_θ(c_in(σ)x, c_noise(σ))` gives the
   network an explicit identity path so smoothness comes from a smoothness-biased
   architecture drawing the curve, not from cancelling noise to 520x the precision it
   currently achieves. Same conv architecture, same dataset, ~78 min on the current Colab
   setup. Decide `σ_data` at the same time — 0.31 under the current global min-max
   normalization vs 0.055 per-curve; per-curve normalization conditions the preconditioning
   much better.
3. **Re-tune the NAG solver's hyperparameters** (`LR_MAX`/`LR_MIN`/`REG_WEIGHT` in the
   presets) against whichever network is retrained — they are currently tuned against the
   old network's gradient magnitudes.
4. **Separately, the NAG solver collapsing to the mean curve has three already-identified,
   still-unfixed causes**, independent of generation quality above: (a) the solver queries
   the network without the VP mean-scale factor `a_t` (measured effect: correct Tweedie RMSE
   0.13 vs solver-style 0.75, correlation-to-mean −0.54 in the solver-style case); (b) global
   min-max normalization squashes ~28% of training curves into a span narrower than `σ_t` at
   the solver's own `T0`; (c) the adaptive score/data rescaling discards the network's
   calibrated score magnitude every step.
5. **Recalibrate the roughness acceptance target.** The original conv1d plan's target of
   ≤3e-3 with no smoothing is unreachable for *any* score/ε-parameterized model on this data
   — the measured floor with the best possible readout is ~0.13. That target belongs to the
   x0-prediction rewrite (item 2), not to the current checkpoint.

## Key numbers worth remembering

| quantity | value |
|---|---|
| real training curves, roughness | 0.000142 |
| d500, raw SDE state (old, buggy readout) | 1.727 |
| d500, after Tweedie readout + T_STOP fix | **0.118** |
| floor of denoised roughness across any `T_STOP` swept | ~0.13 (broad, flat optimum over 0.005–0.1) |
| score calibration ratio at `t=0.005` / `t=1e-4` | 0.93 / 0.11 |
| conv1d net params vs MLP | 1.7M vs 15.6M |
| PCA variance carried by 4 components (training set) | 99.99% |

## Sources consulted

- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM,
  2022) — https://arxiv.org/abs/2206.00364
- Li & He, *Back to Basics: Let Denoising Generative Models Denoise* (2025) —
  https://arxiv.org/abs/2511.13720
- Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*
  (NCSN, 2019); Kong et al., *DiffWave* (2021); van den Oord et al., *WaveNet* (2016);
  Dhariwal & Nichol, *Diffusion Models Beat GANs* (ADM, 2021); Perez et al., *FiLM* (2018) —
  architecture rationale for the conv1d rewrite.
- Robbins (1956) / Efron, *Tweedie's Formula and Selection Bias* (2011) — origin of the
  identity the readout fix relies on.
- Diffusers docs, EDM Euler and DDIM schedulers — reference implementations of the final
  denoising step other codebases use.
