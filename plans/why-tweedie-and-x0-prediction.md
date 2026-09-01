# Why Tweedie is still needed, and what x₀-prediction would change

Answers four questions raised after the readout fix landed. Companion to
`score-model-noise-diagnosis.md` (the measurements) and
`diffusion-readout-fix-walkthrough.md` (the fix). Sources at the bottom.

---

## 1. Tweedie is not a patch for a weak network

The noise in the raw sampler output was **put there by the sampler, not by the network**.
The reverse SDE lands you at a sample from $p_t$, and $p_t$ is *defined* by
$x_t = a_t x_0 + \sigma_t z$ — such a sample carries $\sigma_t$ of white noise by
construction. A mathematically perfect score model returning the raw state at $t = 0.02$
still scores roughness 1.93 against real curves' 0.000142.

So this was never an architecture question. It is the gap between two different objects:
*a point on the noisy manifold at level t* and *a clean sample*.

Everyone closes that gap explicitly. In EDM's Algorithm 2 the final Euler step runs from
$\sigma_{N-1}$ to $\sigma_N = 0$; since $d_i = (x - D(x;\sigma))/\sigma$, that step
evaluates to $x_N = D(x;\sigma_{N-1})$ — **the last step literally is the denoiser
output**, and the Heun correction is skipped there. DDPM does the equivalent by omitting
the $\sigma z$ injection on its last step. We had no such step at all.

## 2. Was the conv rewrite needed? Honestly, not for this bug

The conv net shows the **same signature** the MLP did: more accurate than an exact
empirical Tweedie denoiser built from held-out curves, yet ~400× rougher. Had
depth-permutation-invariance been the cause, that gap would have closed. It did not — the
rewrite bought ~10× against a ~3000× problem.

The repeatable error: `score-network-conv1d-explanation.md` identified a **real** property
of the MLP (no structural locality) and assumed it explained the symptom, without an
experiment that could separate it from alternatives. The experiment that would have —
*"what roughness would a perfect model score under this readout?"* — costs nothing and
involves no network. It was run only after the retrain.

The conv is still worth keeping: 1.7 M params vs 15.6 M, 7 MB vs 62 MB, same or better
accuracy, the flagged GroupNorm amplitude-collapse risk did not materialize, and it is the
right substrate for x₀-prediction. But it was **not** the fix, and that plan's ≤3e-3
acceptance target was unreachable for reasons unrelated to architecture.

## 3. What "x₀-prediction retrain" means

ε, score and x₀ outputs are interchangeable via Tweedie *mathematically*. They are **not**
interchangeable as regression targets for a network with finite error.

Today the net emits the score and the curve is recovered as
$\hat x_0 = (x_t + \sigma_t^2 s_\theta)/a_t$. Output smoothness is therefore a
**cancellation**: the net must reproduce the high-frequency content of its own noisy input
with gain exactly $1/\sigma_t$, and any shortfall lands in the sample as white noise.
Measured at $t = 0.02$: predicted roughness 0.0738 from the shortfall, observed 0.0735.

x₀-prediction inverts the burden — the net draws the curve, so smoothness comes from a
smoothness-biased architecture doing what it is good at, instead of from near-perfect noise
cancellation, which it is structurally bad at.

**EDM preconditioning** is what makes that work across noise levels:

$$D(x;\sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\!\left(c_{\text{in}}(\sigma)x,\; c_{\text{noise}}(\sigma)\right)$$

with $c_{\text{skip}} = \sigma_{\text{data}}^2/(\sigma^2+\sigma_{\text{data}}^2)$ and
$c_{\text{out}} = \sigma\sigma_{\text{data}}/\sqrt{\sigma^2+\sigma_{\text{data}}^2}$. At low
noise $c_{\text{skip}} \to 1$ and $c_{\text{out}} \to 0$: the input passes through on an
explicit identity path and the network supplies only a small correction whose error is
*multiplied down* by $c_{\text{out}}$ — chosen by Karras specifically to bound that
amplification. Exactly our failure mode, targeted by construction.

Practically: same conv architecture, same dataset, change the loss target, add the
preconditioning wrapper in the training script, change the sampler readout. ~78 min on the
current Colab setup. Decide $\sigma_{\text{data}}$ at the same time — 0.31 under global
min–max normalization vs 0.055 per-curve; per-curve amplitude normalization conditions the
preconditioning far better.

## 4. What standard practice does that we do not

| Practice | Us |
|---|---|
| Final step is a denoise, not the raw state | was missing — now fixed |
| x₀- or v-prediction rather than ε/score, especially at low noise | still raw score — the open item |
| EDM preconditioning: never ask the net to reproduce its own input | not done |
| σ-dependent loss weighting so every noise level contributes equally | we use `var_t`; re-examine under x₀ |
| Concentrated noise sampling in training (log σ ~ N(P_mean, P_std)) | uniform in t |
| Stop at $\sigma_{\min} > 0$ because the score diverges | done (`T_STOP = 0.02`) |

## 5. The option worth testing before the EDM retrain

*Back to Basics* (Li & He, 2025) argues clean data lies on a low-dimensional manifold while
noised quantities do not, so predicting the noised quantity can fail catastrophically. Our
data is an extreme case of that premise: **4 PCA components carry 99.99% of the variance**
of a 500-dimensional curve, and real curves have roughness 1.4e-4 — essentially analytic.
We are asking a network to emit a 500-dimensional white-noise field to reconstruct an
object living in ~4 dimensions.

So: **run the diffusion in PCA coefficient space** (4–20 components). Every sample is then a
combination of smooth basis curves, making output smoothness *exact by construction* rather
than achieved to within 520×. Smaller and faster than the EDM retrain, and informative
either way. Worth running first, or alongside.

---

## Sources

- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM,
  2022) — https://arxiv.org/abs/2206.00364
- Li & He, *Back to Basics: Let Denoising Generative Models Denoise* (2025) —
  https://arxiv.org/abs/2511.13720
- Diffusers, EDM Euler scheduler (final-step behaviour) —
  https://huggingface.co/docs/diffusers/en/api/schedulers/edm_euler
- Diffusers, DDIM scheduler (`clip_sample`, x₀ estimation) —
  https://huggingface.co/docs/diffusers/v0.11.0/en/api/schedulers/ddim
- A timeline of sampling methods of diffusion models —
  https://www.blopig.com/blog/2026/05/a-timeline-of-sampling-methods-of-diffusion-models/
