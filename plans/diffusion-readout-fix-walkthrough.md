# How the noisy-generation fix was made — a walkthrough

A teaching-style account of the change to
`src/regularization/score_model/standalones/new_model_toolkit/test_diffusion_generation.py`.
The *why it was broken* lives in `score-model-noise-diagnosis.md`; this file is the *how
we fixed it*, written so you can follow the reasoning step by step and redo it yourself.

The companion file `tweedie-formula-explained.md` derives the one piece of math this fix
leans on. Read that first if the phrase "Tweedie's formula" is not yet familiar — the rest
of this document assumes it.

---

## Step 0 — Understand what the sampler was actually returning

Before touching code, get precise about what the old loop produced.

The forward (noising) process is

$$x_t = a_t\, x_0 + \sigma_t\, z, \qquad z \sim \mathcal{N}(0, I)$$

with, for the VP-SDE,

$$\textstyle\int_0^t \beta = \left(\beta_{\min} + \tfrac{1}{2}(\beta_{\max}-\beta_{\min})t\right)t, \quad a_t = e^{-\frac12 \int_0^t\beta}, \quad \sigma_t = \sqrt{-\mathrm{expm1}\!\left(-\int_0^t\beta\right)}$$

The reverse SDE walks $t$ from ~1 back down. Whatever time you stop at, the state you hold
is a sample from $p_t$ — and by the equation above, **a sample from $p_t$ is a clean curve
plus $\sigma_t$ worth of white noise**. That is not an artifact of an imperfect model; it
is the definition of the distribution you just sampled from.

The old code ended with:

```python
            x = x + drift * h + diffusion_coef * noise

    x_np = x.numpy()
    S_phys = (x_np + 1.0) / norm_scale + d_min
    return S_phys
```

It returned $x$ itself. So it was returning, on purpose, a curve with $\sigma_{\epsilon}$
of noise still on it.

**The lesson to carry:** the reverse SDE's job is to move you to a *point on the noisy
manifold at level $t$*. Turning that point into a clean sample is a separate, final
operation. Skipping it is not "close enough" — at `time_eps = 1e-4` it costs a factor of
~900 in roughness even with a hypothetically perfect score model.

---

## Step 1 — Quantify the instrument before blaming the model

This is the habit worth stealing from this debug session. Before asking "is my model bad?",
ask **"what would a perfect model score on this test?"**

That is a cheap experiment: take real dataset curves, apply the forward process at the
stopping time, and measure the roughness of the result. No network involved.

| $t_{\text{stop}}$ | $\sigma_t$ | roughness of $a_t x_0 + \sigma_t z$ |
|---|---|---|
| 0.02 | 0.0772 | 1.930 |
| 0.005 | 0.0274 | 1.027 |
| $10^{-4}$ | 0.0032 | 0.129 |
| real curves | — | 0.000142 |

The test as written could never return anything close to 0.000142. The metric was
comparing a deliberately-noisy state against clean training data. **Part of the "noise" was
in the measuring stick, not the model.**

---

## Step 2 — Add the missing final step (Tweedie readout)

The fix is one extra network evaluation after the loop ends. Tweedie's formula converts a
score into the posterior mean of the clean signal:

$$\hat{x}_0 = \frac{x_t + \sigma_t^2\, s_\theta(x_t, t)}{a_t}$$

Read it as: *"take the noisy state, push it in the direction the score points (uphill in
probability), scaled by how much noise you believe is present, then undo the shrinkage
$a_t$."*

Implemented as a small helper plus four lines at the end of the sampler:

```python
def _vp_coefficients(t: float, beta_min: float, beta_max: float) -> tuple[float, float]:
    """VP-SDE marginal coefficients (a_t, sigma_t) for x_t = a_t * x_0 + sigma_t * z."""
    int_beta = (beta_min + 0.5 * (beta_max - beta_min) * t) * t
    return math.exp(-0.5 * int_beta), math.sqrt(-math.expm1(-int_beta))
```

```python
        # Tweedie readout: E[x_0 | x_t] = (x_t + sigma_t^2 * score) / a_t.
        t_tensor = torch.full((n_samples, 1), t_stop, dtype=torch.float32, device=DEVICE)
        a_t, sigma_t = _vp_coefficients(t_stop, beta_min, beta_max)
        x0 = (x + sigma_t ** 2 * model(x, t_tensor)) / a_t
```

Two implementation details worth noting:

- `math.expm1` rather than `1 - math.exp(...)`. At small $t$ the integral is tiny and the
  naive form loses most of its significant digits to cancellation. `expm1` is built for
  exactly this.
- The score is evaluated **at the final state**, not reused from the last loop iteration.
  The loop's last score was computed at $t_{\text{val}}$, one step *before* $t_{\text{stop}}$,
  at a different $x$. Reusing it is a subtle off-by-one that would quietly degrade the
  result.

---

## Step 3 — Stop the integration early

Here is the second half of the fix, and it is counter-intuitive: **do not integrate all the
way to $t = 0$.**

The true score of a Gaussian-perturbed distribution behaves like

$$\nabla \log p_t(x_t) \approx -\frac{z}{\sigma_t}$$

which diverges as $\sigma_t \to 0$. No finite network can output an unbounded vector. Ours
tracks the required magnitude beautifully and then flatlines:

| $t$ | $\sigma_t$ | $\|s_\theta\|$ actual | $\|s\|$ ideal | ratio |
|---|---|---|---|---|
| 0.100 | 0.322 | 69.2 | 69.5 | 0.996 |
| 0.020 | 0.077 | 287.2 | 289.9 | 0.991 |
| 0.005 | 0.027 | 762.9 | 818.1 | 0.933 |
| 0.003 | 0.020 | 770.4 | 1134.1 | 0.679 |
| $10^{-4}$ | 0.0032 | 765.0 | 7042.9 | **0.109** |

Below $t \approx 0.005$ the model is answering a question it structurally cannot answer,
and Tweedie's formula — which *multiplies* the score by $\sigma_t^2$ and trusts it — passes
that error straight through.

In this architecture the ceiling is easy to locate: the head is `GroupNorm → SiLU → 1x1
conv`, and every FiLM (the only $t$-dependent modulation) sits *before* a later GroupNorm.
So the output magnitude is $t$-independent by construction. Nothing in the output path
*can* scale like $1/\sigma_t$.

The change:

```python
T_STOP = 0.02
```

and the grid now spans $[t_{\text{stop}}, t_{\text{start}}]$ instead of assuming `time_eps`
at both ends:

```python
def _power_law_time_grid(t_stop, t_start, n_steps, power=3.0):
    u = np.linspace(0.0, 1.0, n_steps + 1)
    return t_stop + (t_start - t_stop) * (1.0 - u) ** power
```

---

## Step 4 — Choose the stopping time by measurement, not argument

I had an argued value ($t \approx 0.02$, where calibration is still 0.99). Arguments are
worth exactly one experiment. The efficient design: run the reverse process **once** down
to the smallest candidate, snapshot the state at every candidate time on the way, then
apply Tweedie at each snapshot. One trajectory, eight answers.

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

Three things this table teaches:

1. **The optimum is broad and flat.** Anywhere in 0.005–0.1 gives 0.12–0.18. This is good
   news: the fix is not sensitive to a magic constant. `T_STOP = 0.02` is fine, and so
   would 0.03 or 0.06 be. Do not over-tune a flat objective.
2. **The raw column falls steadily while the denoised column does not.** Exactly as
   predicted: the raw state's roughness is dominated by $\sigma_t$, which shrinks; the
   denoised curve's roughness is dominated by *model error*, which does not.
3. **The SDE beats the probability-flow ODE, increasingly so at low $t$.** The injected
   noise is doing useful work — it keeps the trajectory on the data manifold rather than
   letting accumulated score error compound deterministically.

---

## Step 5 — Fix the metric too

A debugging session that fixes the code but leaves a lying metric in place has done half a
job. `run()` now reports both numbers:

```python
    samples, samples_raw = reverse_diffusion_sample(..., t_stop=T_STOP)
    ...
    gen_roughness = float(np.median(compute_roughness(samples)))
    raw_roughness = float(np.median(compute_roughness(samples_raw)))
```

Returning `(denoised, raw)` from one call means you always see the before/after together
and can never again mistake one for the other.

---

## Results

Roughness is `rms(second difference) / std` — scale-free point-to-point jitter.

| model | raw SDE state | denoised | real reference |
|---|---|---|---|
| Alon's d32 | 0.647 | 0.180 | 0.0276 |
| My d32 | 1.167 | 0.533 | 0.0276 |
| My d500 | 1.727 | **0.118** | 0.000142 |

The d500 curves went from visually indistinguishable-from-noise to recognizably
SELE-shaped: the surface dip, the peak at a few µm, the decaying tail. Visibly much better,
and honestly still not dataset-indistinguishable.

---

## What this fix does *not* solve

Worth being precise about, so nobody re-litigates it later.

Look again at the sweep table: the denoised column bottoms out around **0.13**, still ~900×
rougher than real curves. The readout fix is worth about 14×, and no choice of $t_{\text{stop}}$
recovers the rest.

The reason is structural. Rearranging Tweedie's formula: for $\hat{x}_0$ to be smooth, the
term $\sigma_t^2 s_\theta$ must cancel the high-frequency content of $x_t$ almost exactly.
The network is not *drawing* a smooth curve — it is producing a near-perfect cancellation
of the noise it was handed. Measured at $t = 0.02$, the white part of the $\epsilon$-error is
0.0357; multiplied by $\sigma_t/a_t$ this predicts $\hat{x}_0$ roughness of 0.0738, against
a measured 0.0735. The mechanism accounts for the residual roughness exactly.

To reach real-curve smoothness, the network's noise prediction would need to be ~520× more
precise in its high-frequency component. That is not a training-budget problem — it is the
output parameterization. The fix is to have the network predict the **clean curve**
directly, with EDM-style preconditioning giving it an explicit $\sigma$-dependent
pass-through of its own input, so the smooth output never has to arise from cancellation.
That needs a retrain (~78 min); see §7 of `score-model-noise-diagnosis.md`.

---

## Transferable habits from this session

1. **Measure what a perfect model would score before blaming your model.** Half the
   apparent failure was in the measuring stick.
2. **Separate "sample from $p_t$" from "clean sample."** They are different objects. Every
   diffusion codebase needs an explicit final denoising step, and it is easy to forget
   because the loop *looks* finished.
3. **When two hypotheses both predict "output is bad," find the experiment that separates
   them.** Here: comparing the network against an *exact* denoiser built from held-out
   data. It showed the net was more accurate yet far rougher — which immediately rules out
   "undertrained" and points at the parameterization.
4. **Sweep once, snapshot many.** Any time you want a curve of "metric vs. stopping point"
   in an iterative process, record along a single run rather than re-running per point.
5. **A flat optimum is a result.** It tells you the constant does not matter and directs
   your attention elsewhere.
