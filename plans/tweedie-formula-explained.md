# Tweedie's formula, from the ground up

Everything in `diffusion-readout-fix-walkthrough.md` rests on one identity. This file
builds it from scratch — no diffusion background assumed — and then shows why it is the
hinge that the whole score-model idea turns on.

---

## 1. The question

You have a quantity you care about, $x$. You cannot see it. You see a noisy version:

$$y = x + \sigma z, \qquad z \sim \mathcal{N}(0, 1)$$

**What is your best guess of $x$ given $y$?**

"Best" here means minimizing expected squared error, and the textbook answer to *that* is
always the posterior mean:

$$\hat{x}(y) = \mathbb{E}[x \mid y]$$

So far, nothing surprising. The catch is that computing $\mathbb{E}[x \mid y]$ normally
requires knowing the prior $p(x)$ — and in most real problems you have no idea what the
prior is. You have data drawn from it, but not the density itself.

Tweedie's formula is the escape hatch. It says you can compute the posterior mean **without
ever touching the prior**, using only the density of the thing you can actually observe.

---

## 2. A first intuition, before any algebra

Suppose you measure $y = 7.3$, and you happen to know that values of $x$ around 7 are
*rare* while values around 5 are *common*.

Which is more likely: a rare $x \approx 7.3$ that got measured accurately, or a common
$x \approx 5$ that got hit by a big positive error? If the noise is large enough, the
second story wins on sheer prior mass.

So your best guess should not be 7.3. It should be **pulled toward where the data is
dense**. And "the direction in which density increases" is precisely a gradient of a log
density. That is the entire content of the formula:

$$\text{best guess} = \text{what you saw} \;+\; (\text{how noisy it is}) \times (\text{direction of increasing density})$$

The remarkable part — the part that is not obvious and needs proof — is that the density
whose gradient you need is the density of the **noisy** $y$, which you can estimate from
observed data, not the density of the unobservable $x$.

---

## 3. The derivation

Write the density of the observation by marginalizing over the unknown:

$$p(y) = \int p(x)\, \mathcal{N}(y; x, \sigma^2)\, dx$$

Differentiate with respect to $y$. Only the Gaussian depends on $y$, and it has the
convenient property

$$\nabla_y \mathcal{N}(y; x, \sigma^2) = \mathcal{N}(y; x, \sigma^2)\cdot\frac{x - y}{\sigma^2}$$

That factor $(x-y)/\sigma^2$ is the whole trick: differentiating a Gaussian in $y$ pulls
down exactly the quantity $x - y$ we want the expectation of. So

$$\sigma^2 \nabla_y p(y) = \int p(x)\, \mathcal{N}(y; x, \sigma^2)\, (x - y)\, dx$$

Divide both sides by $p(y)$. On the left, $\nabla p / p = \nabla \log p$. On the right,
$p(x)\mathcal{N}(y;x,\sigma^2)/p(y)$ is Bayes' rule — it *is* the posterior $p(x \mid y)$:

$$\sigma^2 \nabla_y \log p(y) = \int (x - y)\, p(x \mid y)\, dx = \mathbb{E}[x \mid y] - y$$

Rearrange:

$$\boxed{\;\mathbb{E}[x \mid y] = y + \sigma^2\, \nabla_y \log p(y)\;}$$

That is Tweedie's formula. Three lines, and the prior has vanished.

The vector case is identical with $\nabla$ a gradient and $\mathcal{N}(y; x, \sigma^2 I)$ —
nothing changes, because the Gaussian factorizes and the same identity holds componentwise.

### The object with a name

$$s(y) \;\equiv\; \nabla_y \log p(y)$$

is called the **score**. Note carefully: score of the *noisy* density. It points uphill in
probability. The formula says: step from your observation in the uphill direction, with
step size equal to the noise variance.

---

## 4. Two worked examples

### 4.1 Gaussian prior → linear shrinkage

Let $x \sim \mathcal{N}(0, \tau^2)$. Then $y \sim \mathcal{N}(0, \tau^2 + \sigma^2)$, so

$$\nabla_y \log p(y) = -\frac{y}{\tau^2 + \sigma^2} \quad\Longrightarrow\quad \mathbb{E}[x\mid y] = y - \frac{\sigma^2 y}{\tau^2+\sigma^2} = \frac{\tau^2}{\tau^2 + \sigma^2}\, y$$

This is the classical shrinkage estimator — the same $\tau^2/(\tau^2+\sigma^2)$ factor that
shows up in ridge regression, Kalman gain, and James–Stein. Tweedie reproduces it as a
special case. Sanity: no noise ($\sigma \to 0$) gives $\hat{x} = y$; overwhelming noise
gives $\hat{x} \to 0$, the prior mean. Both correct.

### 4.2 Two-point prior → a nonlinear denoiser

Let $x$ be $-1$ or $+1$ with equal probability. Then

$$p(y) = \tfrac12\mathcal{N}(y;-1,\sigma^2) + \tfrac12\mathcal{N}(y;1,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-(y^2+1)/2\sigma^2}\cosh\!\left(\frac{y}{\sigma^2}\right)$$

$$\nabla_y \log p(y) = -\frac{y}{\sigma^2} + \frac{1}{\sigma^2}\tanh\!\left(\frac{y}{\sigma^2}\right)$$

$$\mathbb{E}[x \mid y] = y + \sigma^2\nabla_y\log p(y) = \tanh\!\left(\frac{y}{\sigma^2}\right)$$

Put numbers on it: $\sigma = 0.5$, so $\sigma^2 = 0.25$. Observe $y = 0.3$. Tweedie returns
$\tanh(1.2) = 0.834$.

Look at what happened. The raw observation sat almost midway between the two possible
truths. The estimator did not return 0.3 — it *confidently* pulled to 0.83, close to the
mode at $+1$, because the prior has no mass near 0.3 and the noise easily explains the gap.
That is a genuinely nonlinear denoiser, and it came out of a formula that never mentioned
the prior.

This is the shape of what happens with SELE curves too, just in 500 dimensions: the
observation lands somewhere off the manifold of plausible curves, and the score pulls it
back onto it.

---

## 5. Where the name comes from

Herbert Robbins introduced this in his 1956 work founding **empirical Bayes** — the idea
that with enough observations you can estimate what you need about the prior *from the
observations themselves*, rather than declaring it in advance. Robbins credited the identity
to a 1956 letter from the statistician Maurice Tweedie, and the name stuck when Bradley
Efron revived it in his 2011 paper *Tweedie's Formula and Selection Bias*.

The name is worth knowing because it explains why the formula feels like it is cheating.
Empirical Bayes is exactly the art of getting prior-dependent answers out of prior-free
quantities.

---

## 6. Why diffusion models are built on it

Here is the payoff, and it is bigger than a "trick for the last step."

A diffusion model's forward process is Tweedie's setup with a knob:

$$x_t = a_t\, x_0 + \sigma_t z$$

There is an extra $a_t$ (the variance-preserving shrinkage), which we handle in §7, but
structurally it is $y = x + \sigma z$ at every noise level $t$ at once.

A score network is trained to output $s_\theta(x_t, t) \approx \nabla \log p_t(x_t)$.
Tweedie's formula then says:

$$\textbf{a score model and a denoiser are the same object.}$$

Not "related to." The same. Give me a network that predicts the score, and I can hand you
the optimal denoiser for free. Give me an optimal denoiser, and I can read off the score by
inverting the formula. This equivalence is what makes the whole enterprise practical:
learning $\nabla \log p$ directly sounds impossible, but training a denoiser is a plain
supervised regression problem you can set up in ten lines. Add noise, predict what you
added, done. That is denoising score matching, and Tweedie is the theorem that says the
resulting network is a score model.

The identity also earns its keep in three concrete places:

| Use | What Tweedie provides |
|-----|----------------------|
| **Final sample readout** | Reverse SDE leaves you at a point on the noisy manifold; Tweedie converts it to a clean sample. This is the fix in the walkthrough. |
| **Training-target equivalence** | Justifies $\epsilon$-prediction, $x_0$-prediction and score-prediction being reparameterizations of one another. |
| **Inverse problems** | Methods like Diffusion Posterior Sampling need $\hat{x}_0$ mid-trajectory to evaluate a data-fidelity term. Directly relevant to this repo — SELE reconstruction *is* an inverse problem. |

---

## 7. The scaled (VP-SDE) version used in this codebase

Our forward process has that $a_t$ in front. Handle it with a change of variable. Define
$u = x_t / a_t$, so that

$$u = x_0 + \frac{\sigma_t}{a_t} z$$

which is exactly the plain Tweedie setup at noise level $\sigma_t/a_t$. So

$$\mathbb{E}[x_0 \mid x_t] = u + \frac{\sigma_t^2}{a_t^2}\nabla_u \log p(u)$$

Now convert the gradient back. Since $p_U(u) = a_t^{D}\, p_{X_t}(a_t u)$, the chain rule
gives $\nabla_u \log p_U(u) = a_t \nabla_{x_t}\log p_{X_t}(x_t)$. Substituting:

$$\boxed{\;\hat{x}_0 = \mathbb{E}[x_0 \mid x_t] = \frac{x_t + \sigma_t^2\, s_\theta(x_t, t)}{a_t}\;}$$

which is the line in `reverse_diffusion_sample`. The VP-SDE coefficients are

$$\int_0^t\!\beta = \left(\beta_{\min} + \tfrac{1}{2}(\beta_{\max}-\beta_{\min})t\right)t, \qquad a_t = e^{-\frac12\int_0^t\beta}, \qquad \sigma_t = \sqrt{-\mathrm{expm1}\!\left(-\textstyle\int_0^t\beta\right)}$$

**Consistency check.** If your network predicts noise instead of score, the two are related
by $s_\theta = -\hat{\epsilon}/\sigma_t$. Substituting gives
$\hat{x}_0 = (x_t - \sigma_t\hat{\epsilon})/a_t$ — which is just the forward equation solved
for $x_0$. Good: the formula degenerates to the obvious thing when you know the noise
exactly.

---

## 8. Pitfalls

**The score blows up as $t \to 0$.** For small $\sigma_t$ the true score is
$\approx -z/\sigma_t$, which diverges. Tweedie *multiplies* the score by $\sigma_t^2$, which
looks like it should tame it — and it does, in exact arithmetic. But with a *learned* score
that has saturated, the product no longer cancels and the error passes straight through.
This is why the fix stops at $t = 0.02$ rather than $10^{-4}$. **Trusting the formula
requires trusting the score at that noise level; verify calibration before you rely on it.**

**Forgetting $a_t$.** Writing $\hat{x}_0 = x_t + \sigma_t^2 s$ without dividing by $a_t$ is
the single most common implementation slip, because at small $t$ you have $a_t \approx 1$
and it *almost* works — then silently mis-scales everywhere else. §7 of the conv1d plan
records this exact bug in the solver.

**Using a stale score.** Evaluate $s_\theta$ at the state and time you are actually reading
out. Reusing the last loop iteration's score means using a score computed at a different
$x$ and a different $t$.

**Precision at small $t$.** Compute $\sigma_t$ with `expm1`, not `1 - exp(...)`. At
$t = 10^{-4}$ the argument is tiny and the naive form loses most of its significant digits
to cancellation.

**A posterior mean is not a sample.** This one is conceptual and matters most. Tweedie
returns $\mathbb{E}[x_0 \mid x_t]$ — an *average over all clean curves compatible with what
you saw*. Averaging destroys detail. When many curves are compatible (large $\sigma_t$), the
average is a blurry consensus, not a plausible draw. That is why you apply it at the *end*
of a trajectory, when the posterior has narrowed to essentially one curve, rather than
jumping straight from noise to $\hat{x}_0$ in one shot.

There is a sharp diagnostic hiding in that last point. A posterior mean should be, if
anything, **over-smooth** relative to real data. Our Tweedie output came out at roughness
0.118 against real curves at 0.000142 — three orders of magnitude *too rough*, in the
direction the estimator structurally cannot produce on its own. That is not a property of
Tweedie's formula. It is proof that the score being fed into it carries high-frequency
error, which is precisely the conclusion §3.4 of `score-model-noise-diagnosis.md` reaches by
a longer route.

---

## 9. One-line summary

> Adding noise blurs a distribution, and the gradient of the blurred log-density points
> back toward where the unblurred mass was. Tweedie's formula makes that statement exact,
> and in doing so proves that a score model and a denoiser are the same thing.
