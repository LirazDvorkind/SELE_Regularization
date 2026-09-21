# How the parametric posterior works — a primer

What the network actually outputs, why it is shaped that way, how to read its calibration
numbers, and what the obvious next improvement is and why it costs nothing to compute.

Companion to `plans/parametric-model-findings.md`, which records what we found and decided.
This one explains the machinery those decisions are about. Code lives in
`src/regularization/parametric_model/model_definition.py`.

## 0. Vocabulary

| term | meaning |
|---|---|
| **ELE** | External Luminescence Efficiency — the 28-wavelength measurement. The network's input. |
| **theta** | The 5 physical parameters: `p0`, `D`, `S`, `tau`, `alpha_scale`. What we want. |
| **normalised space** | Each parameter mapped to `[-1, 1]`, `log10` first. The network never sees physical units. |
| **posterior** | The set of theta values consistent with a measurement, with weights. Not one answer — a distribution. |
| **mu** | Mean vector, 5 numbers. Where the guess is centred. |
| **Sigma** | Covariance matrix, 5x5. How wide the guess is in every direction, and how directions are tied together. |
| **variance** | Spread of one parameter alone. Sigma's diagonal. |
| **covariance** | How two parameters move together. Sigma's off-diagonal. |
| **L** | Cholesky factor: lower-triangular 5x5 with `L @ L.T = Sigma`. What the network actually emits. |
| **PSD** | Positive semi-definite — the property that makes a matrix a legal covariance. No negative variances. |
| **NLL** | Negative log-likelihood. The training loss. |
| **coverage** | Of held-out truths, the fraction landing inside a band the model claims should hold a given percentage. |

## 1. What comes out of the network

One ELE curve in, **20 numbers out**: 5 for `mu`, 15 for `L`.

Those 20 numbers *are* a probability distribution over the five parameters — not a description
of one, the object itself. To use it you sample: draw a random 5-vector `z` from a standard
bell curve and compute `theta = mu + L @ z`. Do that 4000 times and you have 4000 parameter
sets, which is what `sample_sele` feeds to the physics.

### Sigma is cross-variable and same-variable spread in one grid

|  | p0 | D | S | tau | alpha |
|---|---|---|---|---|---|
| **p0** | *var* | cov | cov | cov | cov |
| **D** | cov | *var* | cov | cov | cov |
| **S** | cov | cov | *var* | cov | cov |
| **tau** | cov | cov | cov | *var* | cov |
| **alpha** | cov | cov | cov | cov | *var* |

The diagonal is each parameter with itself: how uncertain is `D` on its own. The off-diagonal
is each pair: positive means they rise together, negative means one rises as the other falls,
zero means knowing one tells you nothing about the other. It is symmetric — cov(D, tau) and
cov(tau, D) are the same number — so only 15 of the 25 entries are free.

**The off-diagonals are the reason for this design.** `D` and `tau` reach the curve shape only
through `Ln = sqrt(D tau_eff)`, so halving one and doubling the other gives the same curve.
Five independent error bars could only say "D is uncertain, tau is uncertain" and stop.
A full covariance says *"I don't know either one, but I know their product"*, and tilts its
ellipsoid along that trade-off.

### Why L instead of Sigma directly

A covariance cannot be just any 15 numbers. It has to be PSD, or it is claiming some direction
has negative variance, which is meaningless. Asking a network to emit a PSD matrix is asking
it to satisfy a constraint it has no way to respect.

`L @ L.T` is automatically PSD for *any* L. So the network emits an unconstrained
lower-triangular L and we take the product: the constraint holds by construction rather than
by penalty. Same spirit as sampling parameters and running the physics instead of penalising
unphysical curves — make the illegal thing unrepresentable rather than discouraged.

The diagonal of L must additionally be strictly positive, so it goes through `softplus` plus a
floor (`MIN_SCALE`). Off-diagonals pass through untouched, because a negative correlation is a
real thing. The floor is not a numerical nicety: without it the likelihood is unbounded and the
network can drive one variance to zero and collect infinite reward on a single sample.

### Why the loss is NLL and not squared error

The NLL of the true theta is two competing terms:

- **distance from mu, measured in units of the network's own claimed uncertainty.** Being 0.1
  away is a disaster if you claimed 0.01 and fine if you claimed 1.0. Pushes the covariance
  **wider**.
- **the volume the distribution occupies.** Pushes it **narrower**.

Training balances them, so the honest width wins. That is what makes the error bars mean
something.

A plain squared error `||mu - theta||^2` would train only the mean. The covariance appears
nowhere in it, so it would receive **no gradient at all** and sit at its initialisation
forever. NLL is what makes uncertainty learnable.

## 2. Reading the coverage numbers

Take the network's Gaussian and draw the ellipsoid that *should* contain 68% of true answers
if the Gaussian were right. Check held-out data: what fraction actually landed inside?

| region | should catch | v1 actually caught | verdict |
|---|---|---|---|
| 68% ellipsoid | 68% | **74.5%** | catches too many -> too big |
| 95% ellipsoid | 95% | **92.6%** | catches too few -> too small |

**Why that combination is the important one.** If the network were simply overconfident, both
numbers would be low, and shrinking everything would fix both. If underconfident, both high,
and inflating fixes both. Here they point in **opposite directions**: widen it and the 68%
gets worse, narrow it and the 95% gets worse.

**When no single rescaling can fix it, the problem is shape, not size.**

In one dimension the intuition is: truth usually very close to the centre, occasionally far
out. Fit a bell curve and the rare far points drag its width up, so its 68% interval is wider
than the real cluster and swallows more than 68% — while the real outliers sit farther than a
bell curve's tail predicts, so the 95% interval still misses some. Fat core, thin tails.

In our five dimensions it is geometry. The valid parameter sets form a thin curved sheet:

- **across the sheet** it is razor thin, but the ellipsoid must be far thicker than that
  because it also has to reach along the sheet -> **over-covers**;
- **along the sheet** valid points run far out and curve away, and a straight ellipsoid cannot
  follow, so it falls short at the ends -> **under-covers**.

One ellipsoid, forced to be too thick in one direction and too short in another at the same
time. That is the 74.5 / 92.6 signature, and it is a fact about ellipsoids versus curved
sheets, not a training failure. `probe_gaussian_fit.py` confirms it directly: a Gaussian
fitted to the *correct answer*, with no training error at all, still draws ~99% rubbish.

**The practical rule: read coverage at two levels, not one, and read it alongside the loss.**
A single NLL is one scalar and cannot distinguish "wrong place" from "wrong shape".

## 3. The recoverable gap

`sample_sele` currently treats the Gaussian as the answer: draw 4000 parameter sets, run the
physics on all of them, take percentiles of the resulting curves. **Every draw counts
equally**, and that is the thing worth changing.

Per draw, from the v2 checkpoint:

| | value |
|---|---|
| best draw fits the measurement to | 0.006-0.2% |
| typical draw fits to | ~9-16% |
| draws fitting better than 1% | 3-6% |

The bag contains excellent answers and ~94% junk, and taking percentiles over the whole bag
lets the junk set the band width. Near the surface the network reports a band of **0.28**
where brute-force search over what genuinely fits spans **0.007** — about **40x too wide**.
That factor is not physics. The physics allows a tight answer; the network cannot point at it
precisely.

**Why it is recoverable without retraining: we own the forward model.** Any draw can be run
through `simulate_ele` and scored against the measurement exactly, thousands at a time, in
milliseconds. The network never has to be right — it only has to hand over a bag that
*contains* good answers, and it does.

So weight each draw by how well it fits instead of counting them equally. The bad draws fade,
the good ones dominate, the band collapses toward what the data supports. Nothing is
retrained; the information was already there and was being averaged away.

Measured at an assumed 1% measurement precision: `srv_1e3`/`1e4`/`1e5` go from ~2.5% to
**0.1-1.8%** error at the surface, with 240-460 effective draws surviving out of 4000.

**The catch.** Applied blindly at every depth it makes the *deep* region worse. Nothing
constrains the answer down there, so the weights chase differences the measurement cannot see
and sharpen confidently around noise. The fix is real; what is still undecided is the scope —
reweight **only where the measurement has sensitivity**, not everywhere.

## 4. Where this leaves the head

The head emits 15 numbers to describe a curved sheet. It can stretch and tilt; it cannot bend.
A mixture would need more than 120 ellipsoids and still discard 82% of its draws
(`probe_mixture_size.py`), so a small change to the head does not rescue it.

That makes the open question not "how do we fix the head" but **whether the head should be an
answer at all, or a proposal that reweighting corrects.** The evidence points at proposal: even
the v1 Gaussian landed in the right region about 12x more often than blind prior sampling, and
v2 finds fits as good as 400k blind draws on every test curve. As a fast way to propose
candidates it is doing its job. As a summary of the posterior it is the wrong shape, and
always will be.
