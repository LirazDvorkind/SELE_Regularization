# Parametric SELE reconstruction — a primer

The machinery, in plain terms: what the network emits, what the bands are, and how to say it
out loud. `plans/parametric-model.md` is what the method delivers and what it may claim.

## Vocabulary

| term | meaning |
|---|---|
| **ELE** | External Luminescence Efficiency. Of the photons absorbed at one wavelength, the fraction returned as photoluminescence. 28 values. The measurement. |
| **SELE** | Spatial ELE. Of the carrier pairs generated at depth z, the fraction that leave as an emitted photon. A profile in depth. The unknown. |
| **G** | The Beer–Lambert operator linking them: `ELE = (1/phi_abs) G SELE`. |
| **theta** | The 5 simulator parameters: `p0`, `D`, `S`, `tau`, `alpha_scale`. |
| **normalised space** | Each parameter mapped to `[-1, 1]`, `log10` first. The network never sees physical units. |
| **mu** | Mean vector, 5 numbers: where the network centres its guess. |
| **Sigma** | Covariance, 5x5: how wide the guess is, and how the parameters trade off. |
| **L** | Cholesky factor, lower-triangular, `L @ L.T = Sigma`. What the network actually emits. |
| **misfit** | Mean squared error between a draw's predicted ELE and the measured ELE. |

## What the network emits

One ELE curve in, **20 numbers out**: 5 for `mu`, 15 for `L`. Those 20 numbers are a
probability distribution over the five parameters. To use it you sample: draw a random
5-vector `z` from a standard bell curve and compute `theta = mu + L @ z`.

### Sigma, in one grid

|  | p0 | D | S | tau | alpha |
|---|---|---|---|---|---|
| **p0** | *var* | cov | cov | cov | cov |
| **D** | cov | *var* | cov | cov | cov |
| **S** | cov | cov | *var* | cov | cov |
| **tau** | cov | cov | cov | *var* | cov |
| **alpha** | cov | cov | cov | cov | *var* |

Diagonal: how uncertain each parameter is on its own. Off-diagonal: how each pair moves
together — positive rises together, negative trades off, zero means independent. Symmetric, so
only 15 of the 25 entries are free.

**The off-diagonals are the reason for this design.** `D` and `tau` reach the curve shape only
through `Ln = sqrt(D tau_eff)`, so halving one and doubling the other gives the same curve.
Five separate error bars could only say "both uncertain". A full covariance says *"I don't know
either one, but I know their product"*.

### Why L rather than Sigma

A covariance cannot be any 15 numbers — it must not claim a negative variance in any
direction. `L @ L.T` satisfies that automatically for *any* lower-triangular L, so the network
emits an unconstrained L and we take the product. The constraint holds by construction rather
than by penalty, the same reflex as sampling parameters and running the physics instead of
penalising unphysical curves.

The diagonal of L goes through `softplus` plus a floor, because it must be strictly positive;
without the floor the likelihood is unbounded and the network can drive one variance to zero
for infinite reward on a single sample. Off-diagonals pass through untouched — a negative
correlation is a real thing.

### Why the loss is a likelihood

The negative log-likelihood balances two terms: distance from `mu` measured in units of the
network's *own claimed* uncertainty, which pushes the covariance wider, and the volume the
distribution occupies, which pushes it narrower. The honest width wins.

A squared error `||mu - theta||^2` would train only the mean — the covariance appears nowhere
in it, so it would get no gradient and never learn.

## From parameters to a band

1. Feed the measured ELE to the network. It answers with a *range* of parameter sets.
2. Draw 4000 parameter sets from that range.
3. Run the real physics on each → 4000 candidate SELE profiles.
4. Push each back through G → 4000 predicted ELE curves → score each by misfit.
5. Keep the best-fitting 68%. The band is the envelope of what survived.

Step 4 is what makes it an uncertainty set: the ordering comes from the data, not from the
network's own spread. And it needs no ground truth — we own the forward model, so any
candidate can be scored against the measurement directly.

### Saying it out loud

> "At each depth, the range spanned by the profiles that best reproduce this measurement.
> Not measurement noise — the measurement can't distinguish them, so all of them are valid
> answers and the width is how much they disagree."

- **Narrow band** — the plausible answers agree here. Trust it.
- **Wide band** — they disagree. The measurement doesn't pin this depth down.

Tight near the surface and wide deep, because light is absorbed in the first few micrometres:
the measurement genuinely knows about that region and genuinely does not know about the rest.

### Three things a reviewer will poke at

**The 68% is a fraction of draws, not a probability.** It says "the best-fitting 68% of our
candidates", not "68% chance the truth is in here". The band is their full envelope, so every
kept candidate lies inside it at every depth — nothing is being cut off at the ends. What
justifies trusting it is the measured coverage against known profiles, not the number itself.

**The band edges are not real curves.** Draws are kept or dropped whole, but the edges are the
smallest and largest kept value at *each depth*, so the top edge can come from one draw at
1 µm and a different one at 10 µm. The plotted boundary is stitched, and the median line is
likewise a per-depth median, not one of the simulated curves.

**It over-states the ambiguity.** The set of parameters that genuinely fit spans about 0.007
near the surface, narrower than the reported band. It errs wide, which is the safe direction,
but it is an upper bound rather than a sharp estimate.

## Reading the axes

Both panels are in percent. Left: `SELE [%]` — of the carriers generated at that depth, the
fraction that escape as light, so 0.84% means 84 in 10000. Right: `ELE [%]` — of the photons
absorbed at that wavelength, the fraction returned as photoluminescence.

The two are linked by `ELE = (1/phi_abs) G SELE`: the right panel is the left one after the
instrument averages it over depth with Beer–Lambert weights. That averaging is why the deep
part of the left panel is poorly determined — it barely moves the right panel.

## The other percentages

| number | how to say it |
|---|---|
| band width `0.45` | "the band spans 45% of the median SELE at that depth, roughly ±22%" |
| ELE misfit `0.007%` | "reproduces the measured curve to 0.007% in relative L2" |
| SELE error `2.5%` | "differs from the known truth by 2.5% over that depth zone" |
| coverage `100%` | "the truth fell inside the band at every depth of the range quoted" |

The last two need ground truth and exist only for the test set. The first two are computable
on any real measurement.
