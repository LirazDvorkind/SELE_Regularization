# From the score prior to a parametric posterior — what we did, found, and should do next

Covers the whole `features/parameter-based-model` branch: 24 commits on top of master plus the
uncommitted parametric-model work. It is the history and the reasoning, not the manual —
`plans/parametric-posterior-primer.md` explains how the machinery works in plain terms, and
`src/forward_model/README.md`, `src/regularization/parametric_model/standalones/README.md` and
`Data/test_set/README.md` own the how-to and are the places to change when behaviour changes.

Worth keeping because several decisions here were reversals, and a reversal whose reason is
lost looks like an arbitrary choice to whoever finds it next.

---

## Where we ended up

An ELE measurement now yields a SELE profile that reproduces the measurement to **0.006-0.2%**
on all 17 ground-truth test curves, is accurate to **0.1-4.6% in the first micrometre**, and
carries bands that contain the truth at every depth on every curve at 95%.

Getting there took one reversal on the physics side (a parameter bound, not a missing term)
and one on the machine-learning side (the output head is the wrong shape and cannot be
trained out of it).

---

## The arc

### Phase 1 — the score prior, and where it ran out

The branch opens with work on `MODEL_SCORE_GRAD`: NAG tuning, warm starts, retraining against
ground truth, and a correction to `create_training_set.m` to randomise the full emission-band
alpha rather than a single wavelength. This is where the score model got as good as it was
going to get in its existing form.

### Phase 2 — one forward model, and something to score against

`1ccade5` unified the optics. `src/optical_constants.py` became the single source of `n` and
`k` with and without the Drude term, read from the paper's own ellipsometry figures via
`src/matlab_fig.py`, and every G in the project — solver and test set alike — is now built
from it. Before this, a benchmark could measure the difference between two G's and call it
reconstruction error.

The same commit added `Data/test_set/`: ground-truth SELE profiles extracted from the paper's
figures, each stored on its own mesh and on the 500-element solver mesh, with the matching G
and ELE for both. Its README records the traps — resample by depth and never by index, a
coarse mesh cannot represent the surface value, and truncating at `W` is free for the forward
model but not for SELE-domain metrics. **Every one of those bit us later in this session
anyway.**

### Phase 3 — the feasibility analysis

`feasibility_demo.py` recomputes, from this repo's own G and training curves, how much the
measurement can possibly say. Two results drove everything after: the SELE curve distribution
is effectively a handful of dimensions rather than 500, and G's singular values fall off a
cliff so only a few directions survive any realistic noise floor. Most of what a 500-point
curve could do is invisible to the instrument.

That is the argument for the whole parametric approach. Asking a network to emit 500 numbers
to describe an object with ~5 degrees of freedom, measured through an operator that resolves
~2, is the wrong level of abstraction.

### Phase 4 — the parametric posterior

A network reads ELE and returns a distribution over the five simulator parameters. SELE curves
and bands come from running the physics on samples, so every curve in a band is physically
admissible by construction rather than by penalty.

**The success criterion is the forward equation.** A profile that does not reproduce the
measured ELE is not an answer, and neither is one whose bands are so wide they say nothing.

| piece | outcome |
|---|---|
| NumPy port of the MATLAB simulator | agrees with `compute_sele_curve.m` to 4.2e-13 |
| optical constants, figures vs `.mat` | bit-identical, 0.000e+00 |
| closed-form ELE, no depth mesh | exact; matches the explicit route to 1.7e-12 |
| 100k-curve dataset with parameters kept | 7 seconds |
| MLP, full-covariance Gaussian head | trained on Colab; v1 reached val NLL -5.465 at 74.5%/92.6% coverage, v2's log was not captured |

The dataset had to be regenerated rather than reused: `create_training_set.m` saves the curves
and discards the parameter values that produced them, which is exactly what a parameter
inference network needs.

#### The performance lesson

The first generator took 27 minutes. SELE is a sum of exponentials in depth and G's entries
are exponentials in depth, so the sum over mesh elements is a geometric series with a closed
form, and every exponential factorises across the two indices — nothing transcendental ever
touches the combined `curves x emission wavelengths x depths` tensor. The depth axis
disappears before it is allocated.

27 minutes became 7 seconds, and the cost stopped depending on the mesh, so the default became
a million elements — within 1.6e-7 of the continuum, where the 500-element solver mesh is off
by as much as 17%. **The fix was to remove the axis, not to parallelise the loop over it.**
Threading came afterwards and bought a further 3x.

Past ~20000 elements it is the *explicit* route that drifts, from summing that many terms in
floating point. The closed form does fixed arithmetic whatever the mesh.

---

## Finding 1: a Gaussian head cannot represent this posterior

The v1 network was beaten by random search by two orders of magnitude. The question was
whether that was a training failure or a representational one.

**Representational, and the oracle test settles it.** Take the parameter vectors that actually
fit a measurement, fit the best possible Gaussian *directly to them* — no training error, no
optimiser — sample it, and push the samples back through the physics:

| curve | members' residual | Gaussian draws: best / median | as good as worst member |
|---|---|---|---|
| srv_1e3 | 0.18-0.36% | 0.20% / 12.9% | 1.2% |
| srv_1e5 | 0.07-0.21% | 0.10% / 16.0% | 0.6% |

A Gaussian fitted to the correct answer still draws ~99% rubbish. The solution set is a thin
curved sheet; the ellipsoid around it is mostly the space the sheet curves through.

This was visible in the training numbers and we read it too late: 74.5% inside the nominal-68%
ellipsoid but only 92.6% inside the nominal-95% one.

Those two numbers point in **opposite directions**, and that is what makes them a shape
diagnosis rather than a size one. Overconfident everywhere would read low on both, and
inflating the covariance would fix both. Underconfident would read high on both. Here,
widening fixes the 95% and breaks the 68%; narrowing does the reverse. **When no single
rescaling can fix both, the problem is the shape.**

The geometry behind it: across the thin sheet the ellipsoid has to be far thicker than the
sheet is, because it also has to reach along it — so it over-covers. Along the sheet the valid
points run far out and curve away, and a straight ellipsoid cannot follow — so it under-covers.
Too thick one way and too short another, at the same time.

**Coverage at two levels diagnoses shape; the likelihood alone does not** — one scalar cannot
tell "wrong place" from "wrong shape". The primer works this through slowly.

### Would a mixture fix it?

| beads | 1 | 3 | 8 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|
| draws that fit | 2.2% | 2.6% | 3.2% | 6.9% | 11.9% | 17.6% |

No elbow. At 120 components it still discards 82% of its draws, and by then there are only ~40
members per component, so part of that last gain is memorising member positions — which makes
the answer more negative, not less. **Decision: no mixture retrain.**

## Finding 2: the ambiguity lives where nobody is looking

The degenerate set is enormous in parameter space and nearly invisible where it matters. For
`srv_1e5`, members span almost the whole allowed range of `D` and most of `tau`, yet their
SELE profiles agree to **0.6% in the first micron** (68% band width relative to the median):

| curve | best fit | 0-1 um | 1-3 um | 3-8 um | 8-30 um |
|---|---|---|---|---|---|
| srv_1e3 | 0.18% | 0.004 | 0.018 | 0.093 | 0.392 |
| srv_1e5 | 0.07% | 0.006 | 0.027 | 0.086 | 0.327 |
| srv_1e7 | 0.75% | 0.063 | 0.171 | 0.333 | 0.559 |

Wildly uncertain *parameters* are compatible with a well-determined *profile*. The parameters
are a means, not the product — which is why parameter-space coverage was never the right score.

## Finding 3: the lifetime gap was a bad bound, not missing physics

The lifetime half of the test set bottomed out at 6-9% residual. The initial suspicion was a
missing physical degree of freedom. **That was wrong.**

Allow each prediction a free multiplier and *every* test curve matches to 0.001-0.15%. The
family makes every shape; it is purely brightness — and not a ceiling either, since the
measurements sit at ~10% of the brightest curve the model can produce.

Among draws that already had the right shape for `tau_8ns`, the multiplier still needed
correlated **-0.93 with `p0`** and **+0.78 with `alpha_scale`**, and the best of them sat at
`D` = 51, hard against the floor of 50. Relaxing one bound at a time:

| box change | srv_1e5 | srv_1e7 | tau_1ns | tau_8ns | tau_60ns |
|---|---|---|---|---|---|
| v1 box (D floor 50) | 0.070% | 0.997% | 3.234% | 7.804% | 8.610% |
| **D floor 50 -> 5** | 0.041% | 0.357% | **0.496%** | **0.512%** | **0.423%** |
| alpha floor -> 3e-3 | 0.075% | 0.674% | 0.910% | 6.046% | 6.991% |
| p0 ceiling -> 1e20 | 0.071% | 1.264% | 4.347% | 8.835% | 9.461% |
| tau ceiling -> 1e-6 | 0.074% | 0.681% | 2.871% | 7.756% | 8.530% |

One bound. All 17 curves then fit to 0.03-0.67%, and the fits ask for `D` = 6-40 cm^2/s, all
below the old floor.

**Why `D`.** It reaches the shape only through `Ln = sqrt(D tau_eff)`, so `D` and `tau` are
interchangeable there. But `tau` also sets brightness on its own, so at fixed shape the only
way to get brighter is a longer lifetime with a smaller `D` — a floor on `D` is a ceiling on
brightness at fixed shape. Confirmed directly: holding `Ln` at 2 um and dropping `D` from 100
to 20 doubles the brightness.

**The general lesson.** The bounds came from the randomisation block of
`create_training_set.m` — a choice about what to put in a training set, not a statement about
what a wafer can do. A systematic residual that survives every parameter combination is
evidence about the *box* before it is evidence about the *physics*.

## Finding 4: the v2 retrain closed the gap

Same architecture, same hyperparameters, dataset regenerated over the widened box.

| | v1 | **v2** |
|---|---|---|
| best draw, worst curve | 14% | **0.202%** |
| best draw, best curve | 5.6% | **0.006%** |
| curves under 0.25% | 0 / 17 | **17 / 17** |

The network now matches what 400k blind prior draws achieve, on every curve. Against the
ground truth, by depth zone (median-curve relative error):

| curve | 0-1 um | 1-3 um | 3-8 um | 8-30 um | in 68% | in 95% |
|---|---|---|---|---|---|---|
| srv_1e3 | 0.1% | 0.3% | 0.4% | 3.3% | 100% | 100% |
| srv_1e5 | 2.5% | 2.6% | 4.7% | 10.9% | 100% | 100% |
| srv_1e7 | 1.6% | 9.4% | 22.4% | 28.4% | 41% | 100% |
| tau_1ns | 2.9% | 5.5% | 16.9% | 24.3% | 100% | 100% |
| tau_60ns | 1.1% | 8.1% | 21.1% | 29.9% | 12% | 100% |

Accuracy degrades with depth exactly as the measurement's sensitivity falls. The 95% band
holds everywhere. **The 68% band collapses on long-tau (12-15%) and should not be quoted
there** — the truth sits consistently in the posterior's tail, which is Finding 1 surviving
into v2 in the one place it still costs something.

Finding 1 and Finding 4 are not in conflict. The Gaussian is still the wrong shape; it is a
poor *summary* and a good *proposal*. Its practical cost dropped because fixing the `D` bound
moved the posterior much closer to the solution set.

## Finding 5: most of the band is one number

Bands look like depth-resolved uncertainty and largely are not. Dividing each draw by its own
mean separates overall brightness from shape:

| curve | z=0.03 raw / shape | z=6 raw / shape | z=30 raw / shape |
|---|---|---|---|
| srv_1e3 | 0.284 / **0.133** | 0.253 / **0.046** | 0.342 / **0.111** |
| srv_1e5 | 0.374 / **0.187** | 0.356 / **0.054** | 0.427 / **0.139** |

For surface-peaked curves the *relative* band is nearly flat with depth (~28% for `srv_1e3`,
shallow minimum near 6 um). It only *looks* like it grows toward z=0 because the plots show
absolute values and those profiles are largest at the surface. Curves that peak away from the
surface — `srv_1e7`, the long-tau set — behave differently: relative width really does grow
with depth, 0.25 to 0.68.

So roughly half to two-thirds of the plotted band is a single brightness multiplier common to
every depth. Take it out and the shape is pinned to ~5% at mid-depth. **If you care about
profile shape rather than absolute magnitude, the reconstruction is much better than the
figures suggest.**

The band the network reports is also far wider than the data requires: at 0-1 um it reports
~0.28 where the true fitting set spans 0.007. That factor of ~40 is network resolution, not
physics, and it is the headroom the next step should claim.

---

## Two mistakes worth remembering

**A depth-integrated SELE metric is misleading and we shipped one briefly.** Scoring with a
single L2 over 0-30 um is dominated by the deep tail nobody can measure. It ranked a
reweighting that improves the surface from 2.5% to 0.1% as a 30% regression. `Data/test_set/`'s
README warns about exactly this and it still happened. `score_test_set.py` now reports per
zone and says why in its docstring.

**Inherited constants are not axioms.** The `D` floor came from a MATLAB training-set script
and silently capped what the whole method could express, for two sessions.

---

## Consequences for the code

- `D` spans 5-200 cm^2/s log-uniformly; every parameter is log-sampled.
- Dataset and checkpoint names carry `v2`. A v1 checkpoint covers a different parameter space
  and is not comparable; `param_spec` travels inside the checkpoint so the mismatch is visible.
- `export_validation_curves.m` samples the new box, so the MATLAB check exercises the short
  diffusion lengths. Still passes at 4.2e-13, including near the `1 - (alpha Ln)^2` pole, and
  the widened box produced no non-finite or negative curves in 200k draws.
- `check_dataset.py` takes a `--name`. v2 is brighter and peaks shallower than the MATLAB
  population, as a lower `D` should be; medians stay close.
- `Data/parametric_model/{datasets,models}` are gitignored following the `score_model`
  convention. They still need `dvc add`.

## Evidence trail

Each probe in `src/forward_model/standalones/` answers one question and prints its own
evidence. They are kept because each backs a decision live in the code, and a number nobody
can regenerate is an assertion rather than evidence.

| script | question |
|---|---|
| `probe_ridge.py` | among parameters that fit equally, how much do the profiles differ, and at what depth? |
| `probe_gaussian_fit.py` | can one Gaussian stand in for that set? |
| `probe_mixture_size.py` | how many Gaussians would it take? |
| `probe_amplitude.py` | is a misfit shape or brightness? |
| `probe_parameter_bounds.py` | which bound is responsible? |
| `../../regularization/parametric_model/standalones/score_test_set.py` | does the band contain the truth, by depth zone? |

---

## Where next

Ranked by value per effort.

**1. Reweight the posterior, restricted to the depths the measurement constrains.** The
biggest win available, and it needs no retraining.

`sample_sele` currently counts all 4000 draws equally when it takes percentiles. But only 3-6%
of them fit the measurement better than 1%, while the rest sit at 9-16% — so the junk is
setting the band width. Near the surface the network reports a band of 0.28 where the true
fitting set spans 0.007, about **40x too wide**. That factor is network resolution, not
physics.

It is recoverable because **we own the forward model**: any draw can be scored exactly against
the measurement in milliseconds. The network does not have to be right, only to propose a bag
that contains good answers — and it does. Weighting by fit instead of counting equally makes
the good draws dominate and collapses the band toward what the data supports.

Measured at an assumed 1% precision: ESS 240-460 of 4000, and `srv_1e3/1e4/1e5` go from ~2.5%
to 0.1-1.8% error at the surface. But applied blindly it makes the *deep* zones worse, because
nothing constrains the answer there and the weights end up sharpening confidently around
differences the measurement cannot see. So it must carry a depth-dependent weight or be
reported only where sensitivity exists. The undecided part is the scope, not whether to do it
— but do not ship the naive global version.

**2. Split amplitude from shape in the figures.** Finding 5 says most of the plotted band is a
scalar. A normalised-band panel beside the absolute one would stop a mostly-scalar ambiguity
reading as depth-resolved uncertainty. Cheap, and it changes how every figure is interpreted.

**3. Stop quoting the 68% band on long-tau.** 95% is sound everywhere; 68% is not. Either
suppress it there or move to a head that can represent the tail. A normalising flow is the
honest fix if one is wanted, but see the mixture result before assuming a small change suffices.

**4. Score the classical methods on the same test set.** Nothing has ever scored
`NON_UNIFORM_MESH` or `TOTAL_VARIATION` against `Data/test_set/`, so we cannot currently say
whether the parametric method beats Tikhonov — only that it satisfies the forward equation
well. Without that comparison the result has no baseline.

**5. Noise-conditioned training, if bands tied to a stated precision are ever wanted.** Add
relative noise to `ele` inside the training loop, resampled each epoch; the reported spread
becomes "uncertainty given a measurement good to X percent". A retrain, not a redesign.

### Tested and rejected

- **Conditioning on known doping.** `p0` correlates -0.93 with the brightness multiplier, so
  fixing it from a known wafer doping looked like it should collapse the band. Measured: it
  does not. At `p0` known to +/-10% the 0-1 um band moves 0.007 to 0.008 and the deep zones
  barely move. The near-surface set is already tight, and the deep spread is not `p0`-driven.
- **A mixture head** (Finding 1).

### Still out of scope

The NAG solver. Nothing in this branch touches it, and the parametric method is not a drop-in
replacement for it — it answers a different question with a different failure mode.
