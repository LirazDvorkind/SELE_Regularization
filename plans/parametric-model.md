# Parametric SELE reconstruction

Recovers a SELE depth profile, with uncertainty, from an ELE measurement — by inferring the
five parameters of the physical simulator rather than regularising a 500-point curve.

How-to lives in `src/forward_model/README.md` and
`src/regularization/parametric_model/standalones/README.md`;
`plans/parametric-model-primer.md` explains the machinery in plain terms. This is what the
method is, what it delivers, and what it cannot claim.

---

## The idea

SELE has roughly five degrees of freedom, and the measurement resolves fewer: the ELE curves
span four directions noise-free, and only two survive a realistic 1-5% noise floor. Asking a
network for 500 numbers is the wrong level of abstraction. Instead:

```
ELE (28 wavelengths)
  → network → a distribution over the 5 simulator parameters
  → draw 4000 parameter sets
  → run the physics on each → 4000 candidate SELE profiles
  → score each against the measurement, keep the best-fitting fraction
  → that set is the answer and its spread is the uncertainty
```

Every curve in the band comes out of the simulator, so it is physically admissible by
construction rather than by penalty. **The success criterion is the forward equation**: a
profile that does not reproduce the measured ELE is not an answer, and neither is one whose
band is so wide it says nothing.

## What the bands are

The band is the envelope of the draws that **best reproduce the measurement**, ranked by mean
squared error in ELE space — not percentiles of every draw. Ranking by data fit is what makes
it an uncertainty set rather than a picture of the network's own spread.

They are **not measurement noise**. Training data is noise-free, so a noisier measurement does
not widen them. They express that the forward map is many-to-one: different parameter sets
produce ELE curves that cannot be told apart, and the band is what those look like in depth.
Same class of statement as the confidence window in the paper's non-uniform-mesh method.

The honest phrasing is **an upper bound on the ambiguity, not a tight estimate of it**. The
set of parameters that genuinely fit spans about 0.007 near the surface; the reported band is
wider than that, and the excess is the network's resolution rather than physics.

## The level, and why 0.68

Keeping the best-fitting fraction `L` of draws, coverage of the truth over 0–3 µm against the
ground-truth test set:

| level | median coverage | median band width |
|---|---|---|
| 0.10 | 40% | 0.074 |
| 0.25 | 72% | 0.152 |
| **0.68** | **100%** | **0.454** |
| plain 95% percentile band | 100% | 0.793 |

**0.68 is the tightest level that still holds the truth at every depth of the first 3 µm, on
every curve**, and it is 1.7x tighter than the percentile band it replaces at identical
coverage. Over the full 0–30 µm even 0.68 lets the truth out on the longest-lifetime curves
(76–89%) — the deep tail no measurement constrains, not a badly chosen level.

Tightening further does not work, and the reason is the physics rather than tuning: a better
ELE fit is not reliably a closer profile. That is the degeneracy. `REPORTED_LEVEL` in
`uncertainty.py` carries this value; `standalones/check_uncertainty.py` re-measures the table.

## What it delivers

Against all 17 ground-truth test curves:

| | |
|---|---|
| ELE reproduced to | **0.006–0.2%** (best draw, every curve) |
| SELE accuracy, first micron | **0.7–3.4%** |
| truth inside the band, 0–3 µm | **100% of depths, every curve** |

Accuracy degrades with depth as the measurement's sensitivity falls — 0.5–9.7% at 1–3 µm,
1.3–30% past 8 µm — and over the full 0–30 µm the band stops holding the truth on the
longest-lifetime curves (76–89%). Trust about the first 3 µm; beyond that the profile is
increasingly prior rather than measurement, and the band widens to say so.
`standalones/score_test_set.py` prints the table.

## Quoting a number

For anything numerical, propagate to the quantity rather than reading it off the band. The
band's edges are the extremes at each depth taken separately, so it loses the correlation
between depths — it cannot express that a draw with a high peak also puts that peak deeper —
and **peak depth cannot be read off a band at all**.

`uncertainty.scalar_intervals` computes surface SELE, peak SELE, peak depth and mean SELE over
0–3 µm directly from the kept draws. Quote those.

## Checks that work without ground truth

All run on a real measurement; `run_inference` prints all of them.

| check | catches | bar |
|---|---|---|
| input in distribution | network extrapolating; nothing it says is meaningful | must pass first |
| best-draw ELE misfit | posterior contains nothing that explains the data | < 0.25% |
| fraction of draws fitting < 1% | posterior centred wrong, only grazing the answer | 3–6% normal |

The input check (`in_distribution.py`) runs two tests, because they fail differently.
**Per-channel range** catches gross extrapolation — an input channel beyond anything in
training. **Off-manifold distance** catches the subtler case: the 29 channels are not
independent, the training curves lie within a 4-dimensional subspace because five parameters
generated them, and a measurement far off that subspace is not a curve the simulator can
make, however ordinary each channel looks alone. (Four is the noise-free span; the two
resolvable directions above are what is left once measurement noise is assumed. The check is
noise-free, so it uses four.)

Both reference the training set rather than a chosen threshold, so the verdict is "unlike the
training data", not "far from the mean in some units". All 17 test curves and the paper's
`paper_gt` pass. An amplitude scaled by 100 is caught by the range test, added noise by both.

## What not to claim

- **Not a noise posterior.** If a stated measurement precision is wanted, add relative noise
  to the ELE inside the training loop, resampled each epoch. A retrain, not a redesign.
- **The 68% is a fraction of draws, not a probability.** It means "the best-fitting 68% of
  candidates", and the band is their full envelope, so each of them lies inside it at every
  depth. What justifies trusting it is the measured coverage against known profiles, not the
  number itself.
- **Depth beyond a few micrometres is not measured.** Light is absorbed in the first few µm.
- **A good ELE fit alone does not prove the profile.** What licenses that step is the
  test-set validation, not the residual.

## Where it stands

The parameter box widens `D` to 5–200 cm^2/s log-uniformly, below the floor of 50 in
`create_training_set.m`: `D` and `tau` reach the curve shape only through
`Ln = sqrt(D tau_eff)`, but `tau` also sets brightness alone, so a floor on `D` caps
achievable brightness at fixed shape. `probe_parameter_bounds.py` re-derives it one bound at a
time.

The network's output is a single Gaussian over the five parameters. That head cannot represent
the true solution set, which is a thin curved sheet — a Gaussian fitted directly to the
correct answer still draws ~99% non-fitting samples, and a mixture needs more than 120
components to help materially. This is why the network is used as a **proposal** and the
misfit ranking does the selecting, rather than the Gaussian being read as the answer.

## Open

1. **Score the classical methods on the same test set.** Nothing has scored
   `NON_UNIFORM_MESH` or `TOTAL_VARIATION` against `Data/test_set/`, so there is no baseline
   to say whether this beats Tikhonov.
2. **Noise-conditioned training**, if bands tied to a stated precision are wanted.

Tested and rejected: a mixture-of-Gaussians head (needs >120 components); conditioning on
known doping (`p0` correlates −0.93 with brightness, but fixing it to ±10% moves the 0–1 µm
band only from 0.007 to 0.008).

The NAG solver is out of scope — a different question with a different failure mode.

## Map

| file | role |
|---|---|
| `src/forward_model/` | NumPy port of the MATLAB simulator; closed-form ELE, no depth mesh |
| `.../parameters.py` | the five parameters and their box |
| `.../standalones/probe_*.py` | diagnostics backing the box and the head choice |
| `.../standalones/generate_dataset.py` | builds the training set, parameters kept with curves |
| `src/regularization/parametric_model/model_definition.py` | the network |
| `.../features.py` | ELE → the 29 input channels; the transform travels in the checkpoint |
| `.../inference.py` | ELE → parameter draws → SELE ensemble |
| `.../uncertainty.py` | misfit ranking, bands, scalar intervals |
| `.../in_distribution.py` | is this measurement one the network was trained for |
| `.../plot_parametric_result.py` | the two-panel figure |
| `.../standalones/parametric_model_training.ipynb` | the Colab notebook that trains it |
| `.../standalones/run_inference.py` | reconstruct and plot; writes `results/parametric_model/` |
| `.../standalones/score_test_set.py` | accuracy vs ground truth, by depth zone |
| `.../standalones/check_uncertainty.py` | coverage vs band width, by level |
