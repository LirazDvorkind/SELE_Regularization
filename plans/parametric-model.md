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
  → score each against the measurement
  → report the best-fitting draw; every draw within 1% in ELE is the set that explains the data
```

Every reported curve comes out of the simulator, so it is physically admissible by
construction rather than by penalty. **The success criterion is the forward equation**: a
profile that does not reproduce the measured ELE is not an answer.

## What is reported

**The curve is the best-fitting draw** — the one whose ELE is closest to the measurement in
relative L2. It is a real simulator curve with its five parameters attached. A depth-wise
median of many draws is not: it stitches different draws together at different depths and
no parameter set produces it.

**The draws that fit are everything within 1% in ELE** (`TOLERANCE` in `uncertainty.py`),
drawn as light lines with their envelope shaded. They all explain the measurement, so where
they fan out in depth the data no longer constrains the profile. If no draw is within the
tolerance, the best draw is reported alone with a warning stating by how much it misses.

The cut is an absolute misfit, not a fraction of draws. A fraction says nothing about fit: the
earlier best-fitting-68% rule admitted draws missing the measurement by 13–23%, and its band
held the truth largely because of them. Its median also scored better on SELE over 0–30 µm
than any tighter selection, for the same reason — draws that do not explain the data happened
to pull the curve toward the truth on this test set. That is not something to report as a
result, so the rule was retired.

The spread is **not measurement noise**. Training data is noise-free, so a noisier measurement
does not widen it. It expresses that the forward map is many-to-one at the chosen tolerance.

## What it delivers

Against all 17 ground-truth test curves, at the default 4000 draws:

| | |
|---|---|
| best draw reproduces the ELE to | **0.01–0.20%** |
| draws within 1% | **134–260** per curve |
| best-draw SELE error, 0–1 µm | **0.1–0.4%** on SRV up to 1e6; **1.7–2.3%** on the lifetime curves and SRV 1e7 |
| best-draw SELE error, 1–3 µm | 0.3–4.0% on SRV up to 1e6; 5–12% on the lifetime curves and SRV 1e7 |
| truth inside the envelope, 0–3 µm | 100% on SRV up to 5e5; 20–38% on the rest |

The 1–3 µm error is shared by every draw that fits, so it is a bias, not ambiguity:
selecting differently does not remove it. It is model mismatch. The test-set truths are
Tamir Yeshurun's COMSOL simulations, which include photon recycling and angle-resolved,
multi-pass escape with back-surface reflection; the simulator here counts only light leaving
the front on a single straight pass. Light from deep carriers is mostly trapped or reabsorbed,
so the missing physics is what gives the truth its deeper, fuller tail, and nothing in the
five-parameter family can produce it. The inferred parameters show the compensation: on the
lifetime sweep the best draws put the doping at 0.5–2.4e18 cm^-3 and D at 7–22 cm^2/s against
the simulation's 1e19 and ~39 — stretching the diffusion length to fake the transport that
recycling provides. Treat the reported parameters as effective, not physical, until the
simulator carries recycling.

Separately, the test-set ELE on the dipped-surface curves was computed from SELE held constant
over the figures' coarse near-surface steps, which puts a ~1.7% floor under every method's ELE
fit on those curves. Loosening the tolerance does raise coverage (median 22% at 1%, 60%
at 5% over 0–3 µm), but only by admitting draws that no longer fit, which is why coverage is
not used to set it. `standalones/score_test_set.py` prints the per-zone errors and coverage;
`standalones/check_uncertainty.py` shows how both move with the tolerance.

## Quoting a number

For anything numerical, propagate to the quantity rather than reading it off the envelope. Its
edges are the extremes at each depth taken separately, so it loses the correlation between
depths — it cannot express that a draw with a high peak also puts that peak deeper — and
**peak depth cannot be read off it at all**.

`uncertainty.scalar_intervals` gives surface SELE, peak SELE, peak depth and mean SELE over
0–3 µm for the best draw, with their range over the draws that fit. Quote those.

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
- **The envelope is not an error bar on the truth.** Every draw in it fits the measurement, so
  a truth outside it means an error all of them share — a bias of the simulator or of the
  measurement — not ambiguity the data leaves open.
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
ELE misfit does the selecting, rather than the Gaussian being read as the answer.

## Open

1. **Score the classical methods on the same test set.** Nothing has scored
   `NON_UNIFORM_MESH` or `TOTAL_VARIATION` against `Data/test_set/`, so there is no baseline
   to say whether this beats Tikhonov.
2. **Noise-conditioned training**, if a spread tied to a stated precision is wanted.
3. **Add photon recycling and back-surface escape to the simulator** and retrain, so the
   family can reach the deep tail the COMSOL truths have. A physics change to agree on first.

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
| `.../uncertainty.py` | best draw, draws within tolerance, scalar intervals |
| `.../in_distribution.py` | is this measurement one the network was trained for |
| `.../plot_parametric_result.py` | the two-panel figure |
| `.../standalones/parametric_model_training.ipynb` | the Colab notebook that trains it |
| `.../standalones/run_inference.py` | reconstruct and plot; writes `results/parametric_model/` |
| `.../standalones/score_test_set.py` | accuracy vs ground truth, by depth zone |
| `.../standalones/check_uncertainty.py` | draws kept, coverage and envelope width, by tolerance |
