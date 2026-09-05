# Is SELE reconstruction from ELE actually feasible? — current assessment

Prompted by a conversation with Guy Ohayon (diffusion researcher, guyohayon.com), who argued
that the score-model reconstruction produces a MAP estimate dominated by the prior, and
attributed this to "concentration of measure in high dimensions" — a phenomenon where, in a
genuinely high-dimensional space, samples of a distribution cluster at almost the same
distance from its mean, so a prior-regularized estimate collapses toward that mean regardless
of the data. He also suggested PMRF (Posterior-Mean Rectified Flow) as a fix.

This was investigated by reading this repo's own generator code and data, running the actual
numbers, and comparing against current literature. See `score-model-status.md` for the
parallel, more implementation-level debugging thread on the diffusion sampler itself — item 1
of that file's open-items list is a direct consequence of the finding here.

## Verdict

**Guy's named mechanism does not literally apply, but his underlying concern is correct for a
different, verified reason.**

- The SELE prior is **not high-dimensional**. `MATLAB SELE Simulation/create_training_set.m`
  draws exactly 5 physical scalars per curve (doping, diffusion length, surface recombination
  velocity, SRH lifetime, an absorption scale) through one deterministic physics solver. PCA
  on the 1000 resulting training curves confirms this statistically: 5 components explain
  99.9998% of the variance, and a participation-ratio dimension estimate comes out to 1.13 —
  close to 1-dimensional in practice, nothing like the setting concentration-of-measure
  describes.
- The real problem is that **the measurement operator G is nearly blind to the directions the
  prior actually varies in.** `G` (28 wavelengths × 500 depth points, built from Beer-Lambert
  absorption) has numerical rank ~4 at realistic noise levels and condition number ~1e16 —
  light is absorbed almost entirely within the first few micrometers, so deep SELE contributes
  almost nothing to any measured wavelength. Projecting the prior's PCA directions onto G's
  row space shows only ~36% of the prior's total variance lies anywhere in G's row space *even
  with zero noise*, and only ~5.2% lies within the noise-surviving rank-4 subspace.
- Put together: an exact linear-Gaussian Bayesian calculation (`C_post = C - CGᵀ(GCGᵀ+σ²I)⁻¹GC`)
  says the measurement resolves about **2 effective degrees of freedom**, against 5 true
  physical parameters. This was cross-checked with a fully nonparametric method (self-
  normalized importance sampling over the empirical prior, no Gaussianity assumed) and a KDE-
  based posterior in 6D PCA space with measured calibration (95% credible intervals covered
  truth 96.7% of the time at 3% assumed noise, 88.2% at 1%) — two independent methods agree.

## Depth-resolution boundary (quantitative, not just qualitative)

At 1% assumed measurement noise, posterior-sd / prior-sd by depth:

| depth | ratio | reading |
|---|---|---|
| 0–3 µm | 0.01–0.09 | trustworthy — genuinely informed by the measurement |
| 3–8 µm | 0.1–0.2 | marginal |
| 8–30 µm | 0.35–0.6+ | prior-dominated — the output there is mostly the prior, not the data |

Cross-referenced against the real 18-curve ground-truth test set (`Data/test_set/index.csv`):
all 18 curves peak within 0.005–5.9 µm, i.e. mostly inside the trustworthy region. This is the
encouraging part of the finding — the physically interesting region and the measurable region
substantially overlap, even though the full 30 µm profile does not.

## Which physical quantities survive the measurement

- **Surface-to-peak ratio (SRV proxy)** — well recoverable (posterior sd ≈ a few % of prior sd).
- **Peak height / peak position** — moderately recoverable, degrades fast as noise rises.
- **Integrated area under the curve** — essentially not recoverable; dominated by the deep tail
  the measurement cannot see. A flat, unweighted SELE-error benchmark would be dominated by
  exactly this unmeasurable region.

## Implementation issues found in `score_model_grad.py`, independent of the above

- `REG_WEIGHT` is cosine-annealed to exactly 0 over the run, so the solver always terminates as
  unregularized least-squares against the rank-4/condition-1e16 operator above — whatever prior
  information was used early is discarded by the final step.
- The per-step adaptive rescaling (`grad_norm_mag / score_mag`) means the update direction isn't
  the gradient of any fixed objective.
- Nothing currently runs the 18-curve ground-truth test set through any solver end-to-end
  (confirmed directly in `README.md`) — there is no current measurement of reconstruction error
  to validate any of this against.

## Recommendations, ranked

1. **Replace the 500-dim score model with a ~6-dim KDE-based posterior over PCA coefficients.**
   Matches the demonstrated true dimensionality, requires no training, and its calibration is
   directly checkable (see numbers above) — unlike a diffusion model's behavior, which is
   opaque without exactly this kind of test.
2. **Consider full simulation-based inference (SBI / neural posterior estimation) directly on
   the 5 physical parameters**, bypassing curve-space entirely, using `calc_Sp2.m` as the
   simulator. Bigger lift (needs the MATLAB forward physics callable from the inference loop),
   but matches the true generative structure exactly and sidesteps curve-space non-identifiability
   altogether.
3. **If a generative/diffusion-flavored solver is kept, report a posterior mean + credible band
   per depth, not a single point estimate** (DPS-style), rather than adopting PMRF. PMRF's
   second stage spends error budget (exactly 2x MMSE, per Blau & Michaeli 2018) buying
   photorealism — appropriate for face restoration, not for a physically-interpretable curve
   where "looks like a plausible sample" isn't the deliverable and a slightly-smooth-but-correct
   mean is preferable to a sample.
4. **Fix the three `score_model_grad.py` issues above, and benchmark against the test set with
   a depth-banded error metric**, not flat MSE — since most SELE *area* lives in the
   unmeasurable deep region, an unweighted metric rewards prior-hugging there and penalizes
   genuinely-informed near-surface reconstructions.

## Assumptions and what's still unverified

- **No measured instrument noise floor exists anywhere in this repo.** Every noise percentage
  used above (0.05–10%, swept) is an assumption. Obtaining a real measured value is the single
  highest-priority open input — it would collapse most of the remaining uncertainty in exactly
  where the feasible/marginal/prior-dominated boundary sits.
- The closed-form linear-Gaussian posterior assumes Gaussian statistics; the PCA coefficients
  are measurably non-Gaussian (checked directly — skew and excess kurtosis are non-trivial).
  The nonparametric KDE-based posterior corroborates the same qualitative picture, but its
  0.3%-noise calibration row is unreliable due to effective-sample-size collapse and should not
  be quoted.
- The actual `MODEL_SCORE_GRAD` solver has never been run end-to-end on the 18-curve test set —
  the critique here is from code reading plus the linear/KDE analyses, not measured
  reconstruction error from the real pipeline.

## Reproducible artifacts

- Full write-up with figures and sources: published as a Claude artifact,
  https://claude.ai/code/artifact/bbeb6181-e00f-4126-827f-39e7497201a6
- **`src/regularization/score_model/standalones/feasibility_demo.py`** — an interactive,
  beginner-friendly version of every claim above, computed live from this repo's real G matrix
  and training dataset (no numbers pasted in from analysis scripts). Seven tabs: a plain-
  language intro plus one tab per claim (prior dimensionality, G's rank, prior/G overlap,
  posterior-vs-depth, recoverable physical quantities, and a no-formulas empirical check), with
  live sliders for assumed noise % and PCA component count. Run with:
  `python -m src.regularization.score_model.standalones.feasibility_demo`
