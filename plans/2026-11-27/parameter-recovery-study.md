# 2026-11-27 — Parameter recovery of the parametric model

## Meeting with Alon

- Discussed the parametric model's results on the 5-parameter synthetic (forward-model) test set.
- The model is given an ELE and outputs a distribution over the five simulator parameters
  (`p0`, `D`, `S`, `tau`, `alpha_scale`). The ELE was matched, but the parameters were not.
- The training loss is on the parameters themselves (Gaussian NLL), so the model does try to
  match them.
- Open question: how much is the model failing, and how much is the ill-posedness of the
  problem (several parameter sets giving the same ELE)?
- Proposed next step: retrain so the model can return several ranges of 5-parameter sets,
  so it can give a correct answer when more than one set produces the same ELE.

## Discussion

- **Current output.** One Gaussian with full covariance (mean plus Cholesky factor) over the
  normalised parameters. Its coverage, about 75% / 93% against 68% / 95%, misses in opposite
  directions. That is the signature of one ellipsoid stretched over a solution set that isn't
  ellipsoid-shaped (a curved ridge, or several islands; not yet known which).
- **Retrain options considered:** a mixture of Gaussians (MDN), a normalizing flow, and K point
  hypotheses (winner-takes-all). A mixture is the smallest step, and its components map
  directly onto "ranges". None of them can make the parameters identifiable. They can only
  report the set of answers honestly, so success must be judged as "truth inside one of the
  ranges", not "best guess equals truth".
- **Decided first:** before any retrain, measure how well the *current* model recovers the
  parameters, in a form we can show collaborators.
- **Metric.** A plain "within 1σ / 2σ" test measures whether the error bars are honest, not
  whether the answer is good. An "anywhere in the box" answer always passes. So the
  distance in σ is paired with the size of σ. σ is the spread (in decades) of the draws that
  reproduce the ELE within 1%, not the network's own σ, which is known to be mis-sized.
  Per parameter, per curve:
  - **recovered**: best draw within 2σ of the truth, and ±2σ spans under half the box;
  - **not determined**: within 2σ, but ±2σ spans half the box or more (the ELE doesn't fix it,
    and the method says so);
  - **wrong**: more than 2σ off (confident and wrong).
- **Considered and deferred:** a model-free search for every parameter set that reproduces
  each ELE, to compare what the model said against what it could have said. It is the cleaner
  test of "model vs problem", but it costs more and depends more on judgement. Kept as a
  possible later step. ELE tolerance fixed at 1% only.

## Done

- `src/regularization/parametric_model/standalones/parameter_recovery_study.py`. It draws 400
  fresh simulator curves (seed 20261127, distinct from training and the test set), runs the
  current checkpoint on each, classifies every parameter, prints a table with binomial errors
  and the 1σ/2σ calibration, and writes figures. `--replot` redraws from the cached results.
- `random_parameters()` in `src/test_set/build_forward_model_set.py` now takes `seed` and `n`.
  The forward-model test set rebuilds unchanged.
- Entry added to the standalones README.
- Output in `results/parametric_model/parameter_recovery/`: `summary.png`,
  `example_recovered.png`, `example_not_determined.png`, `example_wrong.png`, `run_log.txt`,
  `results.npz`.
- Not committed yet.

## Findings (400 curves; each figure ±2%)

Every curve's ELE was reproduced: best-draw misfit 0.001–0.24%, median 0.024%.

| | recovered | not determined | wrong | within 1σ | within 2σ |
|---|---|---|---|---|---|
| p0 | 49% | 26% | 25% | 46% | 75% |
| D | 28% | 48% | 24% | 43% | 76% |
| S | 76% | 1% | 23% | 47% | 77% |
| tau | 3% | 75% | 22% | 46% | 78% |
| alpha_scale | 60% | 16% | 24% | 45% | 76% |
| Ln = √(D·τ_eff) | 71% | 0% | 29% | 43% | 71% |
| S/D | 87% | 0% | 12% | 66% | 88% |

- **The ELE determines combinations, not the raw five.** τ alone is almost never fixed, and D
  rarely, but the diffusion length is recovered 71% of the time. S/D (87%) beats S or D
  alone. This is a clear trend, so the combinations qualify for the figure.
- **"Wrong" is a near-uniform 22–25% across parameters,** and calibration is short (about 45%
  within 1σ, about 76% within 2σ). The draws that fit are too narrow. Hypothesis, not
  confirmed: the single-Gaussian network never proposes the whole ridge of matching parameter
  sets, so its draws undersample it. That is the shortfall the mixture retrain targets.
- The example figures show it directly: ELE curves on top of each other while p0, D, τ and α
  span most of the box (not determined), or the ELE matching to 0.08% with four parameters more
  than 3σ off (wrong).

## Next

1. Add Ln and S/D as bars in `summary.png` (redraw from the cache).
2. Check whether the ~24% "wrong" falls on the same curves or is spread across them.
3. Retrain with a mixture-of-Gaussians head (configurable K, where K = 1 reproduces today's
   model), then rerun this study against the same seed to compare.
4. Possibly later: the model-free solution-set search.
