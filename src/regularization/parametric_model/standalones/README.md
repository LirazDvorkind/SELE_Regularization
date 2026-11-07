# Parametric posterior model — training and inference

A network that reads an ELE measurement and returns a Gaussian over the five simulator
parameters, with the full covariance between them. SELE curves come from running the
physics on samples from that Gaussian, not from a generative model.

## Files

- `parametric_model_training.ipynb` — Colab notebook, the one you run
- `train_parametric_model.py` — the same training run locally (CPU is fine)
- `run_inference.py` — reconstruct SELE from a measurement: best draw plus the draws that fit

## Workflow

### 1. Build the dataset (local, once)

```bash
python -m src.forward_model.standalones.generate_dataset --curves 100000 --seed 7 \n    --name parametric_100k
```

About 8 seconds. Writes two files to `Data/parametric_model/datasets/`. Only
`parametric_100k_train.npz` (29 MB) is needed for training; the `_sele.npz` companion is a
small sample of curves kept locally for sanity plots.

It is fast because the ELE integral is done in closed form rather than on a depth mesh --
see `src/forward_model/README.md`. That also means the measurement is computed at the
equivalent of a million mesh elements, so the training data carries no discretisation error
of its own.

This replaces `MATLAB SELE Simulation/create_training_set.m` for this model. That script
discards the parameter values that produced each curve, which are exactly what a parameter
inference network needs. The Python simulator it uses is checked against MATLAB by
`python -m src.forward_model.standalones.validate_matlab_port` and currently agrees to
4e-13 relative. `check_dataset.py` separately confirms the generated curves have the same
distribution as the MATLAB-made ones already in the repo, which is what would catch a
mis-transcribed sampling range that a per-curve comparison cannot.

### 2. Upload to Drive

Put these three files in one Google Drive folder, alongside each other:

| file | source |
|---|---|
| `parametric_100k_train.npz` | `Data/parametric_model/datasets/` |
| `model_definition.py` | `src/regularization/parametric_model/` |
| `features.py` | `src/regularization/parametric_model/` |

Both Python files are standalone, importing only numpy and torch, so nothing else from the
repo needs to come along.

### 3. Train on Colab

Open `parametric_model_training.ipynb`, set `DRIVE_DIR` in the configuration cell to that
folder, and run every cell. A GPU helps but is not required; the network is around 200k
parameters. Resume checkpoints are written every 25 epochs, so a disconnected session picks
up where it stopped.

**Read the coverage numbers, not only the loss.** Each evaluation prints the fraction of
held-out truths falling inside the predicted 68% and 95% ellipsoids, which should land near
68 and 95. Both low means the covariance is too tight, both high means too loose, and either
is fixed by rescaling.

Expect instead to see them miss in *opposite* directions -- around 75 and 93. No rescaling
fixes that, and it is the signature of an ellipsoid stretched over a solution set that is not
one. It is a property of the output head rather than a bad run, and it is why the reported
curve and spread come from scoring draws by data fit rather than from the network's
covariance directly.

### 4. Bring the checkpoint back

Download `parametric_posterior.pt` and put it in `Data/parametric_model/models/`.

### 5. Reconstruct

```bash
python -m src.regularization.parametric_model.standalones.run_inference --test-set
python -m src.regularization.parametric_model.standalones.run_inference --ele-sim
```

Figures land in `results/parametric_model/`. Around two seconds per curve at the default
4000 samples. Each run prints, in order:

1. **the input check** -- whether the measurement is the kind of curve the network was
   trained on. If this fails nothing below it means anything; see `in_distribution.py`.
2. **the posterior mean parameters**, and the ELE residual of the draws against the
   measurement.
3. **the best draw's parameters, the misfit distribution, and how many draws fit** within the
   tolerance, from `uncertainty.py`. If none does, a warning says so and by how much the best
   draw misses.
4. **scalars** for surface SELE, peak SELE, peak depth and mean SELE over 0-3 um: the best
   draw's value and the range over the draws that fit.

Quote the scalars rather than reading them off the envelope. Its edges are the extremes at
each depth taken separately, so it loses the correlation between depths -- it cannot express
that a draw with a high peak also puts that peak deeper -- and peak depth cannot be read off
it at all.

The residual is the number that matters. The whole method exists to produce a SELE profile
satisfying `ELE = G @ SELE`, and a profile that does not reproduce the measurement is not an
answer regardless of how the figure looks.

Test-set curves are fed their native-mesh `ele.csv`, never `ele_500.csv`. The latter was
built with the same 500-element operator the reconstruction uses, so using it would let the
method invert its own discretisation.

Score against the test set, not against `--ele-sim`. `Data/ELE_sim.csv` disagrees with its own
ground-truth profile pushed through the current G by up to 20%, because that file predates the
unification of the optical constants. Part of any residual there is that gap rather than the
reconstruction. The test-set pairs were built from one forward model and are self-consistent.

## The parameter box

`D` is sampled log-uniformly from 5 cm^2/s, below the floor of 50 that
`create_training_set.m` uses. `D` and `tau` reach the curve shape only through
`Ln = sqrt(D tau_eff)`, but `tau` also sets brightness on its own, so a floor on `D` caps how
bright a curve the family can produce at a given shape. At 50 that cap sits below what the
test set's lifetime sweep needs. With the floor at 5, all 17 test curves fit to under 0.7%.

`probe_parameter_bounds.py` re-derives this one bound at a time.

A checkpoint carries its `param_spec`, so one trained against a different box is detectable
rather than silently wrong.

## What the figure shows, and what it does not

The bold curve is the single draw that best reproduces the measurement -- a real simulator
curve with a parameter set behind it, never a depth-wise median, which would stitch different
draws together and correspond to no parameters. The light lines are every draw whose ELE is
within `uncertainty.TOLERANCE` (relative L2) of the measurement, with their envelope shaded.
`standalones/check_uncertainty.py` shows how that set changes with the tolerance.

Training data is clean: no measurement noise anywhere. So the spread of the light lines is
the ambiguity of the noiseless inverse problem at that tolerance -- how far the parameters can
move while still producing an ELE curve within it of the one given.

It is not an error bar on the truth. Every light line explains the measurement, so if the
truth lies outside all of them the error is one they share: a bias of the simulator, or of
the measurement itself, not ambiguity the data leaves open.

If a spread conditioned on a stated measurement precision is ever wanted, the change is small
and local: add relative noise to `ele` inside the training loop, resampled each epoch, and
the reported spread becomes "uncertainty given a measurement good to X percent". It would
require a retrain, not a redesign.

## Checkpoint format

```python
{
    'model_state_dict': ...,
    'config':      {...},                 # TrainingConfig, including input_mode and hidden_dims
    'input_stats': {'mean': ..., 'std': ...},   # how ELE was standardised
    'param_spec':  {'names', 'lower', 'upper', 'is_log'},
}
```

`input_stats` and `param_spec` travel with the weights on purpose. Inference must reproduce
the training transform exactly, and a mismatch there is silent — it looks like a badly
trained network rather than a bug.
