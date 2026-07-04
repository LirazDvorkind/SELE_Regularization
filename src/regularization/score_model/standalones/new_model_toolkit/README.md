# New Model Toolkit

Test and tune a freshly trained SELE score model in one place.

## Inputs

Set both at the top of `main.py` — just the filename, resolved under
`Data/score_model/models/` and `Data/score_model/datasets/` respectively:

- `MODEL_PATH` — the new `.pt` checkpoint's filename.
- `CURVES_PATH` — filename of a small (~100-curve) `.csv` example from the dataset the
  model was trained on.

## What it does

Running `main.py` reads `target_length` from the checkpoint (auto-selects the `d32`/`d500`
preset) and then runs:

- **`test_diffusion_generation.py`** — unconditionally generates SELE curves via reverse
  diffusion and plots them (a visual "does this model produce plausible shapes?" check).
- **`auto_tune_hyperparameters.py`** — recommends `LR_MAX` / `LR_MIN` / `REG_WEIGHT` and
  prints a paste-ready snippet.

Run it (or either script standalone):

```bash
python -m src.regularization.score_model.standalones.new_model_toolkit.main
```

## Full new-model workflow

**a. Generate the data (MATLAB).** In `MATLAB SELE Simulation/create_training_set.m`, set
`n_samples` and the output extension, and run it twice:
- `n_samples = 1e5`, `.mat` extension → the 100k-curve training corpus.
- `n_samples = 1e2`, `.csv` extension → the example curves for this toolkit.

The filename is auto-built as `sele_simulated_<N>_curves_<x_res>_long.<ext>`. Note the
training script expects the `.mat` named `..._100000_curves_...` under
`Data/score_model/datasets/`.

**b. Train (Colab).** See [`../model_training/README.md`](../model_training/README.md) →
the up-to-date notebook lives in the Google Drive "Thesis" folder.

**c. Copy over + DVC.** Put the `.pt` in `Data/score_model/models/` and the example `.csv`
in `Data/score_model/datasets/`, then track them with DVC — see the "Large Files (DVC)"
section of the [root README](../../../../../README.md) (`dvc add Data/score_model/models`,
`git add` the `.dvc`, `dvc push`).

**d. Run the toolkit.** Set `MODEL_PATH` / `CURVES_PATH` in `main.py`, run it, read the
recommendation and inspect the generated-curve plots.

**e. Plug the values in.** Paste the recommended `LR_MAX` / `LR_MIN` / `REG_WEIGHT` (and the
new `model_path`) into the matching `d32` / `d500` entry of `SCORE_MODEL_PRESETS` in
`src/types/config.py`. The pipeline picks it up via `CONFIG.model_score_grad_config`.
