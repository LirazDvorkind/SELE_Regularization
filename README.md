# SELE Regularization

Reconstructs Spatial External Luminescence Efficiency (SELE) profiles from ELE measurements using regularized inverse problems. Implements three solvers: Tikhonov with non-uniform mesh, Total Variation, and score-based diffusion model gradient descent.

## Setup

1. Install Git Bash.
2. Install PyCharm IDE.
3. Install the latest version of Python.
4. Set up a Python interpreter in PyCharm.
5. Clone the project using `git clone`.
6. Create a new run configuration in PyCharm:
   1. Select Module and write `src.main`.
   2. Select the project directory as the source directory (i.e. the folder one above `src`).
7. Install dependencies (creates `.venv` automatically):
   ```bash
   uv sync
   ```
   Then point PyCharm's interpreter at `.venv/Scripts/python.exe`.
   > **Note:** Install `uv` first if needed: `winget install --id=astral-sh.uv` or `pip install uv`. See [uv docs](https://docs.astral.sh/uv/).
8. On each new machine, run the following command to add the DVC authentication client secret locally (the full command with the secret is saved in the DVC Google Drive folder `My Drive/Thesis/DVC`):
   ```bash
   dvc remote modify --local myremote gdrive_client_secret <secret>
   ```
9. Download large files (model checkpoints, training data): `dvc pull`

## Large Files (DVC)

Model checkpoints and training datasets are managed by [DVC](https://dvc.org) (so they are not included in the git repo). Run `dvc pull` after cloning to download them.

| Directory                    | Tracked by     | Contents                 |
| ---------------------------- | -------------- | ------------------------ |
| `Data/score_model/models/`   | `models.dvc`   | `.pt` model checkpoints  |
| `Data/score_model/datasets/` | `datasets.dvc` | `.mat` training datasets |

The ground-truth test set needs neither git-LFS nor DVC: its one oversized matrix (the
reference profile's 28×100000 G) is not stored at all but recomputed from the optical
constants on demand. See [`Data/test_set/README.md`](Data/test_set/README.md).

To add a new file to a tracked directory and push it:

```bash
# Drop the new file into the appropriate directory, then:
dvc add Data/score_model/models      # or datasets
git add Data/score_model/models.dvc  # or datasets.dvc
git commit -m "Add new model checkpoint"
dvc push
```

To replace an existing file (e.g. after manually swapping a checkpoint):

```bash
dvc add Data/score_model/models      # re-hashes the whole directory
git add Data/score_model/models.dvc
git commit -m "Update model checkpoint"
dvc push
```

> **Important:** Always re-run `dvc add <dir>` after manually placing files so DVC updates its content hash. Never edit `.dvc` files by hand.

> **Note:** DVC is content-addressed, so replacing a file (even with the same name) uploads a new blob and leaves the old one orphaned in the remote. To purge unreferenced blobs from Google Drive, run `dvc gc --cloud -w` — only do this when you're sure no other branch/commit still references the old hash.

The project saves whole directories but it is also possible to dvc individual files.

### Authentication with Google Drive

To successfully give access to DVC to Google Drive I created a [Google Cloud Project](https://console.cloud.google.com/welcome?project=thesis-dvc-project-490306) and followed the instructions
[here](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) to create a Client and OAuth.

## Optical Constants

`src/optical_constants.py` is the project's single source of optics. It reads the paper's
ellipsometry constants — n and k, **with and without the Drude term** — straight from
`Data/Tamir_paper_SELE_figs/Optical constants GaAs.fig`, no MATLAB needed.

Both are needed. Free-carrier absorption attenuates the beam but frees no carrier, so
attenuation follows `alpha` (with Drude) while generation follows `alpha_b` (without), and
G carries the prefactor `alpha_b / alpha`. A row of G therefore sums to `alpha_b/alpha`
(0.976–0.998 over the measurement band), not to 1 — it is a *generation* matrix, not an
absorption matrix.

Every G in the project comes from here: the three solvers, the precomputed
`Data/score_model/G_score_model*.csv`, and the ground-truth test set all build bit-identical
operators. There is no longer a `Data/k.csv`.

## Regularization Modes

Set via `CONFIG.regularization_method` in `src/types/config.py`:

| Mode               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `NON_UNIFORM_MESH` | Tikhonov with adaptive near-surface mesh         |
| `TOTAL_VARIATION`  | Two-parameter Tikhonov, solved with CVXPY        |
| `MODEL_SCORE_GRAD` | Nesterov gradient descent with score model prior |

## Ground-truth Test Set

`Data/test_set/` holds 18 ground-truth SELE profiles — an SRV sweep and an SRH-lifetime
sweep extracted from the paper's MATLAB figures, plus the original `SELE_ground_truth.csv` —
each with the photogeneration matrix and ELE that go with it. It exists so a method can be
scored across a range of profile shapes instead of at a single operating point.

```bash
python -m src.test_set.build_test_set   # re-extract from the figures
python -m src.test_set.plot_test_set    # overview of both sweeps
```

```python
from src.test_set.loader import load_test_set, load_native_G, load_on_solver_mesh
```

The set and the solver build G from the same optics (`src/optical_constants.py`), so the
operators are bit-identical. See [`Data/test_set/README.md`](Data/test_set/README.md) for the
layout and the caveats that matter when scoring reconstructions;
[`Data/Tamir_paper_SELE_figs/README.md`](Data/Tamir_paper_SELE_figs/README.md) for what the
source figures contain.

### Where this stands

The set is built and verified; **nothing runs it through a solver yet**. A benchmark that
loops the 18 curves through `solve_gradient_descent` and reports per-curve error is the
obvious next step, and is deliberately absent rather than overlooked. `loader.py` is the seam
for it.

Two things a benchmark author needs to decide, both explained in
[`Data/test_set/README.md`](Data/test_set/README.md): which ELE counts as the observation
(the native-mesh one is honest but carries a ~2% bias no solver can fit away; the solver-mesh
one is consistent but an inverse crime), and whether to score the first mesh element at all
(a 500-point mesh cannot represent the surface value, which is exactly where SRV is read).

Also unbuilt and deliberately so: **no noise variants.** The set is noiseless (SELE, ELE)
pairs by design.

> **Note on existing results:** `results/` predates the optics change below and was produced
> with the old absorption-only G. Re-run before comparing anything in it against new output.

## Score Model Standalones

Scripts in `src/regularization/score_model/standalones/` are run independently (not as part of the main pipeline). Run them as a module from the project root, e.g.:

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground
```

| Script | Purpose |
| ------ | ------- |
| `hyperparameter_playground.py` | Quick test of one config on a random synthetic curve (prints MSE_ELE/MSE_SELE). |
| `tune_hyperparameters.py` | Brute-force grid search over LR_MAX/REG_WEIGHT/MOMENTUM/T0 (thousands of full solves). |
| `new_model_toolkit/` | One-stop toolkit for a freshly trained checkpoint: generates SELE curves from the model and recommends LR_MAX/REG_WEIGHT. Set `MODEL_PATH`/`CURVES_PATH` in its `main.py` (see its README for the full new-model workflow), then run: |

```bash
python -m src.regularization.score_model.standalones.new_model_toolkit.main
```

The tuner and diffusion-generation utilities also run standalone as
`...new_model_toolkit.auto_tune_hyperparameters` / `...new_model_toolkit.test_diffusion_generation`.
