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

## Regularization Modes

Set via `CONFIG.regularization_method` in `src/types/config.py`:

| Mode               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `NON_UNIFORM_MESH` | Tikhonov with adaptive near-surface mesh         |
| `TOTAL_VARIATION`  | Two-parameter Tikhonov, solved with CVXPY        |
| `MODEL_SCORE_GRAD` | Nesterov gradient descent with score model prior |

## Score Model Standalones

Scripts in `src/regularization/score_model/standalones/` are run independently (not as part of the main pipeline). Run them as a module from the project root, e.g.:

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground
```
