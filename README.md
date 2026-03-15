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

Model checkpoints and training data live in `Data/score_model/` and are managed by [DVC](https://dvc.org) (not included in the git repo). Run `dvc pull` after cloning to download them.

To add or update a large file:

```bash
dvc add Data/score_model/<file>
git add Data/score_model/<file>.dvc Data/score_model/.gitignore
dvc push
```

> **Important:** Always use the DVC CLI for any operation on tracked files. Never manually rename, move, or delete `.dvc` files — they contain content hashes that DVC uses to locate data in the remote cache.

To remove a large file:

```bash
dvc remove Data/score_model/<file>.dvc
git add Data/score_model/<file>.dvc Data/score_model/.gitignore
```

To rename a large file:

```bash
dvc move Data/score_model/<old> Data/score_model/<new>
git add Data/score_model/<old>.dvc Data/score_model/<new>.dvc Data/score_model/.gitignore
dvc push
```

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
