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
7. Install all required Python packages: `pip install -r requirements.txt` or install from the IDE.
8. Download large files (model checkpoints, training data): `dvc pull`

## Large Files (DVC)

Model checkpoints and training data live in `Data/score_model/` and are managed by [DVC](https://dvc.org) (not included in the git repo). Run `dvc pull` after cloning to download them.

To add or update a large file:
```bash
dvc add Data/score_model/<file>
git add Data/score_model/<file>.dvc Data/score_model/.gitignore
dvc push
```

## Regularization Modes

Set via `CONFIG.regularization_method` in `src/types/config.py`:

| Mode | Description |
|------|-------------|
| `NON_UNIFORM_MESH` | Tikhonov with adaptive near-surface mesh |
| `TOTAL_VARIATION` | Two-parameter Tikhonov, solved with CVXPY |
| `MODEL_SCORE_GRAD` | Nesterov gradient descent with score model prior |

## Score Model Standalones

Scripts in `src/regularization/score_model/standalones/` are run independently (not as part of the main pipeline). Run them as a module from the project root, e.g.:

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground
```
