# Score-model standalones

Scripts for the score model that run on their own, outside the main pipeline. Run them as a
module from the repo root, or as a PyCharm "Current File" configuration.

| Script | Purpose |
| ------ | ------- |
| `hyperparameter_playground.py` | Quick test of one config on a random synthetic curve (prints MSE_ELE/MSE_SELE). |
| `hyperparameter_playground_sweep.py` | The playground over a small sweep of settings. |
| `tune_hyperparameters.py` | Brute-force grid search over LR_MAX/REG_WEIGHT/MOMENTUM/T0 (thousands of full solves). |
| `sele_w_score_optimization_example.py` | Alon's original score-model example, kept as written; run as Current File. |
| `Results/tuning/` | Saved output of past tuning runs. |
| `model_training/` | Training scripts and the Colab notebook for the score network. |
| `new_model_toolkit/` | Tests and tunes a freshly trained checkpoint in one run. |

```bash
python -m src.regularization.score_model.standalones.hyperparameter_playground   # one config, one curve
python -m src.regularization.score_model.standalones.tune_hyperparameters        # grid search
python -m src.regularization.score_model.standalones.new_model_toolkit.main      # new checkpoint
```
