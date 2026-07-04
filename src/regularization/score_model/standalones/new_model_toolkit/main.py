"""Entry point for testing and tuning a freshly trained score model.

Set MODEL_PATH / CURVES_PATH below, then run:

    python -m src.regularization.score_model.standalones.new_model_toolkit.main

It generates SELE curves from the model (visual sanity check) and prints a
paste-ready LR_MAX/LR_MIN/REG_WEIGHT recommendation. See README.md for the full
new-model workflow.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from src.regularization.score_model.score_model_grad import load_score_model
from src.regularization.score_model.standalones.new_model_toolkit import (
    auto_tune_hyperparameters as autotune,
)
from src.regularization.score_model.standalones.new_model_toolkit import (
    test_diffusion_generation as diffusion,
)

_DATA_DIR = Path(__file__).resolve().parents[5] / "Data" / "score_model"

# ===========================================================================
# INPUTS -- just the filename; resolved under Data/score_model/{models,datasets}
# ===========================================================================
MODEL_PATH: str = "sele_score_net_d500.pt"   # new .pt checkpoint, e.g. "sele_score_net_d500.pt"
CURVES_PATH: str = "sele_simulated_100_curves_500_long.csv"  # ~100-curve .csv from the dataset it was trained on, e.g.
                        # "sele_simulated_100_curves_500_long.csv"


def main() -> None:
    if not MODEL_PATH or not CURVES_PATH:
        raise SystemExit("Set MODEL_PATH and CURVES_PATH at the top of main.py first.")

    model_path = str(_DATA_DIR / "models" / MODEL_PATH)
    curves_path = str(_DATA_DIR / "datasets" / CURVES_PATH)

    _net, _d_min, _d_max, target_length = load_score_model(model_path)
    preset_name = "d500" if target_length == 500 else "d32"
    print(f"Model target_length={target_length} -> preset '{preset_name}'")

    diffusion.run(model_path, "New model")
    autotune.run(model_path=model_path, dataset_path=curves_path, preset_name=preset_name)

    plt.show(block=True)


if __name__ == "__main__":
    main()
