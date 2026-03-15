from dataclasses import dataclass, field
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[2] / "Data" / "score_model"


@dataclass
class NesterovHyperparams:
    """Hyperparameters of the solve_gradient_descent function in score_model_grad.py"""
    # REG_WEIGHT passed as argument acts as the "score factor" or regularization strength.
    # REG_WEIGHT too high = ignoring data, too low = ignoring physics (score model)
    REG_WEIGHT: float = 5.0

    # MOMENTUM = "mu" Nesterov momentum coefficient. Controls the "inertia" (how much past velocity is kept).
    # MOMENTUM high = plow through noise but may cause overshoot, too low = slower but less overshoot
    MOMENTUM: float = 0.9

    model_path: str = field(default_factory=lambda: str(_DATA_DIR / "sele_score_net_d500.pt"))  # Can be d32

    LR_MAX: float = 1e-2

    LR_MIN: float = 1e-5 # Cosine annealing from LR_MAX to LR_MIN

    MAX_STEPS: int = 5000

    # T0: Larger t means the model expects more noise, so it pushes toward coarser, blurrier features.
    #     Smaller t (like 1e-4) assumes the image is nearly clean, enforcing finer details.
    T0: float = 5e-2               # Small fixed time step to sample "clean" score from the diffusion score model

    # Plot debug plots
    IS_SHOW_DEBUG_PLOT: bool = False

    # Plot MSE plot
    IS_SHOW_MSE_PLOT: bool = True

    # Print useful debug information
    IS_SHOW_DEBUG_DATA: bool = True

    # Stop if error changes less than this
    STOP_CHANGE: float = 1e-8

    # How many steps the mse diff is less than the threshold
    STOP_STEPS: int = 20

    # Minimum steps to run before checking (to let momentum build)
    MIN_STEPS: int = 50

