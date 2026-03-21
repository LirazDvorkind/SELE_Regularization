from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from src.types.enums import RegularizationMethod, LFlag

_SCORE_MODEL_DIR = Path(__file__).resolve().parents[2] / "Data" / "score_model"


@dataclass
class DataPaths:
    # eta_ext="Data/ELE.csv", # path to Tamir's eta_ext (ηₑₓₜ) CSV
    # G="Data/G_matrix.csv", # path to Tamir's optical‑generation matrix

    # path to Tamir's depth vector
    z: str

    # Path to extinction coefficient k(λ), from MATLAB code
    k: str

    # Wavelengths [nm] involved in alpha (optical constants) calculations, in MATLAB it is called n_k_wavelength
    # Reminder α(λ) = 4πk(λ)/λ [cm⁻¹]
    lambda_for_alpha: str

    # eta_ext (J) CSV, this is the experiment simulation results, in MATLAB it is called PLQY_sim
    eta_ext: str

    # the SELE z [cm] mesh, in MATLAB it is called SELE_z_interp
    z_gt: str

    # the real SELE calculated using deltas in simulations, this is the result we are looking for,
    # in MATLAB it is called SELE_interp_for_G0
    sele_gt: str

    # The wavelengths [nm] of eta_ext, in MATLAB it is called wavelength_back,
    wavelengths: str

    # The path to csv of machine learned score curve (L function)
    score_model_curve: str


@dataclass
class NonUniformMeshConfig:
    """For when regularization method is NON_UNIFORM_MESH"""
    # (z_max, z_min) — depth range of the device [cm]
    z_range: Tuple[float, float]

    # Depth at which the mesh transitions from linear to exponential spacing [cm].
    # Below z_turn: uniform fine spacing (lin_mesh_size).
    # Above z_turn: exponentially coarser spacing (base exp_base).
    # Lower z_turn → more fine-resolution budget near the surface.
    z_turn: float

    # Element size in the linear (near-surface) region [cm].
    # Smaller → finer near-surface resolution, more elements, slower solve.
    lin_mesh_size: float

    # Base of the exponential stretch in the deep region.
    # Larger → elements grow faster with depth → fewer deep elements, coarser deep coverage.
    exp_base: float

@dataclass
class TotalVariationTemplateConfig:
    """For when regularization method is TOTAL_VARIATION_TEMPLATE"""
    W: float           # Device width [cm]
    mesh_resolution: int  # Spatial elements in the discretisation; more → finer resolution, slower solve

    # Sweep range for κ₂ [max, min], the weight on the template term ||S - S_model||.
    # κ₂ controls how strongly the solution is pulled toward the score-model template.
    # Too high → solution matches template regardless of data; too low → template is ignored.
    kappa2_range: Tuple[float, float]

    # Number of κ₂ samples (log-spaced between kappa2_range).
    # More samples → finer L-surface resolution, slower sweep.
    n_kappa2: int

@dataclass
class ModelScoreGradConfig:
    """
    All configuration for the MODEL_SCORE_GRAD regularization method.

    Use SCORE_MODEL_PRESETS["d32"] or SCORE_MODEL_PRESETS["d500"] to get a
    ready-made config, then use dataclasses.replace() for one-off overrides:

        from dataclasses import replace
        cfg = replace(SCORE_MODEL_PRESETS["d500"], REG_WEIGHT=5.0, LR_MAX=5e-3)
    """
    # --- Score model checkpoint ---
    # Path to .pt file; the checkpoint's 'target_length' determines expected G column count.
    model_path: str

    # --- Mesh / pipeline settings ---
    W: float               # Device width [cm]
    output_mesh_resolution: int  # Points in the final upsampled SELE output

    # --- Score factor / regularization strength ---
    # Balances data fidelity vs score prior.  Adaptive weighting normalises gradient
    # magnitudes first, so the same value works similarly for d32 and d500.
    # Too high → ignores measurement; too low → ignores learned prior.
    REG_WEIGHT: float

    # --- Nesterov momentum ---
    # Inertia in the NAG update: v = mu*v - lr*grad.
    # Higher → faster convergence but prone to overshoot, especially in high-dim spaces.
    MOMENTUM: float

    # --- Learning rate schedule ---
    # Cosine-annealed from LR_MAX → LR_MIN over MAX_STEPS iterations.
    # Larger models produce larger score gradients, so LR_MAX must shrink accordingly
    # (d500 uses ~3× smaller LR_MAX than d32).
    LR_MAX: float
    LR_MIN: float

    MAX_STEPS: int  # Total optimisation budget (hard cap; early-stopping may fire sooner)

    # --- Diffusion time T0 ---
    # Fixed noise level at which the score network is queried.
    # Larger T → coarser, blurrier prior; smaller T → sharper, more detail-sensitive.
    # d500 uses sinusoidal time embedding and is more sensitive to this than d32.
    T0: float

    # --- Early stopping ---
    # Stops when |MSE[i] - MSE[i-1]| < STOP_CHANGE for STOP_STEPS consecutive steps.
    STOP_CHANGE: float = 1e-8
    STOP_STEPS: int = 20
    MIN_STEPS: int = 50  # Never stop before this many steps, regardless of convergence

    # --- Display flags ---
    IS_SHOW_DEBUG_PLOT: bool = False
    IS_SHOW_MSE_PLOT: bool = True
    IS_SHOW_DEBUG_DATA: bool = True


# ---------------------------------------------------------------------------
# Named presets — one per trained model checkpoint.
# Pick the preset matching your .pt file, then use dataclasses.replace() for
# any per-experiment tweaks.
# ---------------------------------------------------------------------------
SCORE_MODEL_PRESETS: dict[str, "ModelScoreGradConfig"] = {
    # 32-point model: small MLP, fast, good for quick iteration.
    "d32": ModelScoreGradConfig(
        model_path=str(_SCORE_MODEL_DIR / "models" / "sele_score_net_d32.pt"),
        W=30e-4,
        output_mesh_resolution=500,
        REG_WEIGHT=1.0,
        MOMENTUM=0.85,
        LR_MAX=1e-2,
        LR_MIN=1e-5,
        MAX_STEPS=5000,
        T0=1e-1,
    ),
    # 500-point model: residual MLP + LayerNorm + sinusoidal time embedding.
    "d500": ModelScoreGradConfig(
        model_path=str(_SCORE_MODEL_DIR / "models" / "sele_score_net_d500.pt"),
        W=30e-4,
        output_mesh_resolution=500,
        REG_WEIGHT=0.01,
        MOMENTUM=0.9,
        LR_MAX=5e-4,
        LR_MIN=1e-7,
        MAX_STEPS=5000,
        T0=5e-2,
    ),
}


@dataclass
class Config:
    data_paths: DataPaths

    non_uniform_mesh_config: NonUniformMeshConfig
    total_variation_template_config: TotalVariationTemplateConfig
    model_score_grad_config: ModelScoreGradConfig

    # Which regularization mode to run — selects the solver branch in pipeline.py.
    regularization_method: RegularizationMethod

    # Regularisation operator applied to S in the Tikhonov term κ²||LS||².
    # L0 = identity (penalises amplitude), L1 = first-diff (penalises slope),
    # L2 = second-diff (penalises curvature). L2 is the usual smoothness prior.
    # Only used by NON_UNIFORM_MESH and TOTAL_VARIATION modes.
    L_flag: LFlag

    # κ sweep range (κ_max, κ_min), log-spaced.
    # The L-curve knee is searched within this range — if the knee sits at either
    # extreme, widen the range in that direction.
    kappa_range: Tuple[float, float]

    # Number of κ samples (log-spaced). More → smoother L-curve, slower sweep.
    # 40–100 is usually sufficient; increase if the knee is hard to localise.
    n_kappa: int

    # Confidence window half-width around κ_knee (multiplicative).
    # S_mean and S_std are computed over κ ∈ [κ_knee / conf_window, κ_knee × conf_window].
    # Larger → wider average, smoother but potentially biased estimate.
    conf_window: float

    e_charge: float       # Electron charge [C] — physical constant, don't change
    photon_flux: float    # Illumination intensity [photons cm⁻² s⁻¹] — must match simulation

    W: float              # Device width [cm] — used for mesh construction

    # If True, pins the last element of the recovered SELE to zero (no emission at the back contact).
    # Physically motivated for most devices; disable if back-surface emission is expected.
    force_SELE_last_zero: bool

    is_save_plots: bool   # Write PNGs to results/
