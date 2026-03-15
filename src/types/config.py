from dataclasses import dataclass
from typing import Tuple

from src.types.enums import RegularizationMethod, LFlag
from src.types.score_model_params import NesterovHyperparams


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
    # (z_max, z_min)
    z_range: Tuple[float, float]

    # cm: turning depth (linear → exponential mesh)
    z_turn: float

    # cm: linear-part element size (distance between linear mesh elements)
    lin_mesh_size: float

    # base for exponential spacing
    exp_base: float

@dataclass
class TotalVariationTemplateConfig:
    """For when regularization method is TOTAL_VARIATION_TEMPLATE"""
    W: float # Device width in cm
    points_amount: int

    # (κ2_max, κ2_min) - the kappa of the model regularization term, ||S-S_model||
    kappa2_range: Tuple[float, float]

    # Number of κ2 samples
    n_kappa2: int

@dataclass
class ModelScoreGradConfig:
    """For when regularization method is MODEL_SCORE_GRAD"""
    W: float # Device width in cm
    points_amount: int
    longer_points_amount: int
    num_steps: int


@dataclass
class Config:
    data_paths: DataPaths

    non_uniform_mesh_config: NonUniformMeshConfig

    total_variation_template_config: TotalVariationTemplateConfig

    model_score_grad_config: ModelScoreGradConfig

    L_flag: LFlag

    regularization_method: RegularizationMethod

    # Save PNGs to results/
    is_save_plots: bool

    # (κ_max, κ_min)
    kappa_range: Tuple[float, float]

    # Number of κ samples
    n_kappa: int

    # κ_knee ×/÷ CONF_WINDOW
    conf_window: float

    # electron charge [C]
    e_charge: float

    # photons cm⁻² s⁻¹  ← pick any positive value
    photon_flux: float

    # Width of the device
    W: float

    # Should we force the last element of the SELE solution to equal to 0
    force_SELE_last_zero: bool
