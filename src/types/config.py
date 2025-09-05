from dataclasses import dataclass
from typing import Tuple

from src.types.enums import RegularizationMethod, LFlag


@dataclass
class DataPaths:
    # eta_ext="Data/ELE.csv", # path to Tamir's η_ext CSV
    # G="Data/G_matrix.csv", # path to Tamir's optical‑generation matrix

    # path to Tamir's depth vector
    z: str

    # path to extinction coefficient k(λ), from MATLAB code
    k: str

    # wavelengths [nm] involved in alpha (optical constants) calculations, in MATLAB it is called n_k_wavelength
    lambda_for_alpha: str

    # eta_ext (J) CSV, this is the experiment simulation results, in MATLAB it is called PLQY_sim
    eta_ext: str

    # the SELE z [cm] mesh, in MATLAB it is called SELE_z_interp
    z_gt: str

    # the real SELE calculated using deltas in simulations, this is the result we are looking for,
    # in MATLAB it is called SELE_interp_for_G0
    sele_gt: str

    # the wavelengths [nm] of eta_ext, in MATLAB it is called wavelength_back,
    wavelengths: str

    # the machine learned score network (L function)
    L_score_network: str



@dataclass
class Config:
    data_paths: DataPaths

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

    # cm: turning depth (linear → exponential mesh)
    z_turn: float

    # cm: linear-part element size (distance between linear mesh elements)
    lin_mesh_size: float

    # base for exponential spacing
    exp_base: float

    # Width of the device
    W: float
