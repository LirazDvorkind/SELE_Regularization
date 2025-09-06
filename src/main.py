"""Entry point for running in PyCharm or any IDE.

Edit the CONFIG section below and press Run.
"""
from __future__ import annotations

from src.types.enums import RegularizationMethod, LFlag
from src.types.config import DataPaths, Config
from src.types.config import NonUniformMeshParams, ModelScoringParams

# ---------- editable CONFIG ----------
CONFIG = Config(
    data_paths=DataPaths(
        z="Data/z.csv",
        k="Data/k.csv",
        lambda_for_alpha="Data/n_k_wavelength_nm.csv",
        eta_ext="Data/ELE_sim.csv",
        z_gt="Data/z_mesh.csv",
        sele_gt="Data/SELE_ground_truth.csv",
        wavelengths="Data/wavelength_nm.csv",
        L_score_network="Data/sele_score_net_d32.pt",
    ),
    L_flag=LFlag.L2,
    regularization_method=RegularizationMethod.MODEL_SCORING,
    is_save_plots=True,
    kappa_range=(1e-2, 1e-15),
    n_kappa=150,
    conf_window=10 ** 0.5,
    e_charge=1.60217657e-19,
    photon_flux=1e14,
    W=350e-4,
    non_uniform_mesh_params=NonUniformMeshParams(
        z_range=(350e-4, 1.462e-06),
        z_turn=1e-4,
        lin_mesh_size=1e-5,
        exp_base=400.0,
    ),
    model_scoring_params=ModelScoringParams(
        # Grid parameters used by Alon in `sele_w_score_optimization_example.py`
        # Don't touch these!
        W=30e-4,  # cm
        points_amount=32
    )
)

# ----------- do not edit below --------
from src.pipeline import run_regularization

if __name__ == "__main__":
    run_regularization(CONFIG)
