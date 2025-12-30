"""SELE toolbox with non‑uniform mesh support."""
from src.types.config import DataPaths, Config
from src.types.config import NonUniformMeshParams, TotalVariationTemplateParams, ModelScoreGradParams
from src.types.enums import RegularizationMethod, LFlag

# Config defined here so all .py files can import it safely without circular imports (global parameter!)
CONFIG = Config(
    data_paths=DataPaths(
        z="Data/z.csv",
        k="Data/k.csv",
        lambda_for_alpha="Data/n_k_wavelength_nm.csv",
        eta_ext="Data/ELE_sim.csv",
        z_gt="Data/z_mesh.csv",
        sele_gt="Data/SELE_ground_truth.csv",
        wavelengths="Data/wavelength_nm.csv",
        score_model_curve="Data/sele_score_model_curve.csv",
    ),
    L_flag=LFlag.L2,
    regularization_method=RegularizationMethod.MODEL_SCORE_GRAD,
    is_save_plots=True,
    kappa_range=(1e3, 1e-3),
    n_kappa=40, # was 150
    conf_window=10 ** 0.5,
    e_charge=1.60217657e-19,
    photon_flux=1e14,
    W=350e-4,
    force_SELE_last_zero=True,
    non_uniform_mesh_params=NonUniformMeshParams(
        z_range=(350e-4, 1.462e-06),
        z_turn=1e-4,
        lin_mesh_size=1e-5,
        exp_base=400.0,
    ),
    total_variation_template_params=TotalVariationTemplateParams(
        # Grid parameters used by Alon in `sele_w_score_optimization_example.py`
        # Don't touch these!
        W=30e-4,  # cm
        points_amount=32,
        kappa2_range=(1e-2, 1e-6),
        n_kappa2=30
    ),
    model_score_grad_params=ModelScoreGradParams(
        W=30e-4,  # cm
        points_amount=32,
        longer_points_amount=300,
        num_steps=200,
        reg_weight=0.9
    )
)
