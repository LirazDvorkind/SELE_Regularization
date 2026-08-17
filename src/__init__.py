"""SELE toolbox with non‑uniform mesh support."""
from src.types.config import DataPaths, Config, SCORE_MODEL_PRESETS
from src.types.config import NonUniformMeshConfig, TotalVariationConfig
from src.types.enums import RegularizationMethod, LFlag

# Config defined here so all .py files can import it safely without circular imports (global parameter!)
CONFIG = Config(
    data_paths=DataPaths(
        z="Data/z.csv",
        eta_ext="Data/ELE_sim.csv",
        z_gt="Data/z_mesh.csv",
        sele_gt="Data/SELE_ground_truth.csv",
        wavelengths="Data/wavelength_nm.csv",
        score_model_curve="Data/sele_score_model_curve.csv",
    ),
    # Run against a Data/test_set/ curve instead of the default eta_ext/sele_gt/z_gt above,
    # e.g. test_set_curve_id="tau_8ns" (see Data/test_set/index.csv for valid ids or below).
    # SRV sweep (surface recombination velocity): srv_1e3, srv_1e4, srv_1e5, srv_5e5, srv_1e6, srv_1e7
    # τ_SRH sweep (SRH lifetime): tau_0p1ns, tau_0p25ns, tau_0p5ns, tau_1ns, tau_2ns, tau_4ns, tau_8ns, tau_15ns,
    # tau_20ns, tau_40ns, tau_60ns
    # Set to None for original Alon SELE profile
    test_set_curve_id=None,
    L_flag=LFlag.L2,
    regularization_method=RegularizationMethod.MODEL_SCORE_GRAD,
    is_save_plots=True,
    kappa_range=(1e-3, 1e-7),
    n_kappa=40, # was 150
    conf_window=10 ** 0.5,
    e_charge=1.60217657e-19,
    photon_flux=1e14,
    W=350e-4,
    force_SELE_last_zero=True,
    non_uniform_mesh_config=NonUniformMeshConfig(
        z_range=(350e-4, 1.462e-06),
        z_turn=1e-4,
        lin_mesh_size=1e-5,
        exp_base=400.0,
    ),
    total_variation_config=TotalVariationConfig(
        W=30e-4,
        mesh_resolution=500,
    ),
    # Presets are defined in src/types/config.py — one entry per trained checkpoint,
    # each with tuned Nesterov hyperparameters (REG_WEIGHT, MOMENTUM, LR_MAX, T0, …).
    # To switch models: change "d500" → "d32" (or any key in SCORE_MODEL_PRESETS).
    # To tweak a single param without leaving the preset:
    #   from dataclasses import replace
    #   model_score_grad_config=replace(SCORE_MODEL_PRESETS["d500"], REG_WEIGHT=5.0)
    model_score_grad_config=SCORE_MODEL_PRESETS["d500"],
)
