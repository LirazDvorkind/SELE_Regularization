"""High-level orchestration of the SELE extraction workflow (non-uniform mesh + model-scoring)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.__init__ import CONFIG
from src.io import load_eta, load_csv, save_csv, generate_run_report
from src.mesh import calc_mesh_and_G, _linear_mesh
from src.optical_constants import load_optical_constants
from src.operators import build_L
from src.plotting import plot_lcurve, plot_sele, plot_eta
from src.regularization import tikhonov_non_uniform, total_variation
from src.regularization.score_model import score_model_grad
from src.test_set.loader import load_curve
from src.types.G_calculation import GInputData
from src.types.enums import RegularizationMethod
from src.utils import expand_sele


def _load_G_inputs(z, wavelengths_path):
    """Optical inputs for building G, from the paper's own ellipsometry constants.

    Replaces the old `Data/k.csv`, which held one extinction spectrum with no way to express
    the free-carrier split -- so G could only ever be an absorption matrix. The paper's
    constants carry both k and k_bulk, which is what makes it a *generation* matrix.
    """
    constants = load_optical_constants()
    return GInputData(
        k=constants.k,
        k_bulk=constants.k_bulk,
        lambda_for_alpha=constants.wavelength_nm,
        wavelengths=load_csv(wavelengths_path).ravel(),
        z=z,
    )


def _load_ground_truth(data_paths, curve_id: str | None):
    """The (eta_ext, z_gt, sele_gt) triple to reconstruct against, plus a display label.

    Default: the original paper measurement. If CONFIG.test_set_curve_id is set, one
    Data/test_set/ curve instead -- its *native*-mesh ele (the curve's own independently
    generated ELE), not the solver-mesh one from load_on_solver_mesh, to mirror how
    Data/ELE_sim.csv today stands in for an independent measurement rather than something
    recomputed through the solver's own approximate G.
    """
    if curve_id is None:
        return (load_eta(data_paths.eta_ext), load_eta(data_paths.z_gt),
                load_eta(data_paths.sele_gt), "default")
    curve = load_curve(curve_id)
    return curve.ele, curve.z_cm, curve.sele, curve.label


def run_regularization():
    """Run full SELE regularization pipeline."""
    z_path = CONFIG.data_paths.z
    wavelengths_path = CONFIG.data_paths.wavelengths
    curve_id = CONFIG.test_set_curve_id
    L_flag, regularization_method = CONFIG.L_flag, CONFIG.regularization_method
    kappa_max, kappa_min = CONFIG.kappa_range
    conf_fact, n_kappa = CONFIG.conf_window, CONFIG.n_kappa
    e_charge, photon_flux = CONFIG.e_charge, CONFIG.photon_flux
    is_save_plots = CONFIG.is_save_plots

    # --- NON UNIFORM MESH MODE ---------------------------------------------------
    if regularization_method is RegularizationMethod.NON_UNIFORM_MESH:
        # 1. Load data ---------------------------------------------------------
        eta_ext, z_gt, sele_gt, gt_label = _load_ground_truth(CONFIG.data_paths, curve_id)
        z = load_csv(z_path).ravel()

        G_values = _load_G_inputs(z, wavelengths_path)

        G, z = calc_mesh_and_G(regularization_method, G_values)

        # 2. Unit normalisation (A and B must have same units)
        G = G * photon_flux * e_charge
        B = eta_ext * photon_flux * e_charge

        if G.shape[0] != eta_ext.size:
            raise ValueError(f"Row mismatch between G and η_ext: G[0] is {G.shape[0]} but n_ext is {eta_ext.size}")

        # 3. Regularisation operator
        L = build_L(L_flag, len(z) - 1)

        # 4. Tichonov κ‑sweep
        kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
        residuals, seminorms, S_list = tikhonov_non_uniform.sweep_kappa(G, B, L, kappa_vals)

        # 5. Knee detection
        kappa_knee, knee_idx = tikhonov_non_uniform.find_knee(residuals, seminorms, kappa_vals)
        # Use this for debugging different κ values
        # kappa_knee, knee_idx = set_kappa_knee(kappa_vals, desired_kappa_value=3.5e-7)

        # 6. Confidence window
        mask = (kappa_vals >= kappa_knee / conf_fact) & (kappa_vals <= kappa_knee * conf_fact)
        S_stack = np.stack([S_list[i] for i, m in enumerate(mask) if m], axis=1)
        S_mean = S_stack.mean(axis=1)
        S_std = S_stack.std(axis=1)

        # 7. Reconstruction @ kappa_knee
        S_knee = S_list[knee_idx]
        eta_fit = G @ S_knee / (photon_flux * e_charge)

        # 8. Persist results
        z_centres = 0.5 * (z[:-1] + z[1:])  # length M-1
        save_csv("results/raw/S_mean.csv", np.column_stack([z_centres, S_mean]), header="z_cm,S_mean")
        save_csv("results/raw/S_std.csv", np.column_stack([z_centres, S_std]), header="z_cm,S_std")
        save_csv("results/raw/eta_fit.csv", eta_fit, header="eta_fit")
        generate_run_report("results", kappa_knee)

        # 9. Plotting
        plot_lcurve(seminorms, residuals, kappa_vals, knee_idx, mask, save=is_save_plots)
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, gt_label=gt_label, save=is_save_plots)
        plot_eta(G_values.wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)

    # --- TOTAL VARIATION MODE ------------------------------------------------------
    elif regularization_method is RegularizationMethod.TOTAL_VARIATION:
        # 1. Load data
        eta_ext, z_gt, sele_gt, gt_label = _load_ground_truth(CONFIG.data_paths, curve_id)
        z = load_csv(z_path).ravel()

        # Load optical inputs for recomputing G on the new mesh
        G_values = _load_G_inputs(z, wavelengths_path)

        G, z = calc_mesh_and_G(regularization_method, G_values)

        # 2. Unit normalisation
        G *= photon_flux * e_charge
        B = eta_ext * photon_flux * e_charge
        if G.shape[0] != eta_ext.size:
            raise ValueError("Row mismatch between G and η_ext")

        # 3. κ sweep
        kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
        residuals, tv_norms, S_list = total_variation.sweep_kappa_tv(G, B, kappa_vals)

        # 4. Knee detection (reuse non-uniform helper)
        kappa_knee, knee_idx = tikhonov_non_uniform.find_knee(residuals, tv_norms, kappa_vals)

        # 5. Confidence window
        mask = (kappa_vals >= kappa_knee / conf_fact) & (kappa_vals <= kappa_knee * conf_fact)
        S_stack = np.stack([S_list[i] for i, m in enumerate(mask) if m], axis=1)
        S_mean, S_std = S_stack.mean(axis=1), S_stack.std(axis=1)

        # 6. Reconstruction
        S_knee = S_list[knee_idx]
        eta_fit = G @ S_knee / (photon_flux * e_charge)

        # 7. Save & report
        z_centres = 0.5 * (z[:-1] + z[1:])
        save_csv("results/raw/S_mean.csv", np.column_stack([z_centres, S_mean]), header="z_cm,S_mean")
        save_csv("results/raw/S_std.csv", np.column_stack([z_centres, S_std]), header="z_cm,S_std")
        save_csv("results/raw/eta_fit.csv", eta_fit, header="eta_fit")
        generate_run_report("results", kappa_knee=kappa_knee)

        # 8. Plots
        plot_lcurve(tv_norms, residuals, kappa_vals, knee_idx, mask,
                    seminorm_label=r"TV norm $||L_1 S||_1$", save=is_save_plots)
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, gt_label=gt_label, save=is_save_plots)
        plot_eta(G_values.wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)

    # --- MODEL SCORE GRADIENT MODE ------------------------------------------------------
    elif regularization_method is RegularizationMethod.MODEL_SCORE_GRAD:
        # 1. Load data
        eta_ext, z_gt, sele_gt, gt_label = _load_ground_truth(CONFIG.data_paths, curve_id)
        z = load_csv(z_path).ravel()

        G_values = _load_G_inputs(z, wavelengths_path)

        # Derive mesh dimension from the model checkpoint so G always matches the model's expected input.
        _ckpt = torch.load(CONFIG.model_score_grad_config.model_path, map_location='cpu', weights_only=False)
        _target_length = _ckpt['config']['target_length']
        del _ckpt

        G, z = calc_mesh_and_G(regularization_method, G_values, mesh_resolution=_target_length)
        # np.savetxt('src/regularization/score_model/standalones/Data/G_score_model_500.csv', G, delimiter=',')

        # 2. Unit normalization
        unit_factor = photon_flux * e_charge
        G *= unit_factor
        B = eta_ext * unit_factor
        if G.shape[0] != eta_ext.size:
            raise ValueError("Row mismatch between G and η_ext")

        # 3. Regularisation via Gradient Descent with Score Model
        # Set to True to overwrite S to become the ground truth sampled at 'longer_points_amount' points
        override_with_ground_truth = False
        if override_with_ground_truth:
            G_longer, z_longer = _linear_mesh(G_values.wavelengths, G_values.k, G_values.lambda_for_alpha,
                                              CONFIG.model_score_grad_config.W,
                                              CONFIG.model_score_grad_config.output_mesh_resolution,
                                              k_bulk=G_values.k_bulk)
            z_centres = 0.5 * (z[:-1] + z[1:])
            temp_mask = np.searchsorted(z_gt, z_centres, side='right')
            S_rec = sele_gt[temp_mask]
            # Upsample to output_mesh_resolution points, strongly weighted near the surface
            z_centres, S_rec = expand_sele(S_rec, points_amount=CONFIG.model_score_grad_config.output_mesh_resolution,
                                           front_weight=1.0, z_original=z_centres)
            S_mean = S_rec
            S_std = np.zeros_like(S_rec)  # No statistical mean in this method yet

            # 4. Fit
            eta_fit = G_longer @ S_rec / (
                unit_factor if CONFIG.regularization_method != regularization_method.MODEL_SCORE_GRAD else 1)

            # 5. Save & report
            save_csv("results/raw/S_mean.csv", np.column_stack([z_centres, S_mean]), header="z_cm,S_mean")
            save_csv("results/raw/S_std.csv", np.column_stack([z_centres, S_std]), header="z_cm,S_std")
            save_csv("results/raw/eta_fit.csv", eta_fit, header="eta_fit")
        else:
            # Optional TV warm-start: solve TV on the same G/B and feed the result
            # in as the NAG initial point instead of random noise.
            S_init = None
            if CONFIG.model_score_grad_config.warm_start_with_tv:
                kappa_override = CONFIG.model_score_grad_config.warm_start_tv_kappa
                if kappa_override is not None:
                    S_init, _ = total_variation.solve_tv(G, B, kappa_override)
                    print(f"[TV warm-start] solved at κ={kappa_override:.3e}")
                else:
                    kappa_vals_ws = np.logspace(np.log10(kappa_max), np.log10(kappa_min), n_kappa)
                    residuals_ws, tv_norms_ws, S_list_ws = total_variation.sweep_kappa_tv(G, B, kappa_vals_ws)
                    kappa_knee_ws, knee_idx_ws = tikhonov_non_uniform.find_knee(residuals_ws, tv_norms_ws, kappa_vals_ws)
                    S_init = S_list_ws[knee_idx_ws]
                    print(f"[TV warm-start] κ_knee={kappa_knee_ws:.3e}, using S_knee as score-grad init")

            # Sample the ground-truth SELE onto the solver's physical mesh centres so the
            # in-solver MSE/diagnostics compare like-for-like. The GT lives on z_gt (a ~294 µm
            # axis) while the solver mesh spans W (~30 µm); index-based resampling would compress
            # the GT ~10× onto the wrong depths and corrupt every GT-based metric.
            z_centres = 0.5 * (z[:-1] + z[1:])
            sele_gt_on_mesh = np.interp(z_centres, z_gt, sele_gt)

            S_rec = score_model_grad.solve_gradient_descent(
                G,
                B,
                hyperparams=CONFIG.model_score_grad_config,
                S_gt=sele_gt_on_mesh,
                S_init=S_init,
            )
            G_longer, z_longer = _linear_mesh(G_values.wavelengths, G_values.k, G_values.lambda_for_alpha,
                                              CONFIG.model_score_grad_config.W,
                                              CONFIG.model_score_grad_config.output_mesh_resolution,
                                              k_bulk=G_values.k_bulk)
            # Upsample to output_mesh_resolution points, strongly weighted near the surface
            z_centres, S_rec = expand_sele(S_rec, points_amount=CONFIG.model_score_grad_config.output_mesh_resolution,
                                           front_weight=1.0, z_original=z_centres)
            S_mean = S_rec
            S_std = np.zeros_like(S_rec)  # No statistical mean in this method yet

            # 4. Fit
            eta_fit = G_longer @ S_rec / (
                unit_factor if CONFIG.regularization_method != regularization_method.MODEL_SCORE_GRAD else 1)

            # 5. Save & report
            save_csv("results/raw/S_mean.csv", np.column_stack([z_centres, S_mean]), header="z_cm,S_mean")
            save_csv("results/raw/S_std.csv", np.column_stack([z_centres, S_std]), header="z_cm,S_std")
            save_csv("results/raw/eta_fit.csv", eta_fit, header="eta_fit")

        generate_run_report("results")

        # 6. Plots
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, gt_label=gt_label, save=is_save_plots)
        plot_eta(G_values.wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)
