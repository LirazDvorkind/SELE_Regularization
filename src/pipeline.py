"""High-level orchestration of the SELE extraction workflow (non-uniform mesh + model-scoring)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.__init__ import CONFIG
from src.io import load_eta, load_csv, save_csv, generate_run_report
from src.mesh import calc_mesh_and_G, _linear_mesh
from src.operators import build_L
from src.plotting import plot_lcurve, plot_sele, plot_eta
from src.regularization import tikhonov_non_uniform, total_variation
from src.regularization.score_model import score_model_grad
from src.types.G_calculation import GInputData
from src.types.enums import RegularizationMethod
from src.utils import expand_sele


def run_regularization():
    """Run full SELE regularization pipeline."""
    eta_path, z_path, k_path = CONFIG.data_paths.eta_ext, CONFIG.data_paths.z, CONFIG.data_paths.k
    lambda_for_alpha_path, wavelengths_path = CONFIG.data_paths.lambda_for_alpha, CONFIG.data_paths.wavelengths
    z_gt_path, sele_gt_path = CONFIG.data_paths.z_gt, CONFIG.data_paths.sele_gt
    L_flag, regularization_method = CONFIG.L_flag, CONFIG.regularization_method
    kappa_max, kappa_min = CONFIG.kappa_range
    conf_fact, n_kappa = CONFIG.conf_window, CONFIG.n_kappa
    e_charge, photon_flux = CONFIG.e_charge, CONFIG.photon_flux
    is_save_plots = CONFIG.is_save_plots

    # --- NON UNIFORM MESH MODE ---------------------------------------------------
    if regularization_method is RegularizationMethod.NON_UNIFORM_MESH:
        # 1. Load data ---------------------------------------------------------
        z_gt = load_eta(z_gt_path)
        sele_gt = load_eta(sele_gt_path)
        eta_ext = load_eta(eta_path)
        z = load_csv(z_path).ravel()

        # Load optical inputs for recomputing G on the new mesh
        k = load_csv(k_path).ravel()  # extinction coefficient k(λ) [unitless]
        lambda_for_alpha = load_csv(lambda_for_alpha_path).ravel()  # wavelengths for alpha [nm]
        wavelengths = load_csv(wavelengths_path).ravel()  # wavelengths of G [nm]

        # Store in an easy-to-access object :)
        G_values = GInputData(k=k, lambda_for_alpha=lambda_for_alpha, wavelengths=wavelengths, z=z)

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
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, save=is_save_plots)
        plot_eta(wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)

    # --- TOTAL VARIATION MODE ------------------------------------------------------
    elif regularization_method is RegularizationMethod.TOTAL_VARIATION:
        # 1. Load data
        z_gt = load_eta(z_gt_path)
        sele_gt = load_eta(sele_gt_path)
        eta_ext = load_eta(eta_path)
        z = load_csv(z_path).ravel()

        # Load optical inputs for recomputing G on the new mesh
        k = load_csv(k_path).ravel()
        lambda_for_alpha = load_csv(lambda_for_alpha_path).ravel()
        wavelengths = load_csv(wavelengths_path).ravel()

        G_values = GInputData(k=k, lambda_for_alpha=lambda_for_alpha, wavelengths=wavelengths, z=z)

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
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, save=is_save_plots)
        plot_eta(wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)

    # --- MODEL SCORE GRADIENT MODE ------------------------------------------------------
    elif regularization_method is RegularizationMethod.MODEL_SCORE_GRAD:
        # 1. Load data
        z_gt = load_eta(z_gt_path)
        sele_gt = load_eta(sele_gt_path)
        eta_ext = load_eta(eta_path)
        z = load_csv(z_path).ravel()

        # Load optical inputs for recomputing G on the new mesh
        k = load_csv(k_path).ravel()  # extinction coefficient k(λ) [unitless]
        lambda_for_alpha = load_csv(lambda_for_alpha_path).ravel()  # wavelengths for alpha [nm]
        wavelengths = load_csv(wavelengths_path).ravel()  # wavelengths of G [nm]

        # Store values related to calculating G in an easy-to-access object :)
        G_values = GInputData(k=k, lambda_for_alpha=lambda_for_alpha, wavelengths=wavelengths, z=z)

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
                                              CONFIG.model_score_grad_config.output_mesh_resolution)
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
            S_rec = score_model_grad.solve_gradient_descent(
                G,
                B,
                hyperparams=CONFIG.model_score_grad_config,
                S_gt=sele_gt,
            )
            z_centres = 0.5 * (z[:-1] + z[1:])
            G_longer, z_longer = _linear_mesh(G_values.wavelengths, G_values.k, G_values.lambda_for_alpha,
                                              CONFIG.model_score_grad_config.W,
                                              CONFIG.model_score_grad_config.output_mesh_resolution)
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
        plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, save=is_save_plots)
        plot_eta(wavelengths, eta_ext, eta_fit, save=is_save_plots)
        plt.show(block=True)
