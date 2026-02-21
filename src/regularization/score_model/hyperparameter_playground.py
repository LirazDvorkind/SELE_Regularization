from matplotlib import pyplot as plt

from Utils.pickle_save_load import pickle_load
from src.plotting import plot_sele, plot_eta
from src.regularization.score_model.helpers import load_S_B_G
from src.regularization.score_model.score_model_grad import solve_gradient_descent
from src.types.score_model_params import NesterovHyperparams

import numpy as np

config = {
    'reg_weight': 20,
    'lr_max': 0.02,
    'momentum': 0.8
}

if __name__ == "__main__":
    data, G = load_S_B_G()

    for item in data:
        B_target = item['B']
        S_gt = item['S_gt']

        # --- RUN SOLVER ---
        S_est = solve_gradient_descent(
            G=G,
            B=B_target,
            hyperparams=NesterovHyperparams(
                REG_WEIGHT=config['reg_weight'],
                LR_MAX=config['lr_max'],
                MOMENTUM=config['momentum'],
                model_path="../../../Data/sele_score_net_d32.pt",
                IS_SHOW_DEBUG_PLOT=False,
                IS_SHOW_DEBUG_DATA=True,
                IS_SHOW_MSE_PLOT=True
            ),
            S_gt=S_gt,
        )

        # 1. Calculate ELE Error (Data Misfit) - [USER REQUESTED METRIC]
        B_est = G @ S_est
        mse_ele = np.mean((B_est - B_target) ** 2)
        print(f"mse_ele: {mse_ele}")

        # 2. Calculate SELE Error (Ground Truth Misfit) - [FOR SAFETY CHECK]
        mse_sele = np.mean((S_est - S_gt) ** 2)
        print(f"mse_sele: {mse_sele}")

        d1 = pickle_load("plotting_data/sele_plot.p")
        z_gt = d1['z_gt']
        sele_gt = d1['sele_gt']
        z_centres = d1['z_centres']
        S_mean = d1['S_mean']
        S_std = d1['S_std']
        d2 = pickle_load("plotting_data/ele_plot.p")
        wavelengths = d2['wavelengths']
        eta_ext = d2['eta_ext']
        eta_fit = d2['eta_fit']
        mask = z_gt <= np.max(z_centres)
        sele_gt = sele_gt[mask]
        z_gt = z_gt[mask]
        fig, ax = plt.subplots()
        ax.plot(z_centres * 1e4, S_mean, label='SELE (reconstructed)')
        ax.fill_between(z_centres * 1e4,
                        np.asarray(S_mean) - S_std,
                        np.asarray(S_mean) + S_std,
                        alpha=0.3, label=r'$\pm 1\,\sigma$')

        ax.plot(z_gt * 1e4, sele_gt, 'k--', label='SELE ground truth')

        ax.set_xlabel('z $[\\mu m]$')
        ax.set_ylabel('SELE')
        plt.title("SELE vs Ground Truth")
        ax.legend()
        plt.show(block=False)


        fig2, ax = plt.subplots()
        ax.plot(wavelengths, eta_ext, label='Measured')
        ax.plot(wavelengths, eta_fit, '--', label='Reconstructed')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel(r'$\eta_{ext}$')
        ax.legend()
        plt.title("Reconstructed ELE")
        plt.show(block=False)
