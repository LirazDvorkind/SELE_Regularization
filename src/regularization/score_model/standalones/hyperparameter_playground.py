from matplotlib import pyplot as plt

from src.io import load_csv
from src.regularization.score_model.standalones.helpers import load_S_B_G
from src.regularization.score_model.score_model_grad import solve_gradient_descent
from src.types.score_model_params import NesterovHyperparams

import numpy as np

config = {
    'reg_weight': 2.5,
    'lr_max': 0.04,
    'momentum': 0.95
}

if __name__ == "__main__":
    random_sample = 872 # np.random.randint(100, 1000)
    print(f"Random curve number {random_sample}")
    data, G = load_S_B_G(lower_index=random_sample, upper_index=random_sample+1)

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

        W = 30e-4  # cm
        x_res = G.shape[1]
        z_centres = np.linspace(0, W, x_res) * 1e4  # µm
        fig, ax = plt.subplots()
        ax.plot(z_centres, S_est, label='SELE (reconstructed)')
        ax.plot(z_centres, S_gt, 'k--', label='SELE ground truth')

        ax.set_xlabel('z $[\\mu m]$')
        ax.set_ylabel('SELE')
        plt.title("SELE vs Ground Truth")
        ax.legend()
        plt.show(block=False)


        fig2, ax = plt.subplots()
        wavelengths = load_csv("Data/wavelength_nm.csv").ravel()  # wavelengths of G [nm]
        ax.plot(wavelengths, B_target, label='Measured')
        ax.plot(wavelengths, B_est, '--', label='Reconstructed')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel(r'$\eta_{ext}$')
        ax.legend()
        plt.title("Reconstructed ELE")
        plt.show(block=True)
