import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid

from src.interpolation_validation import plot_mesh_elements_position_and_size

def remesh_G(z_old: np.ndarray, G_old: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    z: original mesh in [cm]
    """
    dz0 = np.diff(z_old).mean()  # The original uniform spacing
    z_min = z_old[0]
    z_max = z_old[-1]

    # ---------- The parameters to play with ------------
    # z_turn_relative_location = 0.0005  # Percentage between 0 and 1 of where we turn exponential
    z_turn = 2e-4 # Actual z location of the turn in cm
    lin_mesh_size = 3e-6  # Linear part mesh element size in cm
    exp_base = 20  # The larger the base, the less the number of points in the exp part (n_exp) will be

    n_lin = math.floor((z_turn - z_min) / lin_mesh_size)  # Number of points in linear part
    # z_turn = z_old[0] + (dz0 * len(z_old) * z_turn_relative_location)  # actual z location of the turn

    z_lin = np.linspace(z_min, z_turn, n_lin + 1)  # include both ends

    lin_mesh_size = z_lin[1] - z_lin[0]
    # Number of points in exponential part
    n_exp = math.floor((np.log10(1 + (((exp_base - 1) * lin_mesh_size) / (z_max - z_turn)))) ** -1)

    factor = np.logspace(0.0, 1.0, n_exp, base=exp_base) - 1.0
    z_exp = z_turn + (z_max - z_turn) * (factor / (exp_base - 1))

    z_new = np.hstack([z_lin, z_exp[1:]])  # drop duplicate z_turn

    # Verify plot
    plot_mesh_elements_position_and_size(z_new, z_turn)

    density_old = G_old / dz0  # (λ, N_old)

    G_new = np.zeros((G_old.shape[0], len(z_new)))  # allocate container for new integrated values

    # ---- remesh each wavelength slice ---------------------------------------
    for lmda in range(G_old.shape[0]):
        interp = PchipInterpolator(z_old, density_old[lmda], extrapolate=False)
        for i in range(len(z_new) - 1):
            fine_z = np.linspace(z_new[i], z_new[i + 1], 5)  # was 20
            g_vals = np.clip(interp(fine_z), 0.0, None)
            G_new[lmda, i] = trapezoid(g_vals, fine_z)

    return z_new, G_new
