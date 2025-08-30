"""Entry point for running in PyCharm or any IDE.

Edit the CONFIG section below and press Run.
"""

# ---------- editable CONFIG ----------
DATA_PATHS = {
  # "eta_ext": "Data/ELE.csv",                  # path to Tamir's η_ext CSV
  # "G":       "Data/G_matrix.csv",             # path to Tamir's optical‑generation matrix
    "z":       "Data/z.csv",                    # path to Tamir's depth vector
    "k":       "Data/k.csv",                    # path to extinction coefficient k(λ), from MATLAB code
    "lambda_for_alpha":  "Data/n_k_wavelength_nm.csv",      # wavelengths [nm] involved in alpha (optical constants) calculations, in MATLAB it is called n_k_wavelength
    "eta_ext":  "Data/ELE_sim.csv",             # eta_ext (J) CSV, this is the experiment simulation results, in MATLAB it is called PLQY_sim
    "z_gt":  "Data/z_mesh.csv",                 # the SELE z [cm] mesh, in MATLAB it is called SELE_z_interp
    "sele_gt":  "Data/SELE_ground_truth.csv",   # the real SELE calculated using deltas in simulations, this is the result we are looking for, in MATLAB it is called SELE_interp_for_G0
    "wavelengths":  "Data/wavelength_nm.csv",   # the wavelengths [nm] of eta_ext, in MATLAB it is called wavelength_back,
    "L_score_network": "Data/sele_score_net_d32.pt" # the machine learned score network (L function)
}
L_FLAG        = "L2"            # 'L0', 'L1', 'L2'
IS_SAVE_PLOTS = True            # Save PNGs to results/
KAPPA_RANGE   = (1e-2, 1e-15)   # (κ_max, κ_min)
N_KAPPA       = 150             # Number of κ samples
CONF_WINDOW   = 10 ** 0.5       # κ_knee ×/÷ CONF_WINDOW # TODO: Show this conf window on the l-curve
E_CHARGE      = 1.60217657e-19  # electron charge [C]
PHOTON_FLUX   = 1e14            # photons cm⁻² s⁻¹  ← pick any positive value
Z_TURN        = 1e-4            # cm: turning depth (linear → exponential mesh)
LIN_MESH_SIZE = 1e-5            # cm: linear-part element size (distance between linear mesh elements)
EXP_BASE      = 400.0            # base for exponential spacing

# ----------- do not edit below --------
from src.pipeline import run_regularization

if __name__ == "__main__":
    run_regularization(
        eta_path      = DATA_PATHS["eta_ext"],
        z_path        = DATA_PATHS["z"],
        k_path        = DATA_PATHS["k"],
        lambda_for_alpha_path = DATA_PATHS["lambda_for_alpha"],
        wavelengths_path      = DATA_PATHS["wavelengths"],
        z_gt_path             = DATA_PATHS["z_gt"],
        sele_gt_path          = DATA_PATHS["sele_gt"],
        L_score_network_path  = DATA_PATHS["L_score_network"],
        L_flag        = L_FLAG,
        kappa_max     = KAPPA_RANGE[0],
        kappa_min     = KAPPA_RANGE[1],
        n_kappa       = N_KAPPA,
        conf_fact     = CONF_WINDOW,
        is_save_plots = IS_SAVE_PLOTS,
        e_charge      = E_CHARGE,
        photon_flux   = PHOTON_FLUX,
        z_turn = Z_TURN,
        lin_mesh_size = LIN_MESH_SIZE,
        exp_base = EXP_BASE,
)
