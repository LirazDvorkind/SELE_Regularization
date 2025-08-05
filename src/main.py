"""Entry point for running in PyCharm or any IDE.

Edit the CONFIG section below and press Run.
"""

# ---------- editable CONFIG ----------
DATA_PATHS = {
    "eta_ext": "Data/ELE.csv",                # path to η_ext CSV
    "G":       "Data/G_matrix.csv",           # path to optical‑generation matrix
    "z":       "Data/z.csv",                  # path to depth vector
}
L_FLAG        = "L2"            # 'L0', 'L1', 'L2'
IS_SAVE_PLOTS = True            # Save PNGs to results/
KAPPA_RANGE   = (1e-2, 1e-15)   # (κ_max, κ_min)
N_KAPPA       = 150             # Number of κ samples
CONF_WINDOW   = 10 ** 0.5       # κ_knee ×/÷ CONF_WINDOW
E_CHARGE      = 1.60217657e-19  # electron charge [C]
PHOTON_FLUX   = 1e14            # photons cm⁻² s⁻¹  ← pick any positive value

# ----------- do not edit below --------
from src.pipeline import run_regularization

if __name__ == "__main__":
    run_regularization(
        eta_path      = DATA_PATHS["eta_ext"],
        G_path        = DATA_PATHS["G"],
        z_path        = DATA_PATHS["z"],
        L_flag        = L_FLAG,
        kappa_max     = KAPPA_RANGE[0],
        kappa_min     = KAPPA_RANGE[1],
        n_kappa       = N_KAPPA,
        conf_fact     = CONF_WINDOW,
        is_save_plots = IS_SAVE_PLOTS,
        e_charge      = E_CHARGE,
        photon_flux   = PHOTON_FLUX,
    )
