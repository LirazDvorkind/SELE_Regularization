from pathlib import Path

from src.utils import match_length_interp

_DATA_DIR = Path(__file__).resolve().parents[4] / "Data" / "score_model"


def generate_synthetic_data(S_gt_profiles, G_matrix):
    dataset = []
    for S in S_gt_profiles:
        S_interp = match_length_interp(S, G_matrix.shape[1])
        # B = G * S (Physical measurement)
        B = G_matrix @ S_interp
        # TODO: Something seems off here when we run with the 500 long one.
        #  Test with same index curve so should not be different B for 32 or 500
        dataset.append({'S_gt': S_interp, 'B': B})
    return dataset


def load_S_B_G(points_amount: int = 32, lower_index: int=0, upper_index: int=100):
    # points_amount can be 32 or 500
    from src.io import load_csv
    print(f"Loading SELE and G for {points_amount} points...")

    # Load pre-calculated photogeneration matrix
    # This is G from pipeline.py, before multiplying by unit_factor=photon_flux*e_charge
    suffix = "_500" if points_amount == 500 else ""
    G = load_csv(str(_DATA_DIR / f"G_score_model{suffix}.csv"))

    # Load the dataset of SELE profiles
    # Play around with the amount of S's here
    S_profiles = load_csv(str(_DATA_DIR / f"sele_dataset{suffix}.csv"))[lower_index:upper_index, :]

    # Prepare synthetic measurements (B) for all profiles
    print(f"Preparing synthetic dataset for {len(S_profiles)} profiles...")

    data = generate_synthetic_data(S_profiles, G)
    return data, G
