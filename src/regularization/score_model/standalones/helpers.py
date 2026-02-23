from src.utils import match_length_interp


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


def load_S_B_G(lower_index: int=0, upper_index: int=100):
    from src.io import load_csv
    print("Loading SELE and G...")
    # Load pre-calculated photogeneration matrix
    # This is G from pipeline.py, before multiplying by unit_factor=photon_flux*e_charge
    # Change to G_score_model for the 32 long one
    G = load_csv("Data/G_score_model_500.csv")
    # Load the dataset of SELE profiles
    # Play around with the amount of S's here
    # Change to sele_dataset for the 32 long one
    S_profiles = load_csv("Data/sele_dataset_500.csv")[lower_index:upper_index, :]
    # Prepare synthetic measurements (B) for all profiles
    print(f"Preparing synthetic dataset for {len(S_profiles)} profiles...")
    data = generate_synthetic_data(S_profiles, G)
    return data, G
