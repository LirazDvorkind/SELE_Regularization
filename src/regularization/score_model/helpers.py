from src.utils import match_length_interp


def generate_synthetic_data(S_gt_profiles, G_matrix):
    dataset = []
    for S_100 in S_gt_profiles:
        S_32 = match_length_interp(S_100, 32)
        # B = G * S (Physical measurement)
        B = G_matrix @ S_32
        dataset.append({'S_gt': S_32, 'B': B})
    return dataset


def load_S_B_G():
    from src.io import load_csv
    print("Loading SELE and G...")
    # Load pre-calculated photogeneration matrix
    G = load_csv("../../../Data/G_score_model.csv")
    # Load the dataset of 100-point SELE profiles
    S_profiles_100 = load_csv("../../../Data/sele_dataset.csv")[:100, :]  # Play around with the amount of S's here
    # Prepare synthetic measurements (B) for all profiles
    print(f"Preparing synthetic dataset for {len(S_profiles_100)} profiles...")
    data = generate_synthetic_data(S_profiles_100, G)
    return data, G
