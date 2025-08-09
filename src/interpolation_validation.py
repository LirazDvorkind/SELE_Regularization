import numpy as np
import matplotlib.pyplot as plt


def plot_interpolation_check(
    z_old: np.ndarray,
    z_new: np.ndarray,
    G_old: np.ndarray,
    G_new: np.ndarray,
    wav_idx: int = 100,
) -> None:
    # plot against left edges (no centre shift)
    plt.figure()
    plt.plot(z_old*1e4, G_old[wav_idx], "o-", label="original G")
    plt.plot(z_new*1e4, G_new[wav_idx], ".-", label="interpolated G")
    plt.xlabel("depth z $[\\mu m]$")
    plt.ylabel("integrated ΔG per bin")
    plt.title(f"G interpolation check – λ index {wav_idx}")
    plt.legend()
    plt.show(block=True)


# Like S3.2 figure in SI paper
def plot_mesh_elements_position_and_size(z: np.ndarray, z_turn: float) -> None:
    """
    z: new mesh in [cm]
    z_turn: where we turn from lin to exp
    """
    mesh_sizes = np.diff(z)
    mesh_positions = z[:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Left subplot (full range)
    ax1.plot(mesh_positions*1e4, mesh_sizes*1e4, 'k.', markersize=6)
    ax1.set_xlabel(r'Position [$\mu$m]')
    ax1.set_ylabel(r'Element Size ($\mu$m)')

    # Right subplot (zoom at start)
    ax2.plot(mesh_positions*1e4, mesh_sizes*1e4, 'k.', markersize=6)
    ax2.set_xlabel(r'Position [$\mu$m]')
    ax2.set_ylabel(r'Element Size ($\mu$m)')
    ax2.set_xlim(0, z_turn*1e4*2)  # zoom in horizontally

    fig.suptitle("Mesh elements size and position.")
    ax1.set_title("High level view")
    ax2.set_title("Linear zoomed-in view")
    plt.show(block=False)
