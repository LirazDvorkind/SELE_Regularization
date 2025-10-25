"""Plotting helpers (matplotlib)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import os
from typing import Sequence, Optional, Tuple
import mplcursors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection import side effect


def _ensure_results_dir():
    os.makedirs('results', exist_ok=True)


def plot_lcurve(seminorms: Sequence[float], residuals: Sequence[float], kappa_vals,
                knee_idx: int, mask: Sequence[bool], *, save: bool = False):
    seminorms = np.asarray(seminorms)
    residuals = np.asarray(residuals)
    mask = np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots()

    # Main curve (Line2D)
    line, = ax.loglog(residuals, seminorms, '-o', markersize=3, color="C0")

    # Highlight masked points (PathCollection)
    mask = np.asarray(mask, dtype=bool)
    idx_mask = np.flatnonzero(mask)
    sc_mask = ax.scatter(residuals[mask], seminorms[mask],
                         c="red", s=20, label="Conf window")

    # Knee point (single PathCollection)
    idx_knee = np.array([knee_idx])
    sc_knee = ax.scatter(residuals[knee_idx], seminorms[knee_idx],
                         marker='x', s=60, color="black",
                         label=f'κ_knee = {kappa_vals[knee_idx]:.2e}')

    # One cursor for all artists; show κ on hover
    cursor = mplcursors.cursor([line, sc_mask, sc_knee], hover=True)

    # Build per-artist index mapping back to kappa indices
    index_map = {
        line: np.arange(len(kappa_vals), dtype=int),
        sc_mask: idx_mask.astype(int),
        sc_knee: np.array([knee_idx], dtype=int),
    }

    @cursor.connect("add")
    def _(sel):
        artist = sel.artist
        mapping = index_map.get(artist)
        if mapping is None or len(mapping) == 0:
            sel.annotation.set_text("κ = n/a")
            return
        i_local = sel.index
        i_local = 0 if i_local is None else int(i_local)
        i_global = int(mapping[i_local])
        sel.annotation.set_text(f"κ = {kappa_vals[i_global]:.2e}")

    ax.set_xlabel(r'$\varepsilon = ||\,G S - \eta_{\mathrm{ext}}\,||_2$')
    ax.set_ylabel(r'$||\,L S\,||_2$')
    plt.title("Regularization Loss Curve")
    ax.legend()

    if save:
        _ensure_results_dir()
        fig.savefig('results/lcurve.png', dpi=300)

    plt.show(block=False)


def plot_sele(z_centres, S_mean, S_std, sele_gt, z_gt, *, save: bool = False):
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
    if save:
        _ensure_results_dir()
        fig.savefig('results/sele_profile.png', dpi=300)
    plt.show(block=False)


def plot_eta(lambda_vals, eta_meas, eta_fit, *, save: bool = False):
    fig, ax = plt.subplots()
    ax.plot(lambda_vals, eta_meas, label='Measured')
    ax.plot(lambda_vals, eta_fit, '--', label='Reconstructed')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(r'$\eta_{ext}$')
    ax.legend()
    plt.title("Reconstructed ELE")
    if save:
        _ensure_results_dir()
        fig.savefig('results/eta_fit.png', dpi=300)
    plt.show(block=False)


def plot_interpolation_check(
        z_old: np.ndarray,
        z_new: np.ndarray,
        G_old: np.ndarray,
        G_new: np.ndarray,
        wav_idx: int = 100,
        *,
        save: bool = False
) -> None:
    # plot against left edges (no centre shift)
    fig = plt.figure()
    plt.plot(z_old * 1e4, G_old[wav_idx], "o-", label="original G")
    plt.plot(z_new * 1e4, G_new[wav_idx], ".-", label="interpolated G")
    plt.xlabel("depth z $[\\mu m]$")
    plt.ylabel("integrated ΔG per bin")
    plt.title(f"G interpolation check – λ index {wav_idx}")
    plt.legend()
    if save:
        _ensure_results_dir()
        fig.savefig('results/interpolated_G.png', dpi=300)
    plt.show(block=True)


# Like S3.2 figure in SI paper
def plot_mesh_elements_position_and_size(z: np.ndarray, z_turn: float, *, save: bool = False) -> None:
    """
    z: new mesh in [cm]
    z_turn: where we turn from lin to exp
    """
    mesh_sizes = np.diff(z)
    mesh_positions = z[:-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Left subplot (full range)
    ax1.plot(mesh_positions * 1e4, mesh_sizes * 1e4, 'k.', markersize=6)
    ax1.set_xlabel(r'Position [$\mu$m]')
    ax1.set_ylabel(r'Element Size ($\mu$m)')

    # Right subplot (zoom at start)
    ax2.plot(mesh_positions * 1e4, mesh_sizes * 1e4, 'k.', markersize=6)
    ax2.set_xlabel(r'Position [$\mu$m]')
    ax2.set_ylabel(r'Element Size ($\mu$m)')
    ax2.set_xlim(0, z_turn * 1e4 * 2)  # zoom in horizontally

    fig.suptitle("Mesh elements size and position.")
    ax1.set_title("High level view")
    ax2.set_title("Linear zoomed-in view")
    if save:
        _ensure_results_dir()
        fig.savefig('results/mesh.png', dpi=300)
    plt.show(block=False)


def plot_lsurface_3d(
        residuals: np.ndarray,  # shape (n1, n2)  = || G S - B ||
        seminorms: np.ndarray,  # shape (n1, n2)  = || L S ||
        model_residuals: np.ndarray,  # shape (n1, n2)  = || S - S_model ||
        *,
        kappa1_vals: np.ndarray | None = None,  # optional κ₁ grid (len = n1)
        kappa2_vals: np.ndarray | None = None,  # optional κ₂ grid (len = n2)
        save: bool = False
) -> None:
    """
    3D 'L-surface' visualization:
      X = log10 data residual, Y = log10 seminorm, Z = log10 model residual
    Each point corresponds to a (kappa1, kappa2) pair. If kappa grids are provided,
    a scatter overlay becomes interactive and shows (κ₁, κ₂) on hover.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Try to enable hover tooltips; continue silently if unavailable
    try:
        import mplcursors  # type: ignore
        _HAS_MPLCURSORS = True
    except Exception:
        _HAS_MPLCURSORS = False

    # Safety and positivity checks for log-space
    eps = 1e-300
    R = np.maximum(np.asarray(residuals, float), eps)
    S = np.maximum(np.asarray(seminorms, float), eps)
    M = np.maximum(np.asarray(model_residuals, float), eps)

    X = np.log10(R)  # data residual
    Y = np.log10(S)  # smoothness seminorm
    Z = np.log10(M)  # model residual

    n1, n2 = X.shape

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    dense_enough = (n1 >= 3) and (n2 >= 3)
    surf_obj = None
    if dense_enough:
        surf_obj = ax.plot_surface(
            X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0, antialiased=True
        )
        cbar = fig.colorbar(surf_obj, shrink=0.6, aspect=12)
        cbar.set_label('log10 model residual')

    # Always add a scatter overlay (enables hover if κ grids provided)
    scat = ax.scatter(
        X.ravel(), Y.ravel(), Z.ravel(),
        c=Z.ravel(), cmap='viridis', s=16, depthshade=True, alpha=0.9 if surf_obj is not None else 1.0
    )

    ax.set_xlabel(r'$\log_{10}\,\|\,G S - B\,\|_2$ (data residual)')
    ax.set_ylabel(r'$\log_{10}\,\|\,L S\,\|_2$ (seminorm)')
    ax.set_zlabel(r'$\log_{10}\,\|\,S - S_{\mathrm{model}}\,\|_2$ (model residual)')
    ax.set_title("3D L-surface: data fit vs smoothness vs model proximity")

    # Optional contours projected to the sides (helpful for reading structure)
    try:
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', levels=8, linewidths=1, alpha=0.7)
        ax.contour(X, Y, Z, zdir='x', offset=X.min(), cmap='viridis', levels=8, linewidths=1, alpha=0.7)
        ax.contour(X, Y, Z, zdir='y', offset=Y.min(), cmap='viridis', levels=8, linewidths=1, alpha=0.7)
    except Exception:
        pass

    # Interactivity: show (κ₁, κ₂) and diagnostics on hover if grids provided
    if _HAS_MPLCURSORS and (kappa1_vals is not None) and (kappa2_vals is not None):
        k1 = np.asarray(kappa1_vals, float)
        k2 = np.asarray(kappa2_vals, float)
        if k1.shape == (n1,) and k2.shape == (n2,):
            ii, jj = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
            flat_i = ii.ravel()
            flat_j = jj.ravel()

            cursor = mplcursors.cursor(scat, hover=True)

            @cursor.connect("add")
            def _on_add(sel):
                idx = int(sel.index)
                i = int(flat_i[idx])
                j = int(flat_j[idx])
                sel.annotation.set_text(
                    "κ₁ = {:.2e}\nκ₂ = {:.2e}\n‖GS−B‖ = {:.3e}\n‖LS‖   = {:.3e}\n‖S−Sₘ‖ = {:.3e}".format(
                        k1[i], k2[j], R[i, j], S[i, j], M[i, j]
                    )
                )

    plt.tight_layout()
    if save:
        import os
        os.makedirs('results', exist_ok=True)
        fig.savefig('results/l_surface_3d.png', dpi=300)
    plt.show(block=False)


def plot_lsurface_3d(
        residuals: np.ndarray,
        seminorms: np.ndarray,
        model_residuals: np.ndarray,
        *,
        kappa1_vals: np.ndarray | None = None,
        kappa2_vals: np.ndarray | None = None,
        cross_section: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None,
        save: bool = False
) -> None:
    """3D L-surface plot with faint wireframe cross-section at chosen κ₂."""
    import numpy as np
    import matplotlib.pyplot as plt
    eps = 1e-300
    R = np.maximum(residuals, eps)
    S = np.maximum(seminorms, eps)
    M = np.maximum(model_residuals, eps)
    X, Y, Z = np.log10(R), np.log10(S), np.log10(M)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.6, aspect=12, label="log10 model residual")
    scat = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=Z.ravel(), cmap="viridis", s=16, depthshade=True)

    ax.set_xlabel(r"log10 ||GS−B||")
    ax.set_ylabel(r"log10 ||LS||")
    ax.set_zlabel(r"log10 ||S−Sₘ||")
    ax.set_title("3D L-surface")

    # ---- cross-section at chosen kappa2 (wireframe) ----
    if cross_section is not None:
        Xs, Ys, Zs, idx_star = cross_section
        ax.plot3D(Xs, Ys, Zs, "k-", linewidth=1.0, alpha=0.6)
        ax.scatter(Xs[idx_star], Ys[idx_star], Zs[idx_star], c="k", s=40)

    if save:
        _ensure_results_dir()
        fig.savefig("results/l_surface_3d.png", dpi=300)
    plt.show(block=False)


def plot_heatmap_residual(residuals, kappa1_vals, kappa2_vals, i_star, j_star, i_knee_per_j, *, save=False):
    """Residual heatmap over (κ₁, κ₂) with slice knees and chosen point overlay."""
    eps = 1e-300
    Z = np.log10(np.maximum(residuals, eps)).T
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[np.log10(kappa1_vals[0]), np.log10(kappa1_vals[-1]),
                np.log10(kappa2_vals[0]), np.log10(kappa2_vals[-1])],
        cmap="viridis"
    )
    fig.colorbar(im, ax=ax, label="log10 ||GS−B||")
    ax.plot(np.log10(kappa1_vals[i_knee_per_j]), np.log10(kappa2_vals), "w.", ms=4, label="slice knees")
    ax.plot(np.log10(kappa1_vals[i_star]), np.log10(kappa2_vals[j_star]), "ko", label="chosen (κ₁*, κ₂*)")
    ax.set_xlabel("log10 κ₁")
    ax.set_ylabel("log10 κ₂")
    ax.legend(loc="upper right")
    ax.set_title("Residual heatmap with slice knees")
    if save:
        _ensure_results_dir()
        fig.savefig("results/residual_heatmap.png", dpi=300)
    plt.show(block=False)
