"""Overlay every reconstruction method against the ground truth, one panel per test-set curve.

    python -m src.evaluation.plot_method_comparison

Writes ``all_methods_sele_overlay.png`` (SELE against the truth) and
``all_methods_ele_overlay.png`` (each SELE refit to ELE against the measurement) to
``results/method_comparison/``. Every method is fed the curve's
native-mesh ELE, and none is told anything about the truth:

- parametric model: its best-fitting draw, a real simulator curve;
- score model: the d500 preset on the 500-element solver mesh;
- Tikhonov (L2): the non-uniform mesh, kappa range and confidence window from ``CONFIG``,
  kappa picked by the L-curve knee -- one setup for every curve.

The score-model solves take ~40 minutes, so they are cached under ``.score_cache/`` next to
the figure and reused on later runs; delete that folder after changing the score model.
"""

from __future__ import annotations

from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.__init__ import CONFIG
from src.io import load_csv
from src.mesh import _compute_front_generation, calc_mesh_and_G
from src.optical_constants import load_optical_constants
from src.operators import build_L
from src.pipeline import _load_G_inputs
from src.regularization import tikhonov_non_uniform
from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import GroundTruthCurve, load_curve, load_on_solver_mesh, load_test_set
from src.types.config import SCORE_MODEL_PRESETS
from src.types.enums import RegularizationMethod

_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = _ROOT / "results" / "method_comparison"
SCORE_CACHE = OUTPUT_DIR / ".score_cache"
_SCORE_PRESET = "d500"
_WORKERS = 6

# (key, label, colour, linestyle, width) -- Okabe-Ito; styles vary so hue is not the only cue.
METHODS = (
    ("parametric", "Parametric model", "#0072B2", "-", 1.4),
    ("score", "Score model", "#009E73", "-.", 1.4),
    ("tikhonov", "Tikhonov (L2) non linear mesh", "#CC79A7", "--", 1.8),
)


def _score_solve(curve_id: str) -> str:
    import torch
    from src.regularization.score_model.score_model_grad import load_score_model, solve_gradient_descent

    torch.set_num_threads(1)
    hyperparams = replace(SCORE_MODEL_PRESETS[_SCORE_PRESET], IS_SHOW_DEBUG_DATA=False,
                          IS_SHOW_DEBUG_PLOT=False, IS_SHOW_MSE_PLOT=False)
    model = load_score_model(hyperparams.model_path)
    curve = load_curve(curve_id)
    mesh = load_on_solver_mesh(curve_id, model[3])
    truth = np.interp(mesh.z_centres_cm, curve.z_cm, curve.sele)  # diagnostics only
    sele = solve_gradient_descent(mesh.G, curve.ele, hyperparams, S_gt=truth, preloaded_model=model)
    np.save(SCORE_CACHE / f"{curve_id}.npy", sele)
    return curve_id


def _score_reconstructions(curves: list[GroundTruthCurve]) -> None:
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [c.curve_id for c in curves if not (SCORE_CACHE / f"{c.curve_id}.npy").exists()]
    if not todo:
        return
    with Pool(min(_WORKERS, len(todo))) as pool:
        for done in pool.imap_unordered(_score_solve, todo):
            print(f"  score model: {done}", flush=True)


class _Tikhonov:
    def __init__(self) -> None:
        z = load_csv(CONFIG.data_paths.z).ravel()
        G, z = calc_mesh_and_G(RegularizationMethod.NON_UNIFORM_MESH,
                               _load_G_inputs(z, CONFIG.data_paths.wavelengths))
        self.unit = CONFIG.photon_flux * CONFIG.e_charge
        self.G = G * self.unit
        self.z_cm = 0.5 * (z[:-1] + z[1:])
        self.L = build_L(CONFIG.L_flag, len(z) - 1)
        kappa_max, kappa_min = CONFIG.kappa_range
        self.kappa_vals = np.logspace(np.log10(kappa_max), np.log10(kappa_min), CONFIG.n_kappa)

    def reconstruct(self, ele):
        residuals, seminorms, S_list = tikhonov_non_uniform.sweep_kappa(
            self.G, ele * self.unit, self.L, self.kappa_vals)
        knee, _ = tikhonov_non_uniform.find_knee(residuals, seminorms, self.kappa_vals)
        window = ((self.kappa_vals >= knee / CONFIG.conf_window)
                  & (self.kappa_vals <= knee * CONFIG.conf_window))
        return np.stack([S for S, keep in zip(S_list, window) if keep], axis=1).mean(axis=1)


# 1 nm elements through the absorbing region, then geometric to the back of the wafer.
_REFERENCE_EDGES_CM = np.concatenate([np.linspace(0.0, 20e-4, 20001),
                                      np.geomspace(20e-4, 350e-4, 2000)[1:]])


def reference_G():
    constants = load_optical_constants()
    wavelengths = load_csv(CONFIG.data_paths.wavelengths).ravel()
    G = _compute_front_generation(
        k=np.interp(wavelengths, constants.wavelength_nm, constants.k),
        wavelength_nm=wavelengths,
        z_cm=_REFERENCE_EDGES_CM,
        k_bulk=np.interp(wavelengths, constants.wavelength_nm, constants.k_bulk),
    )
    return G, 0.5 * (_REFERENCE_EDGES_CM[:-1] + _REFERENCE_EDGES_CM[1:])


def refit(G, z_ref, z_cm, sele):
    # Beyond the reconstructed range the profile is set to zero: the light is fully absorbed
    # within a few um, so the deep tail contributes nothing to the ELE.
    return G @ np.interp(z_ref, z_cm, sele, right=0.0)


def reconstruct_all(curves: list[GroundTruthCurve]) -> dict[str, tuple]:
    """``{curve_id: (z_cm, truth, {method: sele})}``, all on the 500-element solver mesh."""
    _score_reconstructions(curves)
    tikhonov = _Tikhonov()
    parametric = load_model(str(_DEFAULT_CHECKPOINT))

    results = {}
    for curve in curves:
        z_cm = load_on_solver_mesh(curve.curve_id).z_centres_cm
        ensemble = sample_sele(parametric, curve.ele)
        selection = unc.select(ensemble, curve.ele)
        results[curve.curve_id] = (z_cm, np.interp(z_cm, curve.z_cm, curve.sele), {
            "parametric": np.interp(z_cm, ensemble.z_cm, selection.best_sele),
            "score": np.load(SCORE_CACHE / f"{curve.curve_id}.npy"),
            "tikhonov": np.interp(z_cm, tikhonov.z_cm, tikhonov.reconstruct(curve.ele)),
        })
    return results


def _panel_grid(curves):
    cols = 4
    rows = int(np.ceil(len(curves) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(17, 3.1 * rows), sharex=True)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    return figure, axes


def _finish(figure, axes, xlabel, ylabel, title, filename):
    # sharex hides tick labels on all but the last row; show them on every panel, and put the
    # axis label on the lowest visible panel of each column since the last row is partly empty.
    for ax in axes.flat:
        ax.xaxis.set_tick_params(labelbottom=True)
    for column in axes.T:
        [ax for ax in column if ax.axison][-1].set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", frameon=False, fontsize=10,
                  bbox_to_anchor=(0.98, 0.04))
    figure.suptitle(title)
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"saved {path}")


def plot_sele(curves, results):
    figure, axes = _panel_grid(curves)
    for ax, curve in zip(axes.flat, curves):
        z_cm, truth, reconstructions = results[curve.curve_id]
        z_um = z_cm * 1e4
        ax.plot(z_um, truth * 100, color="#000000", linewidth=2.2, label="ground truth")
        for key, label, colour, style, width in METHODS:
            ax.plot(z_um, reconstructions[key] * 100, color=colour, linestyle=style,
                    linewidth=width, label=label)
        # Clipped to the truth's range so a method that runs off does not flatten the rest.
        peak = truth.max() * 100
        ax.set_ylim(-0.5 * peak, 2.0 * peak)
        ax.set_xlim(0, 30)
        ax.set_title(curve.label, fontsize=10)
        ax.grid(alpha=0.25)
    _finish(figure, axes, "depth z [um]", "SELE [%]",
            "All methods against the ground truth (y clipped to the truth's range)",
            "all_methods_sele_overlay.png")


def plot_ele(curves, results):
    G, z_ref = reference_G()
    wavelengths = load_csv(CONFIG.data_paths.wavelengths).ravel()
    figure, axes = _panel_grid(curves)
    for ax, curve in zip(axes.flat, curves):
        z_cm, _, reconstructions = results[curve.curve_id]
        ax.plot(wavelengths, curve.ele * 100, color="#000000", linestyle="none", marker="o",
                markersize=3.5, label="measured ELE")
        for key, label, colour, style, width in METHODS:
            ax.plot(wavelengths, refit(G, z_ref, z_cm, reconstructions[key]) * 100,
                    color=colour, linestyle=style, linewidth=width, label=label)
        ax.set_title(curve.label, fontsize=10)
        ax.grid(alpha=0.25)
    _finish(figure, axes, "wavelength [nm]", "ELE [%]",
            "Each method's SELE pushed back through one shared fine-mesh G, against the "
            "measured ELE", "all_methods_ele_overlay.png")


def main() -> None:
    curves = load_test_set(include_reference=False)
    results = reconstruct_all(curves)
    plot_sele(curves, results)
    plot_ele(curves, results)

if __name__ == "__main__":
    main()
