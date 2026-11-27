"""How often the parametric method recovers each parameter, on curves it has never seen.

    python -m src.regularization.parametric_model.standalones.parameter_recovery_study
    python -m src.regularization.parametric_model.standalones.parameter_recovery_study --replot

Every curve is made by the simulator from a fresh seed, so there is no model mismatch: any miss
is the method's or the problem's. Each parameter of each curve is judged against the method's
own uncertainty, sigma, the spread (in decades) of the draws that reproduce the ELE within
``uncertainty.TOLERANCE``:

  * recovered      -- the best draw is within 2 sigma of the truth, and +-2 sigma spans less
                      than ``INFORMATIVE_FRACTION`` of the parameter's box;
  * not determined -- within 2 sigma, but +-2 sigma spans most of the box: the method says the
                      ELE does not fix this parameter, and it is right to;
  * wrong          -- more than 2 sigma off: confident and wrong.

The width check is what keeps "within 2 sigma" from rewarding an answer of "anywhere". The
fractions within 1 and 2 sigma are printed too, as a calibration check (68/95% if sigma is
honest), not as a score. Curves with fewer than ``MIN_KEPT`` draws inside the tolerance have no
usable sigma and are counted apart.

Writes to ``results/parametric_model/parameter_recovery/``: ``summary.png``, one example per
outcome, and ``results.npz``, from which ``--replot`` redraws without re-running inference.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.forward_model.analytic_ele import simulate_ele
from src.forward_model.ele import measurement_wavelengths
from src.forward_model.sele_simulator import effective_lifetime, simulate_sele
from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.build_forward_model_set import random_parameters

_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_DIR = _ROOT / "results" / "parametric_model" / "parameter_recovery"

# Distinct from the training set (7) and the forward-model test set (20261111).
_SEED = 20261127
_N_CURVES = 400
_SAMPLES = 4000
INFORMATIVE_FRACTION = 0.5
MIN_KEPT = 10
_DEPTH_LIMIT_UM = 30.0
_N_DRAWN = 30

QUANTITIES = ("p0", "D", "S", "tau", "alpha_scale", "Ln", "S/D")
_RAW = QUANTITIES[:5]

RECOVERED, NOT_DETERMINED, WRONG, NO_SIGMA = 0, 1, 2, -1
_OUTCOMES = {
    RECOVERED: ("recovered", "#009E73", ""),
    NOT_DETERMINED: ("not determined", "#E69F00", "//"),
    WRONG: ("wrong", "#D55E00", "xx"),
}

# Okabe-Ito, per this repo's plotting conventions.
_TRUTH = "#000000"
_BEST = "#0072B2"
_KEPT = "#56B4E9"


def log_quantities(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(n, 7)`` log10 of the five parameters, the diffusion length and S/D."""
    params = np.atleast_2d(params)
    p0, d, s, tau, _ = params.T
    ln = np.sqrt(d * effective_lifetime(p0, tau))
    return np.log10(np.column_stack([params, ln, s / d]))


def _box_widths(spec) -> NDArray[np.float64]:
    """Each quantity's span over the parameter box, in decades.

    The derived quantities have no box of their own, so their span is read off a dense sample.
    """
    sample = spec.sample(200_000, np.random.default_rng(0))
    return np.ptp(log_quantities(sample), axis=0)


def run(n_curves: int, seed: int, n_samples: int, checkpoint: str) -> Dict[str, np.ndarray]:
    model = load_model(checkpoint)
    _, params = random_parameters(seed=seed, n=n_curves)
    ele = simulate_ele(params)

    truth = log_quantities(params)
    best = np.full_like(truth, np.nan)
    sigma = np.full_like(truth, np.nan)
    n_kept = np.zeros(n_curves, dtype=int)
    misfit = np.zeros(n_curves)

    started = time.time()
    for i in range(n_curves):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selection = unc.select(sample_sele(model, ele[i], n_samples=n_samples), ele[i])
        best[i] = log_quantities(selection.best_params)[0]
        misfit[i] = selection.best_misfit
        if selection.met:
            n_kept[i] = selection.n_kept
            sigma[i] = log_quantities(selection.kept.params).std(axis=0)
        if (i + 1) % 20 == 0 or i + 1 == n_curves:
            print(f"  {i + 1}/{n_curves} curves ({time.time() - started:.0f}s)", flush=True)

    return {
        "params": params, "truth": truth, "best": best, "sigma": sigma, "n_kept": n_kept,
        "misfit": misfit, "box_width": _box_widths(model.spec),
        "seed": np.array(seed), "n_samples": np.array(n_samples),
        "checkpoint": np.array(str(checkpoint)),
    }


def classify(results) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """``(outcome, z)``, both ``(n_curves, n_quantities)``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(results["best"] - results["truth"]) / results["sigma"]
    wide = 4.0 * results["sigma"] / results["box_width"] >= INFORMATIVE_FRACTION
    outcome = np.where(z > 2.0, WRONG, np.where(wide, NOT_DETERMINED, RECOVERED))
    outcome[results["n_kept"] < MIN_KEPT] = NO_SIGMA
    return outcome, z


def _share(hits: NDArray[np.bool_]) -> Tuple[float, float]:
    p = float(hits.mean())
    return p, float(np.sqrt(p * (1.0 - p) / hits.size))


def report(results) -> str:
    outcome, z = classify(results)
    valid = outcome[:, 0] != NO_SIGMA
    misfit = results["misfit"]
    lines = [
        f"{len(misfit)} curves, seed {int(results['seed'])}, {int(results['n_samples'])} draws each",
        f"best-draw ELE misfit {misfit.min():.3%} to {misfit.max():.3%} (median {np.median(misfit):.3%})",
        f"{int((~valid).sum())} curve(s) with fewer than {MIN_KEPT} draws within "
        f"{unc.TOLERANCE:.0%}: no sigma, left out below",
        "",
        f"{'':>12} {'recovered':>14} {'not determined':>16} {'wrong':>14}"
        f" {'within 1 sigma':>16} {'within 2 sigma':>16}",
    ]
    for j, name in enumerate(QUANTITIES):
        cells = [_share(outcome[valid, j] == k) for k in (RECOVERED, NOT_DETERMINED, WRONG)]
        cells += [_share(z[valid, j] <= limit) for limit in (1.0, 2.0)]
        widths = (14, 16, 14, 16, 16)
        lines.append(f"{name:>12} " + " ".join(
            f"{f'{p:.0%} +- {se:.0%}':>{w}}" for (p, se), w in zip(cells, widths)))
        if j == len(_RAW) - 1:
            lines.append(f"{'derived':>12}")
    return "\n".join(lines)


def _summary_figure(results, path: Path) -> None:
    outcome, _ = classify(results)
    valid = outcome[:, 0] != NO_SIGMA
    n_valid = int(valid.sum())
    misfit = results["misfit"]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for row, name in enumerate(_RAW):
        left = 0.0
        for k, (label, color, hatch) in _OUTCOMES.items():
            p, se = _share(outcome[valid, row] == k)
            ax.barh(row, p * 100, left=left, color=color, hatch=hatch, edgecolor="white",
                    linewidth=0.8, label=label if row == 0 else None)
            if p >= 0.08:
                ax.text(left + p * 50, row, f"{p:.0%} ± {se:.0%}", ha="center", va="center",
                        fontsize=8.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
            left += p * 100
    ax.set_yticks(range(len(_RAW)))
    ax.set_yticklabels(_RAW)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of curves [%]")
    ax.set_title(
        f"Parameter recovery on {n_valid} unseen simulator curves\n"
        f"every best draw reproduces the ELE (misfit {misfit.min():.2%}–{misfit.max():.2%});"
        f" judged at 2σ of the draws within {unc.TOLERANCE:.0%} in ELE", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_examples(results) -> Dict[int, int]:
    """For each outcome, the curve with the most raw parameters in it, cleanest ELE fit first."""
    outcome, _ = classify(results)
    raw = outcome[:, :len(_RAW)]
    picks = {}
    for k in _OUTCOMES:
        counts = (raw == k).sum(axis=1)
        counts[outcome[:, 0] == NO_SIGMA] = -1
        candidates = np.flatnonzero(counts == counts.max())
        picks[k] = int(candidates[np.argmin(results["misfit"][candidates])])
    return picks


def _example_figure(results, index: int, kind: int, model, path: Path) -> None:
    params = results["params"][index]
    ele = simulate_ele(params[np.newaxis, :])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble = sample_sele(model, ele, n_samples=int(results["n_samples"]))
        selection = unc.select(ensemble, ele)
    truth_sele = simulate_sele(params[np.newaxis, :], ensemble.z_cm)[0]

    kept = selection.kept
    shown = np.linspace(0, kept.sele.shape[0] - 1, min(_N_DRAWN, kept.sele.shape[0])).astype(int)
    wavelengths = measurement_wavelengths()
    z_um = ensemble.z_cm / 1e-4
    depth = z_um <= _DEPTH_LIMIT_UM

    fig, (ax_ele, ax_sele, ax_par) = plt.subplots(
        1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})

    for j in shown:
        ax_ele.plot(wavelengths, kept.ele[j] * 100, color=_KEPT, linewidth=0.8, alpha=0.5)
        ax_sele.plot(z_um[depth], kept.sele[j, depth] * 100, color=_KEPT, linewidth=0.8,
                     alpha=0.5)
    ax_ele.plot(wavelengths, selection.best_ele * 100, color=_BEST, linewidth=2.0)
    ax_ele.plot(wavelengths, ele * 100, color=_TRUTH, linewidth=1.4, linestyle="--",
                marker="o", markersize=3)
    ax_sele.plot(z_um[depth], selection.best_sele[depth] * 100, color=_BEST, linewidth=2.0)
    ax_sele.plot(z_um[depth], truth_sele[depth] * 100, color=_TRUTH, linewidth=1.4,
                 linestyle="--")

    ax_ele.set_xlabel("excitation wavelength [nm]")
    ax_ele.set_ylabel("ELE [%]")
    ax_ele.set_title(f"ELE: best draw misses the truth by {selection.best_misfit:.2%}",
                     fontsize=10)
    ax_sele.set_xlabel("depth z [µm]")
    ax_sele.set_ylabel("SELE [%]")
    ax_sele.set_xlim(0, _DEPTH_LIMIT_UM)
    ax_sele.set_ylim(bottom=0)
    ax_sele.set_title("SELE behind those ELE curves", fontsize=10)
    for ax in (ax_ele, ax_sele):
        ax.grid(alpha=0.2)
    ax_ele.legend(handles=[
        plt.Line2D([], [], color=_TRUTH, linestyle="--", marker="o", markersize=3,
                   label="truth"),
        plt.Line2D([], [], color=_BEST, linewidth=2.0, label="best draw"),
        plt.Line2D([], [], color=_KEPT, linewidth=0.8,
                   label=f"draws within {unc.TOLERANCE:.0%} ({selection.n_kept} in all)"),
    ], fontsize=8, frameon=False)

    _parameter_panel(ax_par, results, index, model.spec)

    label = _OUTCOMES[kind][0]
    fig.suptitle(f"Example of a '{label}' outcome: same ELE, parameters judged against the "
                 f"method's own spread", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parameter_panel(ax, results, index: int, spec) -> None:
    """Each raw parameter on its own box-wide axis: truth, best draw, and +-1/2 sigma bars."""
    outcome, z = classify(results)
    for row, name in enumerate(_RAW):
        lo, hi = np.log10(spec.lower[row]), np.log10(spec.upper[row])
        to_unit = lambda v: (v - lo) / (hi - lo)
        best = to_unit(results["best"][index, row])
        truth = to_unit(results["truth"][index, row])
        sigma = results["sigma"][index, row] / (hi - lo)
        ax.plot([0, 1], [row, row], color="#BBBBBB", linewidth=0.8)
        for width, k, alpha in ((2, 2.0, 0.45), (6, 1.0, 0.6)):
            ax.plot(np.clip([best - k * sigma, best + k * sigma], 0.0, 1.0), [row, row],
                    color=_BEST, linewidth=width, alpha=alpha, solid_capstyle="butt")
        ax.plot(best, row, marker="o", color=_BEST, markersize=8, linestyle="none",
                markeredgecolor="white")
        ax.plot(truth, row, marker="D", color=_TRUTH, markersize=7, linestyle="none",
                markerfacecolor="none", markeredgewidth=1.6)
        ax.text(-0.02, row - 0.28, f"{10 ** lo:.2g}", ha="left", fontsize=7, color="#555555")
        ax.text(1.02, row - 0.28, f"{10 ** hi:.2g}", ha="right", fontsize=7, color="#555555")
        label, color, _ = _OUTCOMES[outcome[index, row]]
        ax.text(1.06, row, f"{label}\n{z[index, row]:.1f}σ off", va="center", fontsize=8.5,
                color=color, fontweight="bold")
    ax.set_yticks(range(len(_RAW)))
    ax.set_yticklabels(_RAW)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(len(_RAW) - 0.4, -0.7)
    ax.set_xticks([])
    ax.set_xlabel("position in the training box (log scale, ends labelled)")
    ax.set_title("Parameters", fontsize=10)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[
        plt.Line2D([], [], marker="D", color=_TRUTH, markerfacecolor="none", linestyle="none",
                   label="truth"),
        plt.Line2D([], [], marker="o", color=_BEST, linestyle="none", label="best draw"),
        plt.Line2D([], [], color=_BEST, linewidth=6, alpha=0.6, label="±1σ"),
        plt.Line2D([], [], color=_BEST, linewidth=2, alpha=0.45, label="±2σ"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=8, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curves", type=int, default=_N_CURVES)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--samples", type=int, default=_SAMPLES)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the cached results.npz without running inference")
    args = parser.parse_args()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = _OUTPUT_DIR / "results.npz"
    if args.replot:
        results = dict(np.load(cache))
    else:
        results = run(args.curves, args.seed, args.samples, args.checkpoint)
        np.savez(cache, **results)

    print(report(results))

    model = load_model(str(results["checkpoint"]))
    _summary_figure(results, _OUTPUT_DIR / "summary.png")
    for kind, index in _pick_examples(results).items():
        name = _OUTCOMES[kind][0].replace(" ", "_")
        _example_figure(results, index, kind, model, _OUTPUT_DIR / f"example_{name}.png")
    print(f"figures written to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
