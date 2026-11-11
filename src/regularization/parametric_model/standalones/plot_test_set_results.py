"""Consolidated figures for the parametric method on both test sets.

    python -m src.regularization.parametric_model.standalones.plot_test_set_results

Writes to ``results/parametric_model/test_set_comparison/``:
  * ``forward_model_curves.png`` -- every forward-model curve: truth, best draw, and the
    envelope of draws within ``uncertainty.TOLERANCE`` in ELE.
  * ``paper_curves.png`` -- the same for Tamir Yeshurun's paper-figure curves.
  * ``paired_sweeps.png`` -- each paper sweep curve overlaid with its forward-model twin at
    the same SRV or lifetime.
  * ``parameter_recovery.png`` -- forward-model set only, since only it has true parameters:
    best draw against truth, with the range over the draws that fit.
  * ``set_comparison.png`` -- per-zone SELE error and 0-3 um coverage, both sets side by side.
    The gap between them is the model mismatch the paper-figure curves carry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.regularization.parametric_model import uncertainty as unc
from src.regularization.parametric_model.inference import load_model, sample_sele
from src.regularization.parametric_model.standalones.run_inference import _DEFAULT_CHECKPOINT
from src.test_set.loader import load_forward_model_set, load_test_set

_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT_DIR = _ROOT / "results" / "parametric_model" / "test_set_comparison"
_ZONES = ((0.0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 30.0))
_COLUMNS = 6

# Okabe-Ito, per this repo's plotting conventions.
_TRUTH = "#D55E00"
_BEST = "#0072B2"
_KEPT = "#56B4E9"
_PAPER = "#E69F00"
_FORWARD = "#009E73"
_INK_MUTED = "#555555"
# Paired-sweep figure: Okabe-Ito blue and reddish purple, each with a darker shade for the truth.
_PAIR_STYLE = {
    "paper": {"best": "#0072B2", "band": "#56B4E9", "truth": "#004A75", "name": "Yeshurun"},
    "forward": {"best": "#CC79A7", "band": "#CC79A7", "truth": "#8A3D6E",
                "name": "forward model"},
}
_GROUP_STYLE = {
    "sweep_srv": ("#0072B2", "o", "SRV sweep"),
    "sweep_tau": ("#E69F00", "s", "lifetime sweep"),
    "random": ("#009E73", "^", "random draw"),
}


@dataclass
class Scored:
    curve_id: str
    group: str
    z_um: NDArray[np.float64]
    truth: NDArray[np.float64]           # on the solver mesh
    selection: unc.Selection
    zone_errors: NDArray[np.float64]     # relative, one per _ZONES entry
    coverage_0_3: float
    true_params: NDArray[np.float64] | None = None


def _score(model, curve_id, group, ele, truth_z, truth_sele, n_samples, true_params=None):
    ensemble = sample_sele(model, ele, n_samples=n_samples)
    truth = np.interp(ensemble.z_cm, truth_z, truth_sele)
    selection = unc.select(ensemble, ele)
    z_um = ensemble.z_cm / 1e-4

    errors = []
    for low, high in _ZONES:
        zone = (z_um >= low) & (z_um < high)
        errors.append(np.linalg.norm(selection.best_sele[zone] - truth[zone])
                      / np.linalg.norm(truth[zone]))
    inside = (truth >= selection.low) & (truth <= selection.high)
    return Scored(curve_id, group, z_um, truth, selection, np.array(errors),
                  float(inside[z_um < 3.0].mean()), true_params)


def _curve_grid(scored: List[Scored], title: str, path: Path) -> None:
    rows = int(np.ceil(len(scored) / _COLUMNS))
    fig, axes = plt.subplots(rows, _COLUMNS, figsize=(3.0 * _COLUMNS, 2.5 * rows + 0.8),
                             squeeze=False)
    for ax, item in zip(axes.flat, scored):
        x = item.z_um
        ax.fill_between(x, item.selection.low * 100, item.selection.high * 100,
                        color=_KEPT, alpha=0.35, linewidth=0)
        ax.plot(x, item.selection.best_sele * 100, color=_BEST, linewidth=2.0)
        ax.plot(x, item.truth * 100, color=_TRUTH, linewidth=1.6, linestyle="--")
        ax.set_xlim(0, x[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{item.curve_id}\nerror 0–1 µm {item.zone_errors[0]:.1%}, "
                     f"1–3 µm {item.zone_errors[1]:.1%}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(scored):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("depth z [µm]", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("SELE [%]", fontsize=8)

    handles = [
        plt.Line2D([], [], color=_TRUTH, linewidth=1.6, linestyle="--", label="ground truth"),
        plt.Line2D([], [], color=_BEST, linewidth=2.0, label="best draw"),
        plt.Rectangle((0, 0), 1, 1, color=_KEPT, alpha=0.35,
                      label=f"draws within {unc.TOLERANCE:.0%} in ELE"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0 - 0.15 / fig.get_figheight()))
    fig.suptitle(title, fontsize=11, y=1.0 + 0.1 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _paired_sweeps(paper: List[Scored], forward: List[Scored], path: Path) -> None:
    """Paper sweeps with their forward-model twins (same SRV or lifetime) overlaid.

    The twins share only the swept value; the other parameters differ, so the curves are not
    expected to coincide. What compares is how well each set's own truth is reconstructed.
    """
    twins = {item.curve_id.removeprefix("fm_"): item for item in forward
             if item.group != "random"}
    panels = [(p, twins.pop(p.curve_id, None)) for p in paper]
    panels += [(None, f) for f in twins.values()]

    rows = int(np.ceil(len(panels) / _COLUMNS))
    fig, axes = plt.subplots(rows, _COLUMNS, figsize=(3.0 * _COLUMNS, 2.6 * rows + 0.8),
                             squeeze=False)
    for ax, pair in zip(axes.flat, panels):
        lines = []
        for key, item in zip(("paper", "forward"), pair):
            if item is None:
                continue
            style = _PAIR_STYLE[key]
            ax.fill_between(item.z_um, item.selection.low * 100, item.selection.high * 100,
                            color=style["band"], alpha=0.3, linewidth=0)
            ax.plot(item.z_um, item.selection.best_sele * 100, color=style["best"],
                    linewidth=2.0)
            ax.plot(item.z_um, item.truth * 100, color=style["truth"], linewidth=1.5,
                    linestyle="--")
            lines.append(f"{style['name']} {item.zone_errors[0]:.1%} / "
                         f"{item.zone_errors[1]:.1%}")
        name = (pair[0] or pair[1]).curve_id.removeprefix("fm_")
        ax.set_title(name + "\n" + "\n".join(lines), fontsize=8)
        ax.set_xlim(0, (pair[0] or pair[1]).z_um[-1])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("depth z [µm]", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("SELE [%]", fontsize=8)

    handles = []
    for key in ("paper", "forward"):
        style = _PAIR_STYLE[key]
        handles += [
            plt.Line2D([], [], color=style["truth"], linewidth=1.5, linestyle="--",
                       label=f"{style['name']} truth"),
            plt.Line2D([], [], color=style["best"], linewidth=2.0,
                       label=f"{style['name']} best draw"),
            plt.Rectangle((0, 0), 1, 1, color=style["band"], alpha=0.3,
                          label=f"{style['name']} draws within {unc.TOLERANCE:.0%} in ELE"),
        ]
    # Legends fill column-major, so two columns put each set in its own column.
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0 - 0.15 / fig.get_figheight()))
    fig.suptitle("Yeshurun sweeps and forward-model sweeps at the same SRV / lifetime "
                 "(titles: SELE error 0–1 µm / 1–3 µm)",
                 fontsize=11, y=1.0 + 0.1 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parameter_recovery(scored: List[Scored], names, units, path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.6))
    for j, (ax, name, unit) in enumerate(zip(axes, names, units)):
        all_values = []
        for item in scored:
            color, marker, _ = _GROUP_STYLE[item.group]
            true = item.true_params[j]
            kept = item.selection.kept.params[:, j]
            best = item.selection.best_params[j]
            ax.plot([true, true], [kept.min(), kept.max()], color=color, alpha=0.5,
                    linewidth=1.2)
            ax.plot(true, best, marker=marker, color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8, linestyle="none")
            all_values += [true, kept.min(), kept.max()]
        low, high = min(all_values) / 1.5, max(all_values) * 1.5
        ax.plot([low, high], [low, high], color=_INK_MUTED, linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title(f"{name} [{unit}]" if unit != "-" else name, fontsize=10)
        ax.set_xlabel("true", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("best draw (bar: range over draws that fit)", fontsize=9)

    handles = [plt.Line2D([], [], color=c, marker=m, linestyle="none", markersize=6, label=l)
               for c, m, l in _GROUP_STYLE.values()]
    fig.suptitle("Forward-model set: parameters are not recovered, even where SELE is",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _set_comparison(paper: List[Scored], forward: List[Scored], path: Path) -> None:
    fig, (ax_err, ax_cov) = plt.subplots(1, 2, figsize=(11, 4.2),
                                         gridspec_kw={"width_ratios": [3, 1]})
    rng = np.random.default_rng(0)
    sets = (("paper figures (COMSOL)", paper, _PAPER, "o", -0.17),
            ("forward model", forward, _FORWARD, "s", 0.17))

    for label, items, color, marker, offset in sets:
        errors = np.array([item.zone_errors for item in items]) * 100
        for k in range(len(_ZONES)):
            x = k + offset + rng.uniform(-0.06, 0.06, errors.shape[0])
            ax_err.plot(x, errors[:, k], marker=marker, color=color, linestyle="none",
                        markersize=5, alpha=0.8, markeredgecolor="white",
                        markeredgewidth=0.6, label=label if k == 0 else None)
            ax_err.plot([k + offset - 0.12, k + offset + 0.12], [np.median(errors[:, k])] * 2,
                        color=color, linewidth=2.5)

        coverage = np.array([item.coverage_0_3 for item in items]) * 100
        x = (0 if offset < 0 else 1) + rng.uniform(-0.12, 0.12, coverage.size)
        ax_cov.plot(x, coverage, marker=marker, color=color, linestyle="none", markersize=5,
                    alpha=0.8, markeredgecolor="white", markeredgewidth=0.6)

    ax_err.set_yscale("log")
    ax_err.set_xticks(range(len(_ZONES)))
    ax_err.set_xticklabels([f"{lo:g}–{hi:g} µm" for lo, hi in _ZONES])
    ax_err.set_ylabel("best-draw SELE error [%]")
    ax_err.set_title("SELE error by depth zone (bar: median)", fontsize=10)
    ax_err.grid(alpha=0.2, axis="y")
    ax_err.legend(frameon=False, fontsize=9, loc="upper left")

    ax_cov.set_xticks([0, 1])
    ax_cov.set_xticklabels(["paper", "forward\nmodel"])
    ax_cov.set_xlim(-0.5, 1.5)
    ax_cov.set_ylim(-5, 105)
    ax_cov.set_ylabel("depths with truth inside envelope [%]")
    ax_cov.set_title("coverage, 0–3 µm", fontsize=10)
    ax_cov.grid(alpha=0.2, axis="y")

    fig.suptitle("Parametric reconstruction: paper-figure truths vs forward-model truths",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper = [_score(model, c.curve_id, c.param_name, c.ele, c.z_cm, c.sele, args.samples)
             for c in load_test_set(include_reference=False)]
    # true_params is handed to the scorer only for plotting; inference sees c.ele alone.
    forward = [_score(model, c.curve_id, c.group, c.ele, c.z_cm, c.sele, args.samples,
                      true_params=c.true_params)
               for c in load_forward_model_set()]

    _curve_grid(forward, "Forward-model test set: parametric reconstruction",
                _OUTPUT_DIR / "forward_model_curves.png")
    _curve_grid(paper, "Paper-figure test set (Yeshurun): parametric reconstruction",
                _OUTPUT_DIR / "paper_curves.png")
    _paired_sweeps(paper, forward, _OUTPUT_DIR / "paired_sweeps.png")
    _parameter_recovery(forward, model.spec.names, model.spec.units,
                        _OUTPUT_DIR / "parameter_recovery.png")
    _set_comparison(paper, forward, _OUTPUT_DIR / "set_comparison.png")
    print(f"figures written to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
