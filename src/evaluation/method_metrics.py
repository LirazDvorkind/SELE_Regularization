"""Relative RMSE of every reconstruction method, in SELE and ELE, per test-set curve.

    python -m src.evaluation.method_metrics

Uses the same reconstructions as ``plot_method_comparison`` (and its score-model cache).
The error is ||x - ref|| / ||ref|| in percent.

- SELE: against the ground truth over 0-30 um, the range every method reconstructs.
- ELE: against the measured (native-mesh) ELE, with every reconstruction pushed through
  one shared fine-mesh G. Each method's own operator would fold its own discretisation error
  into the score; the shared one keeps the comparison on equal terms. Its floor -- the true
  SELE pushed through the same G -- is reported alongside, so a method cannot look better
  than the operator allows.

Writes ``metrics_per_curve.csv`` and ``summary.txt`` to
``results/method_comparison/``.
"""

from __future__ import annotations

import csv

import numpy as np

from src.evaluation.plot_method_comparison import (
    METHODS, OUTPUT_DIR, reconstruct_all, reference_G, refit)
from src.test_set.loader import load_test_set

def _relative_rmse_pct(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref) * 100)


def _table(title, curve_ids, values):
    """``values[method_key][i]`` as percentages, one row per curve and a mean row."""
    width = max(len(label) for _, label, *_ in METHODS) + 2
    lines = [title, f"{'curve':<12}" + "".join(f"{label:>{width}}" for _, label, *_ in METHODS)]
    for i, curve_id in enumerate(curve_ids):
        lines.append(f"{curve_id:<12}" + "".join(
            f"{f'{values[key][i]:.2f}%':>{width}}" for key, *_ in METHODS))
    lines.append(f"{'mean':<12}" + "".join(
        f"{f'{np.mean(values[key]):.2f}%':>{width}}" for key, *_ in METHODS))
    return lines


def main() -> None:
    curves = load_test_set(include_reference=False)
    results = reconstruct_all(curves)
    G, z_ref = reference_G()
    curve_ids = [c.curve_id for c in curves]

    sele_err = {key: [] for key, *_ in METHODS}
    ele_err = {key: [] for key, *_ in METHODS}
    floor = []
    for curve in curves:
        z_cm, truth, reconstructions = results[curve.curve_id]
        floor.append(_relative_rmse_pct(G @ np.interp(z_ref, curve.z_cm, curve.sele), curve.ele))
        for key, *_ in METHODS:
            sele = reconstructions[key]
            sele_err[key].append(_relative_rmse_pct(sele, truth))
            ele_err[key].append(_relative_rmse_pct(refit(G, z_ref, z_cm, sele), curve.ele))

    with open(OUTPUT_DIR / "metrics_per_curve.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve"] + [f"{key}_sele_pct" for key, *_ in METHODS]
                        + [f"{key}_ele_pct" for key, *_ in METHODS])
        for i, curve_id in enumerate(curve_ids):
            writer.writerow([curve_id] + [f"{sele_err[key][i]:.4g}" for key, *_ in METHODS]
                            + [f"{ele_err[key][i]:.4g}" for key, *_ in METHODS])

    lines = ["Relative RMSE = ||x - ref|| / ||ref||, in percent.", ""]
    lines += _table("SELE, against the true SELE over 0-30 um:", curve_ids, sele_err)
    lines += [""]
    lines += _table("ELE, against the measured ELE (every method refit through one shared "
                    "fine-mesh G):", curve_ids, ele_err)
    lines += ["", f"ELE floor -- the true SELE through that same G -- averages "
                  f"{np.mean(floor):.2f}%, for scale."]

    text = "\n".join(lines)
    (OUTPUT_DIR / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
