"""Convergence study: does B = G @ S converge as mesh resolution increases?

Uses a 10000-point SELE dataset as ground truth. Sweeps resolutions evenly from
500 to 10000, downsample the same SELE curves to R points, build G_R via
Beer-Lambert, and compare B_R = G_R @ S_R against the reference B_10000.

Run from repo root:
    python src/regularization/score_model/standalones/model_training/debug_resolution_convergence.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import win32clipboard
from PIL import Image

from src.io import load_csv
from src.mesh import _linear_mesh
from src.utils import match_length_interp

# ── Paths & constants ─────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[5]
_DATA_DIR    = _REPO_ROOT / "Data"
_SM_DATA_DIR = _DATA_DIR / "score_model"

W = 30e-4  # device width [cm], matches both d32 and d500 presets

RESOLUTIONS   = [int(r) for r in np.linspace(500, 10000, 7)]  # [500, 2083, 3667, 5250, 6833, 8417, 10000]
CURVE_INDICES = [0, 1, 10, 50]

# Wong colorblind-safe palette (7 colours, one per resolution)
COLORS      = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
LINESTYLES  = ["solid", "dashed", "dotted", "dashdot", (0,(3,1,1,1)), "dashed", "dotted"]

# ── Load optical constants ────────────────────────────────────────────────────
k                = load_csv(str(_DATA_DIR / "k.csv")).ravel()
lambda_for_alpha = load_csv(str(_DATA_DIR / "n_k_wavelength_nm.csv")).ravel()
wavelengths      = load_csv(str(_DATA_DIR / "wavelength_nm.csv")).ravel()

# ── Load ground-truth SELE profiles at 500 pts ───────────────────────────────
S_all = load_csv(str(_SM_DATA_DIR / "datasets" / "sele_simulated_100_curves_10000_long.csv"))
print(f"Loaded SELE dataset: {S_all.shape}  (curves × spatial pts)")

# ── Build G_1000 and reference B once ────────────────────────────────────────
G_10000, _ = _linear_mesh(wavelengths, k, lambda_for_alpha, W, 10000)

ref_B = {idx: G_10000 @ S_all[idx] for idx in CURVE_INDICES}

# ── Sweep resolutions, store B vectors ───────────────────────────────────────
B_at_res = {idx: {} for idx in CURVE_INDICES}  # curve_idx -> {R: B_R}

for R in RESOLUTIONS:
    G_R, _ = _linear_mesh(wavelengths, k, lambda_for_alpha, W, R)
    for idx in CURVE_INDICES:
        S_R = match_length_interp(S_all[idx], R)
        B_at_res[idx][R] = G_R @ S_R

# ── Plot: B overlay at all resolutions ───────────────────────────────────────
fig, axes = plt.subplots(len(CURVE_INDICES), 1, figsize=(10, 3.5 * len(CURVE_INDICES)))

for c_idx, idx in enumerate(CURVE_INDICES):
    ax = axes[c_idx]
    for r_idx, R in enumerate(RESOLUTIONS):
        lw = 2.5 if R == 10000 else 1.0
        ax.plot(wavelengths, B_at_res[idx][R], color=COLORS[r_idx],
                linestyle=LINESTYLES[r_idx], linewidth=lw,
                label=f"R={R}" + (" (ref)" if R == 10000 else ""))
    ax.set_title(f"curve {idx}")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("B (ELE)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("B = G @ S at each resolution  (bold = R=10000 reference)")
fig.tight_layout()


def _copy_figure_to_clipboard(event):
    if event.key != "ctrl+c":
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    bmp_buf = io.BytesIO()
    img.save(bmp_buf, format="BMP")
    bmp_data = bmp_buf.getvalue()[14:]  # strip BMP file header, keep DIB header + pixels
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, bmp_data)
    win32clipboard.CloseClipboard()
    print("Figure copied to clipboard.")


fig.canvas.mpl_connect("key_press_event", _copy_figure_to_clipboard)
plt.show()
