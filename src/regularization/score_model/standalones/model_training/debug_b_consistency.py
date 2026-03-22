"""Debug: does G_32 @ S_32 ≈ G_500 @ S_500 for the same physical curve?

Run from repo root:
    python src/regularization/score_model/standalones/model_training/debug_b_consistency.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.io import load_csv
from src.utils import match_length_interp

_DATA_DIR = Path(__file__).resolve().parents[5] / "Data" / "score_model"

G32  = load_csv(str(_DATA_DIR / "G_score_model.csv"))       # (28, 32)
G500 = load_csv(str(_DATA_DIR / "G_score_model_500.csv"))   # (28, 500)

S_all = load_csv(str(_DATA_DIR / "sele_dataset.csv"))       # use d32 dataset as source of truth

print(f"G32  shape : {G32.shape}")
print(f"G500 shape : {G500.shape}")

# ── Figure 1: G matrix heatmaps ───────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
im1 = ax1.imshow(G32,  aspect='auto', interpolation='nearest')
ax1.set_title("G32  (28 × 32)")
ax1.set_xlabel("spatial cell index")
ax1.set_ylabel("wavelength index")
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(G500, aspect='auto', interpolation='nearest')
ax2.set_title("G500  (28 × 500)")
ax2.set_xlabel("spatial cell index")
ax2.set_ylabel("wavelength index")
plt.colorbar(im2, ax=ax2)
fig1.suptitle("G matrix heatmaps")
fig1.tight_layout()

# ── Figure 2: row sums (should match between G32 and G500) ───────────────────
# Each row sum = total absorbed fraction at that wavelength = 1 - e^{-α W}
# If both G matrices encode the same physics this should be identical.
row_sums_32  = G32.sum(axis=1)
row_sums_500 = G500.sum(axis=1)

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.plot(row_sums_32,  label="G32  row sums")
ax3.plot(row_sums_500, label="G500 row sums", linestyle="--")
ax3.set_title("Row sums per wavelength\n(must match if both G's encode same physics)")
ax3.set_xlabel("wavelength index")
ax3.legend()
ax3.grid(True, linestyle="--", alpha=0.5)

diff_sums = np.abs(row_sums_32 - row_sums_500)
ax4.plot(diff_sums, color="red")
ax4.set_title("|row_sum_32 - row_sum_500|")
ax4.set_xlabel("wavelength index")
ax4.grid(True, linestyle="--", alpha=0.5)

print(f"\nRow-sum comparison (G32 vs G500):")
print(f"  max  |Δrow_sum| = {diff_sums.max():.4e}")
print(f"  mean |Δrow_sum| = {diff_sums.mean():.4e}")
print(f"  allclose(rtol=1e-6) = {np.allclose(row_sums_32, row_sums_500, rtol=1e-6)}")
fig2.tight_layout()

# ── Figure 3: single-row profile (G[row, :] vs depth) for a few wavelengths ──
# Interpolate G32 rows to 500 points so both can be overlaid on the same x-axis.
ROWS_TO_PLOT = [0, 5, 14, 27]
fig3, axes3 = plt.subplots(len(ROWS_TO_PLOT), 1, figsize=(10, 3 * len(ROWS_TO_PLOT)))
x32  = np.linspace(0, 1, G32.shape[1])
x500 = np.linspace(0, 1, G500.shape[1])
for i, row in enumerate(ROWS_TO_PLOT):
    axes3[i].plot(x32,  G32[row],  label="G32",  marker='o', markersize=3)
    axes3[i].plot(x500, G500[row], label="G500", linewidth=1, alpha=0.7)
    axes3[i].set_title(f"wavelength index {row} — G value vs normalised depth")
    axes3[i].set_xlabel("normalised depth (0=surface, 1=back)")
    axes3[i].legend(fontsize=8)
    axes3[i].grid(True, linestyle="--", alpha=0.5)
fig3.suptitle("G row profiles: G32 dots vs G500 line\n(should overlap if same physics)")
fig3.tight_layout()

# ── Figure 4: B comparison (original) ────────────────────────────────────────
CURVE_INDICES = [0, 1, 2, 10, 50]
print(f"\n{'curve':>6}  {'max|ΔB|':>12}  {'mean|ΔB|':>12}  {'max rel diff':>14}  {'allclose(1e-3)':>15}")
print("-" * 70)

fig4, axes4 = plt.subplots(len(CURVE_INDICES), 2, figsize=(12, 3 * len(CURVE_INDICES)))
for row, idx in enumerate(CURVE_INDICES):
    S_raw = S_all[idx]
    S32  = match_length_interp(S_raw, G32.shape[1])
    S500 = match_length_interp(S_raw, G500.shape[1])
    B32  = G32  @ S32
    B500 = G500 @ S500
    diff     = np.abs(B32 - B500)
    rel_diff = diff / (np.abs(B32) + 1e-30)
    print(f"{idx:>6}  {diff.max():>12.4e}  {diff.mean():>12.4e}  {rel_diff.max():>14.4e}  {str(np.allclose(B32, B500, rtol=1e-3)):>15}")

    ax = axes4[row, 0]
    ax.plot(B32,  label="B32  (G32 @ S32)")
    ax.plot(B500, label="B500 (G500 @ S500)", linestyle="--")
    ax.set_title(f"curve {idx} — B comparison")
    ax.set_xlabel("wavelength index")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes4[row, 1]
    ax.plot(diff, color="red")
    ax.set_title(f"curve {idx} — |B32 - B500|")
    ax.set_xlabel("wavelength index")
    ax.grid(True, linestyle="--", alpha=0.5)

fig4.tight_layout()
plt.show()
