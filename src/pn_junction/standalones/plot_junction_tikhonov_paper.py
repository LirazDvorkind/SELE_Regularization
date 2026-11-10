"""Paper figures for Tamir's Tikhonov on the synthetic junction, Tikhonov alone.

Same truths, G, noise model and solver as ``run_junction_tikhonov``. The no-dip control is the
plain wafer rather than the dip-filled junction, whose "filled" truth still keeps part of a dip.
Writes PDF and PNG at journal column widths, plus draft captions, to
``results/pn_junction/paper/tikhonov/``.

    fig1  SELE reconstruction at the baseline noise, one panel per junction depth
    fig2  (a) how much the junction changes the ELE against the noise; (b) recovered dip depth
    figS1 every junction depth at every noise level
    figS2 data space: ELE fit and normalised residuals at the baseline noise
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.pn_junction.feasibility import measurement, truth as truth_mod
from src.pn_junction.feasibility.style import (BLACK, BLUE, GREEN, ORANGE, OUT_DIR, SKY, UM, VERMILION,
                                               shade, ylim)
from src.pn_junction.standalones import run_junction_tikhonov as tik

PAPER_DIR = OUT_DIR.parent / "paper" / "tikhonov"
JUNCTION_DEPTHS_UM = tik.JUNCTION_DEPTHS_UM
NOISE_LEVELS = tik.NOISE_LEVELS
BASELINE_NOISE = 0.01
N_DRAWS = tik.N_DRAWS
SEED = 1

SINGLE, DOUBLE = 3.33, 7.0
NOISE_STYLE = {0.005: (SKY, "o"), 0.01: (BLUE, "s"), 0.02: (GREEN, "^"), 0.05: (VERMILION, "D")}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": False, "lines.linewidth": 1.1,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def compute():
    rng = np.random.default_rng(SEED)
    device = truth_mod.THESIS_DEVICE
    fine = measurement.operator(device.length_cm, measurement.TRUTH_SPACING_CM)
    inv = measurement.operator(device.length_cm, measurement.INVERSION_SPACING_CM)
    z = inv.centres_cm

    cases = {}
    for xj in JUNCTION_DEPTHS_UM:
        t = truth_mod.junction_truth(junction_cm=xj / UM)
        s_fine = truth_mod.resample(t, fine.centres_cm)
        s = truth_mod.resample(t, z)
        cases[xj] = dict(truth=t, s=s, s_fill=truth_mod.filled_dip(t, z, s), ele=fine.G @ s_fine,
                         ele_fill=fine.G @ truth_mod.filled_dip(t, fine.centres_cm, s_fine),
                         centre=0.5 * sum(t.depletion_cm))
    wafer = dict(s=truth_mod.wafer_truth(z), ele=fine.G @ truth_mod.wafer_truth(fine.centres_cm))

    data = {}
    for lv in NOISE_LEVELS:
        for xj, c in cases.items():
            data["junction", xj, lv] = measurement.add_noise(c["ele"], lv, N_DRAWS, rng)
        data["wafer", None, lv] = measurement.add_noise(wafer["ele"], lv, N_DRAWS, rng)
    keys = list(data)
    with Pool(os.cpu_count(), initializer=tik._init_worker) as pool:
        flat = pool.map(tik._tikhonov, [d for k in keys for d in data[k]])
    recon = {k: np.array(flat[n * N_DRAWS:(n + 1) * N_DRAWS]) for n, k in enumerate(keys)}
    ell = {lv: measurement.blur_width(inv, lv) for lv in NOISE_LEVELS}
    return dict(inv=inv, z=z, cases=cases, wafer=wafer, data=data, recon=recon, ell=ell)


def metrics(r):
    z, cases = r["z"], r["cases"]
    out = {}
    for xj, c in cases.items():
        out["sensitivity", xj] = float(np.sqrt(np.mean(((c["ele"] - c["ele_fill"]) / c["ele"]) ** 2)))
        for lv in NOISE_LEVELS:
            out["dip", xj, lv] = tik.dip_depth(z, r["recon"]["junction", xj, lv], c["s_fill"], c["centre"])
            out["false dip", xj, lv] = tik.dip_depth(z, r["recon"]["wafer", None, lv], r["wafer"]["s"], c["centre"])
    return out


def _panel_label(ax, text, x=-0.16):
    ax.text(x, 1.04, text, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def _xmax(xj):
    return max(1.0, 2.0 * xj + 0.4)


def _sele_panel(ax, r, xj, lv):
    z, c = r["z"], r["cases"][xj]
    rec = r["recon"]["junction", xj, lv] * 100
    lo, mid, hi = np.percentile(rec, [10, 50, 90], axis=0)
    xmax = _xmax(xj)
    shade(ax, z, r["ell"][lv], xmax, c["truth"].depletion_cm)
    ax.axhline(0, color="0.6", lw=0.5, zorder=1)
    ax.fill_between(z * UM, lo, hi, color=BLUE, alpha=0.25, lw=0, zorder=2)
    ax.plot(z * UM, mid, color=BLUE, zorder=3)
    ax.plot(z * UM, c["s"] * 100, color=BLACK, ls="--", lw=1.0, zorder=4)
    view = z * UM <= xmax
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim(lo[view], hi[view], c["s"][view] * 100))


def _sele_handles():
    return ([Line2D([], [], color=BLACK, ls="--", lw=1.0), Line2D([], [], color=BLUE),
             Patch(facecolor=BLUE, alpha=0.25), Patch(facecolor=ORANGE, alpha=0.18),
             Patch(facecolor="none", edgecolor="0.6", hatch="////")],
            ["true SELE (drift-diffusion)", "Tikhonov, median", "Tikhonov, 10–90% of noise draws",
             "depletion region", "beyond resolution limit"])


def fig1(r):
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6), constrained_layout=True)
    for ax, xj, letter in zip(axes.flat, JUNCTION_DEPTHS_UM, "abcd"):
        _sele_panel(ax, r, xj, BASELINE_NOISE)
        _panel_label(ax, f"({letter})", x=-0.13)
        ax.text(0.98, 0.95, f"$x_j$ = {xj:g} μm", transform=ax.transAxes, ha="right", va="top", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_xlabel("depth $z$ (μm)")
        ax.set_ylabel("SELE (%)")
    fig.legend(*_sele_handles(), loc="outside lower center", ncol=3, frameon=False)
    _save(fig, "fig1_reconstruction")


def fig2(r, m):
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), constrained_layout=True)
    x = np.array(JUNCTION_DEPTHS_UM)

    sens = np.array([m["sensitivity", xj] for xj in JUNCTION_DEPTHS_UM]) * 100
    a.semilogy(x, sens, color=BLACK, marker="o", ms=4)
    for lv in NOISE_LEVELS:
        col, _ = NOISE_STYLE[lv]
        a.axhline(lv * 100, color=col, ls=":", lw=1.0)
        a.text(0.03, lv * 100 * 1.08, f"noise σ = {lv * 100:g}%", color=col, va="bottom", fontsize=7)
    a.set_xlim(0, x[-1] + 0.1)
    a.set_xlabel("junction depth $x_j$ (μm)")
    a.set_ylabel("ELE change if the dip is removed\n(rms over wavelengths, %)")
    _panel_label(a, "(a)")

    offsets = np.linspace(-0.018, 0.018, len(NOISE_LEVELS))
    b.axhspan(1, 2.5, color="0.93", lw=0, zorder=0)
    b.text(0.98, 1.03, "overshoots below zero", transform=b.get_yaxis_transform(), fontsize=7, color="0.4", ha="right",
           va="bottom")
    b.axhline(1, color="0.5", lw=0.6)
    b.axhline(0, color="0.5", lw=0.6)
    for lv, off in zip(NOISE_LEVELS, offsets):
        col, mk = NOISE_STYLE[lv]
        for kind, filled, ls in (("dip", True, "-"), ("false dip", False, ":")):
            vals = [m[kind, xj, lv] for xj in JUNCTION_DEPTHS_UM]
            mid = np.array([np.median(v) for v in vals])
            err = np.array([[mm - np.percentile(v, 10), np.percentile(v, 90) - mm] for v, mm in zip(vals, mid)]).T
            b.errorbar(x + off, mid, yerr=err, color=col, marker=mk, ms=4, lw=0.9, ls=ls, capsize=1.5,
                       mfc=col if filled else "white", elinewidth=0.7)
    b.set_xlim(0, x[-1] + 0.1)
    b.set_ylim(-1.1, 2.1)
    b.set_xlabel("junction depth $x_j$ (μm)")
    b.set_ylabel("recovered dip depth\n(1 = to zero, 0 = none)")
    _panel_label(b, "(b)")
    handles = [Line2D([], [], color=NOISE_STYLE[lv][0], marker=NOISE_STYLE[lv][1], ms=4, ls="none")
               for lv in NOISE_LEVELS]
    handles += [Line2D([], [], color="0.3", marker="o", ms=4, ls="-"),
                Line2D([], [], color="0.3", marker="o", ms=4, mfc="white", ls=":")]
    labels = [f"σ = {lv * 100:g}%" for lv in NOISE_LEVELS] + ["junction", "plain wafer (control)"]
    b.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=1.6)
    _save(fig, "fig2_dip_recovery")


def figS1(r):
    fig, axes = plt.subplots(len(JUNCTION_DEPTHS_UM), len(NOISE_LEVELS), figsize=(DOUBLE, 7.6),
                             constrained_layout=True, sharex="row", sharey="row")
    for i, xj in enumerate(JUNCTION_DEPTHS_UM):
        for j, lv in enumerate(NOISE_LEVELS):
            ax = axes[i, j]
            _sele_panel(ax, r, xj, lv)
            if i == 0:
                ax.set_title(f"σ = {lv * 100:g}%")
            if j == 0:
                ax.set_ylabel(f"$x_j$ = {xj:g} μm\nSELE (%)")
            else:
                ax.tick_params(labelleft=False)
            if i == len(JUNCTION_DEPTHS_UM) - 1:
                ax.set_xlabel("depth $z$ (μm)")
        # sharey uses the last panel's limits; refit to the whole row.
        z, view = r["z"], r["z"] * UM <= _xmax(xj)
        bands = [np.percentile(r["recon"]["junction", xj, lv] * 100, [10, 90], axis=0)[:, view]
                 for lv in NOISE_LEVELS]
        axes[i, 0].set_ylim(*ylim(*[b for pair in bands for b in pair], r["cases"][xj]["s"][view] * 100))
    fig.legend(*_sele_handles(), loc="outside lower center", ncol=3, frameon=False)
    _save(fig, "figS1_all_depths_and_noise")


def figS2(r):
    inv, lv = r["inv"], BASELINE_NOISE
    wl = inv.wavelength_nm
    fig, axes = plt.subplots(2, len(JUNCTION_DEPTHS_UM), figsize=(DOUBLE, 3.6), constrained_layout=True,
                             sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
    for j, (xj, letter) in enumerate(zip(JUNCTION_DEPTHS_UM, "abcd")):
        c = r["cases"][xj]
        data = r["data"]["junction", xj, lv][0]
        fit = inv.G @ r["recon"]["junction", xj, lv][0]
        top, bot = axes[0, j], axes[1, j]
        top.plot(wl, c["ele"] * 100, color=BLACK, ls="--", lw=1.0)
        top.plot(wl, data * 100, ls="none", marker="o", ms=2.5, mfc="white", mec=BLACK, mew=0.6)
        top.plot(wl, fit * 100, color=BLUE)
        top.set_title(f"$x_j$ = {xj:g} μm")
        _panel_label(top, f"({letter})", x=-0.3)
        bot.axhspan(-1, 1, color="0.92", lw=0)
        bot.axhline(0, color="0.5", lw=0.6)
        bot.plot(wl, (data - fit) / (lv * data), ls="none", marker="o", ms=2.5, color=BLUE)
        bot.set_ylim(-3.5, 3.5)
        bot.set_xlabel("wavelength (nm)")
        if j == 0:
            top.set_ylabel("ELE (%)")
            bot.set_ylabel("residual / σ")
    handles = [Line2D([], [], color=BLACK, ls="--", lw=1.0),
               Line2D([], [], ls="none", marker="o", ms=2.5, mfc="white", mec=BLACK),
               Line2D([], [], color=BLUE), Patch(facecolor="0.92")]
    fig.legend(handles, ["noise-free ELE", f"one noisy measurement (σ = {lv * 100:g}%)",
                         "Tikhonov fit", "±1σ"], loc="outside lower center", ncol=4, frameon=False)
    _save(fig, "figS2_data_fit")


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(PAPER_DIR / f"{name}.{ext}")
    plt.close(fig)


def summary(r, m):
    z = r["z"]
    lines = [f"blind beyond: " + ", ".join(f"{lv * 100:g}% -> {measurement.blind_from(z, r['ell'][lv]) * UM:.2f} um"
                                           for lv in NOISE_LEVELS)]
    for xj, c in r["cases"].items():
        neg = r["recon"]["junction", xj, BASELINE_NOISE]
        lines.append(f"\njunction {xj} um: ELE change if dip removed {m['sensitivity', xj] * 100:.1f}% rms; "
                     f"at {BASELINE_NOISE * 100:g}% noise, most negative SELE {np.median(neg.min(axis=1)) * 100:.3f}% "
                     f"vs truth peak {c['s'].max() * 100:.3f}%")
        for lv in NOISE_LEVELS:
            d, f = m["dip", xj, lv], m["false dip", xj, lv]
            lines.append(f"  noise {lv * 100:g}%: dip {np.median(d):.2f} [{np.percentile(d, 10):.2f}, "
                         f"{np.percentile(d, 90):.2f}]; wafer false dip {np.median(f):.2f} "
                         f"[{np.percentile(f, 10):.2f}, {np.percentile(f, 90):.2f}]")
    return "\n".join(lines) + "\n"


def main():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    r = compute()
    m = metrics(r)
    text = summary(r, m)
    (PAPER_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    fig1(r)
    fig2(r, m)
    figS1(r)
    figS2(r)


if __name__ == "__main__":
    main()
