"""Tamir's Tikhonov method, unchanged, on synthetic junction measurements at several junction depths.

For each junction depth: truth from the drift-diffusion solver, ELE through the Beer-Lambert G
with relative white noise, reconstruction by the L-curve-knee Tikhonov used for the wafer. A
control with the dip filled in (a straight line from the emitter peak to past the depletion
region) checks that a reconstructed dip comes from the data and not from the method.
Figures go to ``results/pn_junction/feasibility/``.
"""

import os

# One BLAS thread per worker; the parallelism is across noise draws.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from src.__init__ import CONFIG
from src.operators import build_L
from src.pn_junction.feasibility import measurement, truth as truth_mod
from src.pn_junction.feasibility.style import (BLACK, BLUE, GREEN, OUT_DIR, SKY, UM, VERMILION,
                                               shade, shading_handles, ylim as _ylim)
from src.regularization import tikhonov_non_uniform

JUNCTION_DEPTHS_UM = (0.12, 0.3, 0.5, 0.8)
NOISE_LEVELS = (0.005, 0.01, 0.02, 0.05)
N_DRAWS = 20
SEED = 0

_inv = None


def _init_worker():
    global _inv
    inv = measurement.operator(truth_mod.THESIS_DEVICE.length_cm, measurement.INVERSION_SPACING_CM)
    L = build_L(CONFIG.L_flag, inv.centres_cm.size)
    kappa_vals = np.logspace(*np.log10(CONFIG.kappa_range), CONFIG.n_kappa)
    _inv = (inv.G, L, kappa_vals)


def _tikhonov(ele):
    G, L, kappa_vals = _inv
    unit = CONFIG.photon_flux * CONFIG.e_charge
    residuals, seminorms, S_list = tikhonov_non_uniform.sweep_kappa(G * unit, ele * unit, L, kappa_vals)
    knee, _ = tikhonov_non_uniform.find_knee(residuals, seminorms, kappa_vals)
    window = (kappa_vals >= knee / CONFIG.conf_window) & (kappa_vals <= knee * CONFIG.conf_window)
    return np.stack([S for S, keep in zip(S_list, window) if keep]).mean(axis=0)


def dip_depth(z_cm, recon, reference, centre_cm):
    """How far the reconstruction falls below the dip-free reference at the dip centre (1 = to zero)."""
    i = int(np.argmin(np.abs(z_cm - centre_cm)))
    return 1.0 - recon[..., i] / reference[i]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    device = truth_mod.THESIS_DEVICE
    fine = measurement.operator(device.length_cm, measurement.TRUTH_SPACING_CM)
    inv = measurement.operator(device.length_cm, measurement.INVERSION_SPACING_CM)
    z = inv.centres_cm
    ell = {level: measurement.blur_width(inv, level) for level in NOISE_LEVELS}

    cases = []
    for xj_um in JUNCTION_DEPTHS_UM:
        truth = truth_mod.junction_truth(junction_cm=xj_um / UM)
        s_fine = truth_mod.resample(truth, fine.centres_cm)
        cases.append(dict(xj_um=xj_um, truth=truth,
                          s=truth_mod.resample(truth, z),
                          s_fill=truth_mod.filled_dip(truth, z, truth_mod.resample(truth, z)),
                          ele=fine.G @ s_fine,
                          ele_fill=fine.G @ truth_mod.filled_dip(truth, fine.centres_cm, s_fine)))

    jobs = []
    for c in cases:
        for level in NOISE_LEVELS:
            for kind in ("ele", "ele_fill"):
                jobs.append((c["xj_um"], level, kind, measurement.add_noise(c[kind], level, N_DRAWS, rng)))
    with Pool(os.cpu_count(), initializer=_init_worker) as pool:
        flat = pool.map(_tikhonov, [d for *_, draws in jobs for d in draws])
    recon = {}
    for n, (xj_um, level, kind, _) in enumerate(jobs):
        recon[xj_um, level, kind] = np.array(flat[n * N_DRAWS:(n + 1) * N_DRAWS])

    print("blind beyond (blur width > 2 x depth): "
          + ", ".join(f"{lv * 100:g}% -> {measurement.blind_from(z, ell[lv]) * UM:.2f} um" for lv in NOISE_LEVELS))
    summary = {}
    for c in cases:
        t = c["truth"]
        centre = 0.5 * sum(t.depletion_cm)
        c["centre"] = centre
        ell_here = {lv: ell[lv][np.argmin(np.abs(z - centre))] * UM for lv in NOISE_LEVELS}
        print(f"\njunction {c['xj_um']} um: depletion {t.depletion_cm[0] * UM:.3f}-{t.depletion_cm[1] * UM:.3f} um, "
              f"relative ELE change from filling the dip: "
              f"{np.sqrt(np.mean(((c['ele'] - c['ele_fill']) / c['ele']) ** 2)) * 100:.1f}% rms")
        for lv in NOISE_LEVELS:
            d = dip_depth(z, recon[c["xj_um"], lv, "ele"], c["s_fill"], centre)
            f = dip_depth(z, recon[c["xj_um"], lv, "ele_fill"], c["s_fill"], centre)
            summary[c["xj_um"], lv] = (d, f)
            print(f"  noise {lv * 100:g}%: blur width at dip {ell_here[lv]:.2f} um; recovered dip depth "
                  f"{np.median(d):.2f} [{np.percentile(d, 10):.2f}, {np.percentile(d, 90):.2f}], "
                  f"dip-free control {np.median(f):.2f} [{np.percentile(f, 10):.2f}, {np.percentile(f, 90):.2f}]")

    _plot_grid(cases, recon, z, ell, "ele", "tikhonov_by_depth_and_noise.png",
               "Tamir's Tikhonov on a junction: reconstruction vs. solver truth")
    _plot_control(cases, recon, z, ell)
    _plot_summary(summary)
    _plot_full_depth(cases[0], recon, z, ell)


def _shade(ax, c, z, ell_level, xmax):
    shade(ax, z, ell_level, xmax, c["truth"].depletion_cm)


def _xmax(c):
    return max(1.0, 2.0 * c["xj_um"] + 0.4)


def _band(ax, z, recon, color, label):
    lo, mid, hi = np.percentile(recon, [10, 50, 90], axis=0) * 100
    ax.fill_between(z * UM, lo, hi, color=color, alpha=0.25, lw=0, label=f"{label}, 10–90% of draws")
    ax.plot(z * UM, mid, color=color, lw=1.6, label=f"{label}, median")
    return lo, hi


def _plot_grid(cases, recon, z, ell, kind, name, title):
    fig, axes = plt.subplots(len(cases), len(NOISE_LEVELS), figsize=(13, 2.6 * len(cases)),
                             constrained_layout=True, squeeze=False)
    for r, c in enumerate(cases):
        xmax = _xmax(c)
        view = z * UM <= xmax
        rows = []
        for col, lv in enumerate(NOISE_LEVELS):
            ax = axes[r, col]
            lo, hi = _band(ax, z, recon[c["xj_um"], lv, kind], BLUE, "Tikhonov")
            ax.plot(z * UM, c["s"] * 100, color=BLACK, ls="--", lw=1.4, label="truth (solver)", zorder=5)
            _shade(ax, c, z, ell[lv], xmax)
            ax.set_xlim(0, xmax)
            rows.append((lo[view], hi[view], c["s"][view] * 100))
            if r == 0:
                ax.set_title(f"noise σ = {lv * 100:g}%")
            if col == 0:
                ax.set_ylabel(f"junction at {c['xj_um']} μm\nSELE (%)")
            if r == len(cases) - 1:
                ax.set_xlabel("depth z (μm)")
        lim = _ylim(*[a for row in rows for a in row])
        for ax in axes[r]:
            ax.set_ylim(*lim)
    _legend(fig, axes[0, 0])
    fig.suptitle(title, fontsize=11)
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    extra_h, extra_l = shading_handles()
    handles, labels = handles + extra_h, labels + extra_l
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))


def _plot_control(cases, recon, z, ell):
    levels = (0.01, 0.05)
    fig, axes = plt.subplots(len(cases), len(levels), figsize=(9, 2.6 * len(cases)),
                             constrained_layout=True, squeeze=False)
    for r, c in enumerate(cases):
        xmax = _xmax(c)
        view = z * UM <= xmax
        rows = []
        for col, lv in enumerate(levels):
            ax = axes[r, col]
            a = _band(ax, z, recon[c["xj_um"], lv, "ele"], BLUE, "with junction")
            b = _band(ax, z, recon[c["xj_um"], lv, "ele_fill"], VERMILION, "dip filled")
            ax.plot(z * UM, c["s"] * 100, color=BLUE, ls="--", lw=1.1, zorder=5, label="truth, with junction")
            ax.plot(z * UM, c["s_fill"] * 100, color=VERMILION, ls=":", lw=1.4, zorder=5, label="truth, dip filled")
            _shade(ax, c, z, ell[lv], xmax)
            ax.set_xlim(0, xmax)
            rows += [a[0][view], a[1][view], b[0][view], b[1][view], c["s"][view] * 100]
            if r == 0:
                ax.set_title(f"noise σ = {lv * 100:g}%")
            if col == 0:
                ax.set_ylabel(f"junction at {c['xj_um']} μm\nSELE (%)")
            if r == len(cases) - 1:
                ax.set_xlabel("depth z (μm)")
        lim = _ylim(*rows)
        for ax in axes[r]:
            ax.set_ylim(*lim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Control: the same measurement with the junction dip filled in", fontsize=11)
    fig.savefig(OUT_DIR / "tikhonov_control_dip_filled.png", dpi=150)
    plt.close(fig)


def _plot_summary(summary):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    colors = [SKY, BLUE, GREEN, VERMILION]
    markers = ["o", "s", "^", "D"]
    offsets = np.linspace(-0.015, 0.015, len(NOISE_LEVELS))
    x = np.array(JUNCTION_DEPTHS_UM)
    for lv, col, mk, off in zip(NOISE_LEVELS, colors, markers, offsets):
        for which, filled in ((0, True), (1, False)):
            vals = [summary[xj, lv][which] for xj in JUNCTION_DEPTHS_UM]
            mid = np.array([np.median(v) for v in vals])
            err = np.array([[m - np.percentile(v, 10), np.percentile(v, 90) - m] for v, m in zip(vals, mid)]).T
            ax.errorbar(x + off, mid, yerr=err, color=col, marker=mk, ms=6, lw=1, capsize=2,
                        mfc=col if filled else "white", ls="-" if filled else ":",
                        label=f"σ = {lv * 100:g}%, with junction" if filled else f"σ = {lv * 100:g}%, dip filled")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.text(x[-1] + 0.03, 1, "truth: dip to zero", va="center", fontsize=8, color="0.35")
    ax.text(x[-1] + 0.03, 0, "no dip", va="center", fontsize=8, color="0.35")
    ax.set_xlabel("junction depth (μm)")
    ax.set_ylabel("recovered dip depth at the junction\n(1 = falls to zero, 0 = no dip)")
    ax.set_xlim(0, x[-1] + 0.2)
    handles, labels = ax.get_legend_handles_labels()
    order = [i for i in range(len(labels)) if "with" in labels[i]] + [i for i in range(len(labels)) if "filled" in labels[i]]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center",
               ncol=2, fontsize=8, title="filled markers: real junction · hollow markers: dip filled in (control)",
               title_fontsize=8)
    ax.set_title("Can Tamir's Tikhonov see the junction? (median, 10–90% of 20 draws)")
    fig.savefig(OUT_DIR / "dip_recovery_vs_junction_depth.png", dpi=150)
    plt.close(fig)


def _plot_full_depth(c, recon, z, ell):
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    lo, hi = _band(ax, z, recon[c["xj_um"], 0.01, "ele"], BLUE, "Tikhonov")
    ax.plot(z * UM, c["s"] * 100, color=BLACK, ls="--", lw=1.4, label="truth (solver)", zorder=5)
    xmax = z[-1] * UM
    _shade(ax, c, z, ell[0.01], xmax)
    ax.set_xlim(0, xmax)
    ax.set_ylim(*_ylim(lo, hi, c["s"] * 100))
    ax.set_xlabel("depth z (μm)")
    ax.set_ylabel("SELE (%)")
    ax.set_title(f"Whole device, junction at {c['xj_um']} μm, noise σ = 1% (hatched: no depth information)")
    ax.legend(loc="upper left")
    fig.savefig(OUT_DIR / "tikhonov_full_depth.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
