"""The model-free profile sampler on the synthetic junction measurements, with its checks.

Same junction truths, noise and G as ``run_junction_tikhonov``. For each case the sampler draws
thousands of SELE profiles that reproduce the measured ELE within the noise (four independent
chains), and the spread of those profiles is the answer. Checks:

- does a dip appear where the truth has one, and only there (the dip-filled junctions and a
  plain wafer from the wafer simulator are the no-dip controls);
- does the answer survive another noise draw, a few-percent error in G, and halving or doubling
  the two learned settings (step scale and tail weight).

Figures and a printed summary go to ``results/pn_junction/feasibility/``.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pickle
import sys
import tempfile
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.pn_junction import profile_sampler
from src.pn_junction.feasibility import measurement, truth as truth_mod
from src.pn_junction.feasibility.style import (BLACK, BLUE, GREEN, ORANGE, OUT_DIR, PURPLE, SKY, UM,
                                               VERMILION, shade, shading_handles, ylim)
from src.pn_junction.standalones import run_junction_tikhonov as tikhonov_run

JUNCTION_DEPTHS_UM = (0.12, 0.3, 0.5, 0.8)
NOISE_LEVELS = (0.01, 0.05)
N_NOISE_DRAWS = 2
N_CHAINS = 4
N_BURN, N_KEEP, THIN = 2000, 500, 10
ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE = 0.3, 0.01
G_ERRORS = {"k × 0.97": (0.97, 0.0), "k × 1.03": (1.03, 0.0),
            "k tilted −5% → +5%": (1.0, 0.1), "k tilted +5% → −5%": (1.0, -0.1)}
SEED = 0
CACHE = Path(tempfile.gettempdir()) / "junction_sampler_samples.pkl"


@lru_cache(maxsize=None)
def _G(k_scale, k_tilt):
    return measurement.operator(truth_mod.THESIS_DEVICE.length_cm, measurement.INVERSION_SPACING_CM,
                                k_scale, k_tilt).G


def _chain(job):
    ele, level, k_scale, k_tilt, step, tail, seed = job
    return profile_sampler.sample(_G(k_scale, k_tilt), ele, level * np.abs(ele), N_KEEP, N_BURN,
                                  np.random.default_rng(seed), THIN, step, tail)


def _run(pool, cases, seed0):
    """``cases``: {key: (ele, level, k_scale, k_tilt, step, tail)} -> {key: pooled Samples}."""
    jobs = [(*spec, seed0 + 1000 * n + c) for n, spec in enumerate(cases.values()) for c in range(N_CHAINS)]
    chains = pool.map(_chain, jobs, chunksize=1)
    out = {}
    for n, key in enumerate(cases):
        mine = chains[n * N_CHAINS:(n + 1) * N_CHAINS]
        out[key] = (profile_sampler.Samples(*(np.concatenate(f) for f in zip(*mine))),
                    _rhat(np.array([c.sele for c in mine])))
    return out


def _rhat(per_chain):
    """Worst Gelman-Rubin ratio over depth: near 1 when the chains agree."""
    m = per_chain.shape[1]
    within = per_chain.var(axis=1, ddof=1).mean(axis=0)
    between = m * per_chain.mean(axis=1).var(axis=0, ddof=1)
    ok = within > 0
    return float(np.max(np.sqrt(((m - 1) / m * within[ok] + between[ok] / m) / within[ok])))


def interior_dip(sele, seen):
    """How deep a fall-then-rise each profile has inside the seen zone (0 = none, 1 = to zero).

    At each depth, the drop below the highest point before it and below the highest point after
    it; the smaller of the two is how deep a dip there is, and the largest over depth is the
    profile's. It is blind to where the dip is and to why.
    """
    s = sele[..., seen]
    before = np.maximum.accumulate(s, axis=-1)
    after = np.flip(np.maximum.accumulate(np.flip(s, -1), axis=-1), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dip = np.minimum(1.0 - s / before, 1.0 - s / after)
    return np.nanmax(np.nan_to_num(dip, nan=0.0), axis=-1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    device = truth_mod.THESIS_DEVICE
    fine = measurement.operator(device.length_cm, measurement.TRUTH_SPACING_CM)
    inv = measurement.operator(device.length_cm, measurement.INVERSION_SPACING_CM)
    z = inv.centres_cm
    ell = {lv: measurement.blur_width(inv, lv) for lv in NOISE_LEVELS}
    seen = {lv: z < measurement.blind_from(z, ell[lv]) for lv in NOISE_LEVELS}

    truths, cases = {}, {}
    for xj in JUNCTION_DEPTHS_UM:
        t = truth_mod.junction_truth(junction_cm=xj / UM)
        s_fine = truth_mod.resample(t, fine.centres_cm)
        s = truth_mod.resample(t, z)
        truths[xj] = dict(truth=t, s=s, s_fill=truth_mod.filled_dip(t, z, s))
        ele, ele_fill = fine.G @ s_fine, fine.G @ truth_mod.filled_dip(t, fine.centres_cm, s_fine)
        for lv in NOISE_LEVELS:
            for d, noisy in enumerate(measurement.add_noise(ele, lv, N_NOISE_DRAWS, rng)):
                cases["junction", xj, lv, d] = (noisy, lv, 1.0, 0.0, None, None)
        cases["filled", xj, ROBUSTNESS_NOISE, 0] = (
            measurement.add_noise(ele_fill, ROBUSTNESS_NOISE, 1, rng)[0], ROBUSTNESS_NOISE, 1.0, 0.0, None, None)
    wafer = truth_mod.wafer_truth(z)
    wafer_ele = fine.G @ truth_mod.wafer_truth(fine.centres_cm)
    for lv in NOISE_LEVELS:
        cases["wafer", None, lv, 0] = (measurement.add_noise(wafer_ele, lv, 1, rng)[0], lv, 1.0, 0.0, None, None)
    base_ele = cases["junction", ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE, 0][0]
    for name, (ks, kt) in G_ERRORS.items():
        cases["G error", name, ROBUSTNESS_NOISE, 0] = (base_ele, ROBUSTNESS_NOISE, ks, kt, None, None)

    if "--replot" in sys.argv and CACHE.exists():
        with open(CACHE, "rb") as f:
            res = pickle.load(f)
    else:
        res = _sample_all(cases, base_ele)
    base = res["junction", ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE, 0][0]
    step, tail = float(np.median(base.step_scale)), float(np.median(base.tail_weight))

    tikhonov_run._init_worker()
    tik = {k: tikhonov_run._tikhonov(spec[0]) for k, spec in cases.items() if k[0] == "junction"}

    lines = _summary(res, truths, wafer, z, seen, inv.G, cases, tik, step, tail)
    (OUT_DIR / "sampler_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))

    _plot_grid(res, truths, tik, z, ell)
    _plot_controls(res, truths, wafer, z, ell)
    _plot_robustness(res, truths, z, ell)
    _plot_full_depth(res, truths, z, ell)
    _plot_dip_summary(res, truths, tik, z, seen)


def _sample_all(cases, base_ele):
    """Every case, then the settings check (which needs the learned settings). Saved, outside the
    repo since it is large, so the figures can be redrawn with ``--replot``."""
    with Pool(os.cpu_count()) as pool:
        res = _run(pool, cases, SEED)
        base = res["junction", ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE, 0][0]
        step, tail = float(np.median(base.step_scale)), float(np.median(base.tail_weight))
        settings = {"step scale × 2": (2 * step, None), "step scale × ½": (0.5 * step, None),
                    "tail weight × 2": (None, 2 * tail), "tail weight × ½": (None, 0.5 * tail)}
        res.update(_run(pool, {("settings", name, ROBUSTNESS_NOISE, 0):
                               (base_ele, ROBUSTNESS_NOISE, 1.0, 0.0, *st) for name, st in settings.items()},
                        SEED + 7))
    with open(CACHE, "wb") as f:
        pickle.dump(res, f)
    return res


def _junction_dip(z, sele, t):
    return tikhonov_run.dip_depth(z, sele, t["s_fill"], 0.5 * sum(t["truth"].depletion_cm))


def _summary(res, truths, wafer, z, seen, G, cases, tik, step, tail):
    lines = ["blind beyond: " + ", ".join(f"{lv * 100:g}% noise -> {z[~seen[lv]][0] * UM:.2f} um"
                                            for lv in NOISE_LEVELS),
             "dip at junction: 1 - SELE/dip-free reference at the depletion centre (1 = to zero)",
             "interior dip: deepest fall-then-rise anywhere in the seen zone (0 = none)",
             "covered: share of seen depths where the truth is inside the 5-95% band",
             "chi2/L: mean misfit of the drawn profiles, ~1 means they fit to the noise", ""]
    for key, (smp, rhat) in res.items():
        kind, tag, lv, d = key
        ele = cases[key][0] if key in cases else cases["junction", ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE, 0][0]
        chi2 = np.mean(np.sum(((smp.sele @ G.T - ele) / (lv * ele)) ** 2, axis=1)) / ele.size
        ref = wafer if kind == "wafer" else truths[tag if kind in ("junction", "filled") else ROBUSTNESS_DEPTH_UM][
            "s_fill" if kind == "filled" else "s"]
        lo, hi = np.percentile(smp.sele, [5, 95], axis=0)
        covered = np.mean(((ref >= lo) & (ref <= hi))[seen[lv]])
        idip = interior_dip(smp.sele, seen[lv])
        text = (f"{kind:9s} {str(tag):20s} noise {lv * 100:g}% draw {d}: "
                f"interior dip {np.median(idip):.2f} [{np.percentile(idip, 5):.2f}, {np.percentile(idip, 95):.2f}]")
        if kind != "wafer":
            t = truths[tag if kind in ("junction", "filled") else ROBUSTNESS_DEPTH_UM]
            jd = _junction_dip(z, smp.sele, t)
            text += f"; dip at junction {np.median(jd):.2f} [{np.percentile(jd, 5):.2f}, {np.percentile(jd, 95):.2f}]"
            if key in tik:
                text += f" (Tikhonov {float(_junction_dip(z, tik[key], t)):.2f})"
        text += (f"; covered {covered:.2f}; chi2/L {chi2:.2f}; step {np.median(smp.step_scale) * 100:.1e}%"
                 f"; tail {np.median(smp.tail_weight):g}; Rhat {rhat:.3f}"
                 + (f"; Q nudged {smp.repairs.sum()}x" if smp.repairs.any() else ""))
        lines.append(text)
    lines.append(f"\nsettings learned on junction {ROBUSTNESS_DEPTH_UM} um, {ROBUSTNESS_NOISE * 100:g}%: "
                 f"step scale {step * 100:.2e}% per slice, tail weight {tail:g}")
    return lines


def _band(ax, z, smp, color, label, n_lines=0, rng=None):
    q = np.percentile(smp.sele, [5, 25, 50, 75, 95], axis=0) * 100
    ax.fill_between(z * UM, q[0], q[4], color=color, alpha=0.18, lw=0, label=f"{label}: 5–95% of profiles")
    ax.fill_between(z * UM, q[1], q[3], color=color, alpha=0.35, lw=0, label=f"{label}: 25–75%")
    ax.plot(z * UM, q[2], color=color, lw=1.5, label=f"{label}: median")
    if n_lines:
        for i in (rng or np.random.default_rng(0)).choice(smp.sele.shape[0], n_lines, replace=False):
            ax.plot(z * UM, smp.sele[i] * 100, color=color, lw=0.5, alpha=0.6)
    return q[0], q[4]


def _xmax(xj):
    return max(1.0, 2.0 * xj + 0.4)


def _plot_grid(res, truths, tik, z, ell):
    fig, axes = plt.subplots(len(JUNCTION_DEPTHS_UM), len(NOISE_LEVELS), figsize=(10, 2.7 * len(JUNCTION_DEPTHS_UM)),
                             constrained_layout=True, squeeze=False)
    for r, xj in enumerate(JUNCTION_DEPTHS_UM):
        t, xmax = truths[xj], _xmax(xj)
        view = z * UM <= xmax
        curves = []
        for col, lv in enumerate(NOISE_LEVELS):
            ax = axes[r, col]
            smp = res["junction", xj, lv, 0][0]
            lo, hi = _band(ax, z, smp, BLUE, "sampler", n_lines=4)
            ax.plot(z * UM, tik["junction", xj, lv, 0] * 100, color=VERMILION, ls="-.", lw=1.2,
                    label="Tamir's Tikhonov, same data")
            ax.plot(z * UM, t["s"] * 100, color=BLACK, ls="--", lw=1.4, label="truth (solver)", zorder=5)
            shade(ax, z, ell[lv], xmax, t["truth"].depletion_cm)
            ax.set_xlim(0, xmax)
            curves += [lo[view], hi[view], t["s"][view] * 100]
            if r == 0:
                ax.set_title(f"noise σ = {lv * 100:g}%")
            if col == 0:
                ax.set_ylabel(f"junction at {xj} μm\nSELE (%)")
            if r == len(JUNCTION_DEPTHS_UM) - 1:
                ax.set_xlabel("depth z (μm)")
        lim = ylim(*curves)
        for ax in axes[r]:
            ax.set_ylim(*lim)
    _legend(fig, axes[0, 0], extra_line=BLUE)
    fig.suptitle("Model-free sampler on a junction: the spread of profiles that fit the data", fontsize=11)
    fig.savefig(OUT_DIR / "sampler_by_depth_and_noise.png", dpi=150)
    plt.close(fig)


def _legend(fig, ax, extra_line=None, ncol=4, shading=True):
    handles, labels = ax.get_legend_handles_labels()
    if extra_line:
        handles.append(plt.Line2D([], [], color=extra_line, lw=0.5, alpha=0.6))
        labels.append("a few single drawn profiles")
    if shading:
        h, l = shading_handles()
        handles, labels = handles + h, labels + l
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, fontsize=8)


def _plot_controls(res, truths, wafer, z, ell):
    lv = ROBUSTNESS_NOISE
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, xj in zip(axes.flat, JUNCTION_DEPTHS_UM):
        t, xmax = truths[xj], _xmax(xj)
        view = z * UM <= xmax
        lo, hi = _band(ax, z, res["filled", xj, lv, 0][0], BLUE, "sampler", n_lines=4)
        ax.plot(z * UM, t["s_fill"] * 100, color=BLACK, ls="--", lw=1.4, label="truth", zorder=5)
        shade(ax, z, ell[lv], xmax, t["truth"].depletion_cm)
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim(lo[view], hi[view], t["s_fill"][view] * 100))
        ax.set_title(f"junction at {xj} μm with its dip filled in, σ = {lv * 100:g}%")
    for ax, level in zip(axes.flat[len(JUNCTION_DEPTHS_UM):], NOISE_LEVELS):
        xmax = 2.0
        view = z * UM <= xmax
        lo, hi = _band(ax, z, res["wafer", None, level, 0][0], BLUE, "sampler", n_lines=4)
        ax.plot(z * UM, wafer * 100, color=BLACK, ls="--", lw=1.4, label="truth", zorder=5)
        shade(ax, z, ell[level], xmax)
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim(lo[view], hi[view], wafer[view] * 100))
        ax.set_title(f"plain wafer (no junction), σ = {level * 100:g}%")
    for ax in axes[1]:
        ax.set_xlabel("depth z (μm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SELE (%)")
    _legend(fig, axes[0, 0], extra_line=BLUE)
    fig.suptitle("Controls with no junction dip: does the sampler invent one?", fontsize=11)
    fig.savefig(OUT_DIR / "sampler_controls.png", dpi=150)
    plt.close(fig)


def _outline(ax, z, smp, color, ls, label):
    lo, mid, hi = np.percentile(smp.sele, [5, 50, 95], axis=0) * 100
    ax.plot(z * UM, mid, color=color, ls=ls, lw=1.5, label=label)
    ax.plot(z * UM, lo, color=color, ls=ls, lw=0.7)
    ax.plot(z * UM, hi, color=color, ls=ls, lw=0.7)
    return lo, hi


def _plot_robustness(res, truths, z, ell):
    xj, lv = ROBUSTNESS_DEPTH_UM, ROBUSTNESS_NOISE
    t, xmax = truths[xj], _xmax(xj)
    view = z * UM <= xmax
    base = res["junction", xj, lv, 0][0]
    groups = [("another noise draw", [("noise draw 2", res["junction", xj, lv, 1][0])]),
              ("G a few percent wrong", [(k, res["G error", k, lv, 0][0]) for k in G_ERRORS]),
              ("learned settings halved and doubled", [(k, res["settings", k, lv, 0][0]) for k in
                                                        ("step scale × 2", "step scale × ½",
                                                         "tail weight × 2", "tail weight × ½")])]
    styles = [(VERMILION, "-"), (GREEN, "--"), (PURPLE, ":"), (ORANGE, "-.")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True, sharey=True)
    curves = []
    for ax, (title, variants) in zip(axes, groups):
        lo, hi = _band(ax, z, base, BLUE, "as run")
        curves += [lo[view], hi[view]]
        for (name, smp), (col, ls) in zip(variants, styles):
            a, b = _outline(ax, z, smp, col, ls, f"{name}: median, thin = 5% and 95%")
            curves += [a[view], b[view]]
        ax.plot(z * UM, t["s"] * 100, color=BLACK, ls="--", lw=1.4, label="truth (solver)", zorder=5)
        shade(ax, z, ell[lv], xmax, t["truth"].depletion_cm)
        ax.set_xlim(0, xmax)
        ax.set_title(title)
        ax.set_xlabel("depth z (μm)")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylim(*ylim(*curves, t["s"][view] * 100))
    axes[0].set_ylabel("SELE (%)")
    h, l = shading_handles()
    fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(f"Does the answer hold? Junction at {xj} μm, σ = {lv * 100:g}%", fontsize=11)
    fig.savefig(OUT_DIR / "sampler_robustness.png", dpi=150)
    plt.close(fig)


def _plot_full_depth(res, truths, z, ell):
    xj, lv = JUNCTION_DEPTHS_UM[0], ROBUSTNESS_NOISE
    t = truths[xj]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    lo, hi = _band(ax, z, res["junction", xj, lv, 0][0], BLUE, "sampler", n_lines=4)
    ax.plot(z * UM, t["s"] * 100, color=BLACK, ls="--", lw=1.4, label="truth (solver)", zorder=5)
    xmax = z[-1] * UM
    shade(ax, z, ell[lv], xmax, t["truth"].depletion_cm)
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim(lo, hi, t["s"] * 100))
    ax.set_xlabel("depth z (μm)")
    ax.set_ylabel("SELE (%)")
    ax.set_title(f"Whole device, junction at {xj} μm, σ = {lv * 100:g}%: the band widens where the data stop")
    _legend(fig, ax, extra_line=BLUE, ncol=3)
    fig.savefig(OUT_DIR / "sampler_full_depth.png", dpi=150)
    plt.close(fig)


def _plot_dip_summary(res, truths, tik, z, seen):
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    x = np.array(JUNCTION_DEPTHS_UM)
    for lv, col, mk, off in zip(NOISE_LEVELS, (BLUE, SKY), ("o", "s"), (-0.012, 0.012)):
        for kind, filled, ls in (("junction", True, "-"), ("filled", False, ":")):
            if kind == "filled" and lv != ROBUSTNESS_NOISE:
                continue
            vals = [_junction_dip(z, res[kind, xj, lv, 0][0].sele, truths[xj]) for xj in JUNCTION_DEPTHS_UM]
            mid = np.array([np.median(v) for v in vals])
            err = np.array([[m - np.percentile(v, 5), np.percentile(v, 95) - m] for v, m in zip(vals, mid)]).T
            ax.errorbar(x + off, mid, yerr=err, color=col, marker=mk, ms=6, lw=1, capsize=2, ls=ls,
                        mfc=col if filled else "white",
                        label=f"sampler, σ = {lv * 100:g}%" + ("" if filled else ", dip filled in (control)"))
        tk = [float(_junction_dip(z, tik["junction", xj, lv, 0], truths[xj])) for xj in JUNCTION_DEPTHS_UM]
        ax.plot(x + off, tk, color=VERMILION, marker="x" if lv == NOISE_LEVELS[0] else "+", ms=7, ls="none",
                label=f"Tamir's Tikhonov, σ = {lv * 100:g}%, same data")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.text(x[-1] + 0.03, 1, "truth: dip to zero", va="center", fontsize=8, color="0.35")
    ax.text(x[-1] + 0.03, 0, "no dip", va="center", fontsize=8, color="0.35")
    ax.set_xlim(0, x[-1] + 0.2)
    ax.set_xlabel("junction depth (μm)")
    ax.set_ylabel("dip at the junction\n(1 = falls to zero, 0 = no dip)")
    ax.set_title("Can each method see the junction? (sampler: median, bars 5–95% of profiles)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    fig.savefig(OUT_DIR / "sampler_dip_vs_junction_depth.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
