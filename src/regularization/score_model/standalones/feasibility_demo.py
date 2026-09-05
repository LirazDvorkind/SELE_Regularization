"""
Interactive walkthrough of the score-model feasibility analysis.

Every panel here recomputes a claim from the SELE/MAP feasibility write-up directly from
this repo's own G matrix and training-curve dataset -- nothing is hard-coded from the
earlier analysis scripts. Move the sliders; the numbers and shapes update live so you can
see *why* each claim holds, not just read that it does. Start on view 0 for a plain-language
introduction to the terms used everywhere else (ELE, SELE, G, prior, posterior).

Run as: python -m src.regularization.score_model.standalones.feasibility_demo
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider

from src.io import load_csv

_DATA_DIR = Path(__file__).resolve().parents[4] / "Data"
_SCORE_DIR = _DATA_DIR / "score_model"

# Colorblind-safe categorical palette (Okabe-Ito), per this repo's plotting conventions.
ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILLION, PURPLE, BLACK = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000",
)

# ---------------------------------------------------------------------------
# Load exactly what the pipeline itself uses: G on the 500-point solver mesh,
# and the 1000-curve synthetic dataset the score model is trained on.
# ---------------------------------------------------------------------------
G = load_csv(str(_SCORE_DIR / "G_score_model_500.csv"))                       # (28, 500)
X = load_csv(str(_SCORE_DIR / "datasets" / "sele_simulated_1000_curves_500_long.csv"))  # (1000, 500)
N_CURVES, M_DEPTH = X.shape
W_UM = 30.0  # solver mesh width in um -- see src/mesh.py / CONFIG.model_score_grad_config.W
Z = np.linspace(0.0, W_UM, M_DEPTH)

try:
    TEST_SET_PEAKS = pd.read_csv(_DATA_DIR / "test_set" / "index.csv")["peak_position_um"].to_numpy()
except FileNotFoundError:
    TEST_SET_PEAKS = np.array([])

# ---------------------------------------------------------------------------
# One-time linear algebra: everything else in this file is a live function of
# these matrices plus whatever the sliders are currently set to.
# ---------------------------------------------------------------------------
mu = X.mean(0)
Xc = X - mu
_, s_prior, Vt_prior = np.linalg.svd(Xc, full_matrices=False)          # prior PCA (500-dim curve space)
var_explained = s_prior ** 2 / np.sum(s_prior ** 2)
cum_var = np.cumsum(var_explained)
participation_ratio = (np.sum(s_prior ** 2) ** 2) / np.sum(s_prior ** 4)

_, s_G, Vt_G = np.linalg.svd(G, full_matrices=False)                   # G's SVD: 28 singular values/vectors
tot_prior_var = (Xc ** 2).sum()
CAP_K = np.array([((Xc @ Vt_G[:k].T) ** 2).sum() / tot_prior_var for k in range(1, G.shape[0] + 1)])

E = X @ G.T                                                             # (1000, 28) noiseless ELE, every prior curve
ele_rms = np.linalg.norm(E, axis=1).mean() / np.sqrt(G.shape[0])

C = np.cov(X, rowvar=False)                                             # (500, 500) prior covariance
prior_sd_depth = np.sqrt(np.diag(C))


def derived_quantities(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Physically meaningful scalar readouts from a batch of SELE curves, arr: (n, 500)."""
    peak = arr.max(1)
    return dict(
        surface=arr[:, 0],
        peak=peak,
        peak_pos=Z[arr.argmax(1)],
        area=np.trapezoid(arr, Z, axis=1),
        surf_ratio=arr[:, 0] / peak,
    )


DERIVED_ALL = derived_quantities(X)


def effective_rank(noise_pct: float) -> int:
    """How many of G's singular directions survive this noise floor (relative to s_G[0])."""
    return int(max(1, np.sum(s_G / s_G[0] > noise_pct / 100.0)))


def caption(title: str, body: str, width: int = 98) -> None:
    """Bold headline + a plain-language paragraph, on a light card behind ax_caption."""
    ax_caption.add_patch(Rectangle((0, 0), 1, 1, transform=ax_caption.transAxes,
                                    facecolor="#f4f6f8", edgecolor="#c9d2d8", lw=1, zorder=0))
    ax_caption.text(0.015, 0.90, title, va="top", ha="left", fontsize=12, fontweight="bold",
                     color="#1a1a1a", transform=ax_caption.transAxes, zorder=1)
    ax_caption.text(0.015, 0.68, textwrap.fill(textwrap.dedent(body).strip(), width),
                     va="top", ha="left", fontsize=9.6, color="#2b2b2b", linespacing=1.35,
                     transform=ax_caption.transAxes, zorder=1)


# ---------------------------------------------------------------------------
# Figure layout: a persistent control rail on the left, a teaching caption
# across the top, two content axes underneath that every view repurposes.
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(16, 9.5), facecolor="#fbfbfb")
fig.suptitle("Can we actually recover SELE from a measurement, or are we just seeing our own\n"
             "assumptions reflected back?", fontsize=13.5, fontweight="bold", y=0.995)

ax_caption = fig.add_axes((0.27, 0.77, 0.71, 0.19))
ax_caption.axis("off")

ax_left = fig.add_axes((0.27, 0.08, 0.32, 0.63))
ax_right = fig.add_axes((0.66, 0.08, 0.32, 0.63))
for ax in (ax_left, ax_right):
    ax.set_facecolor("#fdfdfd")

ax_radio = fig.add_axes((0.02, 0.50, 0.22, 0.40))
ax_radio.set_title("pick a claim to test", fontsize=9.5, loc="left", fontweight="bold")
ax_noise = fig.add_axes((0.085, 0.40, 0.145, 0.03))
ax_noise.set_title("assumed noise %", fontsize=8, loc="left")
ax_k = fig.add_axes((0.085, 0.33, 0.145, 0.03))
ax_k.set_title("k = # components kept", fontsize=8, loc="left")
ax_button = fig.add_axes((0.05, 0.24, 0.17, 0.05))
ax_info = fig.add_axes((0.02, 0.02, 0.22, 0.18))
ax_info.axis("off")

VIEWS = [
    "0. Start here",
    "A. Is the answer simple?",
    "B. Can we even see it?",
    "C. Do the two line up?",
    "D. Which depths can we trust?",
    "E. Which numbers survive?",
    "F. Prove it, no formulas",
]
radio = RadioButtons(ax_radio, VIEWS, active=0)
slider_noise = Slider(ax_noise, "", 0.05, 10.0, valinit=1.0, color=BLUE)
slider_k = Slider(ax_k, "", 1, 15, valinit=5, valstep=1, color=ORANGE)
button_resample = Button(ax_button, "↻  try a different curve", color="#eef1f3")

rng = np.random.default_rng()
state = dict(view=VIEWS[0], idx=int(rng.integers(N_CURVES)))

ax_info.add_patch(Rectangle((0, 0), 1, 1, transform=ax_info.transAxes,
                             facecolor="#fff8e6", edgecolor="#e8d9a0", lw=1))
ax_info.text(
    0.06, 0.90,
    "how to use this",
    va="top", ha="left", fontsize=9.5, fontweight="bold", transform=ax_info.transAxes,
)
ax_info.text(
    0.06, 0.74,
    textwrap.fill(
        "Nobody has actually measured this instrument's real noise level -- 'noise %' is a "
        "guess you get to test. Slide it and watch every panel react. The button on the left "
        "swaps which example curve the demo is using as the 'true' answer.",
        34,
    ),
    va="top", ha="left", fontsize=8.3, transform=ax_info.transAxes, linespacing=1.35,
)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def view_intro():
    caption("Start here: what are we even trying to do?", f"""
    We have a wafer of semiconductor material. Shine light on it, and some comes back out as
    a glow (photoluminescence) -- how much glow you get back, per color of light (ELE, on the
    right of the diagram below), depends on how much useful recombination happens at every
    depth inside the material (SELE, on the left -- the profile we actually want, from the
    surface at z=0 down to {W_UM:.0f} micrometers deep). The two are linked by the physics of
    how light is absorbed with depth -- that link is a matrix called G. We only ever get to
    observe the right-hand box. Everything on the other five tabs asks the same question from
    a different angle: how much of the left-hand box can we actually work backward to, and how
    much are we just guessing based on what curves usually look like (the "prior")?
    """)

    ax_left.axis("off")
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    boxes = [
        (0.06, 0.30, "SELE(z)\n\nthe unknown\ndepth profile\nwe want", GREEN),
        (0.40, 0.30, "G\n\nthe physics:\nlight absorption\nvs. depth", SKY),
        (0.72, 0.30, "ELE(λ)\n\nwhat we\nactually\nmeasure", ORANGE),
    ]
    for x, y, text, color in boxes:
        ax_left.add_patch(Rectangle((x, y), 0.24, 0.40, facecolor=color, alpha=0.25,
                                     edgecolor=color, lw=2))
        ax_left.text(x + 0.12, y + 0.20, text, ha="center", va="center", fontsize=9.5)
    for x0, x1 in [(0.30, 0.40), (0.64, 0.72)]:
        ax_left.add_patch(FancyArrowPatch((x0, 0.50), (x1, 0.50), arrowstyle="-|>",
                                           mutation_scale=18, color="#444444", lw=1.5))
    ax_left.text(0.5, 0.90, "forward direction (easy, known physics)", ha="center", fontsize=9,
                 color="#444444")
    ax_left.annotate("", xy=(0.30, 0.16), xytext=(0.94, 0.16),
                      arrowprops=dict(arrowstyle="-|>", color=VERMILLION, lw=1.5))
    ax_left.text(0.5, 0.06, "what we're trying to reverse (hard -- this demo asks how hard)",
                 ha="center", fontsize=9, color=VERMILLION)

    ax_right.axis("off")
    roadmap = [
        ("A", "Is the true answer actually simple?", "yes -- effectively 5 numbers, not 500"),
        ("B", "Can our instrument even see all of it?", "no -- only ~4 independent readings"),
        ("C", "Do those two facts line up, or fight?", "they fight: measurable directions ≠ "
                                                        "the ones curves vary in"),
        ("D", "So which depths can we trust?", "shallow: yes. deep (>~10 µm): mostly no."),
        ("E", "Which real-world numbers survive?", "surface behavior: yes. total light: no."),
        ("F", "Can you show me without the math?", "yes -- very different curves, same reading"),
    ]
    ax_right.text(0, 1.0, "the six tabs, in one line each:", fontsize=10, fontweight="bold",
                  transform=ax_right.transAxes)
    y = 0.88
    for tag, q, a in roadmap:
        ax_right.text(0, y, tag, fontsize=10, fontweight="bold", color=VERMILLION,
                       transform=ax_right.transAxes)
        ax_right.text(0.06, y, textwrap.fill(q, 46), fontsize=9.3, fontweight="bold",
                      transform=ax_right.transAxes, va="top")
        y -= 0.07
        ax_right.text(0.06, y, textwrap.fill(a, 46), fontsize=9, color="#333333",
                      transform=ax_right.transAxes, va="top")
        y -= 0.14


def view_a():
    caption("Claim 1 -- the true answer is simple, even though the curve has 500 numbers", """
    A SELE curve is stored as 500 numbers, one per depth slice -- but that doesn't mean there
    are 500 independent things to learn. The code that generates realistic example curves for
    training actually draws just 5 real-world settings per curve (doping, diffusion length,
    surface recombination velocity, bulk lifetime, an absorption scale) and runs them through
    one physics solver. Left: across 1000 example curves, how much variation is left unexplained
    once you keep only the top k "principal components" -- the few directions curves actually
    vary along -- it's already negligible by k=5, matching those 5 real settings. Right: rebuild
    a random example curve from only its top-k components; drag k down and see how little survives.
    """)

    ax_left.semilogy(np.arange(1, len(cum_var) + 1), 1 - cum_var, "-o", ms=3, color=BLUE)
    ax_left.axvline(5, color=VERMILLION, ls="--", lw=1.3)
    ax_left.text(5.3, 0.3, "5 real physical\nsettings", color=VERMILLION, fontsize=8.5,
                 transform=ax_left.get_xaxis_transform())
    ax_left.set_xlabel("k = how many components we keep")
    ax_left.set_ylabel("how much curve variation is still missing")
    ax_left.set_xlim(0, 20)
    ax_left.grid(alpha=0.25)
    ax_left.set_title(f"an 'effective dimension' score of {participation_ratio:.2f}"
                       " (close to 1 = almost 1-dimensional)", fontsize=8.8)

    k = int(slider_k.val)
    idx = state["idx"]
    coeff = (X[idx] - mu) @ Vt_prior[:k].T
    rec = mu + coeff @ Vt_prior[:k]
    err = np.linalg.norm(rec - X[idx]) / np.linalg.norm(X[idx])
    ax_right.plot(Z, X[idx], color=BLACK, lw=1.6, label="original example curve")
    ax_right.plot(Z, rec, color=ORANGE, lw=1.8, ls="--", label=f"rebuilt from just k={k} numbers")
    ax_right.set_xlabel("depth z (µm)")
    ax_right.set_ylabel("SELE")
    ax_right.grid(alpha=0.25)
    ax_right.set_title(f"curve #{idx}  |  leftover error = {err:.2e}", fontsize=9)
    ax_right.legend(fontsize=8.3, loc="upper right")


def view_b():
    caption("Claim 2 -- our instrument can't see most directions a curve could vary in", """
    G is the matrix that turns a guessed SELE curve into a predicted measurement: 28 rows, one
    per wavelength we measure, and 500 columns, one per depth slice. Its "singular values" (left,
    log scale) measure how strongly a change along some direction of the curve actually shows up
    in the measurement -- most fall off a cliff, meaning most directions are almost invisible.
    The "noise %" slider is our guess at measurement error; only directions above that floor are
    trustworthy, giving the "effective rank" -- usually just a handful, out of 28 possible. Right:
    a few individual rows of G plotted against depth -- they die out within a few micrometers,
    because light is absorbed almost entirely near the surface. That's real physics, not a flaw.
    """)

    noise_pct = slider_noise.val
    r = effective_rank(noise_pct)
    ax_left.semilogy(np.arange(1, len(s_G) + 1), s_G / s_G[0], "-o", ms=4, color=BLUE)
    ax_left.axhline(noise_pct / 100.0, color=VERMILLION, ls="--", lw=1.3,
                     label=f"assumed noise floor ({noise_pct:.2f}%)")
    ax_left.axvline(r, color=VERMILLION, lw=1.3)
    ax_left.set_xlabel("which singular direction (1 = strongest)")
    ax_left.set_ylabel("how visible that direction is (relative)")
    ax_left.grid(alpha=0.25)
    ax_left.set_title(f"trustworthy directions today: {r} of 28   |   "
                       f"worst-to-best ratio: {s_G[0]/s_G[-1]:.1e}", fontsize=8.6)
    ax_left.legend(fontsize=8.3, loc="upper right")

    for i, c in zip([0, 9, 19, 27], [ORANGE, SKY, GREEN, VERMILLION]):
        ax_right.plot(Z, G[i], color=c, lw=1.4, label=f"row {i} (one wavelength)")
    ax_right.set_xlabel("depth z (µm)")
    ax_right.set_ylabel("how strongly this row 'feels' that depth")
    ax_right.grid(alpha=0.25)
    ax_right.set_title("light dies out with depth -- deep SELE is nearly invisible", fontsize=9)
    ax_right.legend(fontsize=8.3)


def view_c():
    caption("Claim 3 -- the directions we CAN measure aren't the ones curves vary in", """
    Even if the true curve has only 5 real degrees of freedom, that only helps if the
    measurement happens to be sensitive to those same directions. Left: if we could measure
    perfectly, with zero noise, how much of all the variation seen across 1000 example curves
    would still be visible using every one of G's 28 directions -- it maxes out around a third,
    not all of it. Restricting to only the noise-surviving directions (dashed line, from the
    slider) shrinks that further. Right: for each of the 6 strongest directions curves actually
    vary along, how much of that specific direction is visible to the measurement -- a short bar
    means that entire way a curve can vary is nearly invisible, however clever the algorithm is.
    """)

    noise_pct = slider_noise.val
    r = effective_rank(noise_pct)
    ks = np.arange(1, len(CAP_K) + 1)
    ax_left.plot(ks, CAP_K, "-o", ms=3, color=BLUE)
    ax_left.axhline(CAP_K[-1], color="#555555", ls=":", lw=1.2)
    ax_left.text(1, CAP_K[-1] + 0.012, f"best possible (all 28 directions): {CAP_K[-1]*100:.1f}%",
                 fontsize=8.3)
    ax_left.axvline(r, color=VERMILLION, ls="--", lw=1.3)
    ax_left.text(r + 0.3, max(CAP_K[r - 1] - 0.06, 0.02),
                 f"today's noise ({noise_pct:.2f}%): {CAP_K[r-1]*100:.1f}%", color=VERMILLION,
                 fontsize=8.3)
    ax_left.set_xlabel("k (using G's k strongest directions)")
    ax_left.set_ylabel("fraction of curve variation still visible")
    ax_left.set_ylim(0, 0.5)
    ax_left.grid(alpha=0.25)

    n_pc = 6
    ov_full = np.array([np.linalg.norm(Vt_G[:28] @ Vt_prior[j]) for j in range(n_pc)])
    ov_rank = np.array([np.linalg.norm(Vt_G[:r] @ Vt_prior[j]) for j in range(n_pc)])
    xpos = np.arange(n_pc)
    ax_right.bar(xpos - 0.2, ov_rank, width=0.4, color=VERMILLION, label=f"today's noise (rank {r})")
    ax_right.bar(xpos + 0.2, ov_full, width=0.4, color=BLUE, label="best possible (rank 28)")
    ax_right.set_xticks(xpos)
    ax_right.set_xticklabels([f"variation\nmode {j}" for j in range(n_pc)], fontsize=8)
    ax_right.set_ylabel("how visible this mode is (1 = fully, 0 = invisible)")
    ax_right.set_ylim(0, 1.05)
    ax_right.grid(alpha=0.25, axis="y")
    ax_right.legend(fontsize=8.3)


def view_d():
    caption("Claim 4 -- some depths are trustworthy, others are basically a guess", """
    This combines everything so far into one number per depth: given a real, noisy measurement,
    how much does our belief about the curve at that depth actually narrow, compared to before
    we measured anything? 1 means we learned nothing there (we're just reporting our prior
    assumption back); 0 means we're now confident. Left: that ratio across depth -- shallow
    depths (green band) are genuinely informed by data, deep ones (red band) are not. The tick
    marks show where 18 real reference curves (extracted from the source paper) actually peak --
    reassuringly, mostly inside the trustworthy region. Right: what an honest answer looks like --
    a shaded uncertainty band, not one confident line -- with a real example curve for comparison.
    """)

    noise_pct = slider_noise.val
    sigma = noise_pct / 100.0 * ele_rms
    A = C @ G.T
    M = G @ C @ G.T + sigma ** 2 * np.eye(G.shape[0])
    Sol = np.linalg.solve(M, G @ C)
    diag_cp = np.diag(C) - np.sum(A * Sol.T, axis=1)
    post_sd = np.sqrt(np.clip(diag_cp, 0, None))
    ratio = post_sd / prior_sd_depth

    ax_left.axvspan(0, 3, color=GREEN, alpha=0.15)
    ax_left.axvspan(3, 8, color=YELLOW, alpha=0.25)
    ax_left.axvspan(8, W_UM, color=VERMILLION, alpha=0.13)
    ax_left.text(1.5, 0.94, "trustworthy", ha="center", fontsize=7.8, color="#175c46")
    ax_left.text(5.5, 0.94, "marginal", ha="center", fontsize=7.8, color="#7a6600")
    ax_left.text(19, 0.94, "prior-dominated", ha="center", fontsize=7.8, color="#8a2f0f")
    ax_left.plot(Z, ratio, color=BLACK, lw=1.8)
    if TEST_SET_PEAKS.size:
        ax_left.plot(TEST_SET_PEAKS, np.zeros_like(TEST_SET_PEAKS), "|", color=PURPLE, ms=16, mew=2,
                     label="18 real reference curves peak here")
        ax_left.legend(fontsize=8, loc="center right")
    ax_left.set_ylim(0, 1.02)
    ax_left.set_xlabel("depth z (µm)")
    ax_left.set_ylabel("still-uncertain fraction (1=blind, 0=confident)")
    ax_left.grid(alpha=0.25)
    ax_left.set_title(f"assumed noise = {noise_pct:.2f}%", fontsize=9)

    idx = state["idx"]
    ax_right.plot(Z, mu, color=BLUE, lw=1.5, label="average curve (the prior)")
    ax_right.fill_between(Z, mu - 2 * post_sd, mu + 2 * post_sd, color=BLUE, alpha=0.2,
                           label="honest uncertainty band")
    ax_right.plot(Z, X[idx], color=BLACK, lw=1.3, ls="--", label=f"a real example curve (#{idx})")
    ax_right.set_xlabel("depth z (µm)")
    ax_right.set_ylabel("SELE")
    ax_right.grid(alpha=0.25)
    ax_right.legend(fontsize=8.3)


def view_e():
    caption("Claim 5 -- which real-world numbers survive the measurement?", """
    Instead of the whole 500-point curve, what about numbers you might actually care about: the
    SELE value right at the surface, the peak height, where the peak sits, the total area under
    the curve, and the surface-to-peak ratio (a stand-in for how bad surface damage is)? Pick a
    curve (button on the left), pretend it produced a noisy measurement, then ask: across our
    1000 example curves, which ones are still consistent with that same measurement, and how
    much do they still disagree on each of these five numbers? Short bars are genuinely
    recoverable; long bars are not. "ESS" is a built-in health check -- if it drops too low, the
    answer isn't trustworthy at that noise level, no matter what the bar says.
    """)

    noise_pct = slider_noise.val
    sigma = noise_pct / 100.0 * ele_rms
    idx = state["idx"]
    r = E - E[idx]
    ll = -0.5 * (r ** 2).sum(1) / sigma ** 2
    w = np.exp(ll - ll.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)

    labels = {"surface": "value at the surface", "peak": "peak height", "peak_pos": "peak depth",
              "area": "total area (integral)", "surf_ratio": "surface / peak ratio"}
    names = list(DERIVED_ALL.keys())
    ratios, true_vals, est_vals = [], [], []
    for name in names:
        v = DERIVED_ALL[name]
        m = np.sum(w * v)
        sd = np.sqrt(np.sum(w * (v - m) ** 2))
        ratios.append(sd / v.std())
        true_vals.append(v[idx])
        est_vals.append(m)

    colors = [GREEN if ratio < 0.3 else YELLOW if ratio < 0.7 else VERMILLION for ratio in ratios]
    ax_left.barh([labels[n] for n in names], ratios, color=colors, edgecolor="#333333", lw=0.6)
    ax_left.set_xlim(0, 1.0)
    ax_left.set_xlabel("still-uncertain fraction (short bar = recoverable)")
    ax_left.grid(alpha=0.25, axis="x")
    unreliable = ess < 20
    ax_left.set_title(f"target curve #{idx}   |   how many curves 'count': {ess:.0f} of {N_CURVES}"
                       + ("   -- TOO FEW, WIDEN NOISE" if unreliable else ""), fontsize=8.8,
                       color=VERMILLION if unreliable else "#222222")

    ax_right.axis("off")
    ax_right.add_patch(Rectangle((0, 0), 1, 1, transform=ax_right.transAxes,
                                  facecolor="#fdfdfd", edgecolor="#dddddd"))
    ax_right.text(0.06, 0.94, "true value   vs.   our best guess", fontsize=9.5, fontweight="bold",
                  transform=ax_right.transAxes)
    y = 0.78
    for name, t, e in zip(names, true_vals, est_vals):
        ax_right.text(0.06, y, labels[name], fontsize=9, transform=ax_right.transAxes)
        ax_right.text(0.06, y - 0.055, f"{t:.4g}   vs.   {e:.4g}", fontsize=9.3,
                       color="#333333", family="monospace", transform=ax_right.transAxes)
        y -= 0.155


def view_f():
    caption("Claim 6 -- proof by example, with no formulas at all", """
    Everything so far leaned on some statistical assumptions (Gaussian noise, a Gaussian prior).
    This view skips all of that: pick a curve, then literally search the 1000 example curves for
    every other one that would produce almost the same measurement, within the "noise %"
    tolerance on the slider. Left: every curve that "fools" the measurement, overlaid, with the
    chosen one in black -- the spread between them IS the leftover ambiguity, visible with no
    math needed. Right: how wide that spread is at each depth, confirming the same
    shallow-is-good, deep-is-bad pattern as claim 4 -- but arrived at by direct comparison of
    real curves instead of a formula.
    """)

    tol = slider_noise.val / 100.0
    idx = state["idx"]
    d = np.linalg.norm(E - E[idx], axis=1) / np.linalg.norm(E[idx])
    match = d < tol
    n_match = int(match.sum())

    if n_match < 2:
        for ax in (ax_left, ax_right):
            ax.axis("off")
        ax_left.text(0.5, 0.5,
                     f"only {n_match} curve(s) fool the measurement at {tol*100:.2f}% tolerance\n"
                     f"→ raise the 'assumed noise %' slider to see the ambiguity grow",
                     ha="center", va="center", fontsize=10.5, transform=ax_left.transAxes)
        return

    Xm = X[match]
    ax_left.plot(Z, Xm.T, color="0.78", lw=0.6)
    ax_left.plot(Z, X[idx], color=BLACK, lw=1.9, label=f"the chosen curve (#{idx})")
    ax_left.set_xlabel("depth z (µm)")
    ax_left.set_ylabel("SELE")
    ax_left.grid(alpha=0.25)
    ax_left.set_title(f"{n_match} of {N_CURVES} curves give almost the same reading", fontsize=9)
    ax_left.legend(fontsize=8.3)

    sd_match = Xm.std(0)
    scale = np.abs(Xm).mean(0) + 1e-30
    ax_right.plot(Z, sd_match / scale, color=VERMILLION, lw=1.6)
    ax_right.set_xlabel("depth z (µm)")
    ax_right.set_ylabel("spread among fooling curves (normalized)")
    ax_right.grid(alpha=0.25)
    ax_right.set_title("where the leftover ambiguity actually lives", fontsize=9)


DRAW_FUNCS = {
    VIEWS[0]: view_intro,
    VIEWS[1]: view_a,
    VIEWS[2]: view_b,
    VIEWS[3]: view_c,
    VIEWS[4]: view_d,
    VIEWS[5]: view_e,
    VIEWS[6]: view_f,
}

# Which sliders each view actually reads -- the rest are hidden so the controls on screen
# never imply an effect a view doesn't have.
SLIDERS_USED = {
    VIEWS[0]: set(),
    VIEWS[1]: {"k"},
    VIEWS[2]: {"noise"},
    VIEWS[3]: {"noise"},
    VIEWS[4]: {"noise"},
    VIEWS[5]: {"noise"},
    VIEWS[6]: {"noise"},
}


def redraw(_event=None):
    ax_left.clear()
    ax_right.clear()
    ax_caption.clear()
    ax_caption.axis("off")
    for ax in (ax_left, ax_right):
        ax.set_facecolor("#fdfdfd")

    used = SLIDERS_USED[state["view"]]
    ax_noise.set_visible("noise" in used)
    ax_k.set_visible("k" in used)

    DRAW_FUNCS[state["view"]]()
    fig.canvas.draw_idle()


def on_view(label):
    state["view"] = label
    redraw()


def on_resample(_event):
    state["idx"] = int(rng.integers(N_CURVES))
    redraw()


radio.on_clicked(on_view)
slider_noise.on_changed(redraw)
slider_k.on_changed(redraw)
button_resample.on_clicked(on_resample)

redraw()

if __name__ == "__main__":
    plt.show()
