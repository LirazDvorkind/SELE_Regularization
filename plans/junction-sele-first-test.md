# Junction SELE: first feasibility test

Self-contained brief for a fresh session. Read it, then `plans/pn-junction-sele-primer.md`
(junction SELE physics) and `plans/measurement-noise-open-questions.md` (noise status).

## Goal and motivation

The project extracts SELE from wavelength-resolved ELE. It works for a GaAs wafer, with
Tikhonov (Tamir Yeshurun's method) and with a parametric simulator-based model. The next
target is a **p-n junction**. The plan is a ladder of increasingly assumption-free methods:

1. **Tikhonov** — Tamir's method applied unchanged. It answers "how well does the published
   method fare on a junction?"
2. **Parametric** — the simulator-parameter model, retrained for a junction. Later, not in
   this task.
3. **Weak-assumption / "no physical model"** — for example, SELE as a sum of exponentials, or
   a profile that may or may not contain dips, with the method deciding **whether there is no
   junction, one, or more**. This is the end goal: more general and more robust than a device
   model, provided it converges robustly.

**This task:** the cheapest test that tells us whether rung 3 is viable at all. Make one
synthetic junction measurement, then run rung 1 (Tikhonov) and a minimal rung 3 (a
"smooth profile" vs. "profile with one dip" model comparison). If one dip cannot be detected
at realistic noise, the more advanced methods will not rescue it.

## What is already established (do not redo)

**The information budget is small.** The SVD of the wafer G (`Data/score_model/G_score_model_500.csv`,
28 wavelengths over 400–670 nm, 500 points over 30 µm) gives singular values
s/s0 = 1, 0.33, 0.086, 0.022, 0.0058, 0.0016, ... The number of independent pieces of depth
information that survive is 6, 5, 4 and 3 at 0.1%, 0.5%, 1% and 5% noise.

**The depth limit (re-checked on the wafer G).** This uses Backus–Gilbert blur width
ℓ(z) = dz / P_ii, with P = V_r V_rᵀ. At 1% noise: ℓ ≈ 0.12 µm at 0.1 µm depth, 0.36 µm at
0.3 µm, 0.47 µm at 0.5 µm, and about 8 µm at 1 µm, so there is no information there. The data
determine SELE to about 0.5–0.7 µm. The mesh spacing (0.06 µm) bounds the near-surface
figures. Script: `verify_res.py` in the previous session's scratchpad (not kept). It is about
20 lines, so rewrite it if needed.

**Degrees of freedom.** Wafer SELE curves from the 5-parameter simulator span about 5
dimensions, and G sees about 5 (noise-free). The two subspaces only partly overlap (see
`feasibility_demo.py`, Claim 3). The parametric work found only about 2 parameter combinations
resolvable (`plans/parametric-model.md`). A junction is assumed to have about 10 degrees of
freedom, so **about half are unrecoverable from ELE alone**. Any rung-3 model must therefore
spend at most about 4 free numbers. A dip costs about 2 (position, width) or 3 (with depth).

**The junction-G numbers in the primer are unverified.** `pn-junction-sele-primer.md` §8 quotes
a blur of 0.04–0.1 µm at the junction and a horizon of 0.7 µm for the thesis device. Those need
transfer-matrix optics (TMM), and **the repo has no TMM code**. Treat them as unverified.

**Noise.** It is not measured. Thesis App. D.5's "5%" is a common power-meter scale factor
(Fig. D.4), which does not affect the depth limit. Repeatability is at most about 1–2%, read by
eye from Fig. D.3, at a single wavelength. **Use 1% white noise as the baseline and sweep
0.5–5%.** Details and open questions: `plans/measurement-noise-open-questions.md`.

**Tamir's own work never quantifies a depth limit.** The paper says only that the extracted and
simulated SELE agree "within the first 500 nm". SI §S3 and thesis §2.4.1 say only, qualitatively,
that there is no information where generation is small. The thesis simulates the junction's SELE
forward (COMSOL, eq. 5) and **never inverts measured junction ELE**. A quantified depth and
resolution limit for this setup is a publishable result in its own right.

## Literature background (abstracts unless noted)

- **Trans-dimensional Bayesian inversion** (geophysics; reversible-jump MCMC) infers a layered
  profile with an unknown number of layers and reports the probability of each count. It is
  the closest structural analogue to "how many junctions". Sources: Dosso & Dettmer et al.,
  Inverse Problems 2014 (doi 10.1088/0266-5611/30/11/114018); Piana Agostinetti & Malinverno,
  GJI 2010 (doi 10.1111/j.1365-246x.2010.04530.x).
- **Bayesian-mixture distribution-of-relaxation-times analysis** (energy devices, impedance
  spectroscopy). It inverts an ill-posed kernel and decides the number of peaks automatically,
  with uncertainty. Sources: Liu & Ciucci, J. Phys. Chem. C 2025 (doi 10.1021/acs.jpcc.5c04766;
  the ChemRxiv preprint is behind a bot wall); Gaussian-process versions in Maradesa, Py &
  Ciucci, J. Power Sources 2024.
- **Evidence-based choice between one and two exponentials** in fluorescence lifetime imaging
  (Rowley et al., PLoS One 2016). It is the template for the minimal rung 3 here.
- **Regularized inverse Laplace (CONTIN) with an L-curve** (Pajuelo et al. 2026,
  arXiv 2606.14761) is essentially our Tikhonov. It gives a smooth profile and no feature count.
- **Caveat, read in full: Mosegaard, arXiv 2308.05858v2** (v3 has been withdrawn in favour of a
  newer Mosegaard & Curtis paper, not yet read). The Bayes factor between "1 layer" and
  "2 layers" can flip with the *width of the parameter prior* alone, with the data fit and the
  posterior unchanged. So the feature count partly reflects the prior. This becomes acceptance
  criterion (a) below.

## The task

### Step 0 — Synthetic junction SELE (the ground truth)

`src/drift_diffusion/` solves the steady-state p-n junction under arbitrary G(x) and bias, but
**it does not compute SELE or PL yet** (its README says so). Two options:

- **(A) Quick:** a hand-built SELE with the primer's shape: an emitter peak at about
  0.05–0.1 µm, a near-zero dip across the depletion region around the junction, then a base
  plateau that decays. Fast. It tests detectability, not the physics.
- **(B) Proper:** add SELE to the drift-diffusion solver by perturbation (paper eq. 5). Add a
  small generation pulse at each depth z, then compute the change in emitted PL divided by the
  added generation. PL ≈ ∫ R_rad(x) · P_esc(x) dx, with P_esc a simple escape probability (no
  photon recycling at first; say so). Run at short circuit or open circuit and state which.

Recommendation: start with (A) to get the pipeline running, then (B). Caveats for (B): the
solver's front contact is an **ideal ohmic contact** (infinite SRV) at x = 0, unlike the thesis
device's AlInP window, so the emitter SELE will be pessimistic. The thesis device geometry is in
the primer (n-GaAs emitter 0–0.12 µm at 2e18, p-base to 3.62 µm at 9e16).

### Step 1 — Measurement

- G: use the project's Beer–Lambert G with GaAs optics (`src/mesh.py` and
  `src/optical_constants.py`; pass `k_bulk`) on the existing 28 wavelengths. State plainly that
  this ignores the window/BSF layers and interference (no TMM).
- ELE = (1/phi_abs) · G @ S, plus 1% white noise (sweep 0.5–5%), several noise draws.

### Step 2 — Rung 1: Tikhonov

Run the existing Tikhonov (`src/regularization/tikhonov_non_uniform.py`: `sweep_kappa`,
`find_knee`). Report whether the dip appears, where, how wide, and how the result varies across
noise draws. Compare against ℓ(z) at the junction depth.

### Step 3 — Minimal rung 3: smooth vs. one dip

Two models on the same mesh, each within the budget of about 4 free numbers:

- M0: a smooth base profile (for example 2–3 exponentials, or a low-order spline);
- M1: M0 multiplied by one dip (position, width, depth).

Compare them by BIC first (cheap), then Bayesian evidence if BIC is promising. Also run M1 on
data generated *without* a dip, to check for false positives.

### Acceptance criteria — when a detected dip counts as real

(a) It survives a factor of about 2–3 change in the prior widths (Mosegaard).
(b) It survives different noise draws at the assumed noise level.
(c) It survives a few percent of error in G (perturb the k(λ) used to build G but not the one
    that generated the data).
(d) There are no false positives on dip-free data.

## Deliverables

- One standalone script (for example `src/junction_feasibility/` or a standalone under
  `src/evaluation/`), plus colour-blind-safe figures (Okabe-Ito; see CLAUDE.md).
- A short write-up of the results: whether the dip was detected, at which noise level, and how
  sharp it was compared with ℓ(z).
- If (B) is built, a `compute_sele` entry in `src/drift_diffusion/` with a check in
  `check_drift_diffusion.py`.

## Out of scope for now

- TMM optics and the full thesis layer stack (the next step if this test passes).
- The parametric junction model (rung 2).
- Full trans-dimensional or Bayesian-mixture methods. Only after the minimal test works.
