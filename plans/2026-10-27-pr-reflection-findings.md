# Photon recycling and back reflection: how much do they move SELE? (2026-10-27)

Question: the method-comparison figure shows reconstructed SELE below the ground truth. The
truth (Tamir's COMSOL) includes photon recycling (PR) and back-surface reflection; the repo's
forward model does not. Is that the cause, and how large are the two effects?

Method and assumptions: `docs/pr_reflection_estimate.md`. Scripts:
`src/forward_model/standalones/estimate_pr_reflection.py` and `refit_without_recycling.py`.
Nothing was trained; no network was rerun. Neither script is validated against COMSOL.

## 1. What the figure data show

Signed deviation of the SELE integral, (method - truth) / truth, from the cached
reconstructions behind `results/method_comparison/all_methods_sele_overlay.png`.

| window | parametric | Tikhonov |
|---|---|---|
| 0-1 µm | at most -0.7% | at most -1.5% |
| 1-2 µm | -6% (tau >= 1 ns) | about +1.5% |
| 2-3 µm | about -12% (tau >= 1 ns) | about -13% (tau >= 2 ns) |
| 3-10 µm | about -29% | about -45% |
| 10-30 µm | about -45% | about -44% |

- The shortfall is not universal. For tau >= ~0.5 ns both methods are ~37-41% low over
  0-30 µm. At low SRV and low tau, Tikhonov and the score model are above the truth.
- Peak SELE ratio, reconstruction / truth: parametric 0.77, Tikhonov 0.73 for tau >= 4 ns.
  Those peaks sit at 5-6 µm, deeper than the ELE reaches.
- The ELE is blind to depth. Fraction of the ELE kernel deeper than 3 µm: 5e-87 at 400 nm,
  8e-5 at 670 nm. Deeper than 10 µm: below 1e-13. Everything below ~3 µm in every
  reconstruction is regularisation or prior, not data.
- Tikhonov is fed `G @ S_truth` and inverts the same G, so missing PR cannot bias it. Only
  the parametric model, whose simulator cannot make the truth's shape, can be affected.

## 2. Independent estimate of reflection and PR

Four SELE variants on one finite-volume mesh of the 350 µm wafer:
V0 = repo simulator escape, V1 = angle-resolved direct escape, V2 = V1 + reflection and
multiple passes (SI S4.9), V3 = V2 + self-consistent PR (SI S4.1-S4.6).

Checks run: photon conservation (absorbed + front + back escape = 1) closes to 4e-7; V0 matches
`simulate_sele` to 1e-4 (stride 4); loop gain below 1 everywhere; strides 4 and 8 agree.
Checks not run: any comparison with COMSOL or the paper's own SELE.

Operating point: SI Table S5.1 (p0 = 1e19, tau_SRH = 0.4 ns, S = 2e7 cm/s, D = 38.7 cm^2/s from
1500 cm^2/Vs by Einstein). Recycling is quoted for the simulator's emission spectrum as given
and scaled up to B0 (the simulator's spectrum integrates to 1.15e-10 cm^3/s, its radiative
lifetime uses 2.5e-10).

| effect | size |
|---|---|
| angle-resolved escape vs the simulator's (V1/V0) | 0.96-0.98 |
| back/front reflection and multiple passes (V2/V1) | 1.000-1.002 |
| PR at S5.1 (V3/V2): surface / 3 µm / 10-30 µm | 1.07 / 1.13 / 1.17 |
| PR at S5.1, spectrum scaled to B0 | 1.17 / 1.34 / 1.44 |
| PR effect on the ELE at S5.1 | +5% (+15% scaled) |

PR depends strongly on doping and lifetime (V3/V0, spectrum as given, S = 2e7):

| p0 [cm^-3] | tau_SRH | loop gain | 1 µm | 10 µm | 30 µm |
|---|---|---|---|---|---|
| 1e17 | 0.4 ns | 0.01 | 0.98 | 0.98 | 0.98 |
| 1e17 | 60 ns | 0.79 | 1.47 | 2.04 | 3.35 |
| 3e17 | 60 ns | 0.91 | 1.93 | 3.44 | 7.47 |
| 1e18 | 0.4 ns | 0.08 | 1.02 | 1.05 | 1.05 |
| 1e18 | 60 ns | 0.78 | 1.64 | 2.90 | 4.47 |
| 3e18 | 60 ns | 0.52 | 1.33 | 1.93 | 2.07 |

Reading: reflection is small everywhere tested (<=0.2%). PR is small for short lifetimes and
large for long lifetimes at low doping.

## 3. Does leaving PR out move the retrieved SELE? (refit)

Fit the five-parameter recycling-free simulator to the ELE of V3, noise-free, same G on both
sides. Ratio = refit SELE integral / V3 SELE integral.

| target | 0-2 µm | 2-3 µm | 3-10 µm | 10-30 µm | ELE misfit |
|---|---|---|---|---|---|
| p0 1e18, tau 60 ns | 0.99-1.00 | 0.98 | 0.90 | 0.79 | 0.01% |
| same, spectrum scaled to B0 | 1.00 | 0.98 | 0.91 | 0.75 | 0.01% |
| p0 1e18, tau 0.4 ns | 1.00-1.01 | 1.02 | 1.02 | 0.97 | 0.01% |
| p0 1e19 (S5.1), tau 0.4 ns | 1.00-1.02 | 1.04 | 1.02 | 0.87 | 0.04% |
| same, spectrum scaled to B0 | 1.00 | 1.01 | 1.03 | 1.06 | 0.00% |

In the strong-PR case the peak comes out 11-14% low and 1.9-2.4 µm too shallow. The fitted
parameters are not physical: p0 1.7e17 against a true 1e18, D 8.8 against 38.7, alpha_scale
at its floor of 0.1 (so the box limits are being used).

## 4. What this does and does not establish

- **Confirmed (for this model):** reflection is negligible; PR is a few percent to ~25% at
  S5.1 and can reach several-fold at low doping with long lifetime; with free parameters the
  recycling-free simulator recovers the 0-3 µm SELE to within 0-4%, and loses mainly the
  region beyond 3 µm that the ELE does not constrain.
- **Not explained:** the figure's parametric deficit (about -12% at 2-3 µm, about -45% at
  10-30 µm) is larger than the refit loses (-2% and -21 to -25%) even in the strongest-PR case.
  PR alone is not shown to be the cause.
- **Unknown:** the doping and lifetime the sweep figures used. At S5.1 the model is ~5x fainter
  than the truth and tau_eff saturates at 0.25 ns, so the figures probably used something
  else (a lower doping around 1e18 reproduces the peak position; this was not fitted and is a
  hypothesis).
- **Assumptions not validated:** no native oxide or TMM, low injection, D held fixed,
  Fresnel only at the GaAs/air interface.

## 5. Suggested next checks

1. Read the parametric model's best-fit parameters for `tau_60ns` and `srv_1e7` and see whether
   they sit at the parameter-box limits (D = 5, alpha_scale = 0.1, tau, p0), which would point
   to the box rather than missing physics.
2. Pin down the sweep figures' operating point, then repeat section 2 there.
3. Compare V3 against a COMSOL curve if one can be obtained.
