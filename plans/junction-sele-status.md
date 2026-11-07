# Junction SELE: what we agreed, what was done, what we learned

Status as of 2026-11-07. This supersedes the method plan in `junction-sele-first-test.md`
(its "smooth vs. one dip" model comparison and BIC were dropped, see below); the background
there and in `pn-junction-sele-primer.md` still holds. Code lives in `src/pn_junction/` (see its
README); figures and printed summaries in `results/pn_junction/feasibility/`.

## What we agreed

**Where things live.** Everything junction-related sits under `src/pn_junction/`; the
drift-diffusion solver moved there from `src/drift_diffusion/`.

**The truth comes from a simulator, not a hand-drawn dip.** The drift-diffusion solver gives the
junction's SELE. The reconstruction methods never see the solver; they only get ELE = G @ SELE
plus noise. The data are made on a finer mesh than any method inverts on.

**Three methods, in this order.**

1. Tamir's Tikhonov, unchanged.
2. The parametric (simulator-parameter) model, retrained for a junction. Deferred until the
   other two are understood, because it needs a fast junction simulator and only answers
   "which junction parameters", not "is there a junction".
3. A generic method, with these rules:
   - It assumes no device model and no number of junctions (wafer, one, or several).
   - It is not a "test for a dip" method: it must be able to return a curve nobody anticipated,
     as long as it fits the data.
   - It knows its limits: it shows where the data stop saying anything, and states every
     assumption it makes.
   - Sum-of-exponentials was rejected as a hidden model.

**What the generic method assumes.** SELE is never negative. Neighbouring depths usually have
similar SELE, but a sudden jump or drop may happen anywhere, any number of times. How small the
usual steps are and how often big ones happen are learned from the data, then halved and
doubled as a check. The answer is the spread of thousands of curves that all fit the data
within the noise.

**How it is judged.** It must find the dip where there is one, not invent one where there is
not (dip-filled junction and a plain wafer as controls), and survive another noise draw, a
few-percent error in G, and the learned settings halved and doubled.

**Noise.** 1 % white relative noise as the baseline, swept 0.5–5 %. Not yet measured (see
`measurement-noise-open-questions.md`).

## What was done

- **Solver additions.** Finite surface recombination at minority-carrier contacts, and SELE
  (and the collected-current profile) from one adjoint solve. Checked against perturbing one
  depth at a time and against global balance (`check_drift_diffusion`).
- **Truth.** The thesis' thin-film GaAs cell: 0.12 µm n-emitter at 2e18 on a 3.5 µm p-base at
  9e16, slow back surface (100 cm/s) in place of the AlGaAs back-surface field, ohmic front, no
  mirror, no photon recycling. Junction depth swept 0.12, 0.3, 0.5, 0.8 µm.
- **Measurement.** Beer–Lambert G on the 28 measurement wavelengths (400–670 nm), noise added
  per wavelength. Blur width from the singular vectors of G marks where the data stop placing
  features (blur width > 2 × depth): beyond 0.83 µm at 0.5–1 % noise, 0.59 µm at 2 %, 0.35 µm
  at 5 %.
- **Tikhonov** (`run_junction_tikhonov`): 20 noise draws per junction depth and noise level.
- **Generic method** (`profile_sampler.py`, run by `run_junction_sampler`): Student-t law on
  neighbour steps, written as a Gaussian with a random width per step; the profile is drawn
  exactly from the resulting Gaussian restricted to SELE ≥ 0 by bouncing Hamiltonian motion off
  the zero wall; step size and tail weight learned on a grid. Four independent chains per case.
  About 2 ms per step; the full study runs in ~7 minutes on 16 cores.

## What we learned

**The junction is only visible near the surface.** Almost all the light is absorbed in the top
half-micron, so the data barely feel a deeper junction, whatever the method. Filling the dip
changes the ELE by 51 % (junction at 0.12 µm), 14 % (0.3), 4 % (0.5) and 0.3 % (0.8).

**Tikhonov sees the junction down to about 0.3 µm**, half-sees it at 0.5 µm at low noise, and
not at 0.8 µm. It routinely goes negative, which is unphysical, and after the dip it does not
recover the way the truth does.

**The generic method:**

- Never goes negative.
- Finds the dip without being told to look: clearly at 0.12 µm at 1 % and 5 % noise, and at
  0.3 and 0.5 µm at 1 %. At 0.5 µm and 1 % it recovers about 0.75 of the dip depth against
  Tikhonov's 0.45.
- Invents no dip on the plain wafer, whose truth lies inside its spread everywhere.
- Widens its spread where the data say nothing, and claims no dip at 0.8 µm.
- With G a few percent wrong, its curves no longer fit the data (misfit 3–4 × the noise), so a
  bad optical model shows up instead of passing silently.
- **Is overconfident on smooth shapes.** The learned rule prefers flat stretches joined by
  sudden steps (it settles on the heaviest tails allowed), so smooth humps come out as
  flat-topped blocks with a narrow spread. At 1 % noise, for junctions at 0.12–0.5 µm, the truth
  sits inside the 90 % band at only 50–80 % of the seen depths.
- **Sometimes has not settled.** In a few cases the four chains disagree (worst: the tilted-G
  case); those need longer runs.

**The dip-filled control is flawed** (for both methods). The straight line from the emitter
peak to just past the depletion region still ends where the base SELE is close to zero, so the
"filled" truth keeps a dip. The plain wafer is the clean no-dip control.

**The truth is not thesis Fig. 3.11.** It is about 15× smaller (no mirror, no photon recycling,
different material values), and the emitter is weaker relative to the base. A deeper junction
also means a thicker emitter, so emitter size and junction depth change together in the sweep.

## Next

1. Put the "usually steady, sometimes sharp" rule on the slope instead of the level, so smooth
   humps cost nothing while sudden drops stay allowed; run longer chains; re-check the 90 %
   band actually covers 90 %.
2. Replace the dip-filled control with one that is dip-free by construction.
3. Then the parametric model.
