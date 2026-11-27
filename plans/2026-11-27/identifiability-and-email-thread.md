# 2026-11-27 — Identifiability, fitting without a network, and the email thread

A discussion that ran alongside the parameter-recovery study (`parameter-recovery-study.md` in
this folder). Nothing here was implemented; it records what was discussed and concluded.

## Fitting the five parameters without a network

- Possible, and cheap: the closed-form ELE simulates about 100k parameter sets in roughly 8 s.
- **Bank screening**: simulate one large bank of random parameter sets (for example 1M) once,
  and keep every set whose ELE is within 1% of the measurement.
  `src/forward_model/standalones/probe_parameter_bounds.py` already does a small version of
  this (300k draws) to find best fits.
- **Fitting from many starting points**: least-squares fits started from the bank survivors.
  Each start converges onto the solution set, so together they trace its shape (one ridge or
  several islands). Bank screening alone undersamples a thin ridge.
- The network's only real advantage is speed per measurement. A direct search finds the
  ambiguity with no model in the way, so it is the yardstick any retrained network should
  be measured against, and it may make the retrain unnecessary.

Terms, for reference: **K** is the number of Gaussian components ("ranges") a mixture network
would output; K = 1 is today's model. The success criteria proposed for a retrain were truth
inside some component, every component fitting the ELE, calibration close to 68/95, and SELE
quality no worse than today.

## Is the ambiguity real, and does it matter for SELE?

- **Real**, and established before today. On the forward-model test set (noise-free, no model
  mismatch), best draws fit the ELE to a few hundredths of a percent and sit up to 1.4 decades
  from the true parameters. Going from 4k to 40k draws moved the best draw along the solution
  set, not towards the truth. The feasibility analysis found about 4 independent ELE directions
  noise-free against 5 parameters, so at least one direction leaves the ELE unchanged, and
  about 2 survive 1–5% noise.
- **Mostly not in the measured region.** On the same set, with the parameters off, the best
  draw's SELE error is ≤ 0.2% at 0–1 µm and ≤ ~3% at 1–3 µm, but up to 50% at 8–30 µm.
  Incident light is absorbed within a few µm, so solutions agree where carriers are generated
  and diverge where G is near zero.
- Today's recovery study points the same way: the diffusion length is recovered 71% of the
  time and S/D 87%, against 3% for τ alone. The ELE determines combinations, not the raw five.
- A limitation of that study: it only sees parameter sets the network proposed, so its
  ambiguity is a lower bound on the problem's.

## The email thread (Liraz, Tamir, Gideon; Alon reviewing the code)

**Positions**

- **Liraz**: several parameter sets fit the ELE even on curves made by the same simulator, so
  without a more accurate model there is no metric to choose between them.
- **Tamir**: separate the fitting model from the prior. The 5-parameter solutions could
  initialise a more physical model. Proposes min ||y − G(x)|| + R(x) over the 5 parameters,
  trying L2 and L1, but expects to return to diffusion models.
- **Gideon**: agrees there are two issues. The pressing one is non-uniqueness inside the region
  the data sees (where G is non-zero). Proposes finding which parameter sets give similar
  solutions, and whether they are physically plausible, which might make it a non-issue.

**Assessment**

- Liraz's first claim is confirmed by the data. The second goes too far: a prior _can_ choose
  between equal fits (that is what regularisation does), it just can't verify that its choice
  is true. "Distinguish" is being used both as "select" and as "verify".
- A more accurate forward model does exist: the COMSOL simulations behind the paper-figure test
  set include photon recycling and back-surface multi-pass escape. The 1–3 µm error rises
  from < 3% (forward-model set) to 5–12% (paper set) because of that mismatch.
- Three different things are being called "the model": the forward physics (5-parameter
  simulator or COMSOL), the unknown (5 parameters or a 500-point S), and the inference
  machinery (network or optimiser). Tamir's "longer vector s" is the curve-space unknown used
  by the Tikhonov, TV and score-model modes.
- The parametric method already is Tamir's formula: the unknown is the 5 parameters, the forward
  model is the simulator then G, and the prior is the uniform training box.
- Reservations about Tamir's plan:
  - L2 or L1 on five normalised parameters has no physical meaning; it just pulls towards
    the centre of the box.
  - A diffusion prior trained on this simulator's own curves learns roughly the same box
    pushed through the simulator, so it adds no new information and would choose along the
    ridge by training density.
  - Real external information beats any prior: an independently measured doping p0 would
    likely collapse much of the ridge, since p0, D and S move together along it
    (hypothesis, untested).
- Gideon's plausibility check connects to the code: D's lower bound was dropped from 50 to
  5 cm²/s so the paper's lifetime curves could be fitted, and those fits ask for D of 6–40,
  which looks like compensation for missing photon recycling. A physical range on D might
  remove part of the ridge.
- The deep tail can't be honestly fixed by any prior. Only a measurement that reaches there
  could, such as near-bandgap wavelengths or back-side illumination.

## Suggested next check (not started)

Using the draws that fit each fresh curve's ELE:

1. Find the depth at which their SELEs start to disagree by more than about 10%, against how
   deep the light reaches, and against the true diffusion length.
2. Check whether the parameter sets that fit fall outside physically plausible ranges (D
   especially), and how much fixing p0 shrinks the spread.

This answers Gideon's question with numbers, for the meeting with Alon on Tuesday at 14:00.
