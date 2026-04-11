# Direction A — Total Variation + Discrepancy Principle

This is the direction to try **first**. It is the honest, principled baseline that any learned method has to beat. Read `GENERIC_GUIDELINES.md` first for context on why we are stepping away from the score-grad approach.

---

## The core idea

Replace the second-derivative penalty in the paper's Tikhonov formulation with a **Total Variation** penalty:

```
min over S of    ||GS - B||²   +   λ · TV(S)
```

where `TV(S) = sum_i |S_{i+1} - S_i|` (the L¹ norm of the first difference). The two terms have very different jobs:

- The data term `||GS - B||²` says "the reconstructed S, pushed back through the forward model, should match the measurement."
- The TV term says "S should be piecewise-smooth — small variation almost everywhere, with the freedom to occasionally jump."

That second behavior is what we need for multi-layer stacks. A second-derivative penalty actively *fights* an interface; a TV penalty *allows* one. Crucially, TV doesn't *force* an interface either — on a smooth single-material curve like GaAs, the minimizer of the TV-regularized problem is still smooth, because there's no measurement evidence pushing toward a jump.

This is why TV is the right baseline regardless of material: it degrades gracefully to "smooth" when smoothness is appropriate, and gracefully to "smooth-with-edges" when edges are appropriate. A second-derivative penalty doesn't have that property.

---

## How to choose λ automatically

Hand-tuning λ per material is exactly the trap we are trying to escape. The classical solution is the **Morozov discrepancy principle**:

> Pick the largest λ such that `||GS_λ - B|| ≤ δ`, where δ is the noise level on the measurement.

The intuition is: if you know the measurement has noise of size δ, then any solution that fits the data more tightly than δ is fitting noise, and you should regularize harder. The largest λ that still fits within the noise band is the most regularized solution that is still consistent with the data.

This needs a noise estimate δ. Three ways to get one, in decreasing order of preference:

1. **From the measurement protocol.** The user knows the spectrometer's per-wavelength noise (photon counting, detector, calibration). This is the right answer when it's available.
2. **From repeated measurements.** If the user took multiple ELE scans of the same sample, the spread between scans is δ.
3. **From a residual estimator.** Run an unregularized least-squares fit on a wavelength range where the system is well-conditioned, look at the residual, and use that as a proxy for δ. This is the fallback when the user can give us nothing.

The discrepancy principle then gives us λ without any hand-tuning, which is the property we need for a generic tool.

---

## How to solve it

TV is convex but not smooth (because of the `|·|` inside the sum). The standard solver families are:

- **CVXPY with a conic solver (ECOS, SCS).** Easiest. The `tikhonov_total_variation.py` file in this repo already does something close to this for the two-parameter case. For our problem size (G is ~28 × 500), CVXPY is fast enough that we don't need to think about it.
- **Primal-dual splitting (Chambolle-Pock).** Faster, more control, scales better. Worth doing if CVXPY becomes a bottleneck.
- **ADMM.** A middle ground. Useful later if we want to plug in a learned prior in place of TV (see `ML_DIRECTION.md` — Plug-and-Play ADMM is exactly this).

Start with CVXPY. The runtime is irrelevant at our problem size and the code is short. Optimize only if there's a measured reason.

---

## How to get uncertainty out

The paper's κ-sweep + averaging gives a free uncertainty band, and we want the same property. Two ways to do it for TV:

- **Bootstrap over noise realizations.** Generate ~50 perturbed copies of B by adding noise of size δ, solve the TV problem for each, take the mean and standard deviation across solutions. This is honest: it directly measures "how much would the answer change under plausible noise?"
- **Sweep λ within the discrepancy band.** Instead of picking the single largest λ that satisfies the discrepancy, sweep over a window of λ values that all satisfy it, and average their solutions. This is more directly analogous to the paper's L-curve averaging.

The bootstrap is more rigorous; the λ-sweep is cheaper. Both are acceptable. Pick whichever is easier to implement first; switch later if the uncertainty bands look obviously wrong.

---

## What this gives us

- **Material-agnostic.** Nothing in the formulation references GaAs, diffusion length, or any material parameter. Plug in a different G matrix and a different B and it just works.
- **Self-tuning.** The discrepancy principle picks λ from the data + noise estimate. No hand-tuning.
- **Edge-aware.** TV preserves jumps if they're supported by the data, smooths otherwise.
- **Uncertainty-aware.** Bootstrap or λ-sweep gives a band.
- **Convergent and predictable.** CVXPY (or any of the alternatives) solves a convex problem with a global optimum. No momentum, no LR schedule, no preconditioner brittleness.
- **Cheap.** No training. No GPU. The solver is faster than loading the score model checkpoint.

---

## Where it might fail

This is the section that matters most for deciding whether we ever need the ML direction. TV will probably struggle on:

- **Curves with smooth gradients that look locally like jumps.** TV's bias is toward piecewise-constant solutions, so a slowly-varying region near a strong feature can get "absorbed" into a stair-step. This is the classic TV staircase artifact.
- **Very thin layers.** If a layer is thinner than the depth resolution of the mesh, TV can't represent it as a clean jump and may miss it entirely.
- **Strongly self-absorbing materials at long depths.** The G matrix loses sensitivity to deep z, and no regularizer can recover what the data doesn't see. This is an information-theoretic limit, not a TV problem, but TV will hit it earlier than a smarter prior would.

The right way to find out which of these matter in practice is to test TV+discrepancy on:

1. The GaAs synthetic curves we already have (sanity check — should match the paper).
2. Hand-constructed two-layer stacks with a known interface (does TV find the interface?).
3. Real published SELE measurements from other materials, if we can get them.

If TV passes (1) and (2), we are probably done and the ML direction is unnecessary. If TV passes (1) but fails (2), the ML direction becomes worth investing in. If TV fails (1), something is wrong with the implementation, not with TV as a method.

---

## What this *isn't*

This is not a step-by-step plan. It's a direction. The actual implementation will involve choices about mesh, normalization, and how to wire the noise model into the existing pipeline, and those choices should be made when implementing, not now. The point of writing this down is so that whoever (you, me, future Claude) starts implementing it has the *why* in front of them, and doesn't accidentally drift back into hand-tuning a learned model when the simpler thing would have worked.
