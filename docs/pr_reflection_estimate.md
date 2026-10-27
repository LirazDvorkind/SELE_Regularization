# Sizing photon recycling and back-surface reflection

Script: `src/forward_model/standalones/estimate_pr_reflection.py`.
Question it answers: how much does SELE change when the forward model includes what the repo's
simulator leaves out (angle-resolved escape with back-surface reflection, and photon
recycling)? It is an estimate for sizing the effect, not a replacement for the COMSOL model of
the paper (SI section S4-S5), and it was not fitted to any ground-truth curve.

## What is computed

Four SELE profiles on one finite-volume mesh of the 350 um wafer (1 nm cells at both surfaces,
growing 3% per cell):

| name | escape of emitted photons | recycling |
|------|---------------------------|-----------|
| V0 | the repo simulator's: normal-incidence transmission, one escape cone, `exp(-alpha z)` | no |
| V1 | angle-resolved, direct path to the front only | no |
| V2 | V1 plus reflection at both surfaces and multiple passes (SI S4.9) | no |
| V3 | V2 | yes (SI S4.1-S4.6) |

V1/V0 isolates the simulator's simplified escape, V2/V1 the reflections, V3/V2 the recycling.
A fifth profile, recycling without reflections, separates the two inside V3.

## Equations

**Carriers (low injection, steady state, linear).** With `Dn(z)` the excess minority density,

    D Dn'' - Dn / tau_eff + G0(z) + G_PR(z) = 0

with surface recombination at both faces (`D Dn' = S Dn` at the front, the mirror image at the
back). `tau_eff` is the simulator's (SRH + radiative with `B0` + Auger).

**SELE.** The photon flux leaving the front per carrier generated at depth `x`:

    SELE(x) = integral over z of  p0 * Dn_x(z) * integral over E of  r(E) * P_esc(z, E)  dE dz

where `Dn_x` is the response to one unit of generation at `x`, `r(E)` is the simulator's
spontaneous-emission coefficient (generalised Planck with `alpha_b`), and `P_esc` the
probability that a photon emitted at `z` leaves by the front. Because everything is linear,
all `x` come from one adjoint solve.

**Escape (SI S4.9, with `mu = cos(theta)`, `kappa = alpha / mu`).** Photons are emitted
isotropically; half head toward the front. Summing the reflection orders gives a geometric
series with `q = R_f R_b exp(-2 kappa d)`:

    P_esc,front(z) = 1/2 * integral_0^1 dmu  T(mu) [ exp(-kappa z) + R_b exp(-kappa (2d - z)) ] / (1 - q)

`R` and `T` are the unpolarised Fresnel coefficients at a GaAs/air interface and equal 1 and 0
beyond the critical angle. `alpha` includes the Drude (free-carrier) term, so free-carrier
absorption removes photons without freeing carriers.

**Recycling (SI S4.1-S4.6).** The same unfolding of reflection orders gives the probability
`P_ij(E)` that a photon emitted in cell `j` is absorbed in cell `i`; the fraction `alpha_b/alpha`
of those absorptions frees a carrier. Then

    G_PR,i = sum over j, E of  p0 r(E) Dn_j P_ij(E) alpha_b/alpha * (dz_j / dz_i)

which is linear in `Dn`, so the iteration of SI Fig. S4.2 collapses to one solve of
`(A - K) Dn = G0`. The spectral radius of `A^-1 K` is the loop gain; it must be below 1.

## Assumptions (all flagged, none validated against COMSOL)

1. Low injection and a linear response. The paper's COMSOL perturbation is 10% of the
   background generation (SI S5.1) with Fermi-Dirac statistics; at 1e19 cm^-3 doping this
   should be close to linear, at lower doping it is unverified.
2. No native oxide and no thin-film interference: plain Fresnel at GaAs/air on both faces. The
   paper uses a transfer-matrix method with oxide layers. Not in the repo, not assumed.
3. Operating point: SI Table S5.1 best fit (p0 = 1e19, tau_SRH = 0.4 ns, S = 2e7 cm/s,
   d = 350 um), with `D` from the 1500 cm^2/Vs minority mobility by the Einstein relation. That
   the paper's sweep figures used these values is **unverified**.
4. The simulator's spectrum integrates to a smaller radiative coefficient than the `B0` it uses
   for the radiative lifetime. Recycling is reported both ways: the spectrum as given, and the
   spectrum scaled up to `B0` (an upper bound).
5. Emission is at the cell centre; escape and absorption are integrated exactly over each cell.
   A coarser emission-wavelength grid (every n-th point) is used for speed; its effect is
   checked by comparing two strides.

## Checks

* Photon conservation: for every probed emission energy and depth, absorbed + front escape +
  back escape = 1.
* V0 against `simulate_sele` at the same parameters.
* Loop gain below 1 in every configuration.
