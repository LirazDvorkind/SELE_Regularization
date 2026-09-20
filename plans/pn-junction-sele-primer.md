# SELE of a PN junction — a primer from the thesis

Source: `Papers/Ph_D__Thesis___SELE___Tel_Aviv_University___Yeshurun.pdf`, section 3.4
"SELE of PN junctions", PDF pages 70–80 = printed pages 47–57 (printed page = PDF page − 23).

## 0. Acronyms and terms

| term | meaning |
|---|---|
| **PL** | Photoluminescence — light the sample emits after you shine light on it. What we measure. |
| **ELE** (η_ext) | External Luminescence Efficiency — of the photons absorbed, what fraction come back out as PL. Per wavelength. Our measurement vector. |
| **SELE** (S(z)) | Spatial ELE — the probability that a carrier pair generated at depth z ends up as an emitted PL photon. The unknown we reconstruct. |
| **ILE** | Internal Luminescence Efficiency — of the recombination happening at z, the fraction that is radiative (emits a photon), regardless of whether the photon escapes. Explained in §2. |
| **SCE** | Spatial Collection Efficiency — the probability that a carrier pair generated at depth z is collected as *current* at the contacts. SELE's twin: same equation, with current instead of photons. |
| **EQE** | External Quantum Efficiency — of the *incident* photons, what fraction become collected current. Per wavelength. SCE's measurement, the way ELE is SELE's. |
| **SRH** | Shockley–Read–Hall recombination — non-radiative recombination through defect states in the bandgap. A loss. Characterized by a lifetime τ_SRH. |
| **Auger** | Non-radiative recombination where the energy goes to a third carrier instead of a photon. Dominates at very high carrier density. A loss. |
| **Radiative (Rad)** | Recombination that emits a photon. The only path that can give PL. Rate ∝ B·n·p. |
| **SRV** | Surface Recombination Velocity — how fast carriers die at a surface or interface. Sets the SELE dip at z=0. |
| **PN junction** | The boundary between n-doped and p-doped regions. Has a built-in electric field. |
| **Emitter / Base** | The two sides of the junction. Emitter: thin, heavily doped, on top (n-GaAs here). Base: thick, lightly doped, does most of the absorbing (p-GaAs here). |
| **Depletion region / space-charge region** | The thin zone around the junction where the field lives and there are almost no free carriers. |
| **ARC** | Anti-Reflection Coating — thin dielectric layers (MgF₂, Ta₂O₅) on top so light enters instead of reflecting. Transparent; contribute no SELE. |
| **Window** | A thin, wide-bandgap layer (AlInP) between ARC and emitter. Transparent to the light, but its high bandgap keeps carriers from reaching the top surface — it's a passivation layer. |
| **BSF** | Back Surface Field — a layer at the back (p-AlGaAs) with a higher bandgap than the base. Its band offset acts as a wall that reflects minority carriers away from the back contact, where they'd otherwise die. |
| **Mirror** | Silver at the very back. Reflects unabsorbed light for a second pass, and reflects PL photons emitted toward the back so they can exit the front. |
| **Ohmic contact** | The metal contact. In the simulation it's ideal: infinite recombination, so carrier density is pinned to zero there. |
| **Minority carriers** | Electrons in p-type, holes in n-type. They're the ones that have to diffuse to the junction to be collected. |
| **Diffusion length** | How far a minority carrier wanders before recombining. Compare with layer thickness to know whether it reaches the junction. |
| **Quasi-Fermi level splitting** (ΔE_F) | Under illumination, electrons and holes each have their own Fermi level; the gap between them is the local "voltage." Sets the radiative rate. |
| **SC / MPP / OC** | Operating points: Short Circuit (V=0, all current extracted), Maximum Power Point, Open Circuit (no current extracted, V=Voc). |
| **Voc, Jsc, FF, η** | Open-circuit voltage, short-circuit current density, fill factor, cell efficiency — the standard I-V figures of merit. |
| **TMM** | Transfer Matrix Method — optics calculation for multilayer stacks, including interference. How G(λ,z) is computed here instead of Beer–Lambert. |
| **Photon recycling (PR)** | An emitted photon gets reabsorbed and makes a new carrier pair. Raises carrier density deep in the device. |
| **AM1.5G** | The standard solar spectrum used as illumination. |
| **Light bias / G₀** | Steady background illumination (1 sun) the device sits under while you measure. SELE is defined as a small perturbation on top of it. |

## 1. The device (Table 3.3 + Fig 3.8, printed p48)

Light enters from the top. Depth z = 0 is the top of the emitter:

```
z [µm]      layer                      role
--------    -----------------------    -----------------------------------------
 (top)      ARC + window (AlInP)       transparent; just a boundary in the model
 0–0.12     n-GaAs emitter (2e18)      thin, heavily doped
 0.12       ── PN junction ──          built-in field, ~1.8e5 V/cm at short circuit
 0.12–3.62  p-GaAs base (9e16)         the absorber, 3.5 µm
 3.62–4.02  p-AlGaAs BSF               barrier that repels minority electrons from the back
 4.02       ohmic back contact + silver mirror
```

Compare with the wafer: 350 µm, one doping, no field anywhere. Here there are two internal fields (junction at 0.12 µm, BSF at 3.62 µm), a metal contact that eats carriers, and a mirror that bounces photons back out the front.

Fig 3.8b shows the AM1.5 generation profile G(z): it drops 4 decades over the 4 µm, and beyond ~2.5 µm it oscillates — interference fringes from the mirror. G here comes from TMM, not Beer–Lambert.

## 2. What ILE is

ILE is a *local* ratio: of all the recombination happening at depth z, what fraction is radiative?

```
ILE(z) = R_rad(z) / (R_rad(z) + R_SRH(z) + R_Auger(z))
```

It depends on the material *and* the local carrier density. It says nothing about whether the photon escapes or whether the carrier could have been collected as current instead. SELE is the whole chain: a carrier generated at z must (i) *not* be collected as current, (ii) recombine radiatively — that's ILE — and (iii) its photon must escape out the front without being reabsorbed. Roughly:

```
SELE(z) ≈ [1 − SCE(z)] × ILE(z) × P_escape(z)
```

ILE is one link in the chain, and Fig 3.10 exists to show that link on its own.

## 3. Fig 3.10 (printed p50) — reading the ingredients

Panel (b), ILE, cyan = short circuit: ~83% almost everywhere in the base, but in the inset at z ≈ 0.13–0.2 µm it drops to **zero**. That's the depletion region. Panel (d) inset shows why: the field there is 1.8e5 V/cm. A carrier generated inside the field is swept across in picoseconds — no time to recombine at all. Both R_rad and R_SRH collapse (panel a, cyan dip), and the ratio lands near zero because SRH wins at low density.

Compare MPP (black) and OC (blue) in the same inset: the dip gets narrower and shallower as voltage rises. Forward bias flattens the bands (panel c), the field shrinks (panel d inset), carriers pile up near the junction, recombination resumes. This is why SELE depends on the operating point.

The other feature: the ILE spike to ~97% at z = 3.62 µm. That's the BSF interface. The AlGaAs barrier reflects minority electrons back into the base; they pile up against it, density rises, and since R_rad ∝ n·p while SRH is linear in the minority density, the radiative fraction wins locally.

## 4. Fig 3.11 (printed p52) — the SELE itself, at short circuit

Red curve, left axis. Units: **0–0.25 %**, versus 0.3–0.45 % for the wafer — comparable magnitude, not orders apart. Left to right:

- **z = 0: zero.** Surface recombination, as in the wafer.
- **z ≈ 0.05–0.1 µm: sharp peak to 0.12 %.** Inside the emitter. The emitter is doped 2e18, so R_rad ∝ n₀·Δp is ~3 decades above the base (Fig 3.10a, red curve near z=0), and a carrier generated here is far enough from the surface not to die there but hasn't reached the field yet. The thesis says outright that in GaAs and InP "most of the SELE originates from the emitter."
- **z ≈ 0.15–0.3 µm: zero.** The junction. Everything generated here becomes current (SCE, black, hits 100%). Nothing is left to luminesce.
- **z = 0.3 → 3.6 µm: slow rise from 0 to 0.05 %.** The farther from the junction a carrier is born, the longer its random walk before finding the field, and the larger its (still tiny) chance to recombine radiatively on the way. SCE stays at 99+% across the base — the diffusion length is much longer than 3.5 µm — so you're looking at the ~1% that *doesn't* get collected.
- **z ≈ 3.8 µm: second peak, 0.2 %, inside the BSF layer.** Carriers generated here are on the wrong side of the barrier: the wall that protects the base also stops *them* from reaching the junction, so SCE drops (black falls). They pile up at the interface where ILE is 97%. High density × high radiative fraction × a mirror right behind them = big SELE.
- **z = 4.02: zero.** Ideal ohmic contact — infinite surface recombination.

One-line intuition: **SELE and SCE compete for the same carrier.** Wherever collection is efficient (junction, fields), SELE is zero. Wherever collection is blocked (surface, barrier, far from the junction), SELE gets a chance. At short circuit, collection wins almost everywhere, so peak SELE tops out around 0.2 % — somewhat below the wafer's 0.3–0.45 %, not orders of magnitude below it.

## 5. Fig 3.12–3.13 (printed p54–55) — bias dependence

Panel (a) matters for *our* work: the simulated ELE(λ) runs **300–900 nm**, with a peak at ~450 nm (0.06 %) and a shoulder near the band edge at 870 nm. Our wafer dataset stops at 670 nm.

Panels (c–d): as V goes 0 → Voc (yellow → green), SELE in the base rises ~10% relative and SCE drops ~0.5% absolute. Small. Fig 3.13 shows the mechanism: the junction field drops from 1.8e5 to ~7e4 V/cm, and quasi-Fermi splitting in the base climbs from ~0.9 to ~1.1 eV. Practically: SELE must be measured *around a bias light G₀* (Eq. 3.2), as a perturbation, and the operating point is part of the answer.

## 6. Fig 3.14–3.15 (printed p57) — materials

Log scale. Si is 1e-7 %, GaAs 1e-3 %: the SELE ratio tracks the B_rad ratio (5e-15 vs 1.9e-10 cm³/s). SCE is identical for all three — collection is a transport property, not a radiative one. Fig 3.15b integrates SELE into voltage: for GaAs/InP the entire photovoltage is built in the first ~1 µm; for Si it keeps growing with thickness.

## 7. Parameters at play

| parameter | where it shows in the SELE |
|---|---|
| emitter doping, thickness | height/width of the first peak |
| junction position + field (doping, V) | position/width of the zero |
| base diffusion length vs base thickness | slope of the slow rise; how close SCE sits to 100% |
| B_rad, τ_SRH, C_Auger | overall magnitude (via ILE) |
| SRV at front surface; ohmic back contact | zeros at both ends |
| BSF barrier height / presence | second peak; remove it and the base SELE hums up in the middle and falls to 0 at the contact |
| mirror reflectivity, self-absorption | P_escape — weights deep contributions |
| operating point V, bias light G₀ | modulates everything by ~10% |

## 8. How this connects to our problem

**Same equation, different G, different prior.** Eq. 1.7 is unchanged: `η_ext(λ) = (1/φ_abs) ∫ G(λ,z) S(z) dz`. The resolution-length framework (Backus–Gilbert analysis of `G·diag(dz)^{-1/2}`: keep the singular components above the noise, form `P = V_r V_rᵀ`, and read the blur width `ℓ(z) = dz/P_ii` at each depth) applies verbatim — feed it TMM rows instead of Beer–Lambert rows. Three consequences:

1. **The features sit exactly where the horizon analysis said we can and can't look.** The emitter peak (0.05–0.1 µm) and the junction zero (0.15–0.3 µm) are inside the data-determined zone — but at 1% noise the blur ℓ there is 0.04–0.1 µm, comparable to the features themselves. They'd be *seen* but smeared. The BSF peak at 3.8 µm is far past the 0.7 µm horizon: with 400–670 nm light it is invisible.

2. **The wavelength band is the lever.** The thesis simulates ELE out to 900 nm. Near-gap wavelengths (800–870 nm) penetrate microns, and the silver mirror doubles their path — that is what puts information at 3–4 µm depth. The framework can quantify how much each added wavelength pushes the horizon. Whether the optical constants in `Data/` cover that range is the first thing to check.

3. **The learned prior does not transfer.** The score model / PCA prior learned five-parameter wafer curves: one broad peak at 0.5–5 µm, magnitude 0.3–0.45 %. The PN-junction SELE is a different *shape* at similar overall magnitude (0–0.25 %): two sharp peaks flanking a hard zero at the junction, and it depends on V. A wafer-trained prior would confidently reconstruct the wrong shape. Any prior-based method needs a training set per device class — or the prior-free G-only analysis is the honest tool, and it tells you which features are recoverable before you build anything.

A softer fourth point: the thesis obtains the junction SELE by forward simulation (Eq. 3.2, perturb-and-measure in COMSOL), never by inverting measured ELE. Inversion for a PN junction is still open territory.

## Open follow-up

Run the horizon analysis with the band extended toward 900 nm (if the optical constants cover it) to see how much deeper the data can reach.
