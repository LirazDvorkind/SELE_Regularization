## The three recombination channels

A photogenerated electron-hole pair has three ways to die. Each has a different _order_ in carrier density, and that difference in order is the whole story — it's what determines which channel wins where.

**Radiative (Rad):** `R_rad ∝ n·p` — an electron and a hole have to physically meet and directly annihilate, emitting a photon. It's a two-body collision: probability of meeting scales with the density of each partner, so it's the _product_ n·p. This is why doping matters so much: in the heavily-doped emitter, n₀ (background electron density) is huge, so even a small injected Δp gives a large R_rad ∝ n₀·Δp — that's literally why the emitter lights up in Fig 3.10a.

**SRH:** recombination through a defect/trap state in the gap — a two-step process (carrier falls into the trap, then the _other_ carrier falls in and annihilates it) rather than a direct collision. Under normal (low-injection) conditions, the majority carrier is already abundant and instantly available to fill the trap, so the process is bottlenecked by the minority carrier finding an empty trap. That makes it effectively **first-order** in the minority/excess carrier density: `R_SRH ∝ Δn/τ_SRH` (or Δp, whichever's the minority). One power of carrier density, not two.

**Auger:** also a two-body annihilation, but the released energy goes to kick a _third_ carrier into a higher energy state instead of making a photon. Three particles have to be in the same place at once, so it's **third-order**: `R_Auger ∝ C·n²·p` (or n·p², depending which carrier absorbs the energy). This is why Auger only matters under very high injection or very heavy doping — at low density, being cubic makes it fall off fastest of all three.

## Why the power law is the whole intuition

Radiative ~ (density)², SRH ~ (density)¹, Auger ~ (density)³. At **low** carrier density, the lowest power wins — SRH dominates, radiative fraction (ILE) is small. As density **rises**, the quadratic and cubic terms catch up and eventually overtake — ILE rises. This single fact explains essentially every feature you saw in Fig 3.10:

- **Depletion region dip (ILE → 0):** the field sweeps carriers out in picoseconds, so steady-state n and p there are both tiny. Tiny density → SRH (linear) dominates over Rad (quadratic) → almost nothing that does recombine does so radiatively.
- **Emitter peak (ILE, and SELE, high):** n₀ is pinned high by doping (2e18), so R_rad ∝ n₀·Δp is already large even for small injection — quadratic-in-one-carrier beats linear SRH.
- **BSF pileup (ILE ~97%):** the barrier reflects minority carriers back, so they accumulate against it — local density rises, R_rad (∝ n·p) grows faster than R_SRH (∝ Δn), so the radiative fraction wins locally even though nothing "generates" more carriers there — it's a bottleneck effect, not a generation effect.
- **Forward bias narrowing the dip (Fig 3.10, MPP/OC vs SC):** forward bias flattens bands, shrinks the field, lets carrier density near the junction rise back up — same mechanism, density recovers, radiative fraction recovers.

So the recurring theme: **ILE isn't a material constant, it's a local density readout.** Anywhere carriers pile up (doping, barriers, blocked transport) you get high ILE; anywhere carriers are swept away or thin (fields, low injection, low doping) you get SRH-dominated, low ILE. That's also the deep reason SELE for the wafer (no junction, no depletion sweep-out) sits at a comparably _high_ magnitude (0.3-0.45%) versus the PN device's peaks (0-0.25%): the wafer has nowhere for carriers to be swept out of the picture the way the junction field does.
