# The forward model vs. the Zohara manuscript

How this repo's forward model relates to the model derived in
`Papers/Zohara-Full_document_rev4-GS.docx`: where they are the same equation, where the repo
deliberately differs, and what each side assumes.

The short version: the two are the same model. `MATLAB SELE Simulation/calc_Sp2.m` and
`src/forward_model/` implement the manuscript's closed form term for term, and the repo's
closed-form ELE reproduces the manuscript's Eq (16) to discretisation error. The differences
on the repo's side are deliberate choices, not errors. On the manuscript's side there are two
typos in the derivation and two modelling points worth raising with its authors.

---

## 1. The equations match term for term

The manuscript derives a closed-form PL signal by solving the continuity equation for a
semi-infinite wafer with a front-surface boundary condition, following Duggan and Scott.
Every term is in `calc_Sp2.m`:

| Manuscript | `calc_Sp2.m` | Status |
|---|---|---|
| Eq (2)/(3) `B'(E)` | `B=1/(pi^2*hbar^3*c0^2*ni^2).*alpha_b.*n.^2.*E.^2.*exp(-E/(Kb*TK))` | identical (Eq (3)'s low-injection form) |
| Eq (6) `A₀=(D−SLₙ)/(D+SLₙ)` | `A0=(D-S*Ln)/(S*Ln+D)` | identical |
| Eq (11) `Sp = K₁(2e^{−αx} + A₁e^{−x/Lₙ})` | `Sp=K_matrix.*(A1.*exp(-alpha.*x)+A2.*exp(-x/Ln))` | identical |
| Eq (12) `A₁=(A₀−1)−Lₙα(A₀+1)` | `A2=A0-1-Ln*alpha_matrix.*(A0+1)` | identical |
| Eq (13) `K₁ = T·B'·p₀·τ_eff(1−cosθc)/4(1−(αLₙ)²)` | `K2=T.*B.*p0*tau_eff.*(1-cos(theta_c))./4; K1=K2./(1-(alpha*Ln).^2)` | identical |
| Eq (17) `α=4πk/λ` | `alpha=4*pi.*k_interp./(wavelength*1e-7)` | identical (× `alpha_scale`, §2) |

**Naming collision, worth pinning down once:** the manuscript's **A₁** (coefficient of the
diffusion term) is the code's **`A2`/`a2`**. The code's `A1` is the literal constant `2` —
the manuscript's leading `2` in Eq (11). Anyone cross-reading the two will map these wrong
at least once.

That `2` is not an approximation. It comes from Eq (10)'s `1/α⁻ + 1/α⁺ = 2Lₙ/(1−(αLₙ)²)`:
the generation term collects one contribution from each side of the injection point.

Manuscript Eqs (7)–(8), the exact escape probability, have no code counterpart. Both the
manuscript (from Eq (10) on) and the code use its first-order form, `P ≈ (T/2)(1−cosθc)e^{−αx}`,
which is where the `(1−cosθc)` factor in `K2` comes from.

### Reading the first implementation

The first MATLAB commit is the cleanest cross-reference to the manuscript: it predates
`alpha_scale` and the band integral, so it maps to the equations with nothing in between.
Note the quoting — the directory name contains a space.

```bash
# the commit that added the MATLAB
git log --diff-filter=A --format="%h %ad %s" --date=short -- "MATLAB SELE Simulation/calc_Sp2.m"

# every commit that has touched it since
git log --oneline --follow -- "MATLAB SELE Simulation/calc_Sp2.m"

# the file as first written
git show 089e0b2:"MATLAB SELE Simulation/calc_Sp2.m"
git show 089e0b2:"MATLAB SELE Simulation/create_training_set.m"
```

Two things there differ from the current code by design: the first version learned an
absolute absorption at a single emission wavelength instead of today's global `alpha_scale`,
and `create_training_set.m` stored `Sp` at that one wavelength rather than integrating the
emission axis. Both are the §2 differences seen from the other end.

### Eq (16) is the repo's ELE integral

Manuscript Eq (16) integrates `Sp` against a Beer–Lambert generation profile over `[0,∞)`:

```
Φ_PL(λ_ex,E) = α_ex·K₁·( 2/(α_em+α_ex) + A₁/(1/Lₙ+α_ex) )
```

Summed over the emission band with the repo's own trapezoid weights and multiplied by
`α_b/α` at the excitation wavelength, it matches `analytic_ele.simulate_ele` to the
discretisation error of that module's fine mesh. Without the prefactor the two differ by
exactly `α_b/α`. So `analytic_ele.py` **is** Eq (16), with one deliberate addition covered
in §3c.

Finite-mesh discretisation is a separate matter and is **not** always small: on the coarse
solver mesh the gap to the continuum is negligible for many parameter draws and large for
others. Don't treat the solver-mesh ELE as the continuum. Both sides use the same wafer depth
(`W` in the MATLAB scripts, `solver_mesh_edges` in Python).

---

## 2. Where the repo deliberately differs

### The observable: one emission wavelength vs. the whole band

The manuscript's observable is `PL(λ_em, λ_ex)` at a single `λ_em`, with `K₁(λ_em)` and
`A₁(λ_em)` both carrying an emission index. The repo integrates the emission axis out:

```matlab
SELE = trapz(E_sorted, Sp_2d(sort_idx,:), 1);
```

This is the right choice here — SELE *is* the energy-integrated quantity, it is what the 2024
paper extracts and what the solver reconstructs. The consequence to keep in mind is that the
repo and the manuscript **no longer share an observable**, so numbers are not directly
comparable: the manuscript keeps an axis the repo collapses.

### Emission-band α: measured vs. fitted

The manuscript treats `α` from Eq (17) as a known input. The repo declines to trust
ellipsometry over the emission band and learns a global multiplier instead (`alpha_scale`,
bounds in `PARAMETER_SPEC` in `src/forward_model/parameters.py`).

The rationale is stated in-code at `calc_Sp2.m` — *"emission-band absorption is unknown; learn
it as a free scale"* — and it is a deliberate position, not an oversight.

### No doping dependence in α

`α` is built from the measured with-Drude `k`; `α_b` from `k_bulk`. Neither depends on `p0`.
FCA attenuates the beam but frees no carrier, which is exactly what that split encodes.

The assumption this carries: `k` comes from the ellipsometry model, whose Drude oscillator
was set to one fixed doping (SI Table S2.1), while `p0` is a free parameter. The thesis notes
that doping barely changes `k` across the excitation band and matters mainly near the band
edge — which is the emission band, where `alpha_scale` already acts. `alpha_scale` absorbs
the mismatch.

That is adequate because the two are degenerate in the observable: refitting `alpha_scale`
alone reproduces the ELE of a model whose FCA scales with `p0`, to a small fraction of the
change that doping dependence itself causes. The data cannot separate the two, so routing
`p0` into `α` would add a parameter without adding information.

`p0` therefore reaches the curve through `ni`, `tau_eff` and `K1` — setting brightness, and,
through `Lₙ = √(D·τ_eff)`, shape — but not through the optics.

---

## 3. Issues on the manuscript's side

The first two are typos in the derivation; nothing downstream depends on them. The last two
are modelling points.

### 3a. Eq (5) has both signs flipped

It reads `Δn'' = (G₀/D)δ(x−x₀) − Δn/Lₙ²`. That equation has oscillating solutions, and Eq (6)
does not solve it. The steady-state continuity equation is `Δn'' = Δn/Lₙ² − (G₀/D)δ(x−x₀)`;
Eq (6) is correct for that, including the boundary condition and the jump at `x₀`.

### 3b. Eq (8) does not equal Eq (7)

Substituting `t = cosθ` in Eq (7) gives

```
P_esc(x) = (T/2)·[ e^{−αx} − cosθc·e^{−αx/cosθc} − αx·∫_{αx}^{αx/cosθc} e^{−z}/z dz ]
```

Eq (8) as printed drops the `1/2` (possibly what its unexplained prime means — review comment
C37 flags that prime) and integrates over negative limits, where `e^{−z}/z` grows instead of
decaying. The printed form diverges from Eq (7) as `αx` grows; the form above agrees with a
direct numerical evaluation of Eq (7). Since Eq (10) uses the first-order approximation rather
than Eq (8), the error does not propagate.

### 3c. Eq (14) has no `α_b/α` prefactor

The manuscript's `G(λ_ex,x)=α_ex e^{−α_ex x}` is a pure *absorption* profile with `∫G dx = 1`
— generation and attenuation following the same α. The repo's `G` carries `α_b/α`, so a row
sums to `α_b/α` rather than 1.

This is the repo's extension of the published method, not something the papers write down:
SI eq. S4.1 applies `α_b/α` to *re-absorbed luminescence* in the photon-recycling model, and
the main paper and thesis compute incident generation from `α` alone. The repo applies the
same physics — FCA frees no carrier — to incident light too. The prefactor is close to 1 over
the excitation band but varies with `λ_ex`, so it does not cancel into `K₁(λ_em)` and slightly
tilts the fitted shape.

### 3d. `A₁` is fitted globally but Eq (12) makes it λ_em-dependent

Eq (12) makes `A₁` depend on `λ_em` through `α_em`, and the repo honours that (`a2` has shape
`(B, E)`). The manuscript's fitting procedure, and its SRV ratio Eq (21), treat `A₁` as a
**single global parameter** shared across all `λ_em`.

Both can only hold if the `α_em` term in Eq (12) is negligible against the constant one:
`Lₙα_em(A₀+1) ≪ |A₀−1|`. `Lₙα_em ≪ 1` is not enough. At low SRV, `A₀ → 1`, the constant
term vanishes, and `A₁` is dominated by the `α_em` term even when `Lₙα_em` is small. The
approximation holds only at high SRV, where `A₀ → −1`.

None of these four is raised by the manuscript's review comments. C48 and C50 concern the
units of `G` (the missing incident flux), a separate issue from 3c.
