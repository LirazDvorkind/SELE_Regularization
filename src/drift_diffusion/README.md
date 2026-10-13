# Drift-diffusion solver

A steady-state 1-D device simulator for a GaAs p-n homojunction: Poisson's equation coupled
to the electron and hole continuity equations, with SRH and radiative recombination and an
arbitrary optical generation profile G(x). It returns the potential, carrier densities,
quasi-Fermi levels, recombination rates and current densities at a given applied voltage,
and plots band diagrams.

It exists because the closed-form SELE model in `src/forward_model/` assumes a field-free
wafer. In a junction the built-in field is what decides whether a carrier is collected or
luminesces (see `plans/pn-junction-sele-primer.md`), and that needs the coupled equations.
SELE itself is not computed here yet.

> Original prompt:
  ```
  We would like to write a Python numerical simulator that solves the continuity and Poisson equations for electron and holes in a semiconductor.
  The device a GaAs p-n junction with thickness L and junction position xj. The dopant concentrations is Na and Nd. Assume that all dopants are ionized.
  At the simulation boundaries x=0 and x=L there will be ohmic contacts.
  Calculate the carrier distribution for an arbitrary optical generation profile G(x), current density, and an applied voltage Va.
  Consider SRH and radiative recombination. Assign standard parameters for the recombination velocity.
  Plot the energy band diagrams for a given input optical generation and material properties.
  ```

## Model

- Device (`device.py`): abrupt junction, n side on `0 <= x < xj`, p side out to `L`; light
  enters at x = 0. Full ionisation, Boltzmann statistics. Both ends are **ideal ohmic
  contacts**: neutral, in equilibrium with their own Fermi level, infinite recombination
  velocity.
- Material (`material.py`): standard room-temperature GaAs, doping-dependent mobilities.
  Every parameter is a dataclass field, so `dataclasses.replace` gives a variant.
- Generation (`generation.py`): any callable of depth. Helpers for uniform, tabulated, and
  Beer-Lambert illumination from the project's optical constants (`src/optical_constants.py`,
  with the same attenuation-vs-generation split as the rest of the project).

Conventions: Va is applied to the p contact at x = L with the n contact grounded, so Va > 0 is
forward bias. Energies are in eV relative to the grounded contact's Fermi level.
`DeviceState.current_density` is the terminal current into the p contact (forward dark
current positive, photocurrent negative); `jn`, `jp`, `j_total` are profiles along +x.

## Numerics (`solver.py`)

Box-integration finite volumes with Scharfetter-Gummel fluxes on a graded mesh, finest at the
contacts and the junction. The unknowns are the potentials `(psi, phi_n, phi_p)` in units of
Vt rather than n and p: GaAs minority densities span ~25 decades, which would wreck the
conditioning of a Newton system in densities. Fully coupled damped Newton with an analytic
sparse Jacobian; equilibrium is solved first, and bias and generation are reached by
continuation from it.

Each edge current is a small difference of large drift and diffusion terms, so J(x) carries a
roundoff floor wherever majority densities are high. The terminal current is therefore read
at the edge where that floor is lowest, inside the depletion region.

## Run

```bash
python -m src.drift_diffusion.standalones.check_drift_diffusion   # physics + numerics checks
python -m src.drift_diffusion.standalones.plot_band_diagram       # figures -> results/drift_diffusion/
```

```python
from src.drift_diffusion import generation, solver, plotting
from src.drift_diffusion.device import PNJunction

device = PNJunction(length_cm=3e-4, junction_cm=0.5e-4, donor_cm3=1e18, acceptor_cm3=1e17)
state = solver.solve(device, applied_voltage=0.5,
                     generation=generation.beer_lambert(532.0, photon_flux_cm2_s=1e17))
plotting.plot_band_diagram(state)
```

The checks cover: built-in voltage and peak field against the depletion approximation, the
Jacobian against finite differences, constancy of J(x), the dark ideality factor, collection
from the depletion region, exact global particle balance, and mesh convergence. Re-run them
after any change to the solver.

With ideal ohmic contacts at the illuminated surface, holes generated in the emitter are
lost to the front contact, so short-wavelength collection is poor by construction. That is a
property of the boundary condition, not a numerical artefact.
