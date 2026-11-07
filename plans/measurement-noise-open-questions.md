# Measurement noise: what we know and what to ask for

The depth to which SELE can be extracted, and how sharp a feature can be at each depth, is
set by the *random, wavelength-to-wavelength* error in the ELE, not by its overall
calibration error. A common scale error rescales the whole SELE and leaves its shape and the
depth limit unchanged. The thesis does not report the error that matters.

## What the thesis gives (Yeshurun, Appendix D.5)

- **Fig. D.4, about 5% "ELE accuracy".** This is a uniform offset between power meters: the
  calibrated photodiode multiplied by 0.95 lies on the other meters across 400–700 nm. It is
  a scale factor, so it does not enter the resolution analysis. By eye, the scaled curves
  differ by about 1–2% between wavelengths (unverified).
- **Fig. D.3, PL repeatability.** There are 10 repeats per integration time at one excitation
  wavelength, and they overlap within marker size, so the scatter is at most about 1–2%
  (read by eye). That is detector repeatability only.
- **Projected gradient descent (§2.4.4)** perturbs the ELE by ±10% to stand in for
  measurement error. That is an assumed number, not a measured one.

## Open — to fill in (ask a teammate who ran the setup)

- [ ] Repeat ELE spectra of one sample: the full wavelength sweep, measured several times,
      possibly on different days. The per-wavelength standard deviation of these is the noise
      the resolution analysis needs.
- [ ] Whether the error is correlated between neighbouring wavelengths (lamp drift during a
      sweep, monochromator bandwidth). Correlated error removes fewer independent pieces of
      information than white noise does.
- [ ] How large the error in the optical model is (ellipsometry k near the gap, absorbed
      fraction A(λ)). It enters G, not the ELE, and the thesis notes that the fitted
      parameters are very sensitive to it near the band edge.
- [ ] The raw data behind Figs. D.3–D.4, if they still exist.

Until these are in hand, depth limits are quoted as a function of the assumed noise level.
