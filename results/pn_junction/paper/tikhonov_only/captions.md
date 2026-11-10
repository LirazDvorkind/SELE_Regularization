# Draft captions: Tikhonov on a synthetic p-n junction

Figures made by `python -m src.pn_junction.standalones.plot_junction_tikhonov_paper`; numbers in
`summary.txt` next to them.

**Figure 1.** Tikhonov reconstruction of the SELE of a thin-film GaAs p-n junction at 1 % relative
measurement noise, for junction depths x_j of (a) 0.12, (b) 0.3, (c) 0.5 and (d) 0.8 µm. Dashed:
true SELE from the drift-diffusion solver. Solid line and band: median and 10–90 % range of the
L-curve Tikhonov reconstruction over 20 independent noise realisations. Shaded: depletion region
(field above 10 % of its peak). Hatched: depths beyond which the data no longer localise features.

**Figure 2.** (a) Rms relative change in the ELE when the junction dip is bridged by a straight
line, against junction depth; dotted lines mark the per-wavelength noise levels. (b) Recovered dip
depth at the centre of the depletion region, 1 − S_rec/S_ref, where 1 means the reconstruction falls
to zero and values above 1 mean it overshoots to negative SELE. Filled markers: junction. Hollow
markers: the same metric for a plain p-type wafer with no junction, read at the same depths.
Median and 10–90 % range over 20 noise realisations.

**Figure S1.** As Figure 1, for every junction depth (rows) and noise level (columns).

**Figure S2.** Data space at 1 % noise: (top) noise-free ELE, one noisy realisation and the
Tikhonov fit; (bottom) residuals normalised by the noise standard deviation, with ±1σ shaded.
