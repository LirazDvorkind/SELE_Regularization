# Forward-model test set

SELE curves made by the parametric method's own simulator (`src/forward_model/`), for checking
whether the inverse method can reconstruct a curve that its model family can represent
exactly. The paper-figure set in `Data/test_set/` cannot answer that: its COMSOL truths carry
photon recycling and multi-pass escape that the simulator lacks, so error there mixes method
error with model mismatch.

Built by `python -m src.test_set.build_forward_model_set`, which defines the curves; scored by
`python -m src.regularization.parametric_model.standalones.score_test_set --set forward_model`.

## What is in it

- **`fm_srv_*`, `fm_tau_*`**: one-at-a-time sweeps through the centre of the parameter box, at
  the paper's SRV values and at those of its lifetimes that lie inside the box, so they read
  side by side with the paper-figure sweeps.
- **`fm_rand_*`**: draws over the whole box from a seed the training set never used.
  `index.csv` records each curve's distance to the nearest training point, which the builder
  prints next to the training set's own nearest-neighbour spacing. They are comparable: these
  curves are as unseen as any point in the box can be.

## Conventions

- Depth in cm, SELE as a fraction, as everywhere in the project.
- `z_cm.csv` is shared: every curve is stored on the 500-element solver mesh the reconstruction
  reports on, so the comparison needs no interpolation.
- `ele.csv` is the closed-form ELE, the same route that generated the training set. The
  observation carries no discretisation error, so this is deliberately the ideal case.
- No noise.
- `index.csv` holds the true parameters. They are for scoring parameter recovery after
  inference, and nothing that runs inference may read them. Read the index through
  `src/test_set/loader.py:load_forward_model_set`, which returns them separately from `ele`.
