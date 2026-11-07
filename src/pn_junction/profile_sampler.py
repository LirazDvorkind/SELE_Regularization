"""Draw many SELE profiles that all reproduce a measured ELE, without assuming a device model.

What the method assumes, and nothing more:

- ELE = G @ SELE with a known G, and the noise on each wavelength is Gaussian with a known size.
- SELE is never negative.
- Neighbouring depth slices usually have similar SELE, but a sudden jump or drop may happen
  anywhere, any number of times. This is a Student-t law on the step between neighbours: most
  steps are small, a few can be large. How small "usually" is (the step scale) and how often
  large steps happen (the tail weight) are both learned from the data, not set by hand.

It knows nothing about junctions, wafers or exponentials, and it does not test for a feature: a
dip appears only if the data ask for it.

Sampling is Gibbs. The Student-t is written as a Gaussian whose width is itself random, one width
per step; given those widths the profile is a Gaussian restricted to SELE >= 0, which is drawn
exactly by bouncing Hamiltonian motion off the zero wall (Pakman and Paninski 2014). The step
scale and tail weight then move on a grid, scored on the steps alone, and the widths are drawn
given them.

Tails heavier than Cauchy are not learned: that is the regime where
"usually small" collapses to "exactly zero", and the step scale then runs away to nothing.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from scipy.special import gammaln

# From Cauchy (1) to near-Gaussian (1000). Below 1 the law piles up on zero steps and the step
# scale runs off to zero: every flat stretch becomes "exactly flat".
TAIL_WEIGHT_GRID = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1000.0])
# In units where a flat profile reproducing the mean ELE is 1; wide enough to never bind.
STEP_SCALE_GRID = np.logspace(-7, 1, 161)
MAX_STEP_STIFFNESS = 1e8


class Samples(NamedTuple):
    sele: NDArray[np.float64]         # (draws, depth slices)
    step_scale: NDArray[np.float64]   # (draws,), in SELE units per slice
    tail_weight: NDArray[np.float64]  # (draws,), Student-t degrees of freedom: small = jumpier
    bounces: NDArray[np.int64]        # (draws,), wall hits per profile draw
    repairs: NDArray[np.int64]        # (draws,), times Q needed a nudge to factorise, since the last kept draw


def sample(G: NDArray[np.float64], ele: NDArray[np.float64], noise_sd: NDArray[np.float64],
           n_draws: int, n_burn: int, rng: np.random.Generator, thin: int = 1,
           fixed_step_scale: Optional[float] = None,
           fixed_tail_weight: Optional[float] = None) -> Samples:
    """One Gibbs chain. ``fixed_*`` pins a setting instead of learning it (for checking that
    the answer does not hinge on it); the step scale is in SELE units per slice."""
    n = G.shape[1]
    # Work in units where a flat profile reproducing the mean ELE is 1, so everything is O(1).
    unit = float(np.mean(ele / G.sum(axis=1)))
    A = G * unit / noise_sd[:, None]
    b = ele / noise_sd
    AtA, Atb = A.T @ A, A.T @ b
    diag = np.arange(n)

    x = np.full(n, 1.0)
    out = Samples(np.empty((n_draws, n)), np.empty(n_draws), np.empty(n_draws),
                  np.empty(n_draws, dtype=np.int64), np.zeros(n_draws, dtype=np.int64))
    repairs = 0
    scales2 = (np.array([fixed_step_scale / unit]) if fixed_step_scale else STEP_SCALE_GRID) ** 2
    tails = np.array([fixed_tail_weight]) if fixed_tail_weight else TAIL_WEIGHT_GRID
    grid = (scales2.size // 2, 0)
    step2, width2 = 0.01, np.ones(n - 1)
    for it in range(n_burn + n_draws * thin):
        # The step prior is tridiagonal in the profile. Capping its stiffness keeps Q
        # factorisable; a step held to 1e-4 of the mean level is as good as held flat.
        w = np.minimum(1.0 / (step2 * width2), MAX_STEP_STIFFNESS)
        Q = AtA.copy()
        Q[diag[:-1], diag[:-1]] += w
        Q[diag[1:], diag[1:]] += w
        Q[diag[:-1], diag[1:]] -= w
        Q[diag[1:], diag[:-1]] -= w
        x, hits, repaired = _truncated_gaussian(Q, Atb, x, rng)
        repairs += repaired
        d2 = np.diff(x) ** 2
        # Step scale and tail weight from the steps alone, with the per-step widths integrated
        # out; drawing them given the widths instead locks them in place.
        grid = _update_settings(rng, d2, scales2, tails, grid)
        step2, nu = float(scales2[grid[0]]), float(tails[grid[1]])
        width2 = _inv_gamma(rng, 0.5 * (nu + 1.0), 0.5 * (nu + d2 / step2))
        if it >= n_burn and (it - n_burn) % thin == 0:
            k = (it - n_burn) // thin
            out.sele[k], out.step_scale[k] = x * unit, np.sqrt(step2) * unit
            out.tail_weight[k], out.bounces[k], out.repairs[k] = nu, hits, repairs
            repairs = 0
    return out


def _inv_gamma(rng, shape, scale):
    return scale / rng.gamma(shape, 1.0, size=np.shape(scale) or None)


def _log_steps(d2, s2, nu):
    """Log-likelihood of the steps under a Student-t of scale^2 ``s2`` and tail weight ``nu``."""
    return d2.size * (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(np.pi * nu * s2))         - 0.5 * (nu + 1.0) * np.sum(np.log1p(d2 / (nu * s2)))


def _update_settings(rng, d2, scales2, tails, grid, n_moves=12):
    """Metropolis moves of (step scale, tail weight) on their grids, flat over each grid.

    Same target as scoring the whole grid, at a fraction of the cost: the posterior is narrow,
    so a few neighbour moves per sweep cover it.
    """
    i, j = grid
    here = _log_steps(d2, scales2[i], tails[j])
    for _ in range(n_moves):
        i2 = i + int(rng.integers(-3, 4))
        j2 = j + int(rng.integers(-1, 2))
        if not (0 <= i2 < scales2.size and 0 <= j2 < tails.size):
            continue
        there = _log_steps(d2, scales2[i2], tails[j2])
        if np.log(rng.random()) < there - here:
            i, j, here = i2, j2, there
    return i, j


def _truncated_gaussian(Q, h, x0, rng, travel=0.5 * np.pi):
    """A draw from N(Q^-1 h, Q^-1) restricted to x >= 0, starting from a feasible ``x0``.

    The motion x(t) = m + (x0 - m) cos t + v sin t keeps the Gaussian invariant; at the wall the
    velocity is mirrored in the metric of the covariance, which keeps it invariant there too.
    """
    try:
        chol, repaired = scipy.linalg.cho_factor(Q, lower=True), False
    except np.linalg.LinAlgError:
        # Rare: roundoff in a very stiff Q. Lift it by just enough to factorise, and count it.
        lift = 2.0 * max(-np.linalg.eigvalsh(Q)[0], 0.0) + 1e-12 * np.max(np.diag(Q))
        chol, repaired = scipy.linalg.cho_factor(Q + lift * np.eye(Q.shape[0]), lower=True), True
    m = scipy.linalg.cho_solve(chol, h)
    v = scipy.linalg.solve_triangular(chol[0], rng.standard_normal(Q.shape[0]), lower=True, trans="T")
    b = np.maximum(x0, 0.0) - m
    left, hits, last = travel, 0, -1
    while True:
        u = np.hypot(v, b)
        phase = np.arctan2(v, b)
        reach = u > np.abs(m)
        t = np.full_like(u, np.inf)
        t[reach] = np.mod(phase[reach] + np.arccos(-m[reach] / u[reach]), 2.0 * np.pi)
        if last >= 0:
            t[last] = np.inf if t[last] < 1e-9 or t[last] > 2.0 * np.pi - 1e-9 else t[last]
        j = int(np.argmin(t))
        if t[j] >= left:
            return m + b * np.cos(left) + v * np.sin(left), hits, repaired
        c, s = np.cos(t[j]), np.sin(t[j])
        pos = m + b * c + v * s
        vel = v * c - b * s
        pos[j] = 0.0
        e = np.zeros_like(m)
        e[j] = 1.0
        col = scipy.linalg.cho_solve(chol, e)
        vel = vel - 2.0 * vel[j] / col[j] * col
        b, v, left, last, hits = pos - m, vel, left - t[j], j, hits + 1
