import numpy as np
from scipy.interpolate import PchipInterpolator


def expand_sele(
        S,
        points_amount,
        front_weight=3.0,
        z_original=None,
):
    """
    Expand a SELE profile S to a denser grid using shape-preserving cubic interpolation
    and a depth grid that is denser near the front surface (z=0).

    Parameters
    ----------
    S : array_like, shape (N,)
        Original SELE values.
    points_amount : int
        Desired number of output points (M >= len(S)).
    front_weight : float, optional
        >1 gives more points near z=0. 1 gives uniform spacing.
        Typical: 2–4. Larger => stronger concentration near the surface.
    z_original : array_like, shape (N,), optional
        Physical depths corresponding to S. If None, assumes uniform spacing
        between 0 and 1.

    Returns
    -------
    z_new : ndarray, shape (points_amount,)
        New depth coordinates.
    S_new : ndarray, shape (points_amount,)
        Interpolated SELE values on z_new.
    """

    S = np.asarray(S)
    N = S.size
    if N < 2:
        raise ValueError("Need at least 2 points in S to interpolate.")

    # Original depth grid
    if z_original is None:
        z_orig = np.linspace(0.0, 1.0, N)
    else:
        z_orig = np.asarray(z_original)
        if z_orig.shape != S.shape:
            raise ValueError("z_original must have the same shape as S.")
        # Enforce increasing order if needed
        sort_idx = np.argsort(z_orig)
        z_orig = z_orig[sort_idx]
        S = S[sort_idx]

    # New depth grid: front-weighted towards z=0
    # t is uniform; z_new = t ** front_weight compresses spacing near 0
    t = np.linspace(0.0, 1.0, points_amount)
    if front_weight <= 0:
        raise ValueError("front_weight must be > 0.")
    z_min, z_max = z_orig[0], z_orig[-1]
    z_new = z_min + (z_max - z_min) * (t ** front_weight)

    # Shape-preserving cubic interpolation
    interp = PchipInterpolator(z_orig, S)
    S_new = interp(z_new)

    return z_new, S_new


def match_length_interp(from_vect, new_length):
    x_from = np.linspace(0, 1, len(from_vect))
    x_to = np.linspace(0, 1, new_length)
    # Interpolate current estimate onto GT grid
    interp = np.interp(x_to, x_from, from_vect)
    return interp
