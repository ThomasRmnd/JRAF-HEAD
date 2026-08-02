from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def _bw_factor(data: np.ndarray, bandwidth: float) -> float:
    """Convert an absolute bandwidth [same units as data] to gaussian_kde's
    unitless bw_method factor (which multiplies the sample's own std)."""
    return bandwidth / np.std(data, ddof=1)


def _scaled_kde(
    data: np.ndarray,
    centers: np.ndarray,
    bin_width: float,
    bandwidth: float | None,
    bw_method,
) -> np.ndarray:
    """Evaluate a KDE of `data` at `centers`, scaled to expected counts/bin."""
    bw = _bw_factor(data, bandwidth) if bandwidth is not None else bw_method
    kde = gaussian_kde(data, bw_method=bw)
    density = kde(centers)
    return density * len(data) * bin_width


def smooth_kde_difference(
    sig_data: np.ndarray,
    bkg_data: np.ndarray,
    bin_centers: np.ndarray,
    bin_width: float,
    bandwidth: float | None = None,
    bw_method=None,
    n_bootstrap: int = 200,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth the sig-minus-bkg difference by KDE-ing each sample separately.

    Parameters
    ----------
    sig_data, bkg_data : np.ndarray
        Raw per-event values (e.g. prompt energies in MeV) for each sample.
    bin_centers : np.ndarray
        Grid to evaluate the smoothed curves on (e.g. your existing ecenters).
    bin_width : float
        Width of one analysis bin, used to convert normalized KDE density
        back into "expected counts per bin" so the output matches the scale
        of a raw np.histogram difference. Assumes uniform bins.
    bandwidth : float or None
        Absolute kernel width in the SAME units as sig_data/bkg_data (e.g.
        MeV). If given, the same physical bandwidth is used for both sig
        and bkg (converted per-sample via each one's own std), avoiding a
        smoothing-scale mismatch between the two terms. Takes precedence
        over bw_method if both are given.
    bw_method : optional
        Passed through to scipy.stats.gaussian_kde if `bandwidth` is None.
        None -> gaussian_kde's default (Scott's rule), applied independently
        per sample.
    n_bootstrap : int
        Number of bootstrap resamples used to estimate the uncertainty on
        the smoothed difference at each grid point.
    random_state : int or None
        Seed for reproducible bootstrapping.

    Returns
    -------
    smooth_diff : np.ndarray, shape (len(bin_centers),)
        KDE(sig) - KDE(bkg), scaled to counts/bin. Can be negative.
    smooth_diff_err : np.ndarray, shape (len(bin_centers),)
        Bootstrap standard deviation of smooth_diff at each grid point.
    """
    sig_data = np.asarray(sig_data)
    bkg_data = np.asarray(bkg_data)
    rng = np.random.default_rng(random_state)

    smooth_sig = _scaled_kde(sig_data, bin_centers, bin_width, bandwidth, bw_method)
    smooth_bkg = _scaled_kde(bkg_data, bin_centers, bin_width, bandwidth, bw_method)
    smooth_diff = smooth_sig - smooth_bkg

    boot = np.empty((n_bootstrap, len(bin_centers)))
    for i in range(n_bootstrap):
        sig_rs = rng.choice(sig_data, size=len(sig_data), replace=True)
        bkg_rs = rng.choice(bkg_data, size=len(bkg_data), replace=True)
        s = _scaled_kde(sig_rs, bin_centers, bin_width, bandwidth, bw_method)
        b = _scaled_kde(bkg_rs, bin_centers, bin_width, bandwidth, bw_method)
        boot[i] = s - b

    smooth_diff_err = boot.std(axis=0, ddof=1)
    return smooth_diff, smooth_diff_err