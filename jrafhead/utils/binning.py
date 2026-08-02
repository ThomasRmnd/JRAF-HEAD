import numpy as np


def nmo_analysis_bins(edges: np.ndarray, nbins: np.ndarray) -> np.ndarray:
    """
    Non-uniform binning for the NMO shape analysis.

    The bin structure is defined by segment edges and the number of equal-width
    sub-bins within each segment, for example:

        [0.7, 1.0]   ==>  1 bin
        [1.0, 6.6]   ==> 56 bins
        [6.6, 7.4]   ==>  4 bins
        [7.4, 7.7]   ==>  1 bin
        [7.7, 8.1]   ==>  1 bin
        [8.1, 8.6]   ==>  1 bin
        [8.6, 9.4]   ==>  1 bin
        [9.4, 12.0]  ==>  1 bin

    Parameters
    ----------
    edges : np.ndarray
        Edges of the segments.
    nbins : np.ndarray
        Sub-bins within each segment.

    Returns
    -------
    np.ndarray
        Array of bin edges of length sum(nbins) + 1.
    """
    segments = []
    for k in range(len(nbins)):
        segment = np.linspace(edges[k], edges[k + 1], nbins[k] + 1)
        if k > 0:
            segment = segment[1:]
        segments.append(segment)

    return np.concatenate(segments)


def uniform_bins(low: float, high: float, n: int) -> np.ndarray:
    """
    Convenience wrapper for uniform binning.

    Parameters
    ----------
    low : float
        Lower edge of the first bin.
    high : float
        Upper edge of the last bin.
    n : int
        Number of bins.

    Returns
    -------
    np.ndarray
        Array of n+1 bin edges uniformly spaced between low and high.
    """
    return np.linspace(low, high, n + 1)