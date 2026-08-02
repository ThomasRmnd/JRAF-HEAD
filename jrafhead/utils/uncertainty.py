import numpy as np

def relative_uncertainty_rescale(hist: np.ndarray, err: np.ndarray, Y: float, y: float):
    """
    Re-scale the relative uncertainty of an histogram of lifetime y to the lifetime Y.
    """
    relerr = np.zeros_like(hist)
    mask = hist > 0
    relerr[mask] = err[mask] / (np.sqrt(Y / y) * hist[mask]) * 100.0
    return relerr