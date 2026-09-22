import numpy as np

from ..base import BaseFitter, _weighted_std

# -------------------------------------------------------------------
# Gaussian
# -------------------------------------------------------------------

class GaussianFitter(BaseFitter):
    """
    Gaussian fit: A * exp(-(x - mu)^2 / (2 * sigma^2)).

    Initial estimates
    -----------------
    A0     = max(y)              - scale
    mu0    = x[argmax(y)]        - position of peak bin
    sigma0 = weighted_std(x, y)  - count-weighted spread
    """

    nparams: int = 3   # scale, mu, sigma

    @property
    def param_names(self) -> list[str]:
        return ["A", "mu", "sigma"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (None, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        A0     = float(np.max(self.y))
        mu0    = float(self.x[np.argmax(self.y)])
        sigma0 = _weighted_std(self.x, self.y)
        return [A0, mu0, sigma0]

    def _predict(self, A: float, mu: float, sigma: float) -> np.ndarray:
        return A * np.exp(-((self.x - mu) ** 2) / (2.0 * sigma ** 2))

# -------------------------------------------------------------------
# Gaussian density
# -------------------------------------------------------------------

class GaussianDensityFitter(BaseFitter):
    """
    Gaussian density fit: N * exp(-(x - mu)^2 / (2 * sigma^2)) / (sigma * sqrt(2 * pi)).

    Initial estimates
    -----------------
    N0     = sum(y * widths)     - scale
    mu0    = x[argmax(y)]        - position of peak bin
    sigma0 = weighted_std(x, y)  - count-weighted spread
    """

    nparams: int = 3   # scale, mu, sigma

    @property
    def shape_param_names(self) -> list[str]:
        return ["N", "mu", "sigma"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (None, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        N0     = float(np.sum(self.y * self.widths))
        mu0    = float(self.x[np.argmax(self.y)])
        sigma0 = _weighted_std(self.x, self.y)
        return [N0, mu0, sigma0]

    def _predict(self, N: float, mu: float, sigma: float) -> np.ndarray:
        return N * np.exp(-((self.x - mu) ** 2) / (2.0 * sigma ** 2)) / (sigma * np.sqrt(2.0 * np.pi))
