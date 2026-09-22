import numpy as np

from ..base import BaseFitter, _weighted_std

# -------------------------------------------------------------------
# Exponential lifetime
# -------------------------------------------------------------------

class ExponentialFitter(BaseFitter):
    """
    Exponential fit: A * exp(-x / tau).

    Initial estimates
    -----------------
    A0     = y[0]                - scale
    tau0   = weighted_std(x, y)  - count-weighted spread
    """

    nparams: int = 2   # scale, tau

    @property
    def param_names(self) -> list[str]:
        return ["A", "tau"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        A0   = float(self.y[0])
        tau0 = _weighted_std(self.x, self.y)
        return [A0, tau0]

    def _predict(self, A: float, tau: float) -> np.ndarray:
        return A * np.exp(-self.x / tau)

# -------------------------------------------------------------------
# Exponential density
# -------------------------------------------------------------------

class ExponentialDensityFitter(BaseFitter):
    """
    Exponential density fit: N * exp(-x / tau) / tau.

    Initial estimates
    -----------------
    N0     = sum(y * widths)     - scale
    tau0   = weighted_std(x, y)  - count-weighted spread
    """

    nparams: int = 2   # scale, tau

    @property
    def param_names(self) -> list[str]:
        return ["N", "tau"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        N0   = float(np.sum(self.y * self.widths))
        tau0 = _weighted_std(self.x, self.y)
        return [N0, tau0]

    def _predict(self, N: float, tau: float) -> np.ndarray:
        return N * np.exp(-self.x / tau) / tau

# -------------------------------------------------------------------
# Exponential rate
# -------------------------------------------------------------------

class ExponentialRateFitter(BaseFitter):
    """
    Exponential rate fit: A * exp(-lam * x).

    Initial estimates
    -----------------
    A0     = y[0]               - scale
    lam0   = weighted_std(x, y) - count-weighted spread
    """

    nparams: int = 2   # scale, lam

    @property
    def param_names(self) -> list[str]:
        return ["A", "lam"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        A0   = float(self.y[0])
        lam0 = 1.0 / _weighted_std(self.x, self.y)
        return [A0, lam0]

    def _predict(self, A: float, lam: float) -> np.ndarray:
        return A * np.exp(-lam * self.x)

# -------------------------------------------------------------------
# Exponential density rate
# -------------------------------------------------------------------

class ExponentialRateDensityFitter(BaseFitter):
    """
    Exponential rate fit: N * lam * exp(-lam * x).

    Initial estimates
    -----------------
    N0     = sum(y * widths)    - scale
    lam0   = weighted_std(x, y) - count-weighted spread
    """

    nparams: int = 2   # scale, lam

    @property
    def param_names(self) -> list[str]:
        return ["N", "lam"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None)]

    def _initial_params(self) -> list[float]:
        N0   = float(np.sum(self.y * self.widths))
        lam0 = 1.0 / _weighted_std(self.x, self.y)
        return [N0, lam0]

    def _predict(self, N: float, lam: float) -> np.ndarray:
        return N * lam * np.exp(-lam * self.x)

# -------------------------------------------------------------------
# Exponential constant
# -------------------------------------------------------------------

class ExponentialConstantFitter(BaseFitter):
    """
    Exponential constant fit: A * exp(-x / tau) + c.

    Initial estimates
    ------------------
    A0   = y[0]                  - scale
    tau0 = weighted_std(x, y)    - count-weighted spread
    c0   = y[-1]                 - constant
    """

    nparams: int = 3   # A, tau, c

    @property
    def param_names(self) -> list[str]:
        return ["A", "tau", "c"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None), (None, None)]

    def _initial_params(self) -> list[float]:
        A0   = float(self.y[0])
        tau0 = _weighted_std(self.x, self.y)
        c0   = float(self.y[-1])
        return [A0, tau0, c0]

    def _predict(self, A: float, tau: float, c: float) -> np.ndarray:
        return A * np.exp(-self.x / tau) + c

# -------------------------------------------------------------------
# Exponential constant density
# -------------------------------------------------------------------

class ExponentialConstantDensityFitter(BaseFitter):
    """
    Exponential constant fit: N * exp(-x / tau) / tau + c / (xmax - xmin).

    Initial estimates
    ------------------
    N0   = sum(y * widths)       - scale
    tau0 = weighted_std(x, y)    - count-weighted spread
    c0   = y[-1]                 - constant
    """

    nparams: int = 3   # N, tau, c

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None, 
    ) -> None:
        super().__init__(bins, y, yerr, xlim)
        self.xmin = self.centers[0]  - 0.5 * self.widths[0]
        self.xmax = self.centers[-1] + 0.5 * self.widths[-1]

    @property
    def param_names(self) -> list[str]:
        return ["N", "tau", "c"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(0.0, None), (0.0, None), (None, None)]

    def _initial_params(self) -> list[float]:
        N0   = float(np.sum(self.y * self.widths))
        tau0 = _weighted_std(self.x, self.y)
        c0   = float(self.y[-1])
        return [N0, tau0, c0]

    def _predict(self, N: float, tau: float, c: float) -> np.ndarray:
        return N * np.exp(-self.x / tau) / tau + c / (self.xmax - self.xmin)