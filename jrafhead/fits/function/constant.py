import numpy as np

from ..base import BaseFitter

# -------------------------------------------------------------------
# Constant
# -------------------------------------------------------------------

class ConstantFitter(BaseFitter):
    """
    Constant fit: c.

    Initial estimate
    ----------------
    c0 = y[0]
    """

    nparams: int = 1   # c

    @property
    def param_names(self) -> list[str]:
        return ["c"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(None, None)]

    def _initial_params(self) -> list[float]:
        c0 = float(self.y[0])
        return [c0]

    def _predict(self, c: float) -> np.ndarray:
        return np.full_like(self.x, c, dtype=float)

# -------------------------------------------------------------------
# Constant density
# -------------------------------------------------------------------

class ConstantDensityFitter(BaseFitter):
    """
    Constant density fit: c / (xmax - xmin).

    Initial estimate
    ----------------
    c0 = y[0]
    """

    nparams: int = 1   # c

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
        return ["c"]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        return [(None, None)]

    def _initial_params(self) -> list[float]:
        c0 = float(self.y[0])
        return [c0]

    def _predict(self, c: float) -> np.ndarray:
        return np.full_like(self.x, c, dtype=float) / (self.xmax - self.xmin)