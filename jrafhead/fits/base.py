from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings


import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist


# -------------------------------------------------------------------------------------------------
# Shared result container
# -------------------------------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    Output of a successful curve fit.

    Attributes
    ----------
    popt : np.ndarray
        Best-fit parameter vector [p0, p1, ...].
    perr : np.ndarray
        1-sigma uncertainties sqrt(diag(pcov)).
    chi2 : float
        Chi-squared statistic: \sum ((y_model - y) / yerr)^2.
    ndf : int
        Number of degrees of freedom: len(y) - n_params.
    pvalue : float
        p-value, i.e. "chi2.sf(chi2, ndf)".
    """
    popt:   np.ndarray
    perr:   np.ndarray
    chi2:   float
    ndf:    int
    pvalue: float


# -------------------------------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------------------------------

def _weighted_std(x: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted standard deviation of x using integer weights (bin counts).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Values (e.g. bin centres).
    weights : np.ndarray, shape (N,)
        Non-negative weights (e.g. histogram counts).

    Returns
    -------
    float
        Weighted standard deviation, or "np.std(x)" if all weights are zero.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total == 0:
        return float(np.std(x))
    mean = np.dot(w, x) / total
    var  = np.dot(w, (x - mean) ** 2) / total
    return float(np.sqrt(var))


def _goodness_of_fit(y_data: np.ndarray, y_model: np.ndarray, yerr: np.ndarray, n_params: int) -> tuple[float, int, float]:
    """
    Compute \chi^2, ndf, and p-value for a completed fit.

    Parameters
    ----------
    y_data : np.ndarray
        Observed values.
    y_model : np.ndarray
        Model values.
    yerr : np.ndarray
        Per-point uncertainties (must be > 0 for valid \chi^2).
    n_params : int
        Number of free parameters in the model.

    Returns
    -------
    chi2_val, ndf, pvalue : float, int, float
    """
    residuals = (y_model - y_data) / yerr
    chi2_val  = float(np.sum(residuals ** 2))
    ndf       = len(y_data) - n_params
    pvalue    = float(chi2_dist.sf(chi2_val, ndf)) if ndf > 0 else float("nan")
    return chi2_val, ndf, pvalue


# -------------------------------------------------------------------------------------------------
# Abstract base fitter
# -------------------------------------------------------------------------------------------------

class BaseFitter(ABC):
    """
    Abstract base class for all fitters.

    Subclasses must implement:
    - n_params   (class attribute) - number of free parameters
    - model      (static method)   - the fit function f(x, *params)
    - _initial_params (method)     - returns a p0 list

    "fit()" is implemented here and should not be overridden.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values (e.g. bin centres).
    y : np.ndarray
        Observed counts or values.
    yerr : np.ndarray
        Per-point uncertainties. Zero-valued entries are excluded before
        fitting to avoid division-by-zero in χ^2.
    """

    #: Number of free parameters - must be set by every subclass
    n_params: int = 0

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None, 
    ) -> None:
        # Exclude zero-error bins (empty bins) - same as the original "mask = hist > 0"
        valid        = yerr > 0
        self.widths  = np.diff(bins)[valid]
        self.centers = (0.5 * (bins[1:] + bins[:-1]))[valid]
        self.x       = self.centers
        self.y       = np.asarray(y)[valid]
        self.yerr    = np.asarray(yerr)[valid]
        self.xlim    = xlim

        if self.xlim is not None:
            xmin, xmax = self.xlim

            cut = np.ones_like(self.x, dtype=bool)
            if xmin is not None:
                cut &= self.x >= xmin
            if xmax is not None:
                cut &= self.x <= xmax

            self.widths  = self.widths[cut]
            self.centers = self.centers[cut]
            self.x       = self.x[cut]
            self.y       = self.y[cut]
            self.yerr    = self.yerr[cut]

    @abstractmethod
    def model(self, x: np.ndarray, *params: float) -> np.ndarray:
        """The fit function f(x, p0, p1, ...) evaluated at x."""
        ...

    @abstractmethod
    def _initial_params(self) -> list[float]:
        """Return a list of initial parameter guesses p0."""
        ...

    def fit(self) -> FitResult | None:
        """
        Run "scipy.optimize.curve_fit" and return a "FitResult".

        Returns None if the fit fails (e.g. not enough data points,
        convergence failure). A warning is printed in that case.

        Returns
        -------
        FitResult or None
        """
        if len(self.x) <= self.n_params:
            warnings.warn(
                f"{self.__class__.__name__}: not enough data points "
                f"({len(self.x)} ≤ {self.n_params} params) - fit skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        try:
            popt, pcov = curve_fit(
                self.model,
                self.x,
                self.y,
                p0=self._initial_params(),
                sigma=self.yerr,
                absolute_sigma=True,
            )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"{self.__class__.__name__}: curve_fit failed - {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        perr    = np.sqrt(np.diag(pcov))
        y_model = self.model(self.x, *popt)
        chi2_val, ndf, pvalue = _goodness_of_fit(self.y, y_model, self.yerr, self.n_params)

        return FitResult(
            popt=popt, perr=perr,
            chi2=chi2_val, ndf=ndf, pvalue=pvalue,
        )