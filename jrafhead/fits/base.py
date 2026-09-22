import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from iminuit import Minuit
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
        1-sigma uncertainties (from the Minuit covariance matrix / HESSE).
    chi2 : float
        Chi-squared statistic at the minimum.
    ndf : int
        Number of degrees of freedom: npoints - nfreeparams.
    pvalue : float
        p-value, i.e. "chi2.sf(chi2, ndf)".
    names : list of str
        Parameter names, in the same order as popt/perr.
    covariance : np.ndarray or None
        Full covariance matrix (nfree x nfree), or None if unavailable.
    valid : bool
        Whether Minuit considers the minimum valid (fmin.is_valid).
    """
    popt:       np.ndarray
    perr:       np.ndarray
    chi2:       float
    ndf:        int
    pvalue:     float
    names:      list[str]         = field(default_factory=list)
    covariance: np.ndarray | None = None
    valid:      bool              = True

    def as_dict(self) -> dict[str, tuple[float, float]]:
        """ Return {name: (value, error)} for readable printing. """
        return {n: (float(v), float(e)) for n, v, e in zip(self.names, self.popt, self.perr)}

# -------------------------------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------------------------------

def _weighted_std(x: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted standard deviation of x using integer weights (bin counts).
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total == 0:
        return float(np.std(x))
    mean = np.dot(w, x) / total
    var  = np.dot(w, (x - mean) ** 2) / total
    return float(np.sqrt(var))

def _goodness_of_fit(chi2_val: float, npoints: int, nparams: int) -> tuple[float, int, float]:
    """
    Compute ndf and p-value from an already-evaluated chi2.

    Unlike the previous curve_fit-based version, the chi2 itself is computed
    by BaseFitter.chi2() (which daughters customize via the _predict/_errors
    hooks), so this helper only handles the bookkeeping.
    """
    ndf    = npoints - nparams
    pvalue = float(chi2_dist.sf(chi2_val, ndf)) if ndf > 0 else float("nan")
    return float(chi2_val), ndf, pvalue

# -------------------------------------------------------------------------------------------------
# Abstract base fitter
# -------------------------------------------------------------------------------------------------

class BaseFitter(ABC):
    """
    Abstract base class for all fitters, minimized with iminuit.

    The cost handed to Minuit is always BaseFitter.chi2(), a single generic
    chi-squared. What varies between daughter classes is what goes INTO that
    chi-squared, supplied by two overloadable hooks:

        _predict(params) -> model values at the fitted points
        _errors(params)  -> per-point sigma for the chi2 denominator

    Daughter classes normally override those hooks (or just `model`), NOT
    chi2() itself.

    Subclasses must implement:
    - nparams         (class attribute) - number of free parameters
    - model           (method)          - the fit function f(x, *params)
    - _initial_params (method)          - returns a p0 list

    Optional overrides:
    - param_names  -> list[str], physical names shown in Minuit output
    - param_limits -> list of (lo, hi) or None, per parameter
    - fixed_params -> list[bool], per parameter
    - _predict     -> custom prediction (default: model(self.x, *params))
    - _errors      -> custom per-point sigma (default: self.yerr)

    Parameters
    ----------
    bins : np.ndarray
        Bin edges, length nbins + 1.
    y : np.ndarray
        Observed values (counts, or counts/width if fitting a density).
    yerr : np.ndarray
        Per-point uncertainties. Zero-valued entries are excluded.
    xlim : tuple or None
        Optional (xmin, xmax) cut applied to bin centres.
    """

    #: Number of free parameters - must be set by every subclass
    nparams: int = 0

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None,
    ) -> None:
        valid        = np.asarray(yerr) > 0
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

        #: Populated by fit(); exposed for MINOS errors, contours, profiles.
        self.minuit: Minuit | None = None

    # ---------------------------------------------------------------------
    # Optional parameter metadata
    # ---------------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        """
        Names for the free parameters, in p0 order.

        Default p0, p1, ... Override in daughters to get physical names
        (N9li8he, f9li, Nbkg, Rmu) in Minuit's output tables and in
        FitResult.names.
        """
        return [f"p{i}" for i in range(self.nparams)]

    @property
    def param_limits(self) -> list[tuple[float | None, float | None] | None] | None:
        """
        Optional per-parameter (lo, hi) bounds, or None for unbounded.

        Return None (default) for no bounds at all. Useful for keeping the
        minimizer in physical regions -- e.g. a fraction in [0, 1] or a
        positive rate.
        """
        return None

    @property
    def fixed_params(self) -> list[bool] | None:
        """Optional per-parameter fixed flags. None (default) = all free."""
        return None

    # ---------------------------------------------------------------------
    # Overloadable hooks - the customization points for daughter classes
    # ---------------------------------------------------------------------

    @abstractmethod
    def _initial_params(self) -> list[float]:
        """ Return a list of initial parameter guesses p0. """
        ...

    @abstractmethod
    def _predict(self, *params: float) -> np.ndarray:
        """ Model prediction at the fitted points. """
        ...

    def _errors(self, *params: float) -> np.ndarray:
        """
        Per-point sigma used in the chi2 denominator.

        Default: self.yerr (the data uncertainties -- the standard Neyman
        chi2, constant w.r.t. the parameters).

        Override to express parameter-DEPENDENT errors, which the previous
        curve_fit-based design could not represent at all. Examples:

            Pearson chi2 (sigma^2 = model):
                def _errors(self, *params):
                    return np.sqrt(np.maximum(self._predict(*params), 1e-12))

            Data + MC template stat error in quadrature:
                def _errors(self, *params):
                    return np.sqrt(self.yerr**2 + self.template_err**2)
        """
        return self.yerr

    def chi2(self, *params: float) -> float:
        """
        Generic chi-squared cost handed to Minuit.

            chi2 = sum( ((prediction - data) / sigma)^2 )

        Both `prediction` and `sigma` come from the overloadable hooks above,
        so daughter classes customize the cost by overriding those rather
        than rewriting this method. Non-finite values are penalized rather
        than returned as NaN, which would stall the minimizer.
        """
        prediction = np.asarray(self._predict(*params), dtype=float)
        sigma      = np.asarray(self._errors(*params),  dtype=float)

        if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(sigma)):
            return 1.0e12
        if np.any(sigma <= 0):
            return 1.0e12

        residuals = (prediction - self.y) / sigma
        value = float(np.sum(residuals ** 2))
        return value if np.isfinite(value) else 1.0e12

    # ---------------------------------------------------------------------
    # Fit
    # ---------------------------------------------------------------------

    def fit(self, migrad_calls: int = 0, run_hesse: bool = True) -> FitResult | None:
        """
        Minimize self.chi2 with iminuit and return a FitResult.

        Parameters
        ----------
        migrad_calls : int
            Max function calls for MIGRAD (0 = iminuit's default).
        run_hesse : bool
            Run HESSE after MIGRAD for an accurate covariance matrix.

        Returns
        -------
        FitResult or None
            None if there are too few points, or if the minimizer fails.
        """
        if len(self.x) <= self.nparams:
            warnings.warn(
                f"{self.__class__.__name__}: not enough data points "
                f"({len(self.x)} <= {self.nparams} params) - fit skipped.",
                RuntimeWarning, stacklevel=2,
            )
            return None

        p0    = list(self._initial_params())
        names = list(self.param_names)

        if len(p0) != self.nparams:
            warnings.warn(
                f"{self.__class__.__name__}: _initial_params() returned {len(p0)} "
                f"values but nparams = {self.nparams}.",
                RuntimeWarning, stacklevel=2,
            )
            return None
        if len(names) != self.nparams:
            warnings.warn(
                f"{self.__class__.__name__}: param_names has {len(names)} entries "
                f"but nparams = {self.nparams}.",
                RuntimeWarning, stacklevel=2,
            )
            return None

        # Minuit cannot introspect the arity of a *params signature, so the
        # parameter names are supplied explicitly.
        try:
            minuit = Minuit(self.chi2, *p0, name=names)
        except Exception as exc:
            warnings.warn(
                f"{self.__class__.__name__}: could not build Minuit object - {exc}",
                RuntimeWarning, stacklevel=2,
            )
            return None

        # errordef=1 is the correct convention for a chi2 cost (0.5 is NLL);
        # it is what makes the reported 1-sigma errors correct.
        minuit.errordef = Minuit.LEAST_SQUARES

        limits = self.param_limits
        if limits is not None:
            for name, lim in zip(names, limits):
                minuit.limits[name] = lim

        fixed = self.fixed_params
        if fixed is not None:
            for name, isfixed in zip(names, fixed):
                minuit.fixed[name] = bool(isfixed)

        try:
            minuit.migrad(ncall=migrad_calls or None)
            if run_hesse:
                minuit.hesse()
        except Exception as exc:
            warnings.warn(
                f"{self.__class__.__name__}: minimization failed - {exc}",
                RuntimeWarning, stacklevel=2,
            )
            return None

        self.minuit = minuit

        if not minuit.valid:
            warnings.warn(
                f"{self.__class__.__name__}: Minuit reports an INVALID minimum "
                "- results should not be trusted.",
                RuntimeWarning, stacklevel=2,
            )

        popt = np.array(minuit.values, dtype=float)
        perr = np.array(minuit.errors, dtype=float)

        try:
            cov = np.array(minuit.covariance, dtype=float)
        except Exception:
            cov = None

        # ndf must count only FREE parameters -- fixing one gives back a dof.
        nfree = int(np.sum(~np.array(minuit.fixed, dtype=bool)))
        chi2_val, ndf, pvalue = _goodness_of_fit(minuit.fval, len(self.x), nfree)

        return FitResult(
            popt=popt, perr=perr,
            chi2=chi2_val, ndf=ndf, pvalue=pvalue,
            names=names, covariance=cov, valid=bool(minuit.valid),
        )