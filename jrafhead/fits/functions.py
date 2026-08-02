from __future__ import annotations

import numpy as np

from .base import BaseFitter, _weighted_std


class GaussianFitter(BaseFitter):
    """
    Gaussian fit: A * exp(-(x - \mu)² / (2\sigma^2)).

    Initial parameter estimates
    ---------------------------
    A0      = max(y)                              - peak height
    \mu0    = x[argmax(y)]                        - position of peak bin
    \sigma0 = weighted_std(x, y)                  - count-weighted spread
              reproduces np.std(np.repeat(x, y.astype(int)))
    """

    n_params: int = 3   # A, mu, sigma

    def model(self, x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
        """Gaussian: A * exp(-(x - \mu)^2 / (2\sigma^2))."""
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def _initial_params(self) -> list[float]:
        A0     = float(np.max(self.y))
        mu0    = float(self.x[np.argmax(self.y)])
        sigma0 = _weighted_std(self.x, self.y)
        return [A0, mu0, sigma0]


# ---------------------------------------------------------------------------
# Simple exponential  A * exp(−x / \tau)
# ---------------------------------------------------------------------------

class ExponentialFitter(BaseFitter):
    """
    Simple exponential decay: A · exp(-x / \tau).

    Initial estimates
    -----------------
    A0      = max(y after cut)
    \tau0   = weighted_std(x, y)   - count-weighted spread of the distribution
    """

    n_params: int = 2   # A, tau

    def model(self, x: np.ndarray, A: float, tau: float) -> np.ndarray:
        """Simple exponential: A * exp(-x / \tau)."""
        return A * np.exp(-x / tau)

    def _initial_params(self) -> list[float]:
        A0   = float(np.max(self.y))
        tau0 = _weighted_std(self.x, self.y)
        return [A0, tau0]
    

# ---------------------------------------------------------------------------
# Simple exponential rate  A * exp(−\almbda * x)
# ---------------------------------------------------------------------------

class ExponentialRateFitter(BaseFitter):
    """
    Simple exponential decay: A · exp(-\lambda x).

    Initial estimates
    -----------------
    A0       = max(y after cut)
    \lamnda0 = weighted_std(x, y)   - count-weighted spread of the distribution
    """

    n_params: int = 2   # A, lambda

    def model(self, x: np.ndarray, A: float, lambda_: float) -> np.ndarray:
        """Simple exponential: A * exp(-x / \lambda)."""
        return A * np.exp(-lambda_ * x)

    def _initial_params(self) -> list[float]:
        A0   = float(np.max(self.y))
        lambda0 = 1.0 / _weighted_std(self.x, self.y)
        return [A0, lambda0]


# ---------------------------------------------------------------------------
# Exponential + constant  A * exp(−x / \tau) + c
# ---------------------------------------------------------------------------

class ExponentialConstantFitter(BaseFitter):
    """
    Exponential decay above a constant background: A * exp(-x / \tau) + c.

    Initial estimates
    -----------------
    A0      = y[0]    — first bin as proxy for peak amplitude
    \tau0   = weighted_std(x, y)
    c0      = y[-1]   — last bin as proxy for flat background level
    """

    n_params: int = 3   # A, tau, c

    def model(self, x: np.ndarray, A: float, tau: float, c: float) -> np.ndarray:
        """Exponential + constant: A * exp(-x / \tau) + c."""
        return A * np.exp(-x / tau) + c

    def _initial_params(self) -> list[float]:
        A0   = float(self.y[0])
        tau0 = _weighted_std(self.x, self.y)
        c0   = float(self.y[-1])
        return [A0, tau0, c0]


# ---------------------------------------------------------------------------
# Constant (background region)
# ---------------------------------------------------------------------------

class ConstantFitter(BaseFitter):
    """
    Constant (flat) model: f(x) = c.

    Initial estimate
    ----------------
    c0 = y[0]
    """

    n_params: int = 1   # c

    def model(self, x: np.ndarray, c: float) -> np.ndarray:
        """Constant: c (broadcast to the shape of x)."""
        return np.full_like(x, c, dtype=float)

    def _initial_params(self) -> list[float]:
        return [float(self.y[0])]
    

# ---------------------------------------------------------------------------
# Cosmogenic rate estimation (triple exponential decay)
# ---------------------------------------------------------------------------

class Li9He8RateEstimationFitter(BaseFitter):
    """
    9Li/8He rate estimation model: 
        (Ideal case) f(x) = N9li8he * (f9li * lbda9li * exp(-lbda9li * t) + (1 - f9li) * lbda8he * exp(-lbda8he * t)) + Nbkg * Rmu * exp(-Rmu * t)
        f(x) = N9li8he * lmbda9li8he * exp(-lmbda9li8he * t) + Nbkg * Rmu * exp(-Rmu * t)

    Initial estimate
    ----------------
    N9li8he = y[0] / 100    - Number of 9Li/8He
    f9li    = 0.9           - 9Li proportion over 8He
    t9li    = 0.256         - 9Li lifetime
    t8he    = 0.171         - 8He lifetime
    Nbkg    = y[0] / 100    - Number of background
    Rmu     = 1             - Muon rate (cps)
    """

    # n_params: int = 6 # N9li8he, f9li, t9li, t8he, Nbkg, Rmu

    # def model_9li(self, x: np.ndarray, N9li8he: float, f9li: float, t9li: float, Rmu: float) -> np.ndarray:
    #     lbda9li = Rmu + 1.0 / t9li
    #     return N9li8he * f9li * lbda9li * np.exp(-lbda9li * x)
    
    # def model_8he(self, x: np.ndarray, N9li8he: float, f9li: float, t8he: float, Rmu: float) -> np.ndarray:
    #     lbda8he = Rmu + 1.0 / t8he
    #     return N9li8he * (1.0 - f9li) * lbda8he * np.exp(-lbda8he * x)
    
    # def model_bkg(self, x: np.ndarray, Nbkg: float, Rmu: float) -> np.ndarray:
    #     return Nbkg * Rmu * np.exp(-Rmu * x)

    # def model(self, x: np.ndarray, N9li8he: float, f9li: float, t9li: float, t8he: float, Nbkg: float, Rmu: float) -> np.ndarray:
    #     return self.model_9li(x, N9li8he, f9li, t9li, Rmu) + self.model_8he(x, N9li8he, f9li, t8he, Rmu) + self.model_bkg(x, Nbkg, Rmu)

    # def _initial_params(self) -> list[float]:
        # return [
            # self.y[0] / 5.0, 
            # 0.9, 
            # 0.256, 
            # 0.171, 
            # self.y[0] / 5.0, 
            # 1.75, 
        # ]

    n_params: int = 3 # N9li8he, Nbkg, Rmu

    def model_9li8he(self, x: np.ndarray, N9li8he: float, t9li8he: float, Rmu: float) -> np.ndarray:
        lmbda9li8he = Rmu + 1.0 / t9li8he
        # return N9li8he * lmbda9li8he * np.exp(-lmbda9li8he * x)
        t1 = x - self.widths / 2.0
        t2 = x + self.widths / 2.0
        return N9li8he * (np.exp(-lmbda9li8he * t1) - np.exp(-lmbda9li8he * t2)) / self.widths
    
    def model_bkg(self, x: np.ndarray, Nbkg: float, Rmu: float) -> np.ndarray:
        # return Nbkg * Rmu * np.exp(-Rmu * x)
        t1 = x - self.widths / 2.0
        t2 = x + self.widths / 2.0
        return Nbkg * (np.exp(-Rmu * t1) - np.exp(-Rmu * t2)) / self.widths

    def model(self, x: np.ndarray, N9li8he: float, Nbkg: float, Rmu: float) -> np.ndarray:
        t9li8he = 0.256
        return self.model_9li8he(x, N9li8he, t9li8he, Rmu) + self.model_bkg(x, Nbkg, Rmu)
    
    def _initial_params(self) -> list[float]:
        return [
            self.y[0] / 5.0, 
            self.y[0] / 5.0, 
            1.75, 
        ]