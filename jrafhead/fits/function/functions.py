from __future__ import annotations

import numpy as np

from ..base import BaseFitter


class CubicFitter(BaseFitter):
    """
    Cubic fit: A * x^{3}.
    
    Initial parameter estimates
    ---------------------------
    A0 = y / x^{3} 
    """

    nparams: int = 1 # A

    def _predict(self, A: float) -> np.ndarray:
        return A * self.x**3

    def _initial_params(self) -> list[float]:
        A0 = float(self.y[-1] / self.x[-1]**3)
        return [A0]

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

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None, 
    ) -> None:
        super().__init__(bins, y, yerr, xlim)
        self.tmin = self.centers[0]  - 0.5 * self.widths[0]
        self.tmax = self.centers[-1] + 0.5 * self.widths[-1]

    # ######################################## 4 parameters model ########################################
    nparams: int = 4 # N9li8he, f9li, Nbkg, Rmu

    def model_9li(self, x: np.ndarray, N9li8he: float, f9li: float, t9li: float, Rmu: float) -> np.ndarray:
        lmbda = Rmu + 1.0 / t9li
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-lmbda * t1) - np.exp(-lmbda * t2)
        denum = self.widths * (np.exp(-lmbda * self.tmin) - np.exp(-lmbda * self.tmax))
        return N9li8he * f9li * num / denum

    def model_8he(self, x: np.ndarray, N9li8he: float, f9li: float, t8he: float, Rmu: float) -> np.ndarray:
        lmbda = Rmu + 1.0 / t8he
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-lmbda * t1) - np.exp(-lmbda * t2)
        denum = self.widths * (np.exp(-lmbda * self.tmin) - np.exp(-lmbda * self.tmax))
        return N9li8he * (1.0 - f9li) * num / denum

    def model_9li8he(self, x: np.ndarray, N9li8he: float, f9li: float, t9li: float, t8he: float, Rmu: float) -> np.ndarray:
        return self.model_9li(x, N9li8he, f9li, t9li, Rmu) + self.model_8he(x, N9li8he, f9li, t8he, Rmu)

    def model_bkg(self, x: np.ndarray, Nbkg: float, Rmu: float) -> np.ndarray:
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-Rmu * t1) - np.exp(-Rmu * t2)
        denum = self.widths * (np.exp(-Rmu * self.tmin) - np.exp(-Rmu * self.tmax))
        return Nbkg * num / denum

    def _predict(self, N9li8he: float, f9li: float, Nbkg: float, Rmu: float) -> np.ndarray:
        t9li = 0.257
        t8he = 0.172
        return self.model_9li8he(self.x, N9li8he, f9li, t9li, t8he, Rmu) + self.model_bkg(self.x, Nbkg, Rmu)

    def _initial_params(self) -> list[float]:
        mask_bkg = self.centers > 2.0
        mask_sig = self.centers <= 2.0

        counts_sig_region = np.sum(self.y[mask_sig] * self.widths[mask_sig])
        r_bkg_est = np.mean(self.y[mask_bkg]) if np.any(mask_bkg) else 0.0
        t_min = self.centers[0]
        t_max = self.centers[-1]
        Nbkg_est = r_bkg_est * (t_max - t_min)
        t_sig_window = min(2.0, t_max) - t_min
        bkg_in_sig_region = r_bkg_est * t_sig_window
        N9li8he_est = max(0.0, counts_sig_region - bkg_in_sig_region)

        return [
            N9li8he_est,
            0.95,
            Nbkg_est,
            1.0,
        ]

class Li9RateEstimationFitter(BaseFitter):
    """
    9Li rate estimation model: 
        (Ideal case) f(x) = N9li * lbda9li * exp(-lbda9li * t) + Nbkg * Rmu * exp(-Rmu * t)

    Initial estimate
    ----------------
    N9li    = y[0] / 100    - Number of 9Li
    t9li    = 0.256         - 9Li lifetime
    Nbkg    = y[0] / 100    - Number of background
    Rmu     = 1             - Muon rate (cps)
    """

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None, 
    ) -> None:
        super().__init__(bins, y, yerr, xlim)
        self.tmin = self.centers[0]  - 0.5 * self.widths[0]
        self.tmax = self.centers[-1] + 0.5 * self.widths[-1]

    # ######################################## 3 parameters model ########################################
    nparams: int = 3 # N9li, Nbkg, Rmu

    def model_9li(self, x: np.ndarray, N9li: float, t9li: float, Rmu: float) -> np.ndarray:
        lmbda = Rmu + 1.0 / t9li
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-lmbda * t1) - np.exp(-lmbda * t2)
        denum = self.widths * (np.exp(-lmbda * self.tmin) - np.exp(-lmbda * self.tmax))
        return N9li * num / denum
    
    def model_bkg(self, x: np.ndarray, Nbkg: float, Rmu: float) -> np.ndarray:
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-Rmu * t1) - np.exp(-Rmu * t2)
        denum = self.widths * (np.exp(-Rmu * self.tmin) - np.exp(-Rmu * self.tmax))
        return Nbkg * num / denum

    def _predict(self, N9li: float, Nbkg: float, Rmu: float) -> np.ndarray:
        t9li = 0.1782  / np.log(2) # t_{1/2} = 178.2(4) ms
        return self.model_9li(self.x, N9li, t9li, Rmu) + self.model_bkg(self.x, Nbkg, Rmu)

    def _initial_params(self) -> list[float]:
        mask_bkg = self.centers > 2.0
        mask_sig = self.centers <= 2.0
    
        counts_sig_region = np.sum(self.y[mask_sig] * self.widths[mask_sig])
        r_bkg_est = np.mean(self.y[mask_bkg]) if np.any(mask_bkg) else 0.0
        t_min = self.centers[0]
        t_max = self.centers[-1]
        Nbkg_est = r_bkg_est * (t_max - t_min)
        t_sig_window = min(2.0, t_max) - t_min
        bkg_in_sig_region = r_bkg_est * t_sig_window
        N9li8he_est = max(0.0, counts_sig_region - bkg_in_sig_region)
    
        return [
            N9li8he_est,
            Nbkg_est,
            1.0,
        ]

class Li9B12RateEstimationFitter(BaseFitter):
    """
    9Li rate estimation model: 
        (Ideal case) f(x) = N12b * lbda12b * exp(-lbda12b * t) + N9li * lbda9li * exp(-lbda9li * t) + Nbkg * Rmu * exp(-Rmu * t)

    Initial estimate
    ----------------
    N12b    = 100           - Number of 12B
    N9li                    - Number of 9Li
    Nbkg                    - Number of background
    Rmu     = 1             - Muon rate (cps)
    """

    def __init__(
        self,
        bins:   np.ndarray,
        y:      np.ndarray,
        yerr:   np.ndarray,
        xlim:   tuple[float | None, float | None] | None = None, 
    ) -> None:
        super().__init__(bins, y, yerr, xlim)
        self.tmin = self.centers[0]  - 0.5 * self.widths[0]
        self.tmax = self.centers[-1] + 0.5 * self.widths[-1]

    # ######################################## 4 parameters model ########################################
    nparams: int = 4 # N12b, N9li, Nbkg, Rmu

    def model_12b(self, x: np.ndarray, N12b: float, t12b: float, Rmu: float) -> np.ndarray:
        lmbda = Rmu + 1.0 / t12b
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-lmbda * t1) - np.exp(-lmbda * t2)
        denum = self.widths * (np.exp(-lmbda * self.tmin) - np.exp(-lmbda * self.tmax))
        return N12b * num / denum

    def model_9li(self, x: np.ndarray, N9li: float, t9li: float, Rmu: float) -> np.ndarray:
        lmbda = Rmu + 1.0 / t9li
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-lmbda * t1) - np.exp(-lmbda * t2)
        denum = self.widths * (np.exp(-lmbda * self.tmin) - np.exp(-lmbda * self.tmax))
        return N9li * num / denum
    
    def model_bkg(self, x: np.ndarray, Nbkg: float, Rmu: float) -> np.ndarray:
        t1    = x - self.widths / 2.0
        t2    = x + self.widths / 2.0
        num   = np.exp(-Rmu * t1) - np.exp(-Rmu * t2)
        denum = self.widths * (np.exp(-Rmu * self.tmin) - np.exp(-Rmu * self.tmax))
        return Nbkg * num / denum

    def _predict(self, N12b: float, N9li: float, Nbkg: float, Rmu: float) -> np.ndarray:
        t12b = 0.02020 / np.log(2) # t_{1/2} = 20.20(2) ms
        t9li = 0.1782  / np.log(2) # t_{1/2} = 178.2(4) ms
        return self.model_12b(self.x, N12b, t12b, Rmu) + self.model_9li(self.x, N9li, t9li, Rmu) + self.model_bkg(self.x, Nbkg, Rmu)

    def _initial_params(self) -> list[float]:
        mask_bkg = self.centers > 2.0
        mask_sig = self.centers <= 2.0
    
        counts_sig_region = np.sum(self.y[mask_sig] * self.widths[mask_sig])
        r_bkg_est = np.mean(self.y[mask_bkg]) if np.any(mask_bkg) else 0.0
        t_min = self.centers[0]
        t_max = self.centers[-1]
        Nbkg_est = r_bkg_est * (t_max - t_min)
        t_sig_window = min(2.0, t_max) - t_min
        bkg_in_sig_region = r_bkg_est * t_sig_window
        N9li8he_est = max(0.0, counts_sig_region - bkg_in_sig_region)
    
        return [
            100.0,
            N9li8he_est,
            Nbkg_est,
            1.0,
        ]