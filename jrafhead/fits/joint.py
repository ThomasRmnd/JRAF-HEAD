from __future__ import annotations

import numpy as np

from .base import (
    BaseFitter,
)

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def model_time_bkg(x: np.ndarray, N: float, fbkg: float, tmin: float, tmax: float) -> np.ndarray:
    return np.full_like(x, N * fbkg / (tmax - tmin), dtype=float)

def model_time_9li(x: np.ndarray, N: float, fbkg: float, f9li: float, t9li: float, tmin: float, tmax: float) -> np.ndarray:
    norm = np.exp(-tmin / t9li) - np.exp(-tmax / t9li)
    pdf = np.exp(-x / t9li) / t9li / norm
    return N * (1.0 - fbkg) * f9li * pdf

def model_time_8he(x: np.ndarray, N: float, fbkg: float, f9li: float, t8he: float, tmin: float, tmax: float) -> np.ndarray:
    norm = np.exp(-tmin / t8he) - np.exp(-tmax / t8he)
    pdf = np.exp(-x / t8he) / t8he / norm
    return N * (1.0 - fbkg) * (1.0 - f9li) * pdf

# ---------------------------------------------------------------------------
# 9Li/8He
# ---------------------------------------------------------------------------

class Li9He8ContrainedFractionFitter(BaseFitter):
    """
    $^{9}$Li/$^{8}$He contribution fitter:
    E = N * (
        fbkg * Ebkg + 
        (1 - fbkg) * (
            f9li * E9li + (1 - f9li) * E8he
        )
    )
    \Delta t_{\mu-p} = N * (
        fbkg + 
        (1 - fbkg) * (
            f9li * e^{-t / t9li} / t9li + (1 - f9li) * e^{-t / t8he} / t8he
        )
    )

    Initial estimates
    -----------------
    N           = sum(y)    - Number of events
    fbkg        = y[-1] / N - Proportion of background
    f9li        = 0.80      - Proportion of Lithium 9
    """

    n_params: int = 3 # N, fbkg, f9li

    def __init__(
        self, 
        ebkg:   np.ndarray, 
        eli9:   np.ndarray, 
        ehe8:   np.ndarray, 
        ebins:  np.ndarray, 
        ey:     np.ndarray, 
        eyerr:  np.ndarray, 
        dtbins: np.ndarray, 
        dty:    np.ndarray, 
        dtyerr: np.ndarray, 
        exlim:  tuple[float | None, float | None] | None = None,
        dtxlim: tuple[float | None, float | None] | None = None,
    ) -> None:
        evalid    = eyerr > 0
        dtvalid   = dtyerr > 0
        ecenters  = 0.5 * (ebins[1:] + ebins[:-1])
        dtcenters = 0.5 * (dtbins[1:] + dtbins[:-1])

        if exlim is not None:
            xmin, xmax = exlim
            if xmin is not None:
                evalid &= np.asarray(ecenters) >= xmin
            if xmax is not None:
                evalid &= np.asarray(ecenters) <= xmax

        if dtxlim is not None:
            xmin, xmax = dtxlim
            if xmin is not None:
                dtvalid &= np.asarray(dtcenters) >= xmin
            if xmax is not None:
                dtvalid &= np.asarray(dtcenters) <= xmax

        self.ebkg       = np.asarray(ebkg)[evalid]
        self.ebkg_full  = np.asarray(ebkg)
        self.eli9       = np.asarray(eli9)[evalid]
        self.eli9_full  = np.asarray(eli9)
        self.ehe8       = np.asarray(ehe8)[evalid]
        self.ehe8_full  = np.asarray(ehe8)

        self.ecenters   = ecenters[evalid]
        self.ewidths    = np.diff(ebins)[evalid]
        self.dtcenters  = dtcenters[dtvalid]
        self.dtwidths   = np.diff(dtbins)[dtvalid]

        self.nenergy    = np.sum(evalid)

        self.x          = np.concatenate((
            np.asarray(ecenters)[evalid], 
            np.asarray(dtcenters)[dtvalid]
        ))
        self.y          = np.concatenate((
            np.asarray(ey)[evalid], 
            np.asarray(dty)[dtvalid]
        ))
        self.yerr       = np.concatenate((
            np.asarray(eyerr)[evalid], 
            np.asarray(dtyerr)[dtvalid]
        ))

    def _to_energy(self, x: np.ndarray) -> np.ndarray:
        return x[:self.nenergy]
    
    def _to_time(self, x: np.ndarray) -> np.ndarray:
        return x[self.nenergy:]

    def model_energy(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        return N * (
            fbkg * self.ebkg +
            (1.0 - fbkg) * (
                f9li * self.eli9 +
                (1.0 - f9li) * self.ehe8
            )
        )
    
    def model_energy_full(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        return N * (
            fbkg * self.ebkg_full +
            (1.0 - fbkg) * (
                f9li * self.eli9_full +
                (1.0 - f9li) * self.ehe8_full
            )
        )
    
    def model_time(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        t9li = 0.256
        t8he = 0.171
        tmin = self.dtcenters[0] - self.dtwidths[0]
        tmax = self.dtcenters[-1] + self.dtwidths[-1]
        return (
            model_time_bkg(x, N, fbkg, tmin, tmax) + 
            model_time_9li(x, N, fbkg, f9li, t9li, tmin, tmax) +
            model_time_8he(x, N, fbkg, f9li, t8he, tmin, tmax)
        )
    
    def model(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        return np.concatenate((
            self.model_energy(self._to_energy(x), N, fbkg, f9li),
            self.model_time(self._to_time(x), N, fbkg, f9li),
        ))

    def _initial_params(self) -> list[float]:
        N = np.sum(self._to_energy(self.y) * self.ewidths)
        fbkg = 0.35
        f9li = 0.9
        return [N, fbkg, f9li]


class Li9He8FractionFitter(BaseFitter):
    """
    $^{9}$Li/$^{8}$He contribution fitter:
    E = N * (
        fbkg * Ebkg + 
        (1 - fbkg) * (
            f9li * E9li + (1 - f9li) * E8he
        )
    )
    \Delta t_{\mu-p} = N * (
        fbkg + 
        (1 - fbkg) * (
            f9li * e^{-t / t9li} / t9li + (1 - f9li) * e^{-t / t8he} / t8he
        )
    )

    Initial estimates
    -----------------
    N           = sum(y)    - Number of events
    fbkg        = y[-1] / N - Proportion of background
    f9li        = 0.80      - Proportion of Lithium 9
    t9li        = 256       - Lifetime of Lithium 9
    t8he        = 171       - Lifetime of Helium 8
    """

    n_params: int = 5 # N, fbkg, f9li, t9li, t8he

    def __init__(
        self, 
        ebkg:   np.ndarray, 
        eli9:   np.ndarray, 
        ehe8:   np.ndarray, 
        ebins:  np.ndarray, 
        ey:     np.ndarray, 
        eyerr:  np.ndarray, 
        dtbins: np.ndarray, 
        dty:    np.ndarray, 
        dtyerr: np.ndarray, 
        exlim:  tuple[float | None, float | None] | None = None,
        dtxlim: tuple[float | None, float | None] | None = None,
    ) -> None:
        evalid    = eyerr > 0
        dtvalid   = dtyerr > 0
        ecenters  = 0.5 * (ebins[1:] + ebins[:-1])
        dtcenters = 0.5 * (dtbins[1:] + dtbins[:-1])

        if exlim is not None:
            xmin, xmax = exlim
            if xmin is not None:
                evalid &= np.asarray(ecenters) >= xmin
            if xmax is not None:
                evalid &= np.asarray(ecenters) <= xmax

        if dtxlim is not None:
            xmin, xmax = dtxlim
            if xmin is not None:
                dtvalid &= np.asarray(dtcenters) >= xmin
            if xmax is not None:
                dtvalid &= np.asarray(dtcenters) <= xmax

        self.ebkg       = np.asarray(ebkg)[evalid]
        self.ebkg_full  = np.asarray(ebkg)
        self.eli9       = np.asarray(eli9)[evalid]
        self.eli9_full  = np.asarray(eli9)
        self.ehe8       = np.asarray(ehe8)[evalid]
        self.ehe8_full  = np.asarray(ehe8)

        self.nenergy    = np.sum(evalid)

        self.x          = np.concatenate((
            np.asarray(ecenters)[evalid], 
            np.asarray(dtcenters)[dtvalid]
        ))
        self.y          = np.concatenate((
            np.asarray(ey)[evalid], 
            np.asarray(dty)[dtvalid]
        ))
        self.yerr       = np.concatenate((
            np.asarray(eyerr)[evalid], 
            np.asarray(dtyerr)[dtvalid]
        ))

    def _to_energy(self, x: np.ndarray) -> np.ndarray:
        return x[:self.nenergy]
    
    def _to_time(self, x: np.ndarray) -> np.ndarray:
        return x[self.nenergy:]

    def model_energy(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        return N * (
            fbkg * self.ebkg +
            (1.0 - fbkg) * (
                f9li * self.eli9 +
                (1.0 - f9li) * self.ehe8
            )
        )
    
    def model_energy_full(self, x: np.ndarray, N: float, fbkg: float, f9li: float) -> np.ndarray:
        return N * (
            fbkg * self.ebkg_full +
            (1.0 - fbkg) * (
                f9li * self.eli9_full +
                (1.0 - f9li) * self.ehe8_full
            )
        )
    
    def model_time(self, x: np.ndarray, N: float, fbkg: float, f9li: float, t9li: float, t8he: float) -> np.ndarray:

        li9 = np.exp(-x / t9li) / t9li
        li9 /= np.sum(li9)

        he8 = np.exp(-x / t8he) / t8he
        he8 /= np.sum(he8)

        bkg = np.ones_like(x, dtype=float)
        bkg /= np.sum(bkg)

        return N * (
            fbkg * bkg +
            (1.0 - fbkg) * (
                f9li * li9 +
                (1.0 - f9li) * he8
            )
        )
    
    def model(self, x: np.ndarray, N: float, fbkg: float, f9li: float, t9li: float, t8he: float) -> np.ndarray:
        return np.concatenate((
            self.model_energy(self._to_energy(x), N, fbkg, f9li),
            self.model_time(self._to_time(x), N, fbkg, f9li, t9li, t8he),
        ))

    def _initial_params(self) -> list[float]:
        N = np.sum(self._to_energy(self.y))
        fbkg = 0.5
        f9li = 0.8
        t9li = 0.256
        t8he = 0.171
        return [N, fbkg, f9li, t9li, t8he]


class Li9He8ChengzhuoFitter(BaseFitter):
    """
    $^{9}$Li/$^{8}$He contribution fitter:
    E = N * (f0 * E0 + f1 * E1 + f2 * E2 + f3 * E3 + (0.508 - f0 - f1 - f2 - f3 - f4) * E4)

    Initial estimates
    -----------------
    N           = - Number of events
    f0          = - Branching ratio of the 0th branch
    f1          = - Branching ratio of the 1th branch
    f2          = - Branching ratio of the 2th branch
    f3          = - Branching ratio of the 3th branch
    """

    n_params: int = 5 # N, f0, f1, f2, f3

    def __init__(
        self, 
        e0:   np.ndarray, 
        e1:   np.ndarray, 
        e2:   np.ndarray, 
        e3:   np.ndarray, 
        e4:   np.ndarray, 
        bins: np.ndarray, 
        y:    np.ndarray, 
        yerr: np.ndarray, 
        xlim: tuple[float | None, float | None] | None = None,
    ) -> None:
        valid   = yerr > 0
        centers = 0.5 * (bins[1:] + bins[:-1])

        if xlim is not None:
            xmin, xmax = xlim
            if xmin is not None:
                valid &= centers >= xmin
            if xmax is not None:
                valid &= centers <= xmax

        self.e0         = np.asarray(e0)[valid]
        self.e0_full    = np.asarray(e0)
        self.e1         = np.asarray(e1)[valid]
        self.e1_full    = np.asarray(e1)
        self.e2         = np.asarray(e2)[valid]
        self.e2_full    = np.asarray(e2)
        self.e3         = np.asarray(e3)[valid]
        self.e3_full    = np.asarray(e3)
        self.e4         = np.asarray(e4)[valid]
        self.e4_full    = np.asarray(e4)

        self.x          = centers[valid]
        self.y          = np.asarray(y)[valid]
        self.yerr       = np.asarray(yerr)[valid]

    def model_full(self, x: np.ndarray, N: float, f0: float, f1: float, f2: float, f3: float) -> tuple[np.ndarray]:
        f4 = 0.508 - f0 - f1 - f2 - f3
        return N * (f0 * self.e0_full + f1 * self.e1_full + f2 * self.e2_full + f3 * self.e3_full + f4 * self.e4_full)

    def model(self, x: np.ndarray, N: float, f0: float, f1: float, f2: float, f3: float) -> tuple[np.ndarray]:
        f4 = 0.508 - f0 - f1 - f2 - f3
        return N * (f0 * self.e0 + f1 * self.e1 + f2 * self.e2 + f3 * self.e3 + f4 * self.e4)

    def _initial_params(self) -> list[float]:
        N  = np.sum(self.y)
        f0 = 0.32
        f1 = 0.12
        f2 = 0.03
        f3 = 0.01
        return [N, f0, f1, f2, f3]