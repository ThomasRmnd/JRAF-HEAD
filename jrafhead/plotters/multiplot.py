from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from config import (
    PROMPT_ENERGY_BINS_UNIFORM, 
    PROMPT_DELAYED_DT_BINS, 
    CUSTOM_LIGHTBLUE, 
    CUSTOM_PURPLE, 
    CUSTOM_ORANGE, 
    CUSTOM_MARKEDRED, 
)
from fits import (
    Li9He8FractionFitter, 
)


class Li9He8FractionPlotter:
    """
    Plot and fit the energy Li9/He8 prompt spectrum and the isotopes decay time.

    Parameters
    ----------
    """

    def __init__(
        self, 
        li9:   np.ndarray, 
        he8:   np.ndarray, 
        li9color: str                               = CUSTOM_LIGHTBLUE, 
        he8color: str                               = CUSTOM_PURPLE, 
        bkgcolor: str                               = CUSTOM_ORANGE, 
        ebins:  np.ndarray                          = PROMPT_ENERGY_BINS_UNIFORM, 
        dtbins: np.ndarray                          = PROMPT_DELAYED_DT_BINS, 
        exlim:  tuple[float, float] | None          = None, 
        eylim:  tuple[float, float] | None          = None, 
        dtxlim: tuple[float, float] | None          = None, 
        dtylim: tuple[float, float] | None          = None, 
        exlabel: str                                = r"$E_{p}$ (MeV)", 
        eylabel: str                                = r"Entries", 
        dtxlabel: str                               = r"\Delta t_{p-d} (s)", 
        dtylabel: str                               = r"Entries", 
        exscale: str                                = "linear", 
        eyscale: str                                = "linear", 
        dtxscale: str                               = "linear", 
        dtyscale: str                               = "linear", 
    ) -> None:
        self.li9        = li9
        self.he8        = he8
        self.li9color   = li9color
        self.he8color   = he8color
        self.bkgcolor   = bkgcolor
        self.ebins      = ebins
        self.dtbins     = dtbins
        self.ecenters   = (ebins[1:] + ebins[:-1]) / 2.0
        self.dtcenters  = (dtbins[1:] + dtbins[:-1]) / 2.0
        self.ewidths    = ebins[1:] - ebins[:-1]
        self.dtwidths   = dtbins[1:] - dtbins[:-1]
        self.exlim      = exlim
        self.eylim      = eylim
        self.dtxlim     = dtxlim
        self.dtylim     = dtylim
        self.exlabel    = exlabel
        self.eylabel    = eylabel
        self.dtxlabel   = dtxlabel
        self.dtylabel   = dtylabel
        self.exscale    = exscale 
        self.eyscale    = eyscale 
        self.dtxscale   = dtxscale 
        self.dtyscale   = dtyscale 

    def add(
        self,
        e:          np.ndarray, 
        dt:         np.ndarray, 
        linecolor:  str | None  = None,
        fillcolor:  str | None  = None,
        linestyle:  str         = "-", 
        label:      str | None  = None, 
    ) -> None:
        """
        Register one dataset to be plotter.

        Parameters
        ----------
        e : array-like
            Prompt energy.
        dt : array-like
            Prompt-delayed time difference.
        fillcolor : str or None
            If given, the histogram area is filled with this color rather than drawn as a step.
        label : str or None
            Legend label. Pass None to suppress this dataset's legend entry. 
        """
        self.ehist, _  = np.histogram(e, bins=self.ebins)
        self.dthist, _ = np.histogram(dt, bins=self.dtbins)
        self.eerr      = np.sqrt(self.ehist)
        self.dterr     = np.sqrt(self.dthist)
        self.linecolor = linecolor
        self.fillcolor = fillcolor
        self.linestyle = linestyle
        self.label     = label

    def plot(self) -> list[tuple[plt.Figure, plt.Axes]]:
        """
        Draw and fit the registered dataset on a new figure 
        and return [(fig1, ax1), (fig2, ax2), (fig3, ax3)].

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        fig3, ax3 = plt.subplots(figsize=(7, 6))

        # -------------------------------------------------
        # Fit
        # -------------------------------------------------

        hli9 = np.histogram(self.li9, bins=self.ebins)
        hhe8 = np.histogram(self.he8, bins=self.dtbins)
        
        fitter = Li9He8FractionFitter(
            hli9 / np.sum(hli9), 
            hhe8 / np.sum(hhe8), 
            self.ecenters, 
            self.ehist, 
            self.eerr, 
            self.dtcenters, 
            self.dthist, 
            self.dterr, 
        )
        result = fitter.fit()
        if result is None:
            return

        # -------------------------------------------------
        # Prompt energy plot
        # -------------------------------------------------

        fit_full = fitter.model_energy_full(self.ecenters, result.popt[0], result.popt[2])

        ax1.step(
            self.ebins, np.r_[hli9, hli9[-1]], 
            where="post", color=self.li9color, linewidth=1.8, zorder=4, 
            label=r"$^{9}$Li template", 
        )
        ax1.step(
            self.ebins, np.r_[hhe8, hhe8[-1]], 
            where="post", color=self.he8color, linewidth=1.8, zorder=4, 
            label=r"$^{8}$He template", 
        )
        ax1.step(
            self.ebins, np.r_[fit_full, fit_full[-1]], 
            where="post", color=CUSTOM_MARKEDRED, linewidth=1.8, zorder=4, 
            label=r"$^{9}$Li/$^{8}$He template", 
        )

        ax1.errorbar(
            self.ecenters, self.ehist,
            yerr=self.eerr, xerr=self.ewidths / 2,
            label=self.label,
            fmt="o", color=self.linecolor,
            markersize=4.5, zorder=3,
        )
    
        ax1.set_xlabel(self.exlabel)
        ax1.set_ylabel(self.eylabel)

        if self.exlim:
            ax1.set_xlim(*self.exlim)
        if self.eylim:
            ax2.set_ylim(*self.eylim)

        if self.exscale == "linear":
            ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        if self.eyscale == "linear":
            ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax1.set_xscale(self.exscale)
        ax1.set_yscale(self.eyscale)

        ax1.minorticks_on()
        ax1.tick_params(direction="in", which="both", top=True, right=True)

        # -------------------------------------------------
        # Prompt-delayed time difference
        # -------------------------------------------------

        hdtbkg = np.full(self.dtcenters.shape, result.popt[0] * result.popt[1])
        hdtli9 = result.popt[0] * (1.0 - result.popt[1]) * result.popt[2] * np.exp(-self.dtcenters / result.popt[3])
        hdthe8 = result.popt[0] * (1.0 - result.popt[1]) * (1.0 - result.popt[2]) * np.exp(-self.dtcenters / result.popt[4])
        fit_full = hdtbkg + hdtli9 + hdthe8

        ax2.step(
            self.dtbins, np.r_[hdtbkg, hdtbkg[-1]], 
            where="post", color=self.bkgcolor, linewidth=1.8, zorder=4, 
            label=r"Background", 
        )
        ax2.step(
            self.ebins, np.r_[hdtli9, hdtli9[-1]], 
            where="post", color=self.li9color, linewidth=1.8, zorder=4, 
            label=r"$^{9}$Li template", 
        )
        ax2.step(
            self.ebins, np.r_[hdthe8, hdthe8[-1]], 
            where="post", color=self.he8color, linewidth=1.8, zorder=4, 
            label=r"$^{8}$He template", 
        )
        ax2.step(
            self.ebins, np.r_[fit_full, fit_full[-1]], 
            where="post", color=CUSTOM_MARKEDRED, linewidth=1.8, zorder=4, 
            label=r"$^{9}$Li/$^{8}$He template", 
        )

        ax2.errorbar(
            self.dtcenters, self.dthist,
            yerr=self.dterr, xerr=self.dtwidths / 2,
            label=self.label,
            fmt="o", color=self.linecolor,
            markersize=4.5, zorder=3,
        )

        ax2.set_xlabel(self.dtxlabel)
        ax2.set_ylabel(self.dtylabel)

        if self.dtxlim:
            ax2.set_xlim(*self.dtxlim)
        if self.ylim:
            ax2.set_ylim(*self.dtylim)

        if self.dtxscale == "linear":
            ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        if self.dtyscale == "linear":
            ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax2.set_xscale(self.dtxscale)
        ax2.set_yscale(self.dtyscale)

        ax2.minorticks_on()
        ax2.tick_params(direction="in", which="both", top=True, right=True)

        return [(fig1, ax1), (fig2, ax2), (fig3, ax3)]