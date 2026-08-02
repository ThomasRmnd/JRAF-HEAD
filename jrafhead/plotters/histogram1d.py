from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from config import (
    BLACK,
    CUSTOM_MARKEDRED,
    DELAYED_ENERGY_BINS,
    DELAYED_ENERGY_HYDROGEN_BINS,
    MUON_PERFORMANCE_ANGLE_BINS,
    MUON_PERFORMANCE_DISTANCE_BINS,
    PROMPT_DELAYED_DR_BINS,
    PROMPT_DELAYED_DT_BINS,
    PROMPT_ENERGY_BINS_207DAYS,
    PROMPT_ENERGY_BINS_NMO,
    PROMPT_ENERGY_BINS_UNIFORM,
)
from loader import (
    MCChengzhuoTemplateHistogram,
    MCGroupCTemplateHistogram,
)
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
from utils import (
    uniform_bins,
)

from .base import BasePlotter

# -------------------------------------------------------------------------------------------------
# Dataset type alias
# -------------------------------------------------------------------------------------------------

_Dataset = dict[str, Any]


# -------------------------------------------------------------------------------------------------
# Generic 1D histogram plotter
# -------------------------------------------------------------------------------------------------

class Histogram1DPlotter(BasePlotter):
    """
    Generic 1D histogram plotter supporting overlaid datasets, a difference overlay, and an 
    optional MC spectrum overlay.

    Parameters
    ----------
    bins : np.ndarray
        Bin edges array of length n_bins + 1.
    legend_ncol: int
        Number of columns in the legend.
    legend_loc: str
        Placement of the legend in the plot.
    enable_fit: bool
        Boolean setting or not the fitting.
    **kwargs
        Forwarded to "BasePlotter.__init__".
    """

    def __init__(
        self,
        bins:           np.ndarray, 
        legend_ncol:    int = 1, 
        legend_loc:     str = "upper right", 
        show_errors:    bool = True, 
        **kwargs:       Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bins                       = bins
        self.centers                    = 0.5 * (bins[1:] + bins[:-1])
        self.widths                     = bins[1:] - bins[:-1]
        self.datasets: list[_Dataset]   = []
        self.legend_ncol                = legend_ncol
        self.legend_loc                 = legend_loc
        self.show_errors                = show_errors

    # ---------------------------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------------------------

    def add(
        self,
        data:       np.ndarray,
        linecolor:  str,
        fillcolor:  str | None = None,
        linestyle:  str | None = None,
        label:      str | None = None,
    ) -> None:
        """
        Histogram data into the internal bin edges and register it.

        Parameters
        ----------
        data : np.ndarray, shape (N,)
            Raw event values to histogram.
        linecolor : str
            Hex color for the step line and error bars.
        fillcolor : str or None
            If given, the histogram area is filled with this color rather than drawn as a step.
        label : str or None
            Legend label. Pass None to suppress this dataset's legend entry.
        """
        hist, _ = np.histogram(data, bins=self.bins)
        self.add_histogram(
            hist, 
            np.sqrt(hist),
            linecolor, 
            fillcolor=fillcolor, 
            linestyle=linestyle, 
            label=label
        )
    
    def add_diff(
        self,
        data1:      np.ndarray,
        data2:      np.ndarray,
        linecolor:  str,
        fillcolor:  str | None = None,
        linestyle:  str | None = None,
        label:      str | None = None,
    ) -> None:
        """
        Substract the histogram from two dataset into the internal bin edges and register it.

        Parameters
        ----------
        data1 : np.ndarray, shape (N1,)
            Raw event values from the first dataset.
        data2 : np.ndarray, shape (N2,)
            Raw event values from the second dataset.
        linecolor : str
            Hex color for the step line and error bars.
        fillcolor : str or None
            If given, the histogram area is filled with this color rather than drawn as a step.
        label : str or None
            Legend label. Pass None to suppress this dataset's legend entry.
        """
        hist1, _ = np.histogram(data1, bins=self.bins)
        hist2, _ = np.histogram(data2, bins=self.bins)

        # err1 = np.sqrt(hist1)
        # err2 = np.sqrt(hist2)

        hist = hist1 - hist2
        err = np.sqrt(hist1 + hist2)
        # err  = np.sqrt(err1 ** 2 + err2 ** 2)

        self.add_histogram(
            hist,
            err,
            linecolor,
            fillcolor=fillcolor,
            linestyle=linestyle,
            label=label
        )

    def add_histogram(
        self,
        hist:       np.ndarray,
        err:        np.ndarray,
        linecolor:  str,
        fillcolor:  str | None = None,
        linestyle:  str | None = None,
        label:      str | None = None,
    ) -> None:
        """
        Register an already binned histogram.

        Parameters
        ----------
        hist : np.ndarray
            Bin of the histogram.
        err : np.ndarray
            Uncertainties of the histogram
        linecolor : str
            Hex color for the step line and error bars.
        fillcolor : str or None
            If given, the histogram area is filled with this color rather than drawn as a step.
        label : str or None
            Legend label. Pass None to suppress this dataset's legend entry.
        """
        self.datasets.append({
            "hist":         hist,
            "err":          err,
            "linecolor":    linecolor,
            "fillcolor":    fillcolor,
            "linestyle":    linestyle,
            "label":        label,
        })

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Draw all registered datasets on a new figure and return (fig, ax).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        for d in self.datasets:
            self._draw_dataset(ax, d)

        self.apply_style(ax)
        if self.enable_fit:
            self._maybe_fit(ax)
            if self.draw_fit_info and self.fit_result is not None:
                text = self._get_fit_info_text()
                ax.text(
                    *self.fit_info_loc, text,
                    transform=ax.transAxes,
                    fontsize=self.fit_info_fontsize,
                    va=self.fit_info_anchor.split(' ')[0], 
                    ha=self.fit_info_anchor.split(' ')[1], 
                )

        if any(d["label"] for d in self.datasets):
            ax.legend(loc=self.legend_loc, ncol=self.legend_ncol)

        fig.tight_layout()
        return fig, ax

    # ---------------------------------------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------------------------------------

    def _draw_dataset(self, ax: plt.Axes, d: _Dataset) -> None:
        """
        Draw one dataset as a filled area or a stepped line + error bars.
        """
        if self.show_errors:
            ax.errorbar(
                self.centers, d["hist"],
                yerr=d["err"], xerr=self.widths / 2,
                label=d["label"],
                fmt="o", color=d["linecolor"],
                markersize=4.5, zorder=3,
            )

        if d["fillcolor"]:
            ax.fill_between(
                self.bins,
                np.r_[d["hist"], d["hist"][-1]],
                step="post",
                color=d["fillcolor"],
                zorder=1,
                alpha=0.15, 
            )
        else:
            ax.step(
                self.bins,
                np.r_[d["hist"], d["hist"][-1]],
                where="post",
                color=d["linecolor"],
                linestyle=d["linestyle"],
                linewidth=1.2,
                label=d["label"] if not self.show_errors else None,
                zorder=2,
            )

    # ---------------------------------------------------------------------------------------------
    # Hooks - overridden by subclasses
    # ---------------------------------------------------------------------------------------------

    def _maybe_fit(self, ax: plt.Axes) -> None:
        """
        Hook for subclass-specific fits. No-op in the base class.
        """
        pass

    def _get_fit_info_text(self) -> None:
        """
        Hook for sublclass specific fits. No-op in the base class.
        """


# -------------------------------------------------------------------------------------------------
# Prompt energy spectrum
# -------------------------------------------------------------------------------------------------

class PromptEnergyPlotter(Histogram1DPlotter):
    """
    Prompt energy spectrum E_p plotter.

    Parameters
    ----------
    binmode : {"nmo", "207days", "normal"}
        "nmo"     - non-uniform NMO analysis bins.
        "207days" - non-uniform 207 days-like bins.
        "normal"  - uniform 100-bin grid (0–12 MeV).
    **kwargs
        Forwarded to "Histogram1DPlotter.__init__".
    """

    def __init__(self, binmode: str = "nmo", **kwargs: Any) -> None:
        if binmode == "nmo":
            bins = PROMPT_ENERGY_BINS_NMO
        elif binmode == "207days":
            bins = PROMPT_ENERGY_BINS_207DAYS
        else:
            bins = PROMPT_ENERGY_BINS_UNIFORM
        super().__init__(
            bins   = bins,
            xlabel = r"$E_{p}$ (MeV)",
            ylabel = "Entries",
            xlim   = (0, 12.5),
            **kwargs,
        )


class RelativeUncertaintyPromptEnergyPlotter(Histogram1DPlotter):
    """
    Relative uncertainty prompt energy spectrum E_p plotter.

    Parameters
    ----------
    binmode : {"nmo", "207days", "normal"}
        "nmo"     - non-uniform NMO analysis bins.
        "207days" - non-uniform 207 days-like bins.
        "normal"  - uniform 100-bin grid (0–12 MeV).
    **kwargs
        Forwarded to "Histogram1DPlotter.__init__".
    """

    def __init__(
        self, 
        binmode:        str = "nmo", 
        **kwargs:       Any
    ) -> None:
        if binmode == "nmo":
            bins = PROMPT_ENERGY_BINS_NMO
        elif binmode == "207days":
            bins = PROMPT_ENERGY_BINS_207DAYS
        else:
            bins = PROMPT_ENERGY_BINS_UNIFORM
        super().__init__(
            bins        = bins,
            xlabel      = r"$E_{p}$ (MeV)", 
            xlim        = (bins[0], bins[-1]), 
            ylabel      = r"Relative uncertainty (\%)", 
            ylim        = (0.0, 50.0), 
            show_errors = False, 
            enable_fit  = False, 
            **kwargs,
        )


# -------------------------------------------------------------------------------------------------
# Prompt energy spectrum
# -------------------------------------------------------------------------------------------------

class Li9He8ShapeGroupCFitPlotter(PromptEnergyPlotter):
    """
    Prompt energy plotter for the Li9/He8 shape analysis.  Draws the
    signal-background difference spectrum, fits a fixed-shape MC template
    to it with a single free amplitude N, and shows the fit residual (pull) 
    in a bottom panel.

    Layout (GridSpec 1 x 2)
    ---------------------
    ┌──────────┐
    │  main    │
    ├──────────┤
    │  resi    │
    └──────────┘
    """

    def __init__(self, template: MCGroupCTemplateHistogram, **kwargs):
        super().__init__(**kwargs)
        self._template   = template
        self._fit_result = None
        self._y_fit      = None

    def plot(self) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Draw all registered datasets on a new figure and return (fig, ax).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if len(self.datasets) != 1:
            return
        fig = plt.figure(figsize=(7, 8))
        gs = GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.1)
        ax_main = fig.add_subplot(gs[0])
        ax_resi = fig.add_subplot(gs[1], sharex=ax_main)

        d = self.datasets[0]
        self._draw_dataset(ax_main, d)

        self._maybe_fit(ax_main)
        if self._fit_result is not None:
            self._draw_fit_overlay(ax_main)
            self._draw_fit_text(ax_main)
            self._draw_residuals(ax_resi)

        self.apply_style(ax_main)
        ax_main.set_xlabel("")
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if any(ds["label"] for ds in self.datasets) or self._fit_result is not None:
            ax_main.legend(loc="upper right")

        self._style_residual_axis(ax_resi)
        fig.align_ylabels([ax_main, ax_resi])

        return fig, {"main": ax_main, "resi": ax_resi}

    def _maybe_fit(self, ax) -> None:
        from fits import TemplateAmplitudeFitter
        from utils import rebin_histogram

        hist       = self.datasets[0]["hist"]
        err        = self.datasets[0]["err"]
        histrb     = rebin_histogram(self._template.edges, self._template.counts, self.bins)

        fitter = TemplateAmplitudeFitter(histrb, self.bins, hist, err)
        self._fit_result = fitter.fit()
        if self._fit_result is None:
            return

        self._y_fit = fitter.model_full(*self._fit_result.popt)

    def _draw_fit_overlay(self, ax) -> None:
        ax.step(
            self.bins, np.r_[self._y_fit, self._y_fit[-1]],
            where="post", color=CUSTOM_MARKEDRED, linewidth=1.8, zorder=4,
            label=r"Template",
        )

    def _draw_fit_text(self, ax) -> None:
        chi2v, ndf, p_chi2 = self._shape_test()
        text = (
            r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d) = %.3f$" "\n"
            r"$N = %.2f \pm %.2f$"
        ) % (
            chi2v, ndf, p_chi2, 
            self._fit_result.popt[0], self._fit_result.perr[0], 
        )
        ax.text(
            0.42, 0.10, text,
            transform=ax.transAxes,
            fontsize=18,
            ha="center", va="center",
        )

    def _shape_test(self):
        from scipy.stats import chi2 as chi2_dist

        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        model = self._y_fit

        valid = err > 0

        chi2v = np.sum(((hist[valid] - model[valid]) / err[valid]) ** 2)
        ndf = np.count_nonzero(valid) - 1
        p_chi2 = chi2_dist.sf(chi2v, df=ndf)

        # No KS test for now
        # cdf_h = np.cumsum(hist / np.sum(hist))
        # cdf_t = np.cumsum(model / np.sum(model))
        # Dn = np.max(np.abs(cdf_h - cdf_t))
        # p_ks = kstwobign.sf(Dn * np.sqrt(np.sum(hist)))

        return chi2v, ndf, p_chi2
    
    def _draw_residuals(self, ax) -> None:
        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        pull = np.full(err.shape, np.nan)
        mask = err > 0
        pull[mask] = (hist[mask] - self._y_fit[mask]) / err[mask]

        ax.plot(self.centers, pull, "o", color=BLACK, markersize=4.5, zorder=3)
        ax.axhline(0.0, color=CUSTOM_MARKEDRED, linewidth=1.5, linestyle="--", zorder=2)
        ax.axhspan(-1.0, 1.0, color=CUSTOM_MARKEDRED, alpha=0.25, zorder=1)
        ax.axhspan(-2.0, 2.0, color=CUSTOM_MARKEDRED, alpha=0.10, zorder=0)

    def _style_residual_axis(self, ax) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"Resi. ($\sigma$)")
        ax.minorticks_on()
        ax.set_xscale(self.xscale)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(plt.FixedLocator([-3, -2, -1, 0, 1, 2, 3]))
        ax.yaxis.set_major_formatter(plt.FixedFormatter(["", "-2", "", "0", "", "2", ""]))
        ax.tick_params(axis="x", labelbottom=True, direction="in", which="both")
        ax.tick_params(axis="y", direction="in")
        if self.xlim:
            ax.set_xlim(*self.xlim)
        ax.set_ylim(-3.0, 3.0)


class Li9He8ChengzhuoFitPlotter(PromptEnergyPlotter):
    """
    Prompt energy plotter for the Li9/He8 shape analysis.  Draws the
    signal-background difference spectrum, fits a fixed-shape MC template
    to it with a single free amplitude N, and shows the fit residual (pull) 
    in a bottom panel.

    Layout (GridSpec 1 x 2)
    ---------------------
    ┌──────────┐
    │  main    │
    ├──────────┤
    │  resi    │
    └──────────┘
    """

    def __init__(self, template: MCChengzhuoTemplateHistogram, **kwargs):
        super().__init__(**kwargs)
        self._template   = template
        self._fit_result = None
        self._y_fit      = None

    def plot(self) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Draw all registered datasets on a new figure and return (fig, ax).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if len(self.datasets) != 1:
            return
        fig = plt.figure(figsize=(7, 8))
        gs = GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.1)
        ax_main = fig.add_subplot(gs[0])
        ax_resi = fig.add_subplot(gs[1], sharex=ax_main)

        d = self.datasets[0]
        self._draw_dataset(ax_main, d)

        self._maybe_fit(ax_main)
        if self._fit_result is not None:
            self._draw_fit_overlay(ax_main)
            self._draw_fit_text(ax_main)
            self._draw_residuals(ax_resi)

        self.apply_style(ax_main)
        ax_main.set_xlabel("")
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if any(ds["label"] for ds in self.datasets) or self._fit_result is not None:
            ax_main.legend(loc="upper right")

        self._style_residual_axis(ax_resi)
        fig.align_ylabels([ax_main, ax_resi])

        return fig, {"main": ax_main, "resi": ax_resi}

    def _maybe_fit(self, ax) -> None:
        from fits import TemplateAmplitudeFitter
        from utils import rebin_histogram

        hist       = self.datasets[0]["hist"]
        err        = self.datasets[0]["err"]
        histrb     = rebin_histogram(self._template.all.edges, self._template.all.counts, self.bins)
        histrb     /= np.sum(histrb)

        fitter = TemplateAmplitudeFitter(histrb, self.bins, hist, err)
        self._fit_result = fitter.fit()
        if self._fit_result is None:
            return

        self._y_fit = fitter.model_full(*self._fit_result.popt)

    def _draw_fit_overlay(self, ax) -> None:
        ax.step(
            self.bins, np.r_[self._y_fit, self._y_fit[-1]],
            where="post", color=CUSTOM_MARKEDRED, linewidth=1.8, zorder=4,
            label=r"Template",
        )

    def _draw_fit_text(self, ax) -> None:
        chi2v, ndf, p_chi2 = self._shape_test()
        text = (
            r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d) = %.3f$" "\n"
            r"$N = %.2f \pm %.2f$"
        ) % (
            chi2v, ndf, p_chi2, 
            self._fit_result.popt[0], self._fit_result.perr[0], 
        )
        ax.text(
            0.42, 0.10, text,
            transform=ax.transAxes,
            fontsize=18,
            ha="center", va="center",
        )

    def _shape_test(self):
        from scipy.stats import chi2 as chi2_dist

        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        model = self._y_fit

        valid = err > 0

        chi2v = np.sum(((hist[valid] - model[valid]) / err[valid]) ** 2)
        ndf = np.count_nonzero(valid) - 1
        p_chi2 = chi2_dist.sf(chi2v, df=ndf)

        return chi2v, ndf, p_chi2
    
    def _draw_residuals(self, ax) -> None:
        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        pull = np.full(err.shape, np.nan)
        mask = err > 0
        pull[mask] = (hist[mask] - self._y_fit[mask]) / err[mask]

        ax.plot(self.centers, pull, "o", color=BLACK, markersize=4.5, zorder=3)
        ax.axhline(0.0, color=CUSTOM_MARKEDRED, linewidth=1.5, linestyle="--", zorder=2)
        ax.axhspan(-1.0, 1.0, color=CUSTOM_MARKEDRED, alpha=0.25, zorder=1)
        ax.axhspan(-2.0, 2.0, color=CUSTOM_MARKEDRED, alpha=0.10, zorder=0)

    def _style_residual_axis(self, ax) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"Resi. ($\sigma$)")
        ax.minorticks_on()
        ax.set_xscale(self.xscale)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(plt.FixedLocator([-3, -2, -1, 0, 1, 2, 3]))
        ax.yaxis.set_major_formatter(plt.FixedFormatter(["", "-2", "", "0", "", "2", ""]))
        ax.tick_params(axis="x", labelbottom=True, direction="in", which="both")
        ax.tick_params(axis="y", direction="in")
        if self.xlim:
            ax.set_xlim(*self.xlim)
        ax.set_ylim(-3.0, 3.0)


class Li9He8ChengzhuoFitSmearingPlotter(PromptEnergyPlotter):
    """
    Prompt energy plotter for the Li9/He8 shape analysis.  Draws the
    signal-background difference spectrum, fits a fixed-shape MC template
    to it with a single free amplitude N, and shows the fit residual (pull) 
    in a bottom panel.

    Layout (GridSpec 1 x 2)
    ---------------------
    ┌──────────┐
    │  main    │
    ├──────────┤
    │  resi    │
    └──────────┘
    """

    def __init__(self, template: MCChengzhuoTemplateHistogram, **kwargs):
        super().__init__(**kwargs)
        self._template   = template
        self._fit_result = None
        self._y_fit      = None

    def plot(self) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Draw all registered datasets on a new figure and return (fig, ax).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if len(self.datasets) != 1:
            return
        fig = plt.figure(figsize=(7, 8))
        gs = GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.1)
        ax_main = fig.add_subplot(gs[0])
        ax_resi = fig.add_subplot(gs[1], sharex=ax_main)

        d = self.datasets[0]
        self._draw_dataset(ax_main, d)

        self._maybe_fit(ax_main)
        if self._fit_result is not None:
            self._draw_fit_overlay(ax_main)
            self._draw_fit_text(ax_main)
            self._draw_residuals(ax_resi)

        self.apply_style(ax_main)
        ax_main.set_xlabel("")
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if any(ds["label"] for ds in self.datasets) or self._fit_result is not None:
            ax_main.legend(loc="upper right")

        self._style_residual_axis(ax_resi)
        fig.align_ylabels([ax_main, ax_resi])

        return fig, {"main": ax_main, "resi": ax_resi}

    def _maybe_fit(self, ax) -> None:
        from fits import Li9He8ChengzhuoFitter
        from utils import rebin_histogram

        hist       = self.datasets[0]["hist"]
        err        = self.datasets[0]["err"]
        integr0    = np.sum(self._template.branch0.counts * np.diff(self._template.branch0.edges))
        histrb0    = rebin_histogram(self._template.branch0.edges, self._template.branch0.counts, self.bins)
        histrb0    *= integr0 / np.sum(histrb0 * self.widths)
        # print(f"[Integral0]: Old = {integr0}, New = {np.sum(histrb0 * (self.bins[1:] - self.bins[:-1]))}")
        integr1    = np.sum(self._template.branch1.counts * np.diff(self._template.branch1.edges))
        histrb1    = rebin_histogram(self._template.branch1.edges, self._template.branch1.counts, self.bins)
        histrb1    *= integr1 / np.sum(histrb1 * self.widths)
        # print(f"[Integral1]: Old = {integr1}, New = {np.sum(histrb1 * (self.bins[1:] - self.bins[:-1]))}")
        integr2    = np.sum(self._template.branch2.counts * np.diff(self._template.branch2.edges))
        histrb2    = rebin_histogram(self._template.branch2.edges, self._template.branch2.counts, self.bins)
        histrb2    *= integr2 / np.sum(histrb2 * self.widths)
        # print(f"[Integral2]: Old = {integr2}, New = {np.sum(histrb2 * (self.bins[1:] - self.bins[:-1]))}")
        integr3    = np.sum(self._template.branch3.counts * np.diff(self._template.branch3.edges))
        histrb3    = rebin_histogram(self._template.branch3.edges, self._template.branch3.counts, self.bins)
        histrb3    *= integr3 / np.sum(histrb3 * self.widths)
        # print(f"[Integral3]: Old = {integr3}, New = {np.sum(histrb3 * (self.bins[1:] - self.bins[:-1]))}")
        integr4    = np.sum(self._template.branch4.counts * np.diff(self._template.branch4.edges))
        histrb4    = rebin_histogram(self._template.branch4.edges, self._template.branch4.counts, self.bins)
        histrb4    *= integr4 / np.sum(histrb4 * self.widths)
        # print(f"[Integral4]: Old = {integr4}, New = {np.sum(histrb4 * (self.bins[1:] - self.bins[:-1]))}")

        fitter = Li9He8ChengzhuoFitter(
            histrb0, histrb1, histrb2, histrb3, histrb4, 
            self.bins, hist, err, 
        )
        self._fit_result = fitter.fit()
        if self._fit_result is None:
            return

        self._y_fit = fitter.model_full(self.centers, *self._fit_result.popt)

    def _draw_fit_overlay(self, ax) -> None:
        ax.step(
            self.bins, np.r_[self._y_fit, self._y_fit[-1]],
            where="post", color=CUSTOM_MARKEDRED, linewidth=1.8, zorder=4,
            label=r"Template",
        )

    def _draw_fit_text(self, ax) -> None:
        chi2v, ndf, p_chi2 = self._shape_test()

        N,  f0,  f1,  f2,  f3  = self._fit_result.popt
        Ne, f0e, f1e, f2e, f3e = self._fit_result.perr

        f4 = 0.508 - f0 - f1 - f2 - f3
        f4e = np.sqrt(
            f0e**2 +
            f1e**2 +
            f2e**2 +
            f3e**2
        )

        text = (
            r"$N = %.0f \pm %.0f$" "\n"
            r"$f_{0} = %.3f \pm %.3f$" "\n"
            r"$f_{1} = %.3f \pm %.3f$" "\n"
            r"$f_{2} = %.3f \pm %.3f$" "\n"
            r"$f_{3} = %.3f \pm %.3f$" "\n"
            r"$f_{4} = %.3f \pm %.3f$" "\n"
            r"$P(\chi^{2}/\mathrm{ndf}=%.1f/%d)=%.3f$"
        ) % (
            N, Ne,
            f0, f0e,
            f1, f1e,
            f2, f2e,
            f3, f3e,
            f4, f4e,
            chi2v, ndf, p_chi2,
        )

        ax.text(
            0.42,
            0.05,
            text,
            transform=ax.transAxes,
            fontsize=18,
            ha="center",
            va="bottom",
        )

    def _shape_test(self):
        from scipy.stats import chi2 as chi2_dist

        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        model = self._y_fit

        valid = err > 0

        chi2v = np.sum(((hist[valid] - model[valid]) / err[valid]) ** 2)
        ndf = np.count_nonzero(valid) - 1
        p_chi2 = chi2_dist.sf(chi2v, df=ndf)

        return chi2v, ndf, p_chi2
    
    def _draw_residuals(self, ax) -> None:
        d = self.datasets[0]
        hist, err = d["hist"], d["err"]
        pull = np.full(err.shape, np.nan)
        mask = err > 0
        pull[mask] = (hist[mask] - self._y_fit[mask]) / err[mask]

        ax.plot(self.centers, pull, "o", color=BLACK, markersize=4.5, zorder=3)
        ax.axhline(0.0, color=CUSTOM_MARKEDRED, linewidth=1.5, linestyle="--", zorder=2)
        ax.axhspan(-1.0, 1.0, color=CUSTOM_MARKEDRED, alpha=0.25, zorder=1)
        ax.axhspan(-2.0, 2.0, color=CUSTOM_MARKEDRED, alpha=0.10, zorder=0)

    def _style_residual_axis(self, ax) -> None:
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(r"Resi. ($\sigma$)")
        ax.minorticks_on()
        ax.set_xscale(self.xscale)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(plt.FixedLocator([-3, -2, -1, 0, 1, 2, 3]))
        ax.yaxis.set_major_formatter(plt.FixedFormatter(["", "-2", "", "0", "", "2", ""]))
        ax.tick_params(axis="x", labelbottom=True, direction="in", which="both")
        ax.tick_params(axis="y", direction="in")
        if self.xlim:
            ax.set_xlim(*self.xlim)
        ax.set_ylim(-3.0, 3.0)



# -------------------------------------------------------------------------------------------------
# Delayed energy spectrum (with Gaussian fit)
# -------------------------------------------------------------------------------------------------

class DelayedEnergyPlotter(Histogram1DPlotter):
    """
    Delayed energy spectrum.

    When exactly one dataset is registered, "_maybe_fit" overlays a Gaussian
    fit and displays \chi^2/ndf, p-value, and fit parameters as a text box.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("bins",   DELAYED_ENERGY_BINS)
        kwargs.setdefault("xlim",   (1.98, 5.52))
        kwargs.setdefault("xlabel", r"$E_{d}$ (MeV)")
        kwargs.setdefault("ylabel", r"Entries")
        super().__init__(**kwargs)

    def _maybe_fit(self, ax: plt.Axes) -> None:
        if len(self.datasets) != 1:
            return

        from fits.functions import GaussianFitter

        hist       = self.datasets[0]["hist"]
        err        = self.datasets[0]["err"]

        mask  = (2.0 <= self.bins) & (self.bins <= 2.5)
        hmask = mask[:-1] & mask[1:]

        fitter = GaussianFitter(self.bins[mask], hist[hmask], err[hmask])
        self.fit_result = fitter.fit()
        if self.fit_result is None:
            return

        x_smooth = np.linspace(2.0, 2.5, 500)
        y_smooth = fitter.model(x_smooth, *self.fit_result.popt)

        ax.plot(
            x_smooth, y_smooth,
            linestyle="--", linewidth=1.8, color=CUSTOM_MARKEDRED, zorder=4,
        )

    def _get_fit_info_text(self):
        text = (
            r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d)  = %.3f$" "\n\n"
            r"$A = %.1f \pm %.1f$" "\n"
            r"$\mu = %.4f \pm %.4f~\mathrm{MeV}$" "\n"
            r"$\sigma = %.4f \pm %.4f~\mathrm{MeV}$"
        ) % (
            self.fit_result.chi2, self.fit_result.ndf, self.fit_result.pvalue,
            self.fit_result.popt[0], self.fit_result.perr[0],
            self.fit_result.popt[1], self.fit_result.perr[1],
            self.fit_result.popt[2], self.fit_result.perr[2],
        )
        return text


class DelayedEnergyHydrogenPlotter(Histogram1DPlotter):
    """
    Delayed energy spectrum.

    When exactly one dataset is registered, "_maybe_fit" overlays a Gaussian
    fit and displays \chi^2/ndf, p-value, and fit parameters as a text box.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            bins   = DELAYED_ENERGY_HYDROGEN_BINS,
            xlabel = r"$E_{d}$ (MeV)",
            ylabel = "Entries",
            xlim   = (1.98, 2.52),
            **kwargs,
        )


# -------------------------------------------------------------------------------------------------
# Prompt–delayed time difference (with exponential fit)
# -------------------------------------------------------------------------------------------------

class PromptDelayedTimePlotter(Histogram1DPlotter):
    """
    Prompt–delayed time difference \Delta t_{p-d} in milliseconds.

    When exactly one dataset is registered, "_maybe_fit" overlays a simple
    exponential A * exp(-t/\tau) fitted for t > 0.025 ms to avoid trigger dead-time
    effects. \tau is reported in \mus in the text box.
    """

    def __init__(
        self, 
        bins:           np.ndarray                  = PROMPT_DELAYED_DT_BINS, 
        xlim:           tuple[float, float] | None  = (0, 1.02), 
        accidental_fit: bool                        = False, 
        **kwargs:       Any
    ) -> None:
        super().__init__(
            bins   = bins,
            xlabel = r"$\Delta t_{p-d}$ (ms)",
            ylabel = "Entries",
            xlim   = xlim,
            **kwargs,
        )
        self.accidental_fit = accidental_fit

    def _maybe_fit(self, ax: plt.Axes) -> None:
        if len(self.datasets) != 1:
            return

        from fits import ConstantFitter, ExponentialFitter

        hist            = self.datasets[0]["hist"]
        err             = self.datasets[0]["err"]
        threshold       = 0.005 # us
        first_fit_bin   = np.searchsorted(self.bins, threshold, side="left")
        t_min_fit       = self.bins[first_fit_bin]

        if self.accidental_fit:
            fitter = ConstantFitter(self.bins, hist, err, xlim=(t_min_fit, None))
        else:
            fitter = ExponentialFitter(self.bins, hist, err, xlim=(t_min_fit, None))
        self.fit_result = fitter.fit()
        if self.fit_result is None:
            return

        x_smooth = uniform_bins(t_min_fit, self.bins[-1], 500)
        y_smooth = fitter.model(x_smooth, *self.fit_result.popt)

        ax.plot(
            x_smooth, y_smooth,
            linestyle="--", linewidth=1.8, color=CUSTOM_MARKEDRED, zorder=4,
        )

    def _get_fit_info_text(self):
        if self.accidental_fit:
            A     = self.fit_result.popt
            A_err = self.fit_result.perr
            text = (
                r"$P(\chi^2/\mathrm{ndf} = %.1f / %d) = %.3f$" "\n\n"
                r"$A = %.1f \pm %.1f$"
            ) % (
                self.fit_result.chi2, self.fit_result.ndf, self.fit_result.pvalue,
                A, A_err,
            )
        else:
            A, tau         = self.fit_result.popt
            A_err, tau_err = self.fit_result.perr
            text = (
                r"$P(\chi^2/\mathrm{ndf} = %.1f / %d) = %.3f$" "\n\n"
                r"$A = %.1f \pm %.1f$" "\n"
                r"$\tau = %.1f \pm %.1f~\mu\mathrm{s}$"
            ) % (
                self.fit_result.chi2, self.fit_result.ndf, self.fit_result.pvalue,
                A, A_err,
                tau * 1e3, tau_err * 1e3,   # ms ==> \mus
            )
        return text


# -------------------------------------------------------------------------------------------------
# Prompt–delayed spatial distance (no fit)
# -------------------------------------------------------------------------------------------------

class PromptDelayedDistancePlotter(Histogram1DPlotter):
    """
    Prompt–delayed vertex distance \Delta r_{p-d} in metres.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            bins    = PROMPT_DELAYED_DR_BINS,
            xlabel  = r"$\Delta r_{p-d}$ (m)",
            ylabel  = "Entries",
            xlim    = (0, 1.55),
            **kwargs,
        )


# -------------------------------------------------------------------------------------------------
# 9Li/8He rate estimation ploter performance metric
# -------------------------------------------------------------------------------------------------

class Li9He8RateEstimationPlotter(Histogram1DPlotter):
    """
    9Li/8He rate estimation via time to last muon with a spallation neutron distribution.

    When exactly one dataset is registered, "_maybe_fit" overlays the triple exponential decay fit
    and displays \chi^2/ndf, p-value, and fit parameters as a text box.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("bins",   uniform_bins(0.0, 5.0, 100))
        kwargs.setdefault("xlim",   (0.0, 5.0))
        kwargs.setdefault("xlabel", r"$\Delta t_{\mu-p}$ (s)")
        kwargs.setdefault("ylabel", r"Entries")
        super().__init__(**kwargs)

    def _maybe_fit(self, ax: plt.Axes) -> None:
        if len(self.datasets) != 1:
            return

        from fits import Li9He8RateEstimationFitter

        hist      = self.datasets[0]["hist"]
        err       = self.datasets[0]["err"]
        threshold       = 0.007 # ms
        first_fit_bin   = np.searchsorted(self.bins, threshold, side="left")
        t_min_fit       = self.bins[first_fit_bin]

        fitter = Li9He8RateEstimationFitter(self.bins, hist, err, xlim=(t_min_fit, self.bins[-1]))
        self.fit_result = fitter.fit()
        if self.fit_result is None:
            return
        # from fits import FitResult
        # self.fit_result = FitResult(
        #     popt=fitter._initial_params(),
        #     perr=fitter._initial_params(),
        #     chi2=0.0,
        #     ndf=1,
        #     pvalue=0.0,
        # )
        
        from config import CUSTOM_BLUE, CUSTOM_DARKPINK, CUSTOM_MARKEDRED, CUSTOM_ORANGE

        # N9li8he, f9li, t9li, t8he, Nbkg, Rmu =  self.fit_result.popt
        N9li8he, Nbkg, Rmu = self.fit_result.popt
        t9li8he = 0.256

        bins_smooth   = np.linspace(t_min_fit, self.bins[-1], 501)
        x_smooth      = 0.5 * (bins_smooth[1:] + bins_smooth[:-1])
        fitter_smooth = Li9He8RateEstimationFitter(bins_smooth, np.zeros_like(x_smooth), np.ones_like(x_smooth))
        # ytot_smooth    = fitter.model(x_smooth, N9li8he, f9li, t9li, t8he, Nbkg, Rmu)
        # y9li_smooth    = fitter.model_9li(x_smooth, N9li8he, f9li, t9li, Rmu)
        # y8he_smooth    = fitter.model_8he(x_smooth, N9li8he, f9li, t8he, Rmu)
        # ybkg_smooth    = fitter.model_bkg(x_smooth, Nbkg, Rmu)
        ytot_smooth    = fitter_smooth.model(x_smooth, N9li8he, Nbkg, Rmu)
        y9li8he_smooth = fitter_smooth.model_9li8he(x_smooth, N9li8he, t9li8he, Rmu)
        ybkg_smooth    = fitter_smooth.model_bkg(x_smooth, Nbkg, Rmu)

        ax.plot(
            x_smooth, ytot_smooth,
            linestyle="--", linewidth=1.8, color=CUSTOM_MARKEDRED, zorder=4,
        )

        # ax.plot(
        #     x_smooth, y9li_smooth,
        #     linestyle="--", linewidth=1.8, color=CUSTOM_BLUE, zorder=4,
        # )

        # ax.plot(
        #     x_smooth, y8he_smooth,
        #     linestyle="--", linewidth=1.8, color=CUSTOM_DARKPINK, zorder=4,
        # )

        ax.plot(
            x_smooth, y9li8he_smooth,
            linestyle="--", linewidth=1.8, color=CUSTOM_BLUE, zorder=4,
        )

        ax.plot(
            x_smooth, ybkg_smooth,
            linestyle="--", linewidth=1.8, color=CUSTOM_ORANGE, zorder=4,
        )

    def _get_fit_info_text(self):
        # N9li8he,    f9li,    t9li,    t8he,    Nbkg,    Rmu    = self.fit_result.popt
        # N9li8heerr, f9lierr, t9lierr, t8heerr, Nbkgerr, Rmuerr = self.fit_result.perr
        # text = (
        #     r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d)  = %.3f$"      "\n\n"
        #     r"$N_{^{9}\mathrm{Li}/^{8}\mathrm{He}} = %.1f \pm %.1f$" "\n"
        #     r"$f_{^{9}\mathrm{Li}} = %.3f \pm %.3f$"                 "\n"
        #     r"$\tau_{^{9}\mathrm{Li}} = %.3f \pm %.3f~\mathrm{s}$"   "\n"
        #     r"$\tau_{^{8}\mathrm{He}} = %.3f \pm %.3f~\mathrm{s}$"   "\n"
        #     r"$N_{\mathrm{bkg}} = %.1f \pm %.1f$"                    "\n"
        #     r"$R_{\mu} =  %.2f \pm %.2f~\mathrm{cps}$"
        # ) % (
        #     self.fit_result.chi2, self.fit_result.ndf, self.fit_result.pvalue,
        #     N9li8he, N9li8heerr,
        #     f9li,    f9lierr,
        #     t9li,    t9lierr,
        #     t8he,    t8heerr,
        #     Nbkg,    Nbkgerr,
        #     Rmu,     Rmuerr,
        # )
        N9li8he,    Nbkg,    Rmu    = self.fit_result.popt
        N9li8heerr, Nbkgerr, Rmuerr = self.fit_result.perr
        text = (
            r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d)  = %.3f$"      "\n\n"
            r"$N_{^{9}\mathrm{Li}/^{8}\mathrm{He}} = %.1f \pm %.1f$" "\n"
            r"$N_{\mathrm{bkg}} = %.1f \pm %.1f$"                    "\n"
            r"$R_{\mu} =  %.2f \pm %.2f~\mathrm{cps}$"
        ) % (
            self.fit_result.chi2, self.fit_result.ndf, self.fit_result.pvalue,
            N9li8he, N9li8heerr,
            Nbkg,    Nbkgerr,
            Rmu,     Rmuerr,
        )
        return text


# -------------------------------------------------------------------------------------------------
# Muon performance metric
# -------------------------------------------------------------------------------------------------

class MuonPerformanceAngle(Histogram1DPlotter):
    """
    Angle between the reference and reconstruction track direction (deg).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            bins    = MUON_PERFORMANCE_ANGLE_BINS,
            xlabel  = r"$\alpha$ (deg)",
            ylabel  = "Entries",
            xlim    = (0, 5.05),
            **kwargs,
        )


class MuonPerformanceDistance(Histogram1DPlotter):
    """
    Distance between the reference and reconstruction track middle points (m).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            bins    = MUON_PERFORMANCE_DISTANCE_BINS,
            xlabel  = r"$d_{\mathrm{mid}}$ (m)",
            ylabel  = "Entries",
            xlim    = (0, 2.02),
            **kwargs,
        )


# -------------------------------------------------------------------------------------------------
# Muon performance metric vs clippingness
# -------------------------------------------------------------------------------------------------

class MuonPerformanceMetricClippingness(Histogram1DPlotter):
    """
    Evolution of the angular resolution in function of the clippingness of the reference track.
    """

    def __init__(
        self, 
        ylabel:     str, 
        ylim:       tuple[float, float | None] | None = None,
        **kwargs:   Any
    ) -> None:
        super().__init__(
            bins    = uniform_bins(0.0 * 0.0, 18.0 * 18.0, 9), 
            xlabel  = r"$R_{\mu}^{2}$ (m$^{2}$)", 
            ylabel  = ylabel, 
            xlim    = (0.0 * 0.0, 18.0 * 18.0), 
            ylim    = ylim, 
            **kwargs, 
        )

    def add(
        self,
        metric:         np.ndarray,
        clippingness:   np.ndarray,
        linecolor:      str,
        label:          str,
    ) -> None:
        """
        Histogram data into the internal bin edges and register it.

        Parameters
        ----------
        metric : np.ndarray, shape (N,)
            Raw metric values to histogram.
        clippingness : np.ndarray, shape (N,)
            Associated clippingness to each metric value.
        linecolor : str
            Hex color for the step line and error bars.
        label : str or None
            Legend label. Pass None to suppress this dataset's legend entry.
        """
        perc68  = []
        for k in range(len(self.bins) - 1):
            left    = self.bins[k]
            right   = self.bins[k + 1]
            mask = (left <= clippingness * clippingness) & (clippingness * clippingness < right)
            if np.sum(mask) < 10:
                continue
            perc68.append(np.quantile(metric[mask], 0.68))

        self.datasets.append({
            "perc68":       perc68,
            "linecolor":    linecolor, 
            "label":        label, 
        })

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(7, 6))

        elow = self.centers - self.bins[:-1]
        ehigh = self.bins[1:] - self.centers

        for d in self.datasets:
            ax.errorbar(
                self.centers, 
                d["perc68"], 
                xerr=[elow, ehigh], 
                fmt="o", 
                color=d["linecolor"], 
                lw=1.5, 
                markersize=7, 
                capsize=3, 
                label=d["label"], 
            )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks(self.bins)
        xticklabels = [rf"${np.sqrt(x):.1f}^2$" for x in self.bins]
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper left")

        fig.tight_layout()

        return fig, ax