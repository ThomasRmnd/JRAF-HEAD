from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from config import (
    BLACK,
    CUSTOM_FAINTBLUE,
    SPATIAL_RHO2_BINS,
    SPATIAL_Z_BINS,
)
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from .base import BasePlotter

# -------------------------------------------------------------------------------------------------
# Generic 1D histogram plotter
# -------------------------------------------------------------------------------------------------

class Histogram2DPlotter(BasePlotter):
    """
    Generic 2D histogram plotter.

    Parameters
    ----------
    bins : np.ndarray
        Bin edges array of length n_bins + 1.
    **kwargs
        Forwarded to "BasePlotter.__init__".
    """

    def __init__(
        self,
        xbins:      np.ndarray,
        ybins:      np.ndarray,
        **kwargs:   Any,
    ) -> None:
        super().__init__(**kwargs)
        self.xbins    = xbins
        self.xcenters = 0.5 * (xbins[1:] + xbins[:-1])
        self.xwidths  = xbins[1:] - xbins[:-1]
        self.ybins    = ybins
        self.ycenters = 0.5 * (ybins[1:] + ybins[:-1])
        self.ywidths  = ybins[1:] - ybins[:-1]

    # ---------------------------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------------------------

    def plot(self, x: np.ndarray, y: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
        """
        Draw all registered datasets on a new figure and return (fig, ax).

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        self._draw_dataset(ax, x, y)

        self.apply_style(ax)
        self._maybe_fit(ax)

        fig.tight_layout()
        return fig, ax

    # ---------------------------------------------------------------------------------------------
    # Drawing helpers
    # ---------------------------------------------------------------------------------------------

    def _draw_dataset(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
        """
        Draw one dataset as a filled area or a stepped line + error bars.
        """
        pass

    # ---------------------------------------------------------------------------------------------
    # Hooks - overridden by subclasses
    # ---------------------------------------------------------------------------------------------

    def _maybe_fit(self, ax: plt.Axes) -> None:
        """
        Hook for subclass-specific fits. No-op in the base class.
        """
        pass


# -------------------------------------------------------------------------------------------------
# Spatial distribution (\rho^2 vs z)
# -------------------------------------------------------------------------------------------------

class SpatialDistributionPlotter(Histogram2DPlotter):
    """
    2D histogram of reconstructed vertex positions: \rho^2 (x-axis) vs z (y-axis).

    Overlays two dashed ellipses marking the acrylic vessel boundary (black) and the fiducial 
    volume (red), both derived from "DetectorGeometry".

    Parameters
    ----------
    xbins : np.ndarray, optional
        Bin edges for the \rho^2 axis (m^2). Defaults to "SPATIAL_RHO2_BINS".
    ybins : np.ndarray, optional
        Bin edges for the z axis [m]. Defaults to "SPATIAL_Z_BINS".
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    """

    def __init__(
        self,
        xbins:      np.ndarray = SPATIAL_RHO2_BINS, 
        ybins:      np.ndarray = SPATIAL_Z_BINS, 
        xlabel:     np.ndarray = r"$\rho^{2}$ (m$^2$)", 
        ylabel:     np.ndarray = r"$z$ (m)", 
        **kwargs:   Any, 
    ) -> None:
        super().__init__(
            xbins, 
            ybins, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            **kwargs, 
        )
        # Acrylic boundary curve: \rho^2 = R^2 − z^2
        self._acrylic_z   = np.linspace(-17.7, 17.7, 500)
        self._acrylic_rho2 = 17.7 ** 2 - self._acrylic_z ** 2

        # Fiducial volume boundary curve
        # self._fv_z    = np.linspace(-17.2, 17.2, 500)
        # self._fv_rho2 = 17.2 ** 2 - self._fv_z ** 2

    def plot(
        self,
        rho2: np.ndarray,
        z: np.ndarray,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Draw the 2D spatial distribution on a new figure.

        Parameters
        ----------
        rho2 : np.ndarray, shape (N,)
            Transverse radius squared (m^2).
        z : np.ndarray, shape (N,)
            Axial coordinate (m).

        Returns
        -------
        fig, ax : Figure and Axes
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        h = ax.hist2d(
            rho2, z,
            bins=(self.xbins, self.ybins),
            cmin=1.0,
            cmap="jet",
        )

        # Boundary overlays
        ax.plot(
            self._acrylic_rho2, self._acrylic_z,
            linestyle="--", linewidth=1.6, color="black",
        )
        # ax.plot(
        #     self._fv_rho2, self._fv_z,
        #     linestyle="--", linewidth=1.2, color="red",
        # )

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(direction="in", which="both", top=True, right=True)

        ax.set_xlim(0.0, 320.0)
        ax.set_ylim(-17.8, 17.8)

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("Entries")
        cbar.ax.minorticks_on()
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        cbar.ax.tick_params(direction="in", which="both", width=1.2)

        fig.tight_layout()
        return fig, ax


# -------------------------------------------------------------------------------------------------
# Muon-veto distribution (\Delta t_{\mu} vs d_{\mu}, 4-panel)
# -------------------------------------------------------------------------------------------------

class MuonVetoDistributionPlotter(Histogram2DPlotter):
    """
    Four-panel figure: \Delta t_{\mu-p} vs d_{\mu-p} 2D histogram with marginal
    projections and a fit to the time projection.

    Layout (GridSpec 2 x 2)
    ---------------------
    ┌──────────┬───────┐
    │  top     │ cbar  │
    ├──────────┼───────┤
    │  main    │ right │
    └──────────┴───────┘

    The fit model depends on "is_signal_region":
    - Signal  : A * exp(-t/\tau) + c  (exponential + constant)
    - Background: constant c

    Parameters
    ----------
    xbins : np.ndarray, optional
        Bin edges for the \rho^2 axis (m^2). Defaults to "SPATIAL_RHO2_BINS".
    ybins : np.ndarray, optional
        Bin edges for the z axis [m]. Defaults to "SPATIAL_Z_BINS".
    is_signal_region : bool
        Selects the fit model (exp+const for signal, const for background).
    """

    def __init__(
        self,
        xbins:              np.ndarray,
        ybins:              np.ndarray, 
        xlabel:             str = r"$\Delta t_{\mu-p}$ (s)", 
        ylabel:             str = r"$d_{\mu-p}$ (m)", 
        is_signal_region:   bool = True, 
        **kwargs:           Any, 
    ) -> None:
        super().__init__(
            xbins, 
            ybins, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            **kwargs, 
        )
        self.is_signal_region = is_signal_region

    def plot(
        self,
        dt:     np.ndarray,
        dlat:   np.ndarray,
    ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Draw the 4-panel muon-veto figure.

        Parameters
        ----------
        dt : np.ndarray, shape (N,)
            Time since last muon [s] (positive = signal region,
            negative = background region).
        dlat : np.ndarray, shape (N,)
            Lateral muon-to-prompt distance [mm]. Divided by 1e3 internally
            to convert to metres for display.

        Returns
        -------
        fig : Figure
        axes : dict with keys "main", "top", "right", "cbar"
        """
        dlat_m = dlat / 1e3

        # Pre-compute 2D histogram for projections
        h2d, xedges, yedges = np.histogram2d(dt, dlat_m, bins=(self.xbins, self.ybins))
        proj_x   = h2d.sum(axis=1)
        proj_y   = h2d.sum(axis=0)
        centers_x = 0.5 * (xedges[:-1] + xedges[1:])
        centers_y = 0.5 * (yedges[:-1] + yedges[1:])

        # ── Figure layout ──────────────────────────────────────────────
        fig = plt.figure(figsize=(7, 6))
        gs  = GridSpec(
            2, 2,
            width_ratios=[4, 1.2],
            height_ratios=[1.2, 4],
            hspace=0.1, wspace=0.1,
        )
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_cbar  = fig.add_subplot(gs[0, 1])

        # ── Main 2D histogram ──────────────────────────────────────────
        H = ax_main.hist2d(dt, dlat_m, bins=(self.xbins, self.ybins), cmin=1.0, cmap="jet")
        ax_main.set_xlabel(self.xlabel)
        ax_main.set_ylabel(self.ylabel)
        ax_main.minorticks_on()
        ax_main.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_main.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_main.tick_params(direction="in", which="both", top=True, right=True)

        # ── Top projection + fit ───────────────────────────────────────
        self._draw_top_projection(
            ax_top, proj_x, centers_x, xedges, self.is_signal_region
        )

        # ── Right projection ───────────────────────────────────────────
        ax_right.barh(
            centers_y, proj_y,
            height=np.diff(yedges),
            color=CUSTOM_FAINTBLUE, edgecolor=CUSTOM_FAINTBLUE,
        )
        ax_right.set_xlabel("Entries")
        ax_right.tick_params(axis="y", labelleft=False, direction="in", which="both")
        ax_right.minorticks_on()

        # ── Colorbar ───────────────────────────────────────────────────
        cbar = fig.colorbar(H[3], cax=ax_cbar, orientation="horizontal")
        cbar.set_label("Entries", labelpad=6)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.tick_top()
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(direction="in")

        pos = ax_cbar.get_position()
        ax_cbar.set_position([
            pos.x0, pos.y0 + pos.height * 0.2,
            pos.width, pos.height * 0.2,
        ])

        # ── Uniform spine width ────────────────────────────────────────
        for ax in [ax_main, ax_top, ax_right]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

        axes = {"main": ax_main, "top": ax_top, "right": ax_right, "cbar": ax_cbar}
        return fig, axes

    # ------------------------------------------------------------------
    # Top-projection drawing + fit
    # ------------------------------------------------------------------

    def _draw_top_projection(
        self,
        ax: plt.Axes,
        proj_x: np.ndarray,
        xcenters: np.ndarray,
        xedges: np.ndarray,
        is_signal_region: bool, 
    ) -> None:
        """Fit and draw the time-axis projection on the top panel."""
        ax.bar(
            xcenters, proj_x,
            width=np.diff(xedges),
            color=CUSTOM_FAINTBLUE, edgecolor=CUSTOM_FAINTBLUE,
        )
        ax.set_ylabel("Entries")
        ax.tick_params(axis="x", labelbottom=False, direction="in", which="both")
        ax.tick_params(axis="y", direction="in")
        ax.minorticks_on()

        if is_signal_region:
            self._fit_signal_projection(ax, proj_x, xedges)
        else:
            self._fit_background_projection(ax, proj_x, xedges)

    def _fit_signal_projection(
        self,
        ax:       plt.Axes,
        xproj:    np.ndarray,
        xedges:   np.ndarray,
    ) -> None:
        """A * exp(-t/\tau) + c fit to the signal-region time projection."""
        from fits.functions import ExponentialConstantFitter

        y    = xproj
        yerr = np.sqrt(y)
        # Apply the same dead-time exclusion cut used in the 1D time plotter
        threshold       = 0.007 # us
        first_fit_bin   = np.searchsorted(xedges, threshold, side="left")
        t_min_fit       = xedges[first_fit_bin]

        fitter = ExponentialConstantFitter(xedges, y, yerr, xlim=(t_min_fit, None))
        result = fitter.fit()
        if result is None:
            return

        x_smooth = np.linspace(t_min_fit, xedges[-1], 500)
        y_smooth = fitter.model(x_smooth, *result.popt)
        ax.plot(x_smooth, y_smooth, linestyle="--", linewidth=1.2, color=BLACK)

        A, tau, c = result.popt
        A_err, tau_err, c_err = result.perr

        text = (
            r"$P(\chi^2/\mathrm{ndf} = %.1f / %d) = %.3f$" "\n"
            r"$A = %.1f \pm %.1f$" "\n"
            r"$\tau = %.3f \pm %.3f~\mathrm{s}$" "\n"
            r"$c = %.1f \pm %.1f$"
        ) % (result.chi2, result.ndf, result.pvalue, A, A_err, tau, tau_err, c, c_err)

        ax.text(
            *self.fit_info_loc, text,
            transform=ax.transAxes,
            fontsize=self.fit_info_fontsize,
            va=self.fit_info_anchor.split(' ')[0],
            ha=self.fit_info_anchor.split(' ')[1],
        )

    def _fit_background_projection(
        self,
        ax:       plt.Axes,
        xproj:    np.ndarray,
        xedges:   np.ndarray,
    ) -> None:
        """Constant fit to the background-region time projection."""
        from fits.functions import ConstantFitter

        y    = xproj
        yerr = np.sqrt(y)
        # Apply the same dead-time exclusion cut used in the 1D time plotter
        threshold       = -0.007 # us
        last_fit_bin    = np.searchsorted(xedges, threshold, side="left") - 1
        t_max_fit       = xedges[last_fit_bin]

        fitter = ConstantFitter(xedges, y, yerr, xlim=(None, t_max_fit))
        result = fitter.fit()
        if result is None:
            return

        x_smooth = np.linspace(xedges[0], t_max_fit, 500)
        y_smooth = fitter.model(x_smooth, *result.popt)
        ax.plot(x_smooth, y_smooth, linestyle="--", linewidth=1.2, color=BLACK)

        c     = result.popt[0]
        c_err = result.perr[0]

        text = (
            r"$P(\chi^2/\mathrm{ndf} = %.1f / %d) = %.3f$" "\n"
            r"$c = %.1f \pm %.1f$"
        ) % (result.chi2, result.ndf, result.pvalue, c, c_err)

        ax.text(
            *self.fit_info_loc, text,
            transform=ax.transAxes,
            fontsize=self.fit_info_fontsize,
            va=self.fit_info_anchor.split(' ')[0],
            ha=self.fit_info_anchor.split(' ')[1],
        )