from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from utils import FusionMode, fuze_by_day

# -------------------------------------------------------------------------------------------------
# Dataset container
# -------------------------------------------------------------------------------------------------

@dataclass
class _EvolutionDataset:
    """One series of (x, y \pm err) measurements to overlay on the plot."""
    x:          Any                 # np.ndarray[int] (run IDs) or list[datetime]
    y:          np.ndarray
    err:        np.ndarray
    color:      str
    label:      str | None
    bar_width:  float | None = None # set by subclass at add() time


@dataclass
class _EvolutionRegion:
    """Region to overlay on the plot."""
    xmin:       float
    xmax:       float
    color:      str
    label:      str
    label_x:    float = 0.5   # position within region width
    label_y:    float = 0.5   # relative position within axes height, 0=bottom, 1=top
    fontsize:   float = 10
    rotation:   float = 0


# -------------------------------------------------------------------------------------------------
# Shared base
# -------------------------------------------------------------------------------------------------

class _EvolutionPlotterBase:
    """
    Shared logic for run-evolution and time-evolution plotters.

    Handles dataset registration, mean/std computation, errorbar drawing,
    dashed mean line, shaded band, legend placement, and axis styling.

    Parameters
    ----------
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label (LaTeX accepted).
    xlim : tuple[float, float] or None
        x-axis limits. If None, matplotlib chooses automatically.
    ylim : tuple[float, float] or None
        y-axis limits. If None, matplotlib chooses automatically.
    figsize : tuple[float, float]
        Figure size in inches. Default (14, 6).
    show_mean : bool
        Draw a dashed mean line per dataset. Default True.
    show_band : bool
        Draw a shaded \pm 1\sigma band per dataset. Default True.
    bar_mode : bool
        Force bar chart for all datasets. Default False.
        Also auto-enabled per-dataset when all errors are zero.
    legend_ncol : int or None
        Legend column count. None ==> auto (one column per labeled dataset).
    """

    def __init__(
        self,
        xlabel:      str                        = "",
        ylabel:      str                        = "",
        xlim:        tuple[float, float] | None = None,
        ylim:        tuple[float, float] | None = None,
        figsize:     tuple[float, float]        = (14, 6),
        show_mean:   bool                       = True,
        show_band:   bool                       = True,
        bar_mode:    bool                       = False,
        legend_ncol: int | None                 = None,
        legend_bbox: tuple[float, float]        = (0.5, 1.30),
    ) -> None:
        self.xlabel       = xlabel
        self.ylabel       = ylabel
        self.xlim         = xlim
        self.ylim         = ylim
        self.figsize      = figsize
        self.show_mean    = show_mean
        self.show_band    = show_band
        self.bar_mode     = bar_mode
        self.legend_ncol  = legend_ncol
        self.legend_bbox  = legend_bbox
        self.datasets: list[_EvolutionDataset] = []
        self.regions:  list[_EvolutionRegion]  = []

    # ---------------------------------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------------------------------

    def add(
        self,
        x:      Any,
        y:      np.ndarray,
        err:    np.ndarray,
        color:  str,
        label:  str | None = None,
    ) -> None:
        """
        Register one dataset to be plotted.

        Parameters
        ----------
        x : array-like
            x positions.
        y : np.ndarray, shape (N,)
            Measured values.
        err : np.ndarray, shape (N,)
            Per-point uncertainties.  If "np.zeros_like(y)", take the square root errors.
        color : str
            Hex colour for markers / bars, mean line, and band.
        label : str or None
            Legend label. If None, plotted but not listed in the legend.
        """
        x_plot, y_plot, err_plot, bar_width = self._prepare_dataset(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(err, dtype=float),
        )
        self.datasets.append(
            _EvolutionDataset(x_plot, y_plot, err_plot, color, label, bar_width)
        )

    def add_region(
        self,
        xmin:   float,
        xmax:   float,
        color:  str,
        label:  str,
        label_x:    float = 0.5,
        label_y:    float = 0.5,
        fontsize:   float = 10,
        rotation:   float = 0,
    ) -> None:
        """
        Register one region to be plotted.

        Parameters
        ----------
        xmin : float
            Beginning of the region.
        xmax : float
            End of the region.
        color : str
            Hex colour for markers / bars, mean line, and band.
        label : str or None
            Legend label.
        label_x : float
            Relative horizontal position of the label within [xmin, xmax],
            0 = left edge, 1 = right edge. Default 0.5 (centered).
        label_y : float
            Relative vertical position of the label within the axes height
            (region spans the full y-range), 0 = bottom, 1 = top. Default 0.5.
        fontsize : float
            Label font size. Default 10.
        rotation : float
            Label rotation in degrees. Default 0 (horizontal).
        """
        self.regions.append(
            _EvolutionRegion(xmin, xmax, color, label, label_x, label_y, fontsize, rotation)
        )

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        """Draw all datasets and return (fig, ax)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for ds in self.datasets:
            self._draw_dataset(ax, ds)

        for r in self.regions:
            ax.add_patch(Rectangle(
                (r.xmin, 0), r.xmax - r.xmin, 1,
                transform=ax.get_xaxis_transform(),
                color=r.color, alpha=0.2, zorder=1,
            ))
            ax.text(
                r.xmin + r.label_x,
                r.label_y,
                r.label,
                transform=ax.get_xaxis_transform(),
                color=r.color,
                fontsize=r.fontsize,
                rotation=r.rotation,
                ha="left",
                va="bottom",
                zorder=2,
            )

        self._apply_style(ax)
        self._draw_legend(ax)
        fig.tight_layout()
        return fig, ax

    # ---------------------------------------------------------------------------------------------
    # Hooks - overridden by subclasses
    # ---------------------------------------------------------------------------------------------

    def _prepare_dataset(
        self,
        x:   np.ndarray,
        y:   np.ndarray,
        err: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray, float | None]:
        """
        Pre-process x, y, err before storing.

        Returns (x_plot, y_plot, err_plot, bar_width).
        bar_width is None for errorbar mode; set by subclasses for bar mode.
        """
        return x, y, err, None

    def _configure_xaxis(self, ax: plt.Axes) -> None:
        """Configure x-axis ticks/formatter. Override for date axes."""
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    # ---------------------------------------------------------------------------------------------
    # Per-dataset drawing
    # ---------------------------------------------------------------------------------------------

    def _draw_dataset(self, ax: plt.Axes, ds: _EvolutionDataset) -> None:
        """
        Draw one dataset: points (errorbar or bar) + mean line + band.

        Each element is independently controlled by show_mean/show_band so callers can 
        mix-and-match (e.g. band only, no points).
        """
        mean = np.mean(ds.y)
        std  = np.std(ds.y)

        if self.bar_mode:
            self._draw_bars(ax, ds)
        else:
            self._draw_errorbar(ax, ds)

        if self.show_mean:
            ax.axhline(
                mean,
                color=ds.color, linestyle="--", linewidth=2.0, zorder=2,
            )

        if self.show_band:
            ax.add_patch(Rectangle(
                (0, mean - std), 1, 2 * std,
                transform=ax.get_yaxis_transform(),
                color=ds.color, alpha=0.2, zorder=1,
            ))

    def _draw_errorbar(self, ax: plt.Axes, ds: _EvolutionDataset) -> None:
        ax.errorbar(
            ds.x, ds.y,
            yerr=ds.err,
            fmt="o",
            color=ds.color,
            markersize=7.5,
            capsize=0,
            elinewidth=1.0,
            markeredgecolor="k",
            markeredgewidth=0.25,
            label=ds.label,
            zorder=3,
        )

    def _draw_bars(self, ax: plt.Axes, ds: _EvolutionDataset) -> None:
        """
        Draw a bar chart for datasets without errors (or bar_mode=True).

        Bar width is determined by the subclass at _prepare_dataset() time
        (stored in ds.bar_width).  Falls back to 1.0 for run plotters.
        """
        width = ds.bar_width if ds.bar_width is not None else 1.0
        ax.bar(
            ds.x, ds.y,
            width=width,
            color=ds.color,
            alpha=1.0,
            label=ds.label,
            zorder=3,
        )

    # ---------------------------------------------------------------------------------------------
    # Axis styling + legend
    # ---------------------------------------------------------------------------------------------

    def _apply_style(self, ax: plt.Axes) -> None:
        ax.set_xlabel(self.xlabel, fontdict={"size": 22})
        ax.set_ylabel(self.ylabel, fontdict={"size": 22})
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=22)
        ax.minorticks_on()
        self._configure_xaxis(ax)
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def _draw_legend(self, ax: plt.Axes) -> None:
        labeled = [ds for ds in self.datasets if ds.label is not None]
        if not labeled:
            return
        
        ncol = self.legend_ncol if self.legend_ncol is not None else len(labeled)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=self.legend_bbox,
            ncol=ncol,
            frameon=False,
            handletextpad=0.3,
            columnspacing=0.9,
            borderaxespad=0.2,
            fontsize=22,
        )


# -------------------------------------------------------------------------------------------------
# Run evolution plotter
# -------------------------------------------------------------------------------------------------

class RunEvolutionPlotter(_EvolutionPlotterBase):
    """
    Plot a physical quantity as a function of run number.

    Parameters
    ----------
    ylabel : str
        y-axis label (LaTeX accepted).
    xlim: tuple[float, float] or None
        x-axis limits.
    ylim : tuple[float, float] or None
        y-axis limits.
    show_mean, show_band, bar_mode, legend_ncol
        See ``_EvolutionPlotterBase``.
    """

    _BAR_WIDTH = 1.0 # fraction of one run-ID unit

    def __init__(
        self,
        ylabel: str                        = "",
        xlim:   tuple[float, float] | None = None,
        ylim:   tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(xlabel="Run Number", ylabel=ylabel, xlim=xlim, ylim=ylim, **kwargs)

    def _prepare_dataset(
        self,
        x:   np.ndarray,
        y:   np.ndarray,
        err: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return x, y, err, self._BAR_WIDTH


# -------------------------------------------------------------------------------------------------
# Time evolution plotter
# -------------------------------------------------------------------------------------------------

class TimeEvolutionPlotter(_EvolutionPlotterBase):
    """
    Plot a physical quantity as a function of calendar date.

    Per-run measurements are fused into one point per calendar day (UTC) before plotting. The 
    fusion mode controls how runs within the same day are combined.

    Parameters
    ----------
    ylabel : str
        y-axis label (LaTeX accepted).
    xlim : tuple[str | datetime, str | datetime] | None
        x-axis limits.
    ylim : tuple[float, float] or None
        y-axis limits.
    mode : {"mean", "sum"}
        Daily fusion strategy.
    date_format : str
        ``strftime`` format for x-axis tick labels.  Default ``"%b %Y"``.
    show_mean, show_band, bar_mode, legend_ncol
        See ``_EvolutionPlotterBase``.
    """

    # One matplotlib date unit = one day, so bar width = 1.0 fills each day slot.
    _BAR_WIDTH_DAYS = 0.8

    def __init__(
        self,
        ylabel:      str                                            = "",
        xlim:        tuple[str | datetime, str | datetime] | None   = None,
        ylim:        tuple[float, float] | None                     = None,
        mode:        FusionMode                                     = "mean",
        date_format: str                                            = "%Y-%m-%d",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            xlabel="",    # date axis - label usually omitted
            ylabel=ylabel,
            xlim=self._to_num_xlim(xlim),
            ylim=ylim,
            **kwargs,
        )
        self._mode        = mode
        self._date_format = date_format

    @staticmethod
    def _to_num_xlim(
        xlim: tuple[str | datetime, str | datetime] | None,
    ) -> tuple[float, float] | None:
        """Convert str/datetime bounds to matplotlib date numbers."""
        if xlim is None:
            return None

        def _to_date(v: str | datetime) -> datetime:
            return v if isinstance(v, datetime) else datetime.fromisoformat(v)

        lo, hi = xlim
        return mdates.date2num(_to_date(lo)), mdates.date2num(_to_date(hi))

    def _prepare_dataset(
        self,
        x:   np.ndarray,
        y:   np.ndarray,
        err: np.ndarray,
    ) -> tuple[list[datetime], np.ndarray, np.ndarray, float]:
        """Fuse per-run data into one point per day, then return with bar width."""
        dates, y_fused, err_fused = fuze_by_day(x, y, err, mode=self._mode)
        return dates, y_fused, err_fused, self._BAR_WIDTH_DAYS

    def _configure_xaxis(self, ax: plt.Axes) -> None:
        fmt = self._date_format

        def _formatter(x: float, pos: int) -> str:
            try:
                datestr = mdates.num2date(x).strftime(fmt)
                return datestr.replace("-", r"\mbox{-}")
            except Exception:
                return ""

        ax.xaxis.set_major_formatter(FuncFormatter(_formatter))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())