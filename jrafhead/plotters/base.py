from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class BasePlotter:
    """
    Common axis styling for all plotters in this project.

    Parameters
    ----------
    xlabel : str
        x-axis label.
    ylabel : str
        y-axis label.
    xlim : tuple of (float, float) or None
        If given, sets the x-axis limits.
    ylim : tuple of (float or None, float or None)
        Lower and upper y-axis limits. None means matplotlib decides. Default is (0, None) ==> 
        force origin at zero, free upper bound.
    xscale : str
        Matplotlib scale string for the x-axis ("linear", "log").
    yscale : str
        Matplotlib scale string for the y-axis.
    """

    def __init__(
        self,
        xlabel:            str = "", 
        ylabel:            str = "", 
        xlim:              tuple[float, float] | None = None, 
        ylim:              tuple[float | None, float | None] = (0, None), 
        xscale:            str = "linear", 
        yscale:            str = "linear", 
        enable_fit:        bool = True, 
        draw_fit_info:     bool = True, 
        fit_info_anchor:   str = "top right", 
        fit_info_loc:      tuple[float, float] = (0.95, 0.95), 
        fit_info_fontsize: int = 18, 
    ) -> None:
        self.xlabel            = xlabel
        self.ylabel            = ylabel
        self.xlim              = xlim
        self.ylim              = ylim
        self.xscale            = xscale
        self.yscale            = yscale
        self.enable_fit        = enable_fit
        self.draw_fit_info     = draw_fit_info
        self.fit_info_anchor   = fit_info_anchor
        self.fit_info_loc      = fit_info_loc
        self.fit_info_fontsize = fit_info_fontsize
        self.fit_result        = None

    # ---------------------------------------------------------------------------------------------
    # Axis styling
    # ---------------------------------------------------------------------------------------------

    def apply_style(self, ax: plt.Axes) -> None:
        """
        Apply the standard style to ax.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        """
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if self.xlim:
            ax.set_xlim(*self.xlim)
        if self.ylim:
            ax.set_ylim(*self.ylim)

        if self.xscale == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        if self.yscale == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)

        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)