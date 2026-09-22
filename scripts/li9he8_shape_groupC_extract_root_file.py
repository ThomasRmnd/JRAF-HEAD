import argparse

import matplotlib.pyplot as plt
import numpy as np
import uproot
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from jrafhead.config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_RED,
    setup_style,
)
from jrafhead.plotters import PromptEnergyPlotter
from jrafhead.utils import rebin_histogram

setup_style()

class PromptEnergyMeanShapePlotter(PromptEnergyPlotter):
    """
    Prompt-energy shape comparison plotter.

    The input spectra are normalized to unit integral before calculating
    the mean shape. The lower panel shows the pull with respect to the
    mean shape:

        (mean - y) / yerr

    The residual panel displays 1, 2 and 3 sigma bands in gray.
    """

    def plot(self) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Plot the spectra and their residuals with respect to the mean shape.
        """
        if len(self.datasets) == 0:
            raise RuntimeError("No spectra have been added.")

        fig = plt.figure(figsize=(7, 8))

        gs = GridSpec(2, 1,
            height_ratios=[4, 1.2],
            hspace=0.08,
        )

        ax_main = fig.add_subplot(gs[0])
        ax_resi = fig.add_subplot(gs[1], sharex=ax_main)

        shapes = np.array([
            d["hist"]
            for d in self.datasets
        ])
        mean_shape = np.mean(shapes, axis=0)
        for d in self.datasets:
            self._draw_dataset(ax_main, d)

        ax_main.step(
            self.bins,
            np.r_[mean_shape, mean_shape[-1]],
            where="post",
            color=BLACK,
            linewidth=2.0,
            linestyle="--",
            label="Mean shape",
            zorder=2,
        )

        self.apply_style(ax_main)

        ax_main.set_xlabel("")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_main.legend(
            loc=self.legend_loc,
            ncol=self.legend_ncol,
        )

        for d in self.datasets:
            hist = d["hist"]
            err  = d["err"]

            pull = np.full_like(hist, np.nan, dtype=float)
            valid = err > 0

            pull[valid] = (
                hist[valid] - mean_shape[valid]
            ) / err[valid]

            ax_resi.plot(
                self.centers,
                pull,
                "o",
                color=d["linecolor"],
                markersize=4.0,
                zorder=4,
            )
        ax_resi.axhline(0.0, color=BLACK, linewidth=1.5, linestyle="--", zorder=2)
        ax_resi.axhspan(-1.0, 1.0, color="gray", alpha=0.25, zorder=1)
        ax_resi.axhspan(-2.0, 2.0, color="gray", alpha=0.10, zorder=0)

        self._style_residual_axis(ax_resi)
        fig.align_ylabels([ax_main, ax_resi])

        return fig, {"main": ax_main, "resi": ax_resi}

    def _style_residual_axis(self, ax: plt.Axes) -> None:
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
        ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        if self.xlim:
            ax.set_xlim(*self.xlim)
        ax.set_ylim(-3.0, 3.0)



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input filepath")
parser.add_argument("--comparison", type=str, default=None, help="Comparison filepath")
args = parser.parse_args()

file = uproot.open(args.input)

has_background = ("__changing_veto__" not in args.input)

fitted_spectrum         = file["fitted_spectrum"]
signal_spectrum         = file["signal_spectrum"]
if has_background:
    background_spectrum = file["background_spectrum"]

branches = [
    "run_id", 
    "e_p", "sec_p", "nsec_p", "posx_p", "posy_p", "posz_p", 
    "e_d", "sec_d", "nsec_d", "posx_d", "posy_d", "posz_d",
]
signal_events         = file["signal_events"].arrays(branches, library="np")
if has_background:
    background_events = file["background_events"].arrays(branches, library="np")

plotter = PromptEnergyPlotter(binmode="normal")
esighist, _ = np.histogram(signal_events["e_p"], bins=plotter.bins)
esighist    = esighist.astype(float)
esigerr     = np.sqrt(esighist)
plotter.add_histogram(esighist, esigerr, CUSTOM_BLUE, label="Enriched sample")
if has_background:
    ebkghist, _ = np.histogram(background_events["e_p"], bins=plotter.bins)
    ebkghist    = ebkghist.astype(float)
    ebkgerr     = np.sqrt(ebkghist)
    plotter.add_histogram(ebkghist, ebkgerr, CUSTOM_RED, label="Depleted sample")
    etothist    = esighist - ebkghist
    etoterr     = np.sqrt(esigerr**2 + ebkgerr**2)
    plotter.add_histogram(etothist, etoterr, BLACK, label="Difference")
fig, ax = plotter.plot()
counts, edges = fitted_spectrum.to_numpy()
centers       = 0.5 * (edges[1:] + edges[:-1])
ax.step(centers, counts, color="#FF0000", label="Template")



plotter = PromptEnergyPlotter( # PromptEnergyMeanShapePlotter(
    binmode="normal",
    ylabel=r"P.D.F.",
    show_errors=False,
    show_steps=True,
    legend_loc="upper right",
    legend_ncol=1,
)
widths = plotter.bins[1:] - plotter.bins[:-1]

file          = uproot.open(args.comparison)
hRawV3        = file["hRawV3_all_1keV_counts"]
counts, edges = hRawV3.to_numpy()
counts        = counts.astype(float)
counts        = rebin_histogram(edges, counts, plotter.bins)
print(f"Ze's shape number of entries: {np.sum(counts)}")
err           = np.sqrt(counts) / np.sum(counts * widths)
counts        = counts / np.sum(counts * widths)
plotter.add_histogram(counts, err, CUSTOM_RED, label="Ze's shape")
esighist, _   = np.histogram(signal_events["e_p"], bins=plotter.bins)
esighist      = esighist.astype(float)
esigerr       = np.sqrt(esighist)
ebkghist, _   = np.histogram(background_events["e_p"], bins=plotter.bins)
ebkghist      = ebkghist.astype(float)
ebkgerr       = np.sqrt(ebkghist)
etothist      = esighist - ebkghist
print(f"Thomas's shape number of entries: {np.sum(etothist)}")
etoterr       = np.sqrt(esigerr**2 + ebkgerr**2) / np.sum(etothist * widths)
etothist      = etothist / np.sum(etothist * widths)
plotter.add_histogram(etothist, etoterr, CUSTOM_BLUE, label="Thomas's shape")
fig, _        = plotter.plot()
plt.show()