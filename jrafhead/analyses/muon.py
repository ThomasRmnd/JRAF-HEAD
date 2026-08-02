from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_RED,
    GOOGLE_BLUE,
    GOOGLE_GREEN,
    GOOGLE_YELLOW,
    ReProd26B,
)
from fits import (
    ExponentialRateFitter,
    FitResult,
)
from loader import (
    load_lifetime_daq,
    load_muon_performance,
    load_muon_rate,
)
from plotters import (
    MuonPerformanceAngle,
    MuonPerformanceDistance,
    MuonPerformanceMetricClippingness,
    RunEvolutionPlotter,
    TimeEvolutionPlotter,
)
from utils import (
    save_figure,
)

from .base import BaseAnalysis

# -------------------------------------------------------------------------------------------------
# Muon performance analysis
# -------------------------------------------------------------------------------------------------

class MuonPerformanceAnalysis(BaseAnalysis):
    """
    Muon performance analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    dirpath_daq : str or Path
        Path to the DAQ directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def __init__(
            self,
            filepath: str,
            dirpath: str,
            dirpath_daq: str,
            output_dir:str = ".",
    ) -> None:
        super().__init__(filepath, dirpath, output_dir)
        self.dirpath_daq    = dirpath_daq

    def _load(self) -> None:
        self._data = load_muon_performance(str(self.filepath), str(self.dirpath))
        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))

    def _plot(self) -> None:
        self._plot_angle()
        self._plot_distance()
        self._plot_angle_vs_clippingness()
        self._plot_distance_vs_clippingness()
        self._plot_angle_vs_run()
        self._plot_distance_vs_run()
        self._plot_angle_vs_time()
        self._plot_distance_vs_time()
        plt.show()

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot_angle(self) -> None:
        perc68 = np.quantile(self._data.angle, 0.68)
        plotter = MuonPerformanceAngle()
        plotter.add(self._data.angle, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE, label=rf"Joint $\chi^{2}$ (68\% = {perc68:.2f}~deg)")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_angle", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance(self) -> None:
        perc68 = np.quantile(self._data.distance, 0.68)
        plotter = MuonPerformanceDistance()
        plotter.add(self._data.distance, linecolor=CUSTOM_RED, fillcolor=CUSTOM_RED, label=rf"Joint $\chi^{2}$ (68\% = {perc68:.2f}~m)")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_distance", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_angle_vs_clippingness(self) -> None:
        perc68 = np.quantile(self._data.angle, 0.68)
        plotter = MuonPerformanceMetricClippingness(
            r"$68^{\mathrm{th}}$ percentile of $\alpha$ (deg)", 
            (0.0, 5.0), 
        )
        plotter.add(
            self._data.angle, 
            self._data.ref_clippingness, 
            CUSTOM_BLUE, 
            rf"Joint $\chi^{2}$ (68\% = {perc68:.2f}~deg)", 
        )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_angle_vs_clippingness", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance_vs_clippingness(self) -> None:
        perc68 = np.quantile(self._data.distance, 0.68)
        plotter = MuonPerformanceMetricClippingness(
            r"$68^{\mathrm{th}}$ percentile of $\d_{\mathrm{mid}}$ (m)", 
            (0.0, 2.0), 
        )
        plotter.add(
            self._data.distance, 
            self._data.ref_clippingness, 
            CUSTOM_RED, 
            rf"Joint $\chi^{2}$ (68\% = {perc68:.2f}~deg)", 
        )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_distance_vs_clippingness", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_angle_vs_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $\alpha$ (deg)", 
            xlim=(ReProd26B.phases[0].run_min - 100, ReProd26B.phases[3].run_max + 100),
            ylim=(0.0, 5.0),
            show_mean=False, 
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(self._data.run_id[mask])
            perc68 = np.array([
                np.quantile(
                    self._data.angle[self._data.run_id == run], 
                    0.68
                )
                for run in runs
            ])

            mean_angle  = np.mean(perc68)
            std_angle   = np.std(perc68)

            plotter.add(
                runs, 
                perc68, 
                np.zeros_like(perc68), 
                phase.color,
                f"{phase.name}: ${mean_angle:.2f} \pm {std_angle:.2f}" r"^{\circ}$",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_angle_vs_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance_vs_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $d_{\mathrm{mid}}$ (m)", 
            xlim=(ReProd26B.phases[0].run_min - 100, ReProd26B.phases[3].run_max + 100),
            ylim=(0.0, 2.0),
            show_mean=False, 
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(self._data.run_id[mask])
            perc68 = np.array([
                np.quantile(
                    self._data.distance[self._data.run_id == run], 
                    0.68
                )
                for run in runs
            ])

            mean_angle  = np.mean(perc68)
            std_angle   = np.std(perc68)

            plotter.add(
                runs, 
                perc68, 
                np.zeros_like(perc68), 
                phase.color,
                f"{phase.name}: ${mean_angle:.2f} \pm {std_angle:.2f}" r"^{\circ}$",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()

        save_figure(fig, self.stem, "_distance_vs_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_angle_vs_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $\alpha$ (deg)",
            xlim=(
                datetime.fromisoformat(ReProd26B.phases[0].date_min) - timedelta(days=2),
                datetime.fromisoformat(ReProd26B.phases[3].date_max) + timedelta(days=2),
            ),
            ylim=(0.0, 5.0),
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(self._data.run_id[mask])
            perc68 = np.array([
                np.quantile(
                    self._data.angle[self._data.run_id == run], 
                    0.68
                )
                for run in runs
            ])

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in runs
            ])

            mean_angle  = np.mean(perc68)
            std_angle   = np.std(perc68)

            plotter.add(
                start_sec,
                perc68,
                np.zeros_like(perc68),
                phase.color,
                f"{phase.name}: ${mean_angle:.2f} \pm {std_angle:.2f}" r"^{\circ}$",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_angle_vs_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance_vs_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $d_{\mathrm{mid}}$ (m)",
            xlim=(
                datetime.fromisoformat(ReProd26B.phases[0].date_min) - timedelta(days=2),
                datetime.fromisoformat(ReProd26B.phases[3].date_max) + timedelta(days=2),
            ),
            ylim=(0.0, 2.0),
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(self._data.run_id[mask])
            perc68 = np.array([
                np.quantile(
                    self._data.distance[self._data.run_id == run], 
                    0.68
                )
                for run in runs
            ])

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in runs
            ])

            mean_angle  = np.mean(perc68)
            std_angle   = np.std(perc68)

            plotter.add(
                start_sec,
                perc68,
                np.zeros_like(perc68),
                phase.color,
                f"{phase.name}: ${mean_angle:.2f} \pm {std_angle:.2f}" r"^{\circ}$",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_distance_vs_time", output_dir=self.output_dir)
        plt.close(fig)

# -------------------------------------------------------------------------------------------------
# Muon rate analysis
# -------------------------------------------------------------------------------------------------

class MuonRateAnalysis(BaseAnalysis):
    """
    Muon rate analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    dirpath_daq : str or Path
        Path to the DAQ directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def __init__(
            self,
            filepath: str,
            dirpath: str,
            dirpath_daq: str,
            output_dir:str = ".",
    ) -> None:
        super().__init__(filepath, dirpath, output_dir)
        self.dirpath_daq    = dirpath_daq

    def _load(self) -> None:
        self._data      = load_muon_rate(str(self.filepath), str(self.dirpath))
        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))

        # h = self._data.hist_cd_wp[0]
        # centers = (h.edges[1:] + h.edges[:-1]) / 2.0
        # widths  = h.edges[1:] - h.edges[:-1]
        # hist    = h.counts
        # err     = h.errors
        # fitter = ExponentialFitter(centers, hist, err)
        # results = fitter.fit() # results can be None

        # fig, ax = plt.subplots(figsize=(7, 6))

        # ax.fill_between(
        #     h.edges, 
        #     np.r_[hist, hist[-1]],
        #     step="post",
        #     color=CUSTOM_BLUE,
        #     alpha=0.15, 
        #     zorder=1,
        # )

        # ax.errorbar(
        #     centers, 
        #     hist,
        #     yerr=err, 
        #     xerr=widths / 2,
        #     label="",
        #     fmt="o", 
        #     color=CUSTOM_BLUE,
        #     markersize=4.5, 
        #     zorder=3,
        # )

        # x_smooth = uniform_bins(0.0, h.edges[-1], 500)
        # y_smooth = fitter.model(x_smooth, *results.popt)

        # ax.plot(
        #     x_smooth, y_smooth,
        #     linestyle="--", linewidth=1.6, color=BLACK, zorder=4,
        # )

        # A, tau   = results.popt
        # A_err, tau_err = results.perr

        # text = (
        #     r"$\chi^2/\mathrm{ndf} = %.1f / %d$" "\n"
        #     r"$p = %.3f$" "\n\n"
        #     r"$A = %.2f \pm %.2f$" "\n"
        #     r"$\tau = %.2f \pm %.2f~\mathrm{s}$"
        # ) % (
        #     results.chi2, results.ndf, results.pvalue,
        #     A, A_err,
        #     tau, tau_err,   # ms ==> \mus
        # )
        # ax.text(
        #     0.55, 0.9, text,
        #     transform=ax.transAxes,
        #     fontsize=18,
        #     verticalalignment="top",
        #     horizontalalignment="left",
        # )

        # ax.set_xlabel(r"$\Delta t_{\mu}$ (s)")
        # ax.set_ylabel(r"Entries")
        # ax.set_yscale("log")

        # fig.show()
        
        self._fit_cd_wp: list[FitResult] = []
        for h in self._data.hist_cd_wp:
            centers = (h.edges[1:] + h.edges[:-1]) / 2.0
            hist    = h.counts
            err     = h.errors

            fitter = ExponentialRateFitter(centers, hist, err)
            results = fitter.fit() # results can be None
            self._fit_cd_wp.append(results)

        self._fit_wp_only: list[FitResult] = []
        for h in self._data.hist_wp_only:
            centers = (h.edges[1:] + h.edges[:-1]) / 2.0
            hist    = h.counts
            err     = h.errors

            fitter = ExponentialRateFitter(centers, hist, err, xlim=(0.1, None))
            results = fitter.fit()
            self._fit_wp_only.append(results)

        self._fit_cd_only: list[FitResult] = []
        for run, h in zip(self._data.run_id, self._data.hist_cd_only):
            # No fit because the CD only rate is too low
            runtime = self._data_daq.duration_sec[self._data_daq.run_id == run][0]
            counts = h.underflow + h.overflow + np.sum(h.counts)
            results = FitResult(
                popt    = [0.0, counts / runtime],
                perr    = [0.0, np.sqrt(counts) / runtime],
                chi2    = 0.0,
                ndf     = 0,
                pvalue  = 0
            )
            self._fit_cd_only.append(results)

    def _plot(self) -> None:
        self._plot_rate_per_run()
        self._plot_rate_per_time()
        self._plot_efficiency_per_run()
        self._plot_efficiency_per_time()
        plt.show()

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot_rate_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"Muon rate (cps)", 
            ylim=(0.0, 10.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=2,
        )

        valid_wp_only   = np.array([fit is not None for fit in self._fit_wp_only])
        rate_wp_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        err_wp_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        mean_wp_only    = np.mean(rate_wp_only)
        std_wp_only     = np.sqrt(np.mean(err_wp_only**2))

        plotter.add(
            self._data.run_id[valid_wp_only],
            rate_wp_only[valid_wp_only],
            err_wp_only[valid_wp_only],
            GOOGLE_BLUE,
            rf"WP only rate: ${mean_wp_only:.2f} \pm {std_wp_only:.2f}$~cps",
        )

        valid_cd_wp   = np.array([fit is not None for fit in self._fit_cd_wp])
        rate_cd_wp    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_cd_wp])
        err_cd_wp     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_cd_wp])
        mean_cd_wp    = np.mean(rate_cd_wp)
        std_cd_wp     = np.sqrt(np.mean(err_cd_wp**2))

        plotter.add(
            self._data.run_id[valid_cd_wp],
            rate_cd_wp[valid_cd_wp],
            err_cd_wp[valid_cd_wp],
            GOOGLE_YELLOW,
            rf"CD-WP rate: ${mean_cd_wp:.2f} \pm {std_cd_wp:.2f}$~cps",
        )

        valid_cd_only   = np.array([fit is not None for fit in self._fit_cd_only])
        rate_cd_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_cd_only])
        err_cd_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_cd_only])
        mean_cd_only    = np.mean(rate_cd_only)
        std_cd_only     = np.sqrt(np.mean(err_cd_only**2))

        exp = int(np.floor(np.log10(mean_cd_only)))
        mantissa_mean = mean_cd_only / 10**exp
        mantissa_std = std_cd_only / 10**exp

        plotter.add(
            self._data.run_id[valid_cd_only],
            rate_cd_only[valid_cd_only],
            err_cd_only[valid_cd_only],
            GOOGLE_GREEN,
            rf"CD only rate: $({mantissa_mean:.2f} \pm {mantissa_std:.2f}) \times 10^{{{exp}}}$~cps",
        )

        valid_total = valid_wp_only & valid_cd_wp & valid_cd_only
        rate_total  = rate_wp_only + rate_cd_wp + rate_cd_only
        err_total   = np.sqrt(err_cd_wp**2 + err_cd_only**2 + err_wp_only**2)
        mean_total  = np.mean(rate_total)
        std_total   = np.sqrt(np.mean(err_total**2))

        plotter.add(
            self._data.run_id[valid_total],
            rate_total[valid_total],
            err_total[valid_total],
            BLACK,
            rf"Total rate: ${mean_total:.2f} \pm {std_total:.2f}$~cps",
        )

        for phase in ReProd26B.phases:
            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.1, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_rate_per_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"Muon rate (cps)", 
            ylim=(0.0, 10.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        mask = np.isin(self._data_daq.run_id, run_ids)

        valid_wp_only   = np.array([fit is not None for fit in self._fit_wp_only])
        rate_wp_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        err_wp_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        mean_wp_only    = np.mean(rate_wp_only)
        std_wp_only     = np.sqrt(np.mean(err_wp_only**2))

        plotter.add(
            self._data_daq.start_sec[mask][valid_wp_only],
            rate_wp_only[valid_wp_only],
            err_wp_only[valid_wp_only],
            GOOGLE_BLUE,
            rf"WP only rate: ${mean_wp_only:.2f} \pm {std_wp_only:.2f}$~cps",
        )

        valid_cd_wp   = np.array([fit is not None for fit in self._fit_cd_wp])
        rate_cd_wp    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_cd_wp])
        err_cd_wp     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_cd_wp])
        mean_cd_wp    = np.mean(rate_cd_wp)
        std_cd_wp     = np.sqrt(np.mean(err_cd_wp**2))

        plotter.add(
            self._data_daq.start_sec[mask][valid_cd_wp],
            rate_cd_wp[valid_cd_wp],
            err_cd_wp[valid_cd_wp],
            GOOGLE_YELLOW,
            rf"CD-WP rate: ${mean_cd_wp:.2f} \pm {std_cd_wp:.2f}$~cps",
        )

        valid_cd_only   = np.array([fit is not None for fit in self._fit_cd_only])
        rate_cd_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_cd_only])
        err_cd_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_cd_only])
        mean_cd_only    = np.mean(rate_cd_only)
        std_cd_only     = np.sqrt(np.mean(err_cd_only**2))

        exp = int(np.floor(np.log10(mean_cd_only)))
        mantissa_mean = mean_cd_only / 10**exp
        mantissa_std = std_cd_only / 10**exp

        plotter.add(
            self._data_daq.start_sec[mask][valid_cd_only],
            rate_cd_only[valid_cd_only],
            err_cd_only[valid_cd_only],
            GOOGLE_GREEN,
            rf"CD only rate: $({mantissa_mean:.2f} \pm {mantissa_std:.2f}) \times 10^{{{exp}}}$~cps",
        )

        valid_total = valid_wp_only & valid_cd_wp & valid_cd_only
        rate_total  = rate_wp_only + rate_cd_wp + rate_cd_only
        err_total   = np.sqrt(err_cd_wp**2 + err_cd_only**2 + err_wp_only**2)
        mean_total  = np.mean(rate_total)
        std_total   = np.sqrt(np.mean(err_total**2))

        plotter.add(
            self._data_daq.start_sec[mask][valid_total],
            rate_total[valid_total],
            err_total[valid_total],
            BLACK,
            rf"Total rate: ${mean_total:.2f} \pm {std_total:.2f}$~cps",
        )

        for phase in ReProd26B.phases:
            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.1, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_per_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_efficiency_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"Efficiency (\%)", 
            ylim=(0.90, 1.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=2,
        )

        counts_wp_only_per_run = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_wp_only])
        counts_cd_only_per_run = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_cd_only])
        counts_cd_wp_per_run   = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_cd_wp])

        mask = np.isin(self._data_daq.run_id, self._data.run_id)
        daq_run_id = self._data_daq.run_id[mask]
        duration = self._data_daq.duration_sec[mask]

        efficiency = (
            1.0 - (
                counts_wp_only_per_run * 0.002
                + (counts_cd_wp_per_run + counts_cd_only_per_run) * 0.007
            ) / duration
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= daq_run_id,
                daq_run_id <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            mean_efficiency = np.mean(efficiency[mask])
            std_efficiency  = np.std(efficiency[mask])

            plotter.add(
                self._data.run_id[mask], 
                efficiency[mask], 
                np.zeros_like(efficiency[mask]), 
                phase.color,
                f"{phase.name}: ${mean_efficiency:.2f} \pm {std_efficiency:.2f}" r"\%$",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_efficiency_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_efficiency_per_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"Efficiency (\%)", 
            ylim=(0.9, 1.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=2,
        )

        counts_wp_only_per_run = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_wp_only])
        counts_cd_only_per_run = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_cd_only])
        counts_cd_wp_per_run   = np.array([np.sum(data.counts) + data.underflow + data.overflow for data in self._data.hist_cd_wp])

        mask = np.isin(self._data_daq.run_id, self._data.run_id)
        daq_run_id = self._data_daq.run_id[mask]
        duration = self._data_daq.duration_sec[mask]

        efficiency = (
            1.0 - (
                counts_wp_only_per_run * 0.002
                + (counts_cd_wp_per_run + counts_cd_only_per_run) * 0.007
            ) / duration
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= daq_run_id,
                daq_run_id <= phase.run_max,
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(daq_run_id[mask])
            effs = efficiency[mask]

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in runs
            ])

            mean_angle  = np.mean(effs)
            std_angle   = np.std(effs)

            plotter.add(
                start_sec,
                effs,
                np.zeros_like(effs),
                phase.color,
                f"{phase.name}: ${mean_angle:.2f} \pm {std_angle:.2f}" r"\%$",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_efficiency_per_time", output_dir=self.output_dir)
        plt.close(fig)