from __future__ import annotations
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalysis
from config import (
    ReProd26B, 
)
from fits import (
    FitResult, 
    ExponentialRateFitter, 
)
from loader import (
    load_multiplicity, 
    load_lifetime_daq, 
)
from plotters import (
    RunEvolutionPlotter, 
    TimeEvolutionPlotter, 
)
from utils import (
    save_figure, 
    uniform_bins, 
)


class MultiplicityAnalysis(BaseAnalysis):
    """
    Multiplicity candidate analysis.

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
        self._data      = load_multiplicity(str(self.filepath), str(self.dirpath))
        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))

        bins = uniform_bins(0.0, 1.0, 100)
        centers = 0.5 * (bins[1:] + bins[:-1])

        run_ids = np.unique(self._data.run_id)

        self._fit: list[FitResult] = []

        for run_id in run_ids:
            mask = self._data.run_id == run_id

            sec = self._data.sec[mask].astype(np.float64)
            sec += self._data.nsec[mask] * 1e-9
            ts_diff = np.diff(sec)

            hist, _ = np.histogram(ts_diff, bins=bins)
            err = np.sqrt(hist)

            fitter = ExponentialRateFitter(centers, hist, err)
            results = fitter.fit()
            self._fit.append(results)

        #     fig, ax = plt.subplots(figsize=(7, 6))

        #     ax.fill_between(
        #         bins,
        #         np.r_[hist, hist[-1]],
        #         step="post",
        #         color=_COLOR_MULTIPLICITY,
        #         alpha=0.15,
        #         zorder=1,
        #     )

        #     ax.errorbar(
        #         centers,
        #         hist,
        #         yerr=err,
        #         xerr=widths / 2,
        #         fmt="o",
        #         color=_COLOR_MULTIPLICITY,
        #         markersize=4.5,
        #         zorder=3,
        #     )

        #     x_smooth = uniform_bins(0.0, bins[-1], 500)
        #     y_smooth = fitter.model(x_smooth, *results.popt)

        #     ax.plot(
        #         x_smooth,
        #         y_smooth,
        #         "--",
        #         linewidth=1.6,
        #         color="black",
        #         zorder=4,
        #     )

        #     A, tau = results.popt
        #     A_err, tau_err = results.perr

        #     text = (
        #         r"$P(\chi^{2}/\mathrm{ndf} = %.1f / %d) = %.3f$" "\n\n"
        #         r"$A = %.2f \pm %.2f$" "\n"
        #         r"$\tau = %.2f \pm %.2f~\mathrm{s}$"
        #     ) % (
        #         results.chi2,
        #         results.ndf,
        #         results.pvalue,
        #         A,
        #         A_err,
        #         tau,
        #         tau_err,
        #     )

        #     ax.text(
        #         0.97,
        #         0.97,
        #         text,
        #         transform=ax.transAxes,
        #         fontsize=18,
        #         va="top",
        #         ha="right",
        #     )

        #     ax.set_xlabel(r"$\Delta t$ (s)")
        #     ax.set_ylabel("Entries")
        #     ax.set_yscale("log")

        #     fig.tight_layout()
        #     fig.savefig(f"{self.output_dir}/pdf/test_run_{run_id}.pdf")
        #     plt.close(fig)

    def _plot(self) -> None:
        self._plot_rate_per_run()
        self._plot_rate_per_time()
        self._plot_efficiency_per_run()
        self._plot_efficiency_per_time()

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot_rate_per_run(self):
        plotter = RunEvolutionPlotter(
            r"Multiplicity rate (cps)", 
            ylim=(0.0, 10.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        rate    = np.array([fit.popt[1] for fit in self._fit])
        err     = np.array([fit.perr[1] for fit in self._fit])

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= run_ids,
                run_ids <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            r    = rate[mask]
            e    = err[mask]
            mean = np.mean(r)
            std  = np.sqrt(np.mean(e**2))

            plotter.add(
                run_ids[mask],
                r,
                e,
                phase.color,
                rf"{phase.name}: ${mean:.2f} \pm {std:.2f}$~cps",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.1, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_rate_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_rate_per_time(self):
        plotter = TimeEvolutionPlotter(
            r"Multiplicity rate (cps)",
            xlim=(
                datetime.fromisoformat(ReProd26B.phases[0].date_min) - timedelta(days=2),
                datetime.fromisoformat(ReProd26B.phases[3].date_max) + timedelta(days=2),
            ),
            ylim=(0.0, 10.0),
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        rate    = np.array([fit.popt[1] for fit in self._fit])
        err     = np.array([fit.perr[1] for fit in self._fit])

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= run_ids,
                run_ids <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            r    = rate[mask]
            e    = err[mask]
            mean = np.mean(r)
            std  = np.sqrt(np.mean(e**2))

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in run_ids[mask]
            ])

            plotter.add(
                start_sec,
                r,
                e,
                phase.color,
                rf"{phase.name}: ${mean:.2f} \pm {std:.2f}$~cps",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_rate_per_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_efficiency_per_run(self):
        plotter = RunEvolutionPlotter(
            r"Efficiency (\%)", 
            ylim=(0.98, 0.99),
            show_mean=False, 
            show_band=False,
            legend_ncol=2,
        )

        run_ids   = np.unique(self._data.run_id)
        rate      = np.array([fit.popt[1] for fit in self._fit])
        rate_err  = np.array([fit.perr[1] for fit in self._fit])
        T = 0.001 + 0.001 + 0.000220
        # 1 ms before prompt
        # 1 ms after delayed
        # 220 us mean neutron capture time
        eff       = np.exp(-rate * T) 
        eff_err   = T * eff * rate_err

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= run_ids,
                run_ids <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            x = eff[mask]
            e = eff_err[mask]
            mean = np.mean(x)
            std  = np.sqrt(np.mean(e**2))

            plotter.add(
                run_ids[mask],
                x,
                e,
                phase.color,
                rf"{phase.name}: ${mean:.3f} \pm {std:.3f}$~cps",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.1, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_efficiency_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_efficiency_per_time(self):
        plotter = TimeEvolutionPlotter(
            r"Efficiency (\%)",
            xlim=(
                datetime.fromisoformat(ReProd26B.phases[0].date_min) - timedelta(days=2),
                datetime.fromisoformat(ReProd26B.phases[3].date_max) + timedelta(days=2),
            ),
            ylim=(0.98, 0.99), 
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        run_ids   = np.unique(self._data.run_id)
        rate      = np.array([fit.popt[1] for fit in self._fit])
        rate_err  = np.array([fit.perr[1] for fit in self._fit])
        T = 0.001 + 0.001 + 0.000220
        # 1 ms before prompt
        # 1 ms after delayed
        # 220 us mean neutron capture time
        eff       = np.exp(-rate * T) 
        eff_err   = T * eff * rate_err

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= run_ids,
                run_ids <= phase.run_max,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            x = eff[mask]
            e = eff_err[mask]
            mean = np.mean(x)
            std  = np.sqrt(np.mean(e**2))

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in run_ids[mask]
            ])

            plotter.add(
                start_sec,
                x,
                e,
                phase.color,
                rf"{phase.name}: ${mean:.3f} \pm {std:.3f}$~cps",
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