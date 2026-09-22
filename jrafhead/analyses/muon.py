from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from jrafhead.config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_GREEN,
    CUSTOM_RED,
    GOOGLE_BLUE,
    GOOGLE_GREEN,
    GOOGLE_YELLOW,
    ReProd26B,
)
from jrafhead.fits import (
    ExponentialRateFitter,
    FitResult,
)
from jrafhead.loader import (
    load_lifetime_daq,
    load_muon_length,
    load_muon_multiplicity,
    load_muon_performance,
    load_muon_rate,
)
from jrafhead.plotters import (
    MuonLengthPlotter,
    MuonMultiplicityPlotter,
    MuonPerformanceAngle,
    MuonPerformanceDistance,
    MuonPerformanceMetricClippingness,
    RunEvolutionPlotter,
    TimeEvolutionPlotter,
)
from jrafhead.utils import (
    rebin_histogram,
    save_figure,
    uniform_bins,
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
        self._plot_chi2_vs_run()
        self._plot_angle_vs_time()
        self._plot_distance_vs_time()
        self._plot_chi2_vs_time()
        self._plot_angle_vs_run_subset()
        self._plot_distance_vs_run_subset()
        self._plot_chi2_vs_run_subset()
        self._plot_angle_vs_time_subset()
        self._plot_distance_vs_time_subset()
        self._plot_chi2_vs_time_subset()
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

            mean_distance  = np.mean(perc68)
            std_distance   = np.std(perc68)

            plotter.add(
                runs, 
                perc68, 
                np.zeros_like(perc68), 
                phase.color,
                f"{phase.name}: ${mean_distance:.2f} \pm {std_distance:.2f}" r"$~m",
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

    def _plot_chi2_vs_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}}$", 
            xlim=(ReProd26B.phases[0].run_min - 100, ReProd26B.phases[3].run_max + 100),
            ylim=(0.0, 10.0),
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
            mean = np.array([
                np.mean(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
            std = np.array([
                np.std(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
        
            mean_chi2 = np.mean(mean)
            std_chi2  = np.std(mean)
            stat_chi2 = np.sqrt(np.sum(std**2)) / len(mean)
        
            plotter.add(
                runs, 
                mean, 
                np.zeros_like(mean), 
                phase.color,
                f"{phase.name}: ${mean_chi2:.2f} \pm {std_chi2:.2f}" r"$",
            )
        
            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.05, 20, 0.0
            )
        
        fig, _ = plotter.plot()
        
        save_figure(fig, self.stem, "_chi2_vs_run", output_dir=self.output_dir)
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

    def _plot_chi2_vs_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}}$",
            xlim=(
                datetime.fromisoformat(ReProd26B.phases[0].date_min) - timedelta(days=2),
                datetime.fromisoformat(ReProd26B.phases[3].date_max) + timedelta(days=2),
            ),
            ylim=(0.0, 10.0),
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
            mean = np.array([
                np.mean(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
            std = np.array([
                np.std(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in runs
            ])
        
            mean_chi2 = np.mean(mean)
            std_chi2  = np.std(mean)
            stat_chi2 = np.sqrt(np.sum(std**2)) / len(mean)

            plotter.add(
                start_sec,
                mean,
                np.zeros_like(mean),
                phase.color,
                f"{phase.name}: ${mean_chi2:.2f} \pm {std_chi2:.2f}" r"$",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_chi2_vs_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_angle_vs_run_subset(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $\alpha$ (deg)", 
            xlim=(10550, 11000),
            ylim=(0.0, 3.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
        )

        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
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
                CUSTOM_BLUE,
                r"$\bar{\alpha} = " f"{mean_angle:.1f}" r" \pm " f"{std_angle:.1f}" r"^{\circ}$",
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_subset_angle_vs_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance_vs_run_subset(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $d_{\mathrm{mid}}$ (m)", 
            xlim=(10550, 11000),
            ylim=(0.0, 1.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
        )

        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
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
                CUSTOM_RED,
                r"$\bar{d}_{\mathrm{mid}} = " f"{mean_angle:.2f}" r" \pm " f"{std_angle:.2f}" r"$~m",
            )

        fig, _ = plotter.plot()

        save_figure(fig, self.stem, "_subset_distance_vs_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_chi2_vs_run_subset(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}}$", 
            xlim=(10550, 11000),
            ylim=(0.0, 4.0),
            show_mean=True, 
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
        )
        
        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
            )
            if bool(np.all(np.logical_not(mask))):
                continue
        
            runs = np.unique(self._data.run_id[mask])
            mean = np.array([
                np.mean(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
            std = np.array([
                np.std(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
        
            mean_chi2 = np.mean(mean)
            std_chi2  = np.std(mean)
            stat_chi2 = np.sqrt(np.sum(std**2)) / len(mean)
        
            plotter.add(
                runs, 
                mean, 
                np.zeros_like(mean), 
                CUSTOM_GREEN,
                r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}} = " f"{mean_chi2:.1f}" r" \pm " f"{std_chi2:.1f}" r"$",
            )
        
        fig, _ = plotter.plot()
        
        save_figure(fig, self.stem, "_subset_chi2_vs_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_angle_vs_time_subset(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $\alpha$ (deg)",
            xlim=(
                datetime.fromisoformat("2025-10-08"),
                datetime.fromisoformat("2025-10-31"),
            ),
            ylim=(0.0, 3.0),
            show_mean=True,
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
            fuze_by_date=False,
        )

        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
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
                CUSTOM_BLUE,
                r"$\bar{\alpha} = " f"{mean_angle:.1f}" r" \pm " f"{std_angle:.1f}" r"^{\circ}$",
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_subset_angle_vs_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_distance_vs_time_subset(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$68^{\mathrm{th}}$ percentile of $d_{\mathrm{mid}}$ (m)",
            xlim=(
                datetime.fromisoformat("2025-10-08"),
                datetime.fromisoformat("2025-10-31"),
            ),
            ylim=(0.0, 1.0),
            show_mean=True,
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
            fuze_by_date=False,
        )

        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
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
                CUSTOM_RED,
                r"$\bar{d}_{\mathrm{mid}} = " f"{mean_angle:.2f}" r" \pm " f"{std_angle:.2f}" r"$~m",
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_subset_distance_vs_time", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_chi2_vs_time_subset(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}}$",
            xlim=(
                datetime.fromisoformat("2025-10-08"),
                datetime.fromisoformat("2025-10-31"),
            ),
            ylim=(0.0, 4.0),
            show_mean=True,
            show_band=True,
            legend_ncol=1,
            legend_bbox=(0.85, 0.95),
            fuze_by_date=False,
        )

        for phase in ReProd26B.phases:
            if phase.name != "Phase 1":
                continue
            mask = np.logical_and(
                10550 <= self._data.run_id,
                self._data.run_id <= 11000,
            )
            if bool(np.all(np.logical_not(mask))):
                continue

            runs = np.unique(self._data.run_id[mask])
            mean = np.array([
                np.mean(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])
            std = np.array([
                np.std(
                    self._data.target_quality[self._data.run_id == run]
                )
                for run in runs
            ])

            start_sec = np.array([
                self._data_daq.start_sec[self._data_daq.run_id == run][0]
                for run in runs
            ])
        
            mean_chi2 = np.mean(mean)
            std_chi2  = np.std(mean)
            stat_chi2 = np.sqrt(np.sum(std**2)) / len(mean)

            plotter.add(
                start_sec,
                mean,
                np.zeros_like(mean),
                CUSTOM_GREEN,
                r"$\bar{\chi}^{2}_{\mathrm{CD}+\mathrm{WP}} = " f"{mean_chi2:.1f}" r" \pm " f"{std_chi2:.1f}" r"$",
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_subset_chi2_vs_time", output_dir=self.output_dir)
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
            hist    = h.counts
            err     = h.errors

            fitter = ExponentialRateFitter(h.edges, hist, err)
            results = fitter.fit() # results can be None
            self._fit_cd_wp.append(results)

        self._fit_wp_only: list[FitResult] = []
        for h in self._data.hist_wp_only:
            hist    = h.counts
            err     = h.errors

            fitter = ExponentialRateFitter(h.edges, hist, err, xlim=(0.1, None))
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
            show_grid=False,
            legend_ncol=2,
        )

        valid_wp_only   = np.array([fit is not None for fit in self._fit_wp_only])
        rate_wp_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        err_wp_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        mean_wp_only    = np.mean(rate_wp_only)
        stat_wp_only    = np.sqrt(np.sum(err_wp_only**2)) / len(rate_wp_only)
        spread_wp_only  = np.std(rate_wp_only, ddof=1)
        std_wp_only     = np.sqrt(stat_wp_only**2 + spread_wp_only**2)

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
        stat_cd_wp    = np.sqrt(np.sum(err_cd_wp**2)) / len(rate_cd_wp)
        spread_cd_wp  = np.std(rate_cd_wp, ddof=1)
        std_cd_wp     = np.sqrt(stat_cd_wp**2 + spread_cd_wp**2)

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
        stat_cd_only    = np.sqrt(np.sum(err_cd_only**2)) / len(rate_cd_only)
        spread_cd_only  = np.std(rate_cd_only, ddof=1)
        std_cd_only     = np.sqrt(stat_cd_only**2 + spread_cd_only**2)

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

        valid_total  = valid_wp_only & valid_cd_wp & valid_cd_only
        rate_total   = rate_wp_only + rate_cd_wp + rate_cd_only
        err_total    = np.sqrt(err_cd_wp**2 + err_cd_only**2 + err_wp_only**2)
        mean_total   = np.mean(rate_total)
        stat_total   = np.sqrt(np.sum(err_total**2)) / len(rate_total)
        spread_total = np.std(rate_total, ddof=1)
        std_total    = np.sqrt(stat_total**2 + spread_total**2)

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
            show_grid=False,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        mask = np.isin(self._data_daq.run_id, run_ids)

        valid_wp_only   = np.array([fit is not None for fit in self._fit_wp_only])
        rate_wp_only    = np.array([fit.popt[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        err_wp_only     = np.array([fit.perr[1] if fit is not None else 0.0 for fit in self._fit_wp_only])
        mean_wp_only    = np.mean(rate_wp_only)
        stat_wp_only    = np.sqrt(np.sum(err_wp_only**2)) / len(rate_wp_only)
        spread_wp_only  = np.std(rate_wp_only, ddof=1)
        std_wp_only     = np.sqrt(stat_wp_only**2 + spread_wp_only**2)

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
        stat_cd_wp    = np.sqrt(np.sum(err_cd_wp**2)) / len(rate_cd_wp)
        spread_cd_wp  = np.std(rate_cd_wp, ddof=1)
        std_cd_wp     = np.sqrt(stat_cd_wp**2 + spread_cd_wp**2)

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
        stat_cd_only    = np.sqrt(np.sum(err_cd_only**2)) / len(rate_cd_only)
        spread_cd_only  = np.std(rate_cd_only, ddof=1)
        std_cd_only     = np.sqrt(stat_cd_only**2 + spread_cd_only**2)

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

        valid_total  = valid_wp_only & valid_cd_wp & valid_cd_only
        rate_total   = rate_wp_only + rate_cd_wp + rate_cd_only
        err_total    = np.sqrt(err_cd_wp**2 + err_cd_only**2 + err_wp_only**2)
        mean_total   = np.mean(rate_total)
        stat_total   = np.sqrt(np.sum(err_total**2)) / len(rate_total)
        spread_total = np.std(rate_total, ddof=1)
        std_total    = np.sqrt(stat_total**2 + spread_total**2)

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
            show_grid=False,
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
            show_grid=False,
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

# -------------------------------------------------------------------------------------------------
# Muon length analysis
# -------------------------------------------------------------------------------------------------

class MuonLengthAnalysis(BaseAnalysis):
    """
    Muon length analysis.

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
        self._data      = load_muon_length(str(self.filepath), str(self.dirpath))
        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot(self) -> None:
        self._print_total_length()
        self._plot_accumulated_length()
        self._plot_length_per_run()
        self._plot_length_per_time()
        plt.show()

    def _print_total_length(self) -> None:
        print(f"Total muon length: {np.sum(self._data.total_length)}")
        print(f"Total number of muon: {np.sum(self._data.total_muon)}")
        print(f"Mean muon length: {np.sum(np.sum(self._data.total_length)) / np.sum(self._data.total_muon)}")

    def _plot_accumulated_length(self) -> None:
        bins    = uniform_bins(0.0, 40.0, 100)
        plotter = MuonLengthPlotter(bins=bins)

        mask = np.logical_and(
            10000 <= self._data.run_id,
            self._data.run_id <= 10100
        )
        hist = np.sum([h.counts for h, m in zip(self._data.hist_length, mask) if m], axis=0)
        hist = rebin_histogram(self._data.hist_length[0].edges, hist, bins)
        err  = np.sqrt(hist)
        # err  = np.sqrt(np.sum([h.errors ** 2 for h, m in zip(self._data.hist_length, mask) if m], axis=0))

        plotter.add_histogram(
            hist, err,
            BLACK, fillcolor=BLACK,
        )
        # cmap   = plt.get_cmap("YlGnBu")
        # colors = [mcolors.to_hex(cmap(i / 20.0)) for i in range(20)]
        # for k in range(10):
        #     plotter.add_histogram(
        #         self._data.hist_length[k].counts, self._data.hist_length[k].errors,
        #         colors[k], label=k,
        #         )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_accumulate", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_length_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$L_{\mu}$ (m)",
            ylim=(0.0, 30.0),
            show_mean=True, 
            show_band=True,
            show_grid=False,
            legend_ncol=1,
        )

        length = self._data.total_length / self._data.total_muon
        mean   = np.mean(self._data.total_length / self._data.total_muon)
        err    = np.std(self._data.total_length / self._data.total_muon, ddof=1)

        plotter.add(
            self._data.run_id,
            length,
            np.zeros_like(length),
            BLACK,
            rf"Muon length: ${mean:.3f} \pm {err:.3f}$~m",
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

    def _plot_length_per_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$L_{\mu}$ (m)",
            ylim=(0.0, 30.0),
            show_mean=True, 
            show_band=True,
            show_grid=False,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        mask = np.isin(self._data_daq.run_id, run_ids)

        length = self._data.total_length / self._data.total_muon
        mean   = np.mean(self._data.total_length / self._data.total_muon)
        err    = np.std(self._data.total_length / self._data.total_muon, ddof=1)

        plotter.add(
            self._data_daq.start_sec[mask],
            length,
            np.zeros_like(length),
            BLACK,
            rf"Muon length: ${mean:.3f} \pm {err:.3f}$~m",
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

# -------------------------------------------------------------------------------------------------
# Muon multiplicity analysis
# -------------------------------------------------------------------------------------------------

class MuonMultiplicityAnalysis(BaseAnalysis):
    """
    Muon multiplicity analysis.

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
        self._data      = load_muon_multiplicity(str(self.filepath), str(self.dirpath))
        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot(self) -> None:
        self._print_total_multiplicity()
        self._plot_accumulated_multiplicity()
        self._plot_multiplicity_per_run()
        self._plot_multiplicity_per_time()
        plt.show()

    def _print_total_multiplicity(self) -> None:
        hist = np.sum([h.counts for h in self._data.hist_multipliticy], axis=0)
        print(f"Average muon multiplicity: {np.sum(hist * self._data.hist_multipliticy[0].edges[:-1]) / np.sum(hist)}")

    def _plot_accumulated_multiplicity(self) -> None:
        plotter = MuonMultiplicityPlotter()

        hist = np.sum([h.counts for h in self._data.hist_multipliticy], axis=0)
        err  = np.sqrt(np.sum([h.errors ** 2 for h in self._data.hist_multipliticy], axis=0))

        plotter.add_histogram(
            hist, err,
            BLACK, fillcolor=BLACK,
        )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_accumulate", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_multiplicity_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"$n_{\mathrm{track}}$",
            ylim=(0.0, 2.0),
            show_mean=True, 
            show_band=True,
            show_grid=False,
            legend_ncol=1,
        )

        multiplicity = np.array([np.sum(h.counts * h.edges[:-1]) / np.sum(h.counts) for h in self._data.hist_multipliticy])
        mean         = np.mean(multiplicity)
        err          = np.std(multiplicity, ddof=1)

        plotter.add(
            self._data.run_id,
            multiplicity,
            np.zeros_like(multiplicity),
            BLACK,
            rf"Muon multiplicity: ${mean:.3f} \pm {err:.3f}$",
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

    def _plot_multiplicity_per_time(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"$n_{\mathrm{track}}$",
            ylim=(0.0, 2.0),
            show_mean=True, 
            show_band=True,
            show_grid=False,
            legend_ncol=2,
        )

        run_ids = np.unique(self._data.run_id)
        mask = np.isin(self._data_daq.run_id, run_ids)

        multiplicity = np.array([np.sum(h.counts * h.edges[:-1]) / np.sum(h.counts) for h in self._data.hist_multipliticy])
        mean         = np.mean(multiplicity)
        err          = np.std(multiplicity, ddof=1)

        plotter.add(
            self._data_daq.start_sec[mask],
            multiplicity,
            np.zeros_like(multiplicity),
            BLACK,
            rf"Muon multiplicity: ${mean:.3f} \pm {err:.3f}$",
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