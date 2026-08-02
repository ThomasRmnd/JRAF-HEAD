from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from config import (
    BLACK,
    CUSTOM_BLUE,
    ReProd26B,
)
from loader import (
    load_ibd,
    load_lifetime_daq,
    load_lifetime_veto,
)
from plotters import (
    DelayedEnergyPlotter,
    MuonVetoDistributionPlotter,
    PromptDelayedDistancePlotter,
    PromptDelayedTimePlotter,
    PromptEnergyPlotter,
    SpatialDistributionPlotter,
    TimeEvolutionPlotter,
)
from utils import (
    compute_3d_distance,
    compute_geometry,
    save_figure,
    sum_by_day,
    uniform_bins,
)

from .base import BaseAnalysis

# Default muon-veto bin ranges for the IBD analysis
_MUON_VETO_XBINS = uniform_bins(0.0, 1.2, 50)
_MUON_VETO_YBINS = uniform_bins(0.0, 3.0, 50)


class IBDAnalysis(BaseAnalysis):
    """
    IBD candidate analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    dirpath_daq : str or Path
        Path to the DAQ directory inside the ROOT file.
    dirpath_veto : str or Path
        Path to the veto directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def __init__(
            self,
            filepath: str,
            dirpath: str,
            dirpath_daq: str,
            dirpath_veto: str,
            output_dir:str = ".",
    ) -> None:
        super().__init__(filepath, dirpath, output_dir)
        self.dirpath_daq    = dirpath_daq
        self.dirpath_veto   = dirpath_veto

    def _load(self) -> None:
        self._data          = load_ibd(str(self.filepath), str(self.dirpath))
        self._geom_p        = compute_geometry(self._data.pos_p_mm)
        self._geom_d        = compute_geometry(self._data.pos_d_mm)
        self._distance      = compute_3d_distance(self._data.pos_p_mm, self._data.pos_d_mm)

        self._data_daq      = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))
        self._data_veto     = load_lifetime_veto(str(self.filepath), str(self.dirpath_veto))

    def _plot(self) -> None:
        self._plot_prompt_energy_nmo()
        self._plot_prompt_energy_207days()
        self._plot_prompt_energy_normal()
        self._plot_delayed_energy()
        self._plot_dt()
        self._plot_dr()
        self._plot_spatial()
        self._plot_muon_veto()
        self._plot_cosmo_rate_with_neu()
        self._plot_cosmo_rate()
        self._plot_rate()

        del self._data_daq
        del self._data_veto

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot_prompt_energy_nmo(self) -> None:
        plotter = PromptEnergyPlotter(binmode="nmo")
        plotter.add(self._data.e_p, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_nmo", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_207days(self) -> None:
        plotter = PromptEnergyPlotter(binmode="207days")
        plotter.add(self._data.e_p, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_207days", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_normal(self) -> None:
        plotter = PromptEnergyPlotter(binmode="normal")
        plotter.add(self._data.e_p, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_normal", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_delayed_energy(self) -> None:
        plotter = DelayedEnergyPlotter(ylim=(0.7, None), yscale="log")
        plotter.add(self._data.e_d, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dt(self) -> None:
        plotter = PromptDelayedTimePlotter()
        plotter.add(self._data.dt_p_d_ms, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_p_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dr(self) -> None:
        plotter = PromptDelayedDistancePlotter()
        plotter.add(self._distance, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dr_p_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_spatial(self) -> None:
        plotter = SpatialDistributionPlotter(
            xlabel=r"$\rho_{p}^{2}$ (m$^2$)", 
            ylabel=r"$z_{p}$ (m)", 
        )
        fig, _ = plotter.plot(self._geom_p.rho2_m2, self._geom_p.z_m)
        save_figure(fig, self.stem, "_rho_z_p", output_dir=self.output_dir)
        plt.close(fig)

        plotter = SpatialDistributionPlotter(
            xlabel=r"$\rho_{d}^{2}$ (m$^2$)", 
            ylabel=r"$z_{d}$ (m)", 
        )
        fig, _ = plotter.plot(self._geom_d.rho2_m2, self._geom_d.z_m)
        save_figure(fig, self.stem, "_rho_z_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_muon_veto(self) -> None:
        plotter = MuonVetoDistributionPlotter(
            _MUON_VETO_XBINS,
            _MUON_VETO_YBINS,
            xlabel=r"$\Delta t_{\mu-p}$ (s)", 
            ylabel=r"$d_{\mu-p}$ (m)", 
            is_signal_region=True, 
            fit_info_fontsize=11, 
            fit_info_loc=(0.95, 0.85), 
        )
        fig, _ = plotter.plot(self._data.dt_mu2p, self._data.dlat_mu2p)
        save_figure(fig, self.stem, "_dt_dlat_p", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_cosmo_rate_with_neu(self) -> None:
        from plotters import Li9He8RateEstimationPlotter

        bins    = uniform_bins(0.0, 10.0, 200)
        widths  = bins[1:] - bins[:-1]
        hist, _ = np.histogram(self._data.dt_last_mu_with_neu, bins=bins)
        err     = np.sqrt(hist)
        hist    = hist / widths
        err     = err / widths
        plotter = Li9He8RateEstimationPlotter(
            bins=bins,
            xlim=(bins[0], bins[-1]),
            ylabel=f"Entries / {widths[0]:g}~s",
            yscale="log", 
            ylim=(0.7, 2.0 * np.max(hist)), 
        )
        plotter.add_histogram(hist, err, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_last_mu_with_neu", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_cosmo_rate(self) -> None:
        from plotters import Li9He8RateEstimationPlotter

        bins    = uniform_bins(0.0, 10.0, 200)
        widths  = bins[1:] - bins[:-1]
        hist, _ = np.histogram(self._data.dt_last_mu, bins=bins)
        err     = np.sqrt(hist)
        hist    = hist / widths
        err     = err / widths
        plotter = Li9He8RateEstimationPlotter(
            bins=bins,
            xlim=(bins[0], bins[-1]),
            ylabel=f"Entries / {widths[0]:g}~s",
            yscale="log", 
            ylim=(0.7, 2.0 * np.max(hist)), 
        )
        plotter.add_histogram(hist, err, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_last_mu", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_rate(self) -> None:
        # Calculate the lifetime per run
        daq_time = dict(zip(self._data_daq.run_id, self._data_daq.duration_sec))
        
        veto_time = defaultdict(float)
        for run, veto_sec in zip(self._data_veto.run_id, self._data_veto.sec):
            veto_time[run] += veto_sec

        runs = np.array(sorted(daq_time.keys()))

        lifetime = np.array([
            daq_time[run] - veto_time.get(run, 0.0)
            for run in runs
        ])

        # Calculate the number of IBD per run
        ibd_count = Counter(self._data.run_id)

        nibd = np.array([
            ibd_count.get(run, 0)
            for run in runs
        ])

        plotter = TimeEvolutionPlotter(
            r"IBD rate (cpd)", 
            ylim=(0.0, None),
            mode="sum", 
            show_mean=False, 
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data_daq.run_id,
                self._data_daq.run_id <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            dates, lifetime_day, nibd_day = sum_by_day(
                self._data_daq.start_sec[mask],
                lifetime[mask],
                nibd[mask],
            )

            dates = [d.timestamp() for d in dates]
            lifetime_day /= (24.0 * 3600.0)

            rate = nibd_day / lifetime_day
            err = np.sqrt(nibd_day) / lifetime_day

            mean_rate  = np.mean(rate)
            std_rate   = np.sqrt(np.mean(err**2))

            plotter.add(
                dates,
                rate,
                err,
                phase.color,
                f"{phase.name}: ${mean_rate:.2f} \pm {std_rate:.2f}$~cpd",
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.05, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_per_date", output_dir=self.output_dir)
        plt.close(fig)