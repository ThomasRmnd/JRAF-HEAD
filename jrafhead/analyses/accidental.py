from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_RED,
    DELAYED_ENERGY_HYDROGEN_BINS,
    PROMPT_ENERGY_BINS_207DAYS,
    PROMPT_ENERGY_BINS_NMO,
    PROMPT_ENERGY_BINS_UNIFORM,
    ReProd26B,
)
from loader import (
    load_accidental,
    load_lifetime_daq,
    load_lifetime_veto,
)
from plotters import (
    DelayedEnergyHydrogenPlotter,
    DelayedEnergyPlotter,
    MuonVetoDistributionPlotter,
    PromptDelayedDistancePlotter,
    PromptDelayedTimePlotter,
    PromptEnergyPlotter,
    RelativeUncertaintyPromptEnergyPlotter,
    RunEvolutionPlotter,
    SpatialDistributionPlotter,
    TimeEvolutionPlotter,
)
from utils import (
    Timestamp,
    compute_3d_distance,
    compute_geometry,
    relative_uncertainty_rescale,
    save_figure,
    sum_by_day,
    uniform_bins,
)

from .base import BaseAnalysis

# Default muon-veto bin ranges for the accidental analysis
_MUON_VETO_XBINS = np.linspace(0.0, 1.2, 51)
_MUON_VETO_YBINS = np.linspace(0.0, 3.0, 51)


class AccidentalAnalysis(BaseAnalysis):
    """
    Accidental candidate analysis.

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
        self._data          = load_accidental(str(self.filepath), str(self.dirpath))
        self._geom_p        = compute_geometry(self._data.pos_p_mm)
        self._geom_d        = compute_geometry(self._data.pos_d_mm)
        self._distance      = compute_3d_distance(self._data.pos_p_mm, self._data.pos_d_mm)

        self._data_daq      = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))
        self._data_veto     = load_lifetime_veto(str(self.filepath), str(self.dirpath_veto))

    def _plot(self) -> None:
        self._plot_prompt_energy_nmo()
        self._plot_prompt_energy_207days()
        self._plot_prompt_energy_normal()
        self._plot_relative_uncertainty_prompt_energy_nmo()
        self._plot_relative_uncertainty_prompt_energy_207days()
        self._plot_relative_uncertainty_prompt_energy_normal()
        self._plot_delayed_energy()
        self._plot_dt()
        self._plot_dr()
        self._plot_spatial()
        self._plot_muon_veto()
        self._plot_rate_by_run()
        self._plot_rate_by_date()
        self._plot_prompt_delayed_shape()
        plt.show()

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

    def _plot_relative_uncertainty_prompt_energy_nmo(self) -> None:
        tot_daq      = np.sum([Timestamp(sec, nsec) for sec, nsec in zip(self._data_daq.duration_sec, self._data_daq.duration_nsec)])
        tot_lifetime = tot_daq.to_sec() / 3600.0 / 24.0

        _DAYS_PER_YEAR = 365.25
        periods_days   = [tot_lifetime, 325.0] + [n * _DAYS_PER_YEAR for n in range(2, 11, 2)]
        labels         = [f"{int(tot_lifetime)} days", "325 days"] + [f"{n} years" for n in range(2, 11, 2)]
        cmap   = plt.get_cmap("rainbow")
        colors = [
            mcolors.to_hex(cmap(i / (len(periods_days) - 1)))
            for i in range(len(periods_days))
        ]

        tot, _ = np.histogram(self._data.e_p, bins=PROMPT_ENERGY_BINS_NMO)
        toterr = np.sqrt(tot)

        plotter = RelativeUncertaintyPromptEnergyPlotter(binmode="nmo")
        for Y_days, label, color in zip(periods_days, labels, colors):
            relerr = relative_uncertainty_rescale(tot, toterr, Y_days, tot_lifetime)
            plotter.add_histogram(
                relerr, np.zeros_like(relerr),
                color, linestyle="-", label=label,
            )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_relative_uncertainty_nmo", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_relative_uncertainty_prompt_energy_207days(self) -> None:
        tot_daq      = np.sum([Timestamp(sec, nsec) for sec, nsec in zip(self._data_daq.duration_sec, self._data_daq.duration_nsec)])
        tot_lifetime = tot_daq.to_sec() / 3600.0 / 24.0

        _DAYS_PER_YEAR = 365.25
        periods_days   = [tot_lifetime, 325.0] + [n * _DAYS_PER_YEAR for n in range(2, 11, 2)]
        labels         = [f"{int(tot_lifetime)} days", "325 days"] + [f"{n} years" for n in range(2, 11, 2)]
        cmap   = plt.get_cmap("rainbow")
        colors = [
            mcolors.to_hex(cmap(i / (len(periods_days) - 1)))
            for i in range(len(periods_days))
        ]

        tot, _ = np.histogram(self._data.e_p, bins=PROMPT_ENERGY_BINS_207DAYS)
        toterr = np.sqrt(tot)

        plotter = RelativeUncertaintyPromptEnergyPlotter(binmode="207days")
        for Y_days, label, color in zip(periods_days, labels, colors):
            relerr = relative_uncertainty_rescale(tot, toterr, Y_days, tot_lifetime)
            plotter.add_histogram(
                relerr, np.zeros_like(relerr),
                color, linestyle="-", label=label,
            )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_relative_uncertainty_207days", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_relative_uncertainty_prompt_energy_normal(self) -> None:
        tot_daq      = np.sum([Timestamp(sec, nsec) for sec, nsec in zip(self._data_daq.duration_sec, self._data_daq.duration_nsec)])
        tot_lifetime = tot_daq.to_sec() / 3600.0 / 24.0

        _DAYS_PER_YEAR = 365.25
        periods_days   = [tot_lifetime, 325.0] + [n * _DAYS_PER_YEAR for n in range(2, 11, 2)]
        labels         = [f"{int(tot_lifetime)} days", "325 days"] + [f"{n} years" for n in range(2, 11, 2)]
        cmap   = plt.get_cmap("rainbow")
        colors = [
            mcolors.to_hex(cmap(i / (len(periods_days) - 1)))
            for i in range(len(periods_days))
        ]

        tot, _ = np.histogram(self._data.e_p, bins=PROMPT_ENERGY_BINS_UNIFORM)
        toterr = np.sqrt(tot)

        plotter = RelativeUncertaintyPromptEnergyPlotter(binmode="normal", legend_loc="upper center", legend_ncol=2)
        for Y_days, label, color in zip(periods_days, labels, colors):
            relerr = relative_uncertainty_rescale(tot, toterr, Y_days, tot_lifetime)
            plotter.add_histogram(
                relerr, np.zeros_like(relerr),
                color, linestyle="-", label=label,
            )
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_relative_uncertainty_normal", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_delayed_energy(self) -> None:
        plotter = DelayedEnergyPlotter(ylim=(0.7, None), yscale="log", enable_fit=False)
        plotter.add(self._data.e_d, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dt(self) -> None:
        plotter = PromptDelayedTimePlotter(
            bins=uniform_bins(2.0, 4.0, 100), 
            xlim=(2.0, 4.0), 
            accidental_fit=True, 
            fit_info_anchor="bottom right", 
            fit_info_loc=(0.95, 0.05), 
        )
        plotter.add(self._data.dt_p_d_ms / 1000.0, linecolor=BLACK, fillcolor=BLACK)
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
            is_signal_region=False, 
            fit_info_fontsize=11, 
            fit_info_loc=(0.95, 0.85), 
        )
        fig, _ = plotter.plot(self._data.dt_mu2p, self._data.dlat_mu2p)
        save_figure(fig, self.stem, "_dt_dlat_p", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_rate_by_run(self) -> None:
        # Calculate the lifetime per run
        duration_dict = dict(zip(self._data_daq.run_id, self._data_daq.duration_sec))
        
        veto_dict = defaultdict(float)
        for run, veto_sec in zip(self._data_veto.run_id, self._data_veto.sec):
            veto_dict[run] += veto_sec

        runs = np.array(sorted(duration_dict.keys()))

        lifetime = np.array([
            duration_dict[run] - veto_dict.get(run, 0.0)
            for run in runs
        ])

        # Calculate the number of accidental per run
        accidental_count = Counter(self._data.run_id)

        naccidental = np.array([
            accidental_count.get(run, 0)
            for run in runs
        ])

        plotter = RunEvolutionPlotter(
            r"Accidental rate (cpd)", 
            ylim=(0.0, None),
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= runs,
                runs <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            scale_factor = (4.0 - 2.0) / (1.0e-3 - 5.0e-6)
            rate = naccidental[mask] / (lifetime[mask] / (24.0 * 3600.0)) / scale_factor
            err = np.sqrt(naccidental[mask]) / (lifetime[mask] / (24.0 * 3600.0)) / scale_factor

            mean_rate  = np.mean(rate)
            std_rate   = np.sqrt(np.mean(err**2))

            plotter.add(
                runs[mask],
                rate,
                err,
                phase.color,
                rf"{phase.name}: ${mean_rate:.2f} \pm {std_rate:.2f}$~cpd",
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                2, 0.80, 20, 0.0
            )

        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_rate_by_date(self) -> None:
        # Calculate the lifetime per run
        duration_dict = dict(zip(self._data_daq.run_id, self._data_daq.duration_sec))
        start_dict    = dict(zip(self._data_daq.run_id, self._data_daq.start_sec))
        
        veto_dict = defaultdict(float)
        for run, veto_sec in zip(self._data_veto.run_id, self._data_veto.sec):
            veto_dict[run] += veto_sec

        runs = np.array(sorted(duration_dict.keys()))

        lifetime  = np.array([
            duration_dict[run] - veto_dict.get(run, 0.0)
            for run in runs
        ])
        start_sec = np.array([
            start_dict[run] for run in runs
        ])

        # Calculate the number of accidental per run
        accidental_count = Counter(self._data.run_id)

        naccidental = np.array([
            accidental_count.get(run, 0)
            for run in runs
        ])

        plotter = TimeEvolutionPlotter(
            r"Accidental rate (cpd)", 
            ylim=(0.0, None),
            mode="sum", 
            show_mean=False,
            show_band=False,
            legend_ncol=2,
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= runs,
                runs <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            dates, lifetime_day, naccidental_day = sum_by_day(
                start_sec[mask],
                lifetime[mask],
                naccidental[mask],
            )

            dates = [d.timestamp() for d in dates]
            lifetime_day /= (24.0 * 3600.0)

            scale_factor = (4.0 - 2.0) / (1.0e-3 - 5.0e-6)
            rate = naccidental_day / lifetime_day / scale_factor
            err = np.sqrt(naccidental_day) / lifetime_day / scale_factor

            mean_rate  = np.mean(rate)
            std_rate   = np.sqrt(np.mean(err**2))

            plotter.add(
                dates,
                rate,
                err,
                phase.color,
                rf"{phase.name}: ${mean_rate:.2f} \pm {std_rate:.2f}$~cpd",
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

    def _plot_prompt_delayed_shape(self):
        bins = DELAYED_ENERGY_HYDROGEN_BINS
        widths = bins[1:] - bins[:-1]
        plotter = DelayedEnergyHydrogenPlotter()
        hist_p, _ = np.histogram(self._data.e_p, bins=bins)
        err_p = np.sqrt(hist_p)
        norm_p = np.sum(hist_p * widths)
        hist_p = hist_p / norm_p
        err_p = err_p / norm_p
        plotter.add_histogram(hist_p, err_p, linecolor=CUSTOM_BLUE)
        hist_d, _ = np.histogram(self._data.e_d, bins=bins)
        err_d = np.sqrt(hist_d)
        norm_d = np.sum(hist_d * widths)
        hist_d = hist_d / norm_d
        err_d = err_d / norm_d
        plotter.add_histogram(hist_d, err_d, linecolor=CUSTOM_RED)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_comp_e", output_dir=self.output_dir)
        plt.close(fig)