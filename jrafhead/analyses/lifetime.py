from __future__ import annotations

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseAnalysis
from config import (
    ReProd26B, 
    CUSTOM_BLUE, 
)
from loader import (
    load_lifetime_daq, 
    load_lifetime_veto, 
    VetoType,
)
from plotters.evolution import (
    RunEvolutionPlotter,
    TimeEvolutionPlotter
)
from utils import (
    save_figure, 
)

# ---------------------------------------------------------------------------------------------
# Lifetime DAQ
# ---------------------------------------------------------------------------------------------

class LifetimeDAQAnalysis(BaseAnalysis):
    """
    DAQ analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def _load(self) -> None:
        self._data = load_lifetime_daq(str(self.filepath), str(self.dirpath))

    def _plot(self) -> None:
        self._plot_daq_per_run()
        self._plot_daq_per_date()


    # ---------------------------------------------------------------------------------------------
    # Individual plot methods - one per output figure
    # ---------------------------------------------------------------------------------------------

    def _plot_daq_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"DAQ (hours)", 
            ylim=(0.0, 10.0),
            show_mean=False, 
            show_band=False,
            bar_mode=True,
            legend_bbox=(0.5, 1.15),
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            plotter.add(
                self._data.run_id[mask],
                self._data.duration_sec[mask] / 3600.0,
                np.zeros(len(self._data.duration_sec[mask])),
                phase.color,
                phase.name
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.85, 20, 0.0
            )

        fig, _ = plotter.plot()

        save_figure(fig, self.stem, "_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_daq_per_date(self) -> None:
        plotter = TimeEvolutionPlotter(
            r"DAQ (hours)", 
            ylim=(0.0, 30.0),
            mode="sum", 
            show_mean=False, 
            show_band=False,
            bar_mode=True,
            legend_bbox=(0.5, 1.15),
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            plotter.add(
                self._data.start_sec[mask],
                self._data.duration_sec[mask] / 3600.0,
                np.zeros(len(self._data.duration_sec[mask])),
                phase.color,
                phase.name
            )

            plotter.add_region(
                mdates.date2num(datetime.fromisoformat(phase.date_min)),
                mdates.date2num(datetime.fromisoformat(phase.date_max)),
                phase.color,
                phase.name,
                2, 0.85, 20, 0.0
            )

        fig, _ = plotter.plot()

        save_figure(fig, self.stem, "_per_date", output_dir=self.output_dir)
        plt.close(fig)

# ---------------------------------------------------------------------------------------------
# Lifetime veto
# ---------------------------------------------------------------------------------------------

class LifetimeVetoAnalysis(BaseAnalysis):
    """
    Veto analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def _load(self) -> None:
        self._data = load_lifetime_veto(str(self.filepath), str(self.dirpath))

    def _plot(self) -> None:
        self._plot_veto_per_run()
        self._plot_veto_per_type_per_run()

    # -------------------------------------------------------------------------
    # Individual plot methods — one per output figure
    # -------------------------------------------------------------------------

    def _plot_veto_per_run(self) -> None:
        plotter = RunEvolutionPlotter(
            r"Veto (min)", 
            show_mean=False, 
            show_band=False,
            bar_mode=True,
            legend_bbox=(0.5, 1.15),
        )

        for phase in ReProd26B.phases:
            mask = np.logical_and(
                phase.run_min <= self._data.run_id,
                self._data.run_id <= phase.run_max
            )

            if bool(np.all(np.logical_not(mask))):
                continue

            plotter.add(
                self._data.run_id[mask],
                self._data.sec[mask] / 60.0,
                np.zeros_like(self._data.sec[mask]),
                phase.color,
                phase.name,
            )

            plotter.add_region(
                phase.run_min,
                phase.run_max,
                phase.color,
                phase.name,
                50, 0.85, 20, 0.0
            )

        fig, _ = plotter.plot()

        save_figure(fig, self.stem, "_per_run", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_veto_per_type_per_run(self) -> None:
        veto_labels = [
            "None", "Job veto", "Missing headers", "Big gaps",
            "Muon", "CD muon", "WP muon",
        ]

        for veto_type, label in zip(VetoType, veto_labels):
            mask = self._data.type == veto_type

            if bool(np.all(np.logical_not(mask))):
                continue

            plotter = RunEvolutionPlotter(
                r"Veto (min)", 
                show_mean=False, 
                show_band=False,
                bar_mode=True,
                legend_bbox=(0.5, 1.15),
            )

            plotter.add(
                self._data.run_id[mask],
                self._data.sec[mask] / 60.0,
                np.zeros(len(self._data.sec[mask])),
                CUSTOM_BLUE,
                label,
            )

            # for phase in ReProd26B.phases:
            #     plotter.add_region(
            #         phase.run_min,
            #         phase.run_max,
            #         phase.color,
            #         phase.name,
            #         50, 0.85, 20, 0.0
            #     )

            fig, _ = plotter.plot()

            save_figure(fig, self.stem, f"_{label.replace(' ', '_')}_per_run", output_dir=self.output_dir)
            plt.close(fig)