from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import uproot
from config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_DARKPINK,
    CUSTOM_LIGHTBLUE,
    CUSTOM_MARKEDRED,
    CUSTOM_ORANGE,
    CUSTOM_PURPLE,
    CUSTOM_RED,
    PROMPT_ENERGY_BINS_207DAYS,
    PROMPT_ENERGY_BINS_NMO,
    PROMPT_ENERGY_BINS_UNIFORM,
)
from fits import (
    Li9He8ContrainedFractionFitter,
    model_time_8he,
    model_time_9li,
    model_time_bkg,
)
from loader import (
    Li9He8Data,
    Li9He8ShapeData,
    load_li9he8_rate,
    load_li9he8_shape,
    load_lifetime_daq,
    load_lifetime_veto,
)
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator
from plotters import (
    DelayedEnergyPlotter,
    Li9He8ChengzhuoFitPlotter,
    Li9He8ChengzhuoFitSmearingPlotter,
    Li9He8ShapeGroupCFitPlotter,
    MuonVetoDistributionPlotter,
    PromptDelayedDistancePlotter,
    PromptDelayedTimePlotter,
    PromptEnergyPlotter,
    RelativeUncertaintyPromptEnergyPlotter,
    SpatialDistributionPlotter,
)
from scipy.stats import chi2 as chi2_dist
from utils import (
    PositionGeometry,
    Timestamp,
    compute_3d_distance,
    compute_geometry,
    extract_window,
    relative_uncertainty_rescale,
    save_figure,
    save_json,
    uniform_bins,
)

from .base import BaseAnalysis

# -------------------------------------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------------------------------------

_LABEL_SIG  = "Cosmogenic enriched"
_LABEL_BKG  = "Cosmogenic depleted"


# -------------------------------------------------------------------------------------------------
# Bin-range inference from filename
# -------------------------------------------------------------------------------------------------

def _muon_veto_bins(dirpath: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Infer muon-veto bin edges from the filename encoding the selection cuts.

    The naming convention encodes the lateral-distance and time windows used in the event selection, 
    e.g. "cosmo_shape_3m_1_2s.root".

    Parameters
    ----------
    filepath : 
        str or Path

    Returns
    -------
    xbins_sig, xbins_bkg, ybins : np.ndarray
        Bin edges for the signal-region $\Delta t_{\mu}$ axis (positive), the background-region 
        $\Delta t_{\mu}$ axis (negative), and the d_{lat} axis.

    Raises
    ------
    ValueError
        If the filename does not match any known naming pattern.
    """
    # Work on the stem (no extension)
    stem = Path(dirpath).stem

    d_max = extract_window(str(stem), 'm')
    t_max = extract_window(str(stem), 's')

    ybins       = np.linspace(0.0,   d_max, 51)
    xbins_sig   = np.linspace(0.0,   t_max, 51)
    xbins_bkg   = np.linspace(-t_max, 0.0,  51)

    return xbins_sig, xbins_bkg, ybins


# -------------------------------------------------------------------------------------------------
# Geometry container for one sample
# -------------------------------------------------------------------------------------------------

@dataclass
class _SampleGeometry:
    geom_p:   PositionGeometry
    geom_d:   PositionGeometry
    distance: np.ndarray


def _compute_sample_geometry(sample: Li9He8Data) -> _SampleGeometry:
    return _SampleGeometry(
        geom_p   = compute_geometry(sample.pos_p_mm),
        geom_d   = compute_geometry(sample.pos_d_mm),
        distance = compute_3d_distance(sample.pos_p_mm, sample.pos_d_mm),
    )


# -------------------------------------------------------------------------------------------------
# Analysis class
# -------------------------------------------------------------------------------------------------

class Li9He8RateAnalysis(BaseAnalysis):
    """
    Cosmogenic shape analysis comparing signal and background candidates.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """
    def __init__(
            self,
            filepath:           str,
            dirpath:            str,
            dirpath_daq:        str,
            dirpath_veto:       str,
            output_dir:         str = ".",
        ) -> None:
            super().__init__(filepath, dirpath, output_dir)
            self.dirpath_daq       = dirpath_daq
            self.dirpath_veto      = dirpath_veto

    def _load(self) -> None:
        self._data      = load_li9he8_rate(str(self.filepath), str(self.dirpath))

        self._data_daq  = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))
        self._data_veto = load_lifetime_veto(str(self.filepath), str(self.dirpath_veto))

    def _plot(self) -> None:
        self._plot_muon_veto()
        self._plot_cosmo_rate_with_neu()
        self._plot_cosmo_rate()

        del self._data_daq
        del self._data_veto

    def _plot_muon_veto(self) -> None:
        plotter = MuonVetoDistributionPlotter(
            xbins=uniform_bins(0.0, 1.2, 50), 
            ybins=uniform_bins(0.0, 3.0, 50), 
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
        for k in range(11):
            if k == 0:
                dt = self._data.dt_last_mu_with_neu
                suffix = ""
            else:
                dt = getattr(self._data, f"dt_last_mu_with_neu_{k}m")
                suffix = f"_{k}m"

            hist, _ = np.histogram(dt, bins=bins)
            err = np.sqrt(hist)
            hist = hist / widths
            err = err / widths

            plotter = Li9He8RateEstimationPlotter(
                bins=bins,
                xlim=(bins[0], bins[-1]),
                ylabel=f"Entries / {widths[0]:g}~s",
                yscale="log", 
                ylim=(0.7, 2.0 * np.max(hist)), 
            )
            plotter.add_histogram(hist, err, linecolor=BLACK, fillcolor=BLACK)
            fig, _ = plotter.plot()
            save_figure(fig, self.stem, f"_dt_last_mu_with_neu{suffix}", output_dir=self.output_dir)
            plt.close(fig)

            lifetime = np.sum(self._data_daq.duration_sec) / 24.0 / 3600.0

            if not plotter.fit_result:
                return

            key = f"{k}m"
            data = {
                "N9li8he":      plotter.fit_result.popt[0],
                "N9li8heerr":   plotter.fit_result.perr[0],
                "Nbkg":         plotter.fit_result.popt[1],
                "Nbkgerr":      plotter.fit_result.perr[1],
                "Rmu":          plotter.fit_result.popt[2],
                "Rmuerr":       plotter.fit_result.perr[2],
                "chi2":         plotter.fit_result.chi2,
                "ndf":          plotter.fit_result.ndf,
                "pvalue":       plotter.fit_result.pvalue,
                "lifetime":     lifetime,
            }
            save_json(data, key, "li9he8_rate_muon_with_neu", "", output_dir="output/data")

    def _plot_cosmo_rate(self) -> None:
        from plotters import Li9He8RateEstimationPlotter
        
        bins    = uniform_bins(0.0, 10.0, 200)
        widths  = bins[1:] - bins[:-1]
        for k in range(11):
            if k == 0:
                dt = self._data.dt_last_mu
                suffix = ""
            else:
                dt = getattr(self._data, f"dt_last_mu_{k}m")
                suffix = f"_{k}m"

            hist, _ = np.histogram(dt, bins=bins)
            err = np.sqrt(hist)
            hist = hist / widths
            err = err / widths

            plotter = Li9He8RateEstimationPlotter(
                bins=bins,
                xlim=(bins[0], bins[-1]),
                ylabel=f"Entries / {widths[0]:g}~s",
                yscale="log", 
                ylim=(0.7, 2.0 * np.max(hist)), 
            )
            plotter.add_histogram(hist, err, linecolor=BLACK, fillcolor=BLACK)
            fig, _ = plotter.plot()
            save_figure(fig, self.stem, f"_dt_last_mu{suffix}", output_dir=self.output_dir)
            plt.close(fig)

            lifetime = np.sum(self._data_daq.duration_sec) / 24.0 / 3600.0

            if not plotter.fit_result:
                return

            key = f"{k}m"
            data = {
                "N9li8he":      plotter.fit_result.popt[0],
                "N9li8heerr":   plotter.fit_result.perr[0],
                "Nbkg":         plotter.fit_result.popt[1],
                "Nbkgerr":      plotter.fit_result.perr[1],
                "Rmu":          plotter.fit_result.popt[2],
                "Rmuerr":       plotter.fit_result.perr[2],
                "chi2":         plotter.fit_result.chi2,
                "ndf":          plotter.fit_result.ndf,
                "pvalue":       plotter.fit_result.pvalue,
                "lifetime":     lifetime,
            }
            save_json(data, key, "li9he8_rate_muon", "", output_dir="output/data")


class Li9He8ShapeAnalysis(BaseAnalysis):
    """
    Cosmogenic shape analysis comparing signal and background candidates.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    mc_groupc_path : str
        Path to the MC GroupC cosmogenics ROOT file.
    mc_9li8he_path : str
        Path to the MC 9Li/8He cosmogenics ROOT file.
    mc_chengzhuo_path : str
        Path to the Chengzhuo MC 9Li cosmogenics ROOT file.
    """

    def __init__(
        self,
        filepath:           str,
        dirpath:            str,
        dirpath_daq:        str,
        dirpath_veto:       str,
        mc_groupc_path:     str,
        mc_9li8he_path:     str,
        mc_chengzhuo_path:  str, 
        output_dir:         str = ".",
    ) -> None:
        super().__init__(filepath, dirpath, output_dir)
        self.dirpath_daq       = dirpath_daq
        self.dirpath_veto      = dirpath_veto
        self.mc_groupc_path    = mc_groupc_path
        self.mc_9li8he_path    = mc_9li8he_path
        self.mc_chengzhuo_path = mc_chengzhuo_path

    def _load(self) -> None:
        cosmo: Li9He8ShapeData  = load_li9he8_shape(str(self.filepath), str(self.dirpath))
        self._sig               = cosmo.signal
        self._bkg               = cosmo.background
        self._geom_s            = _compute_sample_geometry(cosmo.signal)
        self._geom_b            = _compute_sample_geometry(cosmo.background)
        self._xbins_sig, self._xbins_bkg, self._ybins = _muon_veto_bins(self.dirpath)

        self._data_daq      = load_lifetime_daq(str(self.filepath), str(self.dirpath_daq))
        self._data_veto     = load_lifetime_veto(str(self.filepath), str(self.dirpath_veto))

    def _plot(self) -> None:
        # self._plot_prompt_energy_nmo()
        # self._plot_prompt_energy_207days()
        # self._plot_prompt_energy_normal_sig()
        # self._plot_prompt_energy_normal_bkg()
        # self._plot_prompt_energy_normal()
        # self._plot_relative_uncertainty_prompt_energy_nmo()
        # self._plot_relative_uncertainty_prompt_energy_207days()
        # self._plot_relative_uncertainty_prompt_energy_normal()
        # self._plot_prompt_energy_diff_mc_group()
        # self._plot_prompt_energy_diff_mc_chengzhuo()
        # self._plot_prompt_energy_diff_mc_chengzhuo_smearing()
        # self._plot_delayed_energy_sig()
        # self._plot_delayed_energy_bkg()
        # self._plot_delayed_energy_diff()
        # self._plot_dt_sig()
        # self._plot_dt_bkg()
        # self._plot_dt_diff()
        # self._plot_dr_sig()
        # self._plot_dr_bkg()
        # self._plot_dr_diff()
        # self._plot_spatial()
        # self._plot_muon_veto()
        # self._plot_cosmo_rate_with_neu()
        # self._plot_cosmo_rate()
        if "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_2s_" in str(self.dirpath):
            self._fraction_fitter()

        del self._data_daq
        del self._data_veto

    # ---------------------------------------------------------------------------------------------
    # Individual plot methods
    # ---------------------------------------------------------------------------------------------

    def _plot_prompt_energy_nmo(self) -> None:
        plotter = PromptEnergyPlotter(binmode="nmo")
        plotter.add(self._sig.e_p, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._bkg.e_p, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, label="Difference", linestyle="-")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_nmo", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_207days(self) -> None:
        plotter = PromptEnergyPlotter(binmode="207days")
        plotter.add(self._sig.e_p, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._bkg.e_p, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, label="Difference", linestyle="-")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_207days", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_normal_sig(self) -> None:
        plotter = PromptEnergyPlotter(binmode="normal")
        plotter.add(self._sig.e_p, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_normal_sig", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_normal_bkg(self) -> None:
        plotter = PromptEnergyPlotter(binmode="normal")
        plotter.add(self._bkg.e_p, linecolor=CUSTOM_RED, fillcolor=CUSTOM_RED)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_normal_bkg", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_normal(self) -> None:
        plotter = PromptEnergyPlotter(binmode="normal")
        plotter.add(self._sig.e_p, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._bkg.e_p, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, label="Difference", linestyle="-")
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

        sig, _ = np.histogram(self._sig.e_p, bins=PROMPT_ENERGY_BINS_NMO)
        bkg, _ = np.histogram(self._bkg.e_p, bins=PROMPT_ENERGY_BINS_NMO)
        tot    = sig - bkg
        toterr = np.sqrt(sig + bkg)

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

        sig, _ = np.histogram(self._sig.e_p, bins=PROMPT_ENERGY_BINS_207DAYS)
        bkg, _ = np.histogram(self._bkg.e_p, bins=PROMPT_ENERGY_BINS_207DAYS)
        tot    = sig - bkg
        toterr = np.sqrt(sig + bkg)

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

        sig, _ = np.histogram(self._sig.e_p, bins=PROMPT_ENERGY_BINS_UNIFORM)
        bkg, _ = np.histogram(self._bkg.e_p, bins=PROMPT_ENERGY_BINS_UNIFORM)
        tot    = sig - bkg
        toterr = np.sqrt(sig + bkg)

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

    def _plot_prompt_energy_diff_mc_group(self) -> None:
        from loader import load_mc_groupc_template

        template = load_mc_groupc_template(self.mc_groupc_path, hist_path="prefit/lihe")

        plotter = Li9He8ShapeGroupCFitPlotter(template, binmode="normal")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, fillcolor=BLACK, label=r"Estimation")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_diff_mc_groupc_fit", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_diff_mc_chengzhuo(self) -> None:
        from loader import load_mc_chengzhuo_template

        template = load_mc_chengzhuo_template(self.mc_chengzhuo_path)

        plotter = Li9He8ChengzhuoFitPlotter(template, binmode="normal")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, fillcolor=BLACK, label=r"Estimation")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_diff_mc_chengzhuo_fit", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_prompt_energy_diff_mc_chengzhuo_smearing(self) -> None:
        from loader import load_mc_chengzhuo_template

        template = load_mc_chengzhuo_template(self.mc_chengzhuo_path)

        plotter = Li9He8ChengzhuoFitSmearingPlotter(template, binmode="normal")
        plotter.add_diff(self._sig.e_p, self._bkg.e_p, linecolor=BLACK, fillcolor=BLACK, label=r"Estimation")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_p_diff_mc_chengzhuo_fit_smearing", output_dir=self.output_dir)
        plt.close(fig)

        if not plotter.fit_result:
            return
        with uproot.recreate(f"{self.output_dir}/data/{self.dirpath}_mc_chengzhuo_fit_smearing.root") as output_file:
            N, f0, f1, f2, f3 = plotter._fit_result.popt
            f4 = 0.508 - f0 - f1 - f2 - f3
        
            fitted_counts = N * (
                f0 * template.branch0.counts
                + f1 * template.branch1.counts
                + f2 * template.branch2.counts
                + f3 * template.branch3.counts
                + f4 * template.branch4.counts
            )
        
            # Fitted spectrum
            output_file["fitted_spectrum"] = (
                fitted_counts,
                template.all.edges,
            )

            # Signal candidates
            output_file["signal_events"] = {
                "run_id":     self._sig.run_id,
                "e_p":        self._sig.e_p,
                "e_d":        self._sig.e_d,
                "posx_p":     self._sig.pos_p_mm[:, 0],
                "posy_p":     self._sig.pos_p_mm[:, 1],
                "posz_p":     self._sig.pos_p_mm[:, 2],
                "dt_p_d_ms":  self._sig.dt_p_d_ms,
                "dt_mu2p":    self._sig.dt_mu2p,
                "dlat_mu2p":  self._sig.dlat_mu2p,
                "posx_d":     self._sig.pos_d_mm[:, 0],
                "posy_d":     self._sig.pos_d_mm[:, 1],
                "posz_d":     self._sig.pos_d_mm[:, 2],
                "dt_mu2d":    self._sig.dt_mu2d,
                "dlat_mu2d":  self._sig.dlat_mu2d,
            }

            # Background candidates
            output_file["background_events"] = {
                "run_id":     self._bkg.run_id,
                "e_p":        self._bkg.e_p,
                "e_d":        self._bkg.e_d,
                "posx_p":     self._bkg.pos_p_mm[:, 0],
                "posy_p":     self._bkg.pos_p_mm[:, 1],
                "posz_p":     self._bkg.pos_p_mm[:, 2],
                "dt_p_d_ms":  self._bkg.dt_p_d_ms,
                "dt_mu2p":    self._bkg.dt_mu2p,
                "dlat_mu2p":  self._bkg.dlat_mu2p,
                "posx_d":     self._bkg.pos_d_mm[:, 0],
                "posy_d":     self._bkg.pos_d_mm[:, 1],
                "posz_d":     self._bkg.pos_d_mm[:, 2],
                "dt_mu2d":    self._bkg.dt_mu2d,
                "dlat_mu2d":  self._bkg.dlat_mu2d,
            }

            # Raw spectrum
            binnings = {
                "uniform": PROMPT_ENERGY_BINS_UNIFORM,
                "207days": PROMPT_ENERGY_BINS_207DAYS,
                "nmo": PROMPT_ENERGY_BINS_NMO,
            }

            for name, bins in binnings.items():

                signal_counts, _ = np.histogram(
                    self._sig.e_p,
                    bins=bins,
                )

                background_counts, _ = np.histogram(
                    self._bkg.e_p,
                    bins=bins,
                )

                output_file[f"signal_{name}"] = (
                    signal_counts,
                    bins,
                )

                output_file[f"background_{name}"] = (
                    background_counts,
                    bins,
                )

    def _plot_delayed_energy_sig(self) -> None:
        plotter = DelayedEnergyPlotter(ylim=(0.7, None), yscale="log")
        plotter.add(self._sig.e_d, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_d_sig", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_delayed_energy_bkg(self) -> None:
        plotter = DelayedEnergyPlotter(ylim=(0.7, None), yscale="log")
        plotter.add(self._bkg.e_d, linecolor=CUSTOM_BLUE, fillcolor=CUSTOM_BLUE)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_d_bkg", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_delayed_energy_diff(self) -> None:
        plotter = DelayedEnergyPlotter(ylim=(0.7, None), yscale="log")
        plotter.add(self._sig.e_d, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._bkg.e_d, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._sig.e_d, self._bkg.e_d, linecolor=BLACK, label="Difference", linestyle="-")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_e_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dt_sig(self) -> None:
        plotter = PromptDelayedTimePlotter()
        plotter.add(self._sig.dt_p_d_ms, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_p_d_sig", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dt_bkg(self) -> None:
        plotter = PromptDelayedTimePlotter()
        plotter.add(self._bkg.dt_p_d_ms, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_p_d_bkg", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dt_diff(self) -> None:
        plotter = PromptDelayedTimePlotter(enable_fit=False)
        plotter.add(self._sig.dt_p_d_ms, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._bkg.dt_p_d_ms, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._sig.dt_p_d_ms, self._bkg.dt_p_d_ms, linecolor=BLACK, label="Difference", linestyle="-")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dt_p_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dr_sig(self) -> None:
        plotter = PromptDelayedDistancePlotter()
        plotter.add(self._geom_s.distance, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dr_p_d_sig", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dr_bkg(self) -> None:
        plotter = PromptDelayedDistancePlotter()
        plotter.add(self._geom_b.distance, linecolor=BLACK, fillcolor=BLACK)
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dr_p_d_bkg", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_dr_diff(self) -> None:
        plotter = PromptDelayedDistancePlotter()
        plotter.add(self._geom_s.distance, linecolor=CUSTOM_BLUE, label=_LABEL_SIG, linestyle=":")
        plotter.add(self._geom_b.distance, linecolor=CUSTOM_RED, label=_LABEL_BKG, linestyle=":")
        plotter.add_diff(self._geom_s.distance, self._geom_b.distance, linecolor=BLACK, label="Difference", linestyle="-")
        fig, _ = plotter.plot()
        save_figure(fig, self.stem, "_dr_p_d", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_spatial(self) -> None:
        cases = [
            (self._geom_b.geom_p, r"$\rho_{p}^{2}$ (m$^2$)", r"$z_{p}$ (m)", "_rho_z_p_bkg"),
            (self._geom_b.geom_d, r"$\rho_{d}^{2}$ (m$^2$)", r"$z_{d}$ (m)", "_rho_z_d_bkg"),
            (self._geom_s.geom_p, r"$\rho_{p}^{2}$ (m$^2$)", r"$z_{p}$ (m)", "_rho_z_p_sig"),
            (self._geom_s.geom_d, r"$\rho_{d}^{2}$ (m$^2$)", r"$z_{d}$ (m)", "_rho_z_d_sig"),
        ]
        for geom, xlabel, ylabel, suffix in cases:
            plotter = SpatialDistributionPlotter(xlabel=xlabel, ylabel=ylabel)
            fig, _ = plotter.plot(geom.rho2_m2, geom.z_m)
            save_figure(fig, self.stem, suffix, output_dir=self.output_dir)
            plt.close(fig)

    def _plot_muon_veto(self) -> None:
        plotter = MuonVetoDistributionPlotter(
            xbins=self._xbins_sig, 
            ybins=self._ybins, 
            xlabel=r"$\Delta t_{\mu-p}$ (s)", 
            ylabel=r"$d_{\mu-p}$ (m)", 
            is_signal_region=True, 
            fit_info_fontsize=11, 
            fit_info_loc=(0.95, 0.85), 
        )
        fig, _ = plotter.plot(self._sig.dt_mu2p, self._sig.dlat_mu2p)
        save_figure(fig, self.stem, "_dt_dlat_p_sig", output_dir=self.output_dir)
        plt.close(fig)

        plotter = MuonVetoDistributionPlotter(
            xbins=self._xbins_bkg, 
            ybins=self._ybins, 
            xlabel=r"$\Delta t_{\mu-p}$ (s)", 
            ylabel=r"$d_{\mu-p}$ (m)", 
            is_signal_region=False, 
            fit_info_fontsize=11, 
            fit_info_anchor="bottom right", 
            fit_info_loc=(0.95, 0.15), 
        )
        fig, _ = plotter.plot(self._bkg.dt_mu2p, self._bkg.dlat_mu2p)
        save_figure(fig, self.stem, "_dt_dlat_p_bkg", output_dir=self.output_dir)
        plt.close(fig)

    def _plot_cosmo_rate_with_neu(self) -> None:
            from plotters import Li9He8RateEstimationPlotter
    
            bins    = uniform_bins(0.0, 10.0, 200)
            widths  = bins[1:] - bins[:-1]
            hist, _ = np.histogram(self._sig.dt_last_mu_with_neu, bins=bins)
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

            lifetime = np.sum(self._data_daq.duration_sec) / 24.0 / 3600.0

            if not plotter.fit_result:
                return
            radius = extract_window(str(self.dirpath), 'm')
            time   = extract_window(str(self.dirpath), 's')
            key    = f"_{radius:g}m_{time:g}s_".replace(".", "_")
            data = {
                "N9li8he":      plotter.fit_result.popt[0],
                "N9li8heerr":   plotter.fit_result.perr[0],
                "Nbkg":         plotter.fit_result.popt[1],
                "Nbkgerr":      plotter.fit_result.perr[1],
                "Rmu":          plotter.fit_result.popt[2],
                "Rmuerr":       plotter.fit_result.perr[2],
                "chi2":         plotter.fit_result.chi2,
                "ndf":          plotter.fit_result.ndf,
                "pvalue":       plotter.fit_result.pvalue,
                "lifetime":     lifetime,
            }
            save_json(data, key, "cosmo_rate_with_neu", "", output_dir="output/data")

    def _plot_cosmo_rate(self) -> None:
            from plotters import Li9He8RateEstimationPlotter
        
            bins    = uniform_bins(0.0, 10.0, 200)
            widths  = bins[1:] - bins[:-1]
            hist, _ = np.histogram(self._sig.dt_last_mu, bins=bins)
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

            if not plotter.fit_result:
                return
            radius = extract_window(str(self.dirpath), 'm')
            time   = extract_window(str(self.dirpath), 's')
            key    = f"_{radius:g}m_{time:g}s_".replace(".", "_")
            data = {
                "N9li8he":      plotter.fit_result.popt[0],
                "N9li8heerr":   plotter.fit_result.perr[0],
                "Nbkg":         plotter.fit_result.popt[1],
                "Nbkgerr":      plotter.fit_result.perr[1],
                "Rmu":          plotter.fit_result.popt[2],
                "Rmuerr":       plotter.fit_result.perr[2],
                "chi2":         plotter.fit_result.chi2,
                "ndf":          plotter.fit_result.ndf,
                "pvalue":       plotter.fit_result.pvalue
            }
            save_json(data, key, "cosmo_rate", "", output_dir="output/data")

    def _fraction_fitter(self) -> None:
        from loader import load_mc_9li8he_cosmogenics, load_mc_chengzhuo_template
        from utils import rebin_histogram

        _li9 = load_mc_9li8he_cosmogenics(self.mc_9li8he_path, "Li9")
        _he8 = load_mc_9li8he_cosmogenics(self.mc_9li8he_path, "He8")

        ebins     = uniform_bins(0.0, 12.0, 100)
        dtbins    = uniform_bins(0.0,  2.0, 100)
        ecenters  = (ebins[1:]  + ebins[:-1])  / 2.0
        dtcenters = (dtbins[1:] + dtbins[:-1]) / 2.0
        ewidths   = ebins[1:]  - ebins[:-1]
        dtwidths  = dtbins[1:] - dtbins[:-1]

        hesig, _  = np.histogram(self._sig.e_p, bins=ebins)
        hebkg, _  = np.histogram(self._bkg.e_p, bins=ebins)
        eesig     = np.sqrt(hesig)
        eebkg     = np.sqrt(hebkg)
        eesig     = np.sqrt(eesig**2 + eebkg**2)

        hdtsig, _ = np.histogram(self._sig.dt_mu2p, bins=dtbins)
        edtsig    = np.sqrt(hdtsig)

        hesig     = hesig / ewidths
        eesig     = eesig / ewidths
        hdtsig    = hdtsig / dtwidths
        edtsig    = edtsig / dtwidths

        label = r"Enriched region"

        exlabel  = r"$E_{p}$ (MeV)"
        eylabel  = f"Entries / {ewidths[0]:g}~MeV"
        dtxlabel = r"$\Delta t_{\mu-p}$ (s)"
        dtylabel = f"Entries / {dtwidths[0]:g}~s"
        exlim    = (0.0, 12.0)
        eylim    = (0.0, 1.5 * np.max(hesig))
        dtxlim   = (0.0, 2.0)
        dtylim   = (0.7, None)
        # dtylim   = (0.7, 4.0 * np.max(hdtsig))
        exscale  = "linear"
        eyscale  = "linear"
        dtxscale = "linear"
        dtyscale = "linear"

        # -------------------------------------------------
        # Fit
        # -------------------------------------------------

        # hli9, _ = np.histogram(_li9.e_p, bins=ebins)
        dli9    = load_mc_chengzhuo_template(self.mc_chengzhuo_path)
        hli9    = rebin_histogram(dli9.all.edges, dli9.all.counts, ebins)
        hhe8, _ = np.histogram(_he8.e_p, bins=ebins)
        hli9    = hli9  / np.sum(hli9  * ewidths)
        hhe8    = hhe8  / np.sum(hhe8  * ewidths)
        hebkg   = hebkg / np.sum(hebkg * ewidths)

        threshold       = 0.007 # ms
        first_fit_bin   = np.searchsorted(dtbins, threshold, side="left")
        dtxlim_fit = (dtbins[first_fit_bin], extract_window(str(self.dirpath), 's'))

        fitter = Li9He8ContrainedFractionFitter(
            hebkg, hli9, hhe8, 
            ebins, hesig, eesig, 
            dtbins, hdtsig, edtsig, 
            dtxlim=dtxlim_fit
        )
        # fitter = Li9He8FractionFitter(
        #     hli9, hhe8, 
        #     ecenters, ehist, eerr, 
        #     dtcenters, dthist, dterr, 
        #     dtxlim=dtxlim_fit
        # )
        result = fitter.fit()
        if result is None:
            return
        
        names = ["N", "fbkg", "f9li", "t9li", "t8he"]
        print("Fit fraction:")
        print(f"  P(chi^2 / ndf = {result.chi2:.3f} / {result.ndf}) = {result.pvalue:.3f}")
        print(f"    chi^2 / ndf = {(result.chi2 / result.ndf):.3f}")
        for name, par, err in zip(names, result.popt, result.perr):
            print(f"  {name} = {par:.3f} +/- {err:.3f}")

        N,    fbkg,    f9li,   = result.popt
        Nerr, fbkgerr, f9lierr = result.perr
        # N,    fbkg,    f9li,    t9li,    t8he    = result.popt
        # Nerr, fbkgerr, f9lierr, t9lierr, t8heerr = result.perr

        radius = extract_window(str(self.dirpath), 'm')
        time   = extract_window(str(self.dirpath), 's')
        key    = f"_{radius:g}m_{time:g}s_".replace(".", "_")

        json_dir = Path("output/data")
        json_dir.mkdir(parents=True, exist_ok=True)

        json_file = json_dir / "li9he9_fraction_fit.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[key] = {
            "N":    {"value": float(N),    "error": float(Nerr)}, 
            "fbkg": {"value": float(fbkg), "error": float(fbkg)}, 
            "f9li": {"value": float(f9li), "error": float(f9li)}, 
            # "t9li": {"value": float(t9li), "error": float(t9li)}, 
            # "t8he": {"value": float(t8he), "error": float(t8he)}, 
            "chi2":   float(result.chi2),
            "ndf":    int(result.ndf),
            "pvalue": float(result.pvalue),
        }

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)

        text = (
            r"$P(\chi^{2} / \mathrm{ndf} = %.1f / %d) = %.3f$" "\n"
            r"$N = %.1f \pm %.1f$"                             "\n"
            r"$f_{\mathrm{bkg}} = %.2f \pm %.2f$"              "\n"
            r"$f_{^{9}\mathrm{Li}} = %.2f \pm %.2f$"
        ) % (
            result.chi2, result.ndf, result.pvalue,
            result.popt[0], result.perr[0],
            result.popt[1], result.perr[1],
            result.popt[2], result.perr[2],
        )

        # -------------------------------------------------
        # Prompt energy plot
        # -------------------------------------------------

        fig1, ax1 = plt.subplots(figsize=(7, 6))

        ebkg_fit = N * fbkg * hebkg
        e9li_fit = N * (1.0 - fbkg) * f9li * hli9
        e8he_fit = N * (1.0 - fbkg) * (1.0 - f9li) * hhe8
        etot_fit = fitter.model_energy_full(ecenters, N, fbkg, f9li)

        emask   = eesig > 0
        echi2   = np.sum(
            ((hesig[emask] - etot_fit[emask]) / eesig[emask]) ** 2
        )
        endf    = np.count_nonzero(emask) - 3
        epvalue = chi2_dist.sf(echi2, endf)
        print("Energy:")
        print(f"  P(chi^2 / ndf = {echi2:.3f} / {endf}) = {epvalue:.3f}")
        print(f"    chi^2 / ndf = {(echi2 / endf):.3f}")

        ys         = [ebkg_fit,           e9li_fit,             e8he_fit,             etot_fit        ]
        colors     = [CUSTOM_ORANGE,      CUSTOM_BLUE,          CUSTOM_DARKPINK,      CUSTOM_MARKEDRED]
        linestyles = ["--",               "--",                 "--",                 "-"             ]
        labels     = [r"Depleted region", r"$^{9}$Li template", r"$^{8}$He template", r"Fit"          ]

        for y, clr, lnstl, lbl in zip(ys, colors, linestyles, labels):
            ax1.step(
                ebins, np.r_[y, y[-1]],  
                where="post", color=clr, linewidth=1.8, linestyle=lnstl, zorder=4, 
                label=lbl, 
            )

        ax1.errorbar(
            ecenters, hesig,
            yerr=eesig, xerr=ewidths / 2.0,
            label=label,
            fmt="o", color=BLACK,
            markersize=4.5, zorder=3,
        )
        ax1.fill_between(
            ebins,
            np.r_[hesig, hesig[-1]],
            step="post",
            color=BLACK,
            zorder=1,
            alpha=0.10, 
        )

        # ax1.text(
        #     0.95, 0.95, text,
        #     transform=ax1.transAxes,
        #     fontsize=18,
        #     va="top", 
        #     ha="right", 
        # )
    
        ax1.set_xlabel(exlabel)
        ax1.set_ylabel(eylabel)

        if exlim:
            ax1.set_xlim(*exlim)
        if eylim:
            ax1.set_ylim(*eylim)

        if exscale == "linear":
            ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        if eyscale == "linear":
            ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax1.set_xscale(exscale)
        ax1.set_yscale(eyscale)

        ax1.minorticks_on()
        ax1.tick_params(direction="in", which="both", top=True, right=True)
        ax1.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax1.legend(loc="upper right")


        fig1.tight_layout()
        save_figure(fig1, self.stem, "_ratio_e_p", output_dir=self.output_dir)
        plt.close(fig1)

        # -------------------------------------------------
        # Prompt-muon time difference
        # -------------------------------------------------

        fig2, ax2 = plt.subplots(figsize=(7, 6))

        t9li = 0.256
        t8he = 0.171
        dtbkg_fit = model_time_bkg(dtcenters, N, fbkg, dtxlim[0], dtxlim[1])
        dt9li_fit = model_time_9li(dtcenters, N, fbkg, f9li, t9li, dtxlim[0], dtxlim[1])
        dt8he_fit = model_time_8he(dtcenters, N, fbkg, f9li, t8he, dtxlim[0], dtxlim[1])
        dttot_fit = fitter.model_time(dtcenters, N, fbkg, f9li)

        dtmask   = edtsig > 0
        dtmask   &= ( (dtxlim_fit[0] <= dtcenters) & (dtcenters <= dtxlim_fit[1]) )
        dtchi2   = np.sum(((hdtsig[dtmask] - dttot_fit[dtmask]) / edtsig[dtmask]) ** 2)
        dtndf    = np.count_nonzero(dtmask) - 3
        dtpvalue = chi2_dist.sf(dtchi2, dtndf)
        print("Time:")
        print(f"  P(chi^2 / ndf = {dtchi2:.3f} / {dtndf}) = {dtpvalue:.3f}")
        print(f"    chi^2 / ndf = {(dtchi2 / dtndf):.3f}")

        ys         = [dtbkg_fit,          dt9li_fit,            dt8he_fit,            dttot_fit       ]
        colors     = [CUSTOM_ORANGE,      CUSTOM_BLUE,          CUSTOM_DARKPINK,      CUSTOM_MARKEDRED]
        linestyles = ["--",               "--",                 "--",                 "-"             ]
        labels     = [r"Background",      r"$^{9}$Li template", r"$^{8}$He template", r"Fit"          ]

        for y, clr, lnstl, lbl in zip(ys, colors, linestyles, labels):
            ax2.plot(
                dtcenters[dtmask], y[dtmask], 
                color=clr, linewidth=1.8, linestyle=lnstl, zorder=4, 
                label=lbl, 
            )

        ax2.errorbar(
            dtcenters, hdtsig,
            yerr=edtsig, xerr=dtwidths / 2,
            label=label,
            fmt="o", color=BLACK,
            markersize=4.5, zorder=3,
        )
        ax2.fill_between(
            dtbins,
            np.r_[hdtsig, hdtsig[-1]],
            step="post",
            color=BLACK,
            zorder=1,
            alpha=0.10, 
        )

        ax2.text(
            0.95, 0.95, text,
            transform=ax2.transAxes,
            fontsize=18,
            va="top", 
            ha="right", 
        )

        ax2.set_xlabel(dtxlabel)
        ax2.set_ylabel(dtylabel)

        if dtxlim:
            ax2.set_xlim(*dtxlim)
        if dtylim:
            ax2.set_ylim(*dtylim)

        if dtxscale == "linear":
            ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        if dtyscale == "linear":
            ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax2.set_xscale(dtxscale)
        ax2.set_yscale(dtyscale)

        ax2.minorticks_on()
        ax2.tick_params(direction="in", which="both", top=True, right=True)
        ax2.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
        ax2.legend(loc="center right")

        fig2.tight_layout()
        save_figure(fig2, self.stem, "_ratio_dt_p_d", output_dir=self.output_dir)
        plt.close(fig2)

        # -------------------------------------------------
        # Contour
        # -------------------------------------------------

        def chi2_from_params(N_: float, fbkg_: float, f9li_: float) -> np.ndarray: #, t9li_: float, t8he_: float) -> tuple[np.ndarray]:
            emodel = fitter.model_energy_full(ecenters, N_, fbkg_, f9li_)
            echi2 = np.sum(
                ((hesig[emask] - emodel[emask]) / eesig[emask])**2
            )

            dtmodel = fitter.model_time(dtcenters, N_, fbkg_, f9li_) # , t9li_, t8he_)
            dtchi2 = np.sum(
                ((hdtsig[dtmask] - dtmodel[dtmask]) / edtsig[dtmask])**2
            )

            totchi2 = echi2 + dtchi2
            return echi2, dtchi2, totchi2

        def plot_1d_contour(best: float, besterr: float, index: int, label: str) -> tuple[plt.Figure, plt.Axes]:
            xlim = (np.max([0.0, best - 3.0 * besterr]), best + 3.0 * besterr)

            x = uniform_bins(*xlim, 200)

            echi2   = np.empty_like(x)
            dtchi2  = np.empty_like(x)
            totchi2 = np.empty_like(x)

            for i, p in enumerate(x):
                pars = result.popt.copy()
                pars[index] = p
                echi2[i], dtchi2[i], totchi2[i] = chi2_from_params(*pars)

            echi2   -= np.min(echi2)
            dtchi2  -= np.min(dtchi2)
            totchi2 -= np.min(totchi2)

            fig, ax = plt.subplots(figsize=(7, 6))

            ax.plot(
                x, echi2, 
                color=CUSTOM_RED, linewidth=1.8, zorder=4, 
                label=r"Energy model", 
            )
            ax.plot(
                x, dtchi2, 
                color=CUSTOM_BLUE, linewidth=1.8, zorder=4, 
                label=r"Time model", 
            )
            ax.plot(
                x, totchi2, 
                color=BLACK, linewidth=1.8, zorder=4, 
                label=r"Energy + time model", 
            )

            # ax.axvline(best, color=BLACK, lw=1.5, linestyle="--")
            # ax.add_patch(Rectangle(
            #     (best - besterr, 0.0), 2.0 * besterr, 1.0,
            #     transform=ax.get_yaxis_transform(),
            #     color=BLACK, alpha=0.2, zorder=1,
            # ))
            # ax.axvline(best - besterr, color=BLACK, lw=1.0, linestyle=":")
            # ax.axvline(best + besterr, color=BLACK, lw=1.0, linestyle=":")
            # ax.axhline(1.0, color="#e05c5c", lw=0.8, linestyle="--", alpha=0.5, label=r"$\Delta\chi^2 = 1$")
            # ax.axhline(4.0, color="#f0a500", lw=0.8, linestyle="--", alpha=0.5, label=r"$\Delta\chi^2 = 4$")

            ax.set_xlabel(label)
            ax.set_ylabel(r"$\Delta \chi^{2}$")

            ax.set_xlim(*xlim)
            ax.set_ylim(bottom=0.0, top=9.0)

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            ax.minorticks_on()
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc="upper center")

            fig.tight_layout()
            return fig, ax

        def plot_2d_contour(
            xbest: float, xbesterr: float, xindex: int, xlabel: str, 
            ybest: float, ybesterr: float, yindex: int, ylabel: str, 
        ) -> tuple[plt.Figure, plt.Axes]:
            xlim = (np.max([0.0, xbest - 6.5 * xbesterr]), xbest + 6.5 * xbesterr)
            ylim = (np.max([0.0, ybest - 6.5 * ybesterr]), ybest + 6.5 * ybesterr)

            x = uniform_bins(*xlim, 100)
            y = uniform_bins(*ylim, 100)
            X, Y = np.meshgrid(x, y)

            totchi2 = np.empty_like(X)

            for iy in range(len(y)):
                for ix in range(len(x)):
                    pars = result.popt.copy()
                    pars[xindex] = X[iy, ix]
                    pars[yindex] = Y[iy, ix]

                    _, _, totchi2[iy, ix] = chi2_from_params(*pars)

            totchi2 -= np.nanmin(totchi2)

            fig, ax = plt.subplots(figsize=(7, 6))

            sigma_levels = [2.30, 6.18, 11.83] # 1\sigma, 2\sigma, 3\sigma

            cf = ax.contourf(
                X,
                Y,
                totchi2,
                levels=[0.0] + sigma_levels,
                colors=["#1a5fa8", "#4a90d9", "#a8c8f0"],
                alpha=0.35,
            )
            cl = ax.contour(
                X,
                Y,
                totchi2,
                levels=sigma_levels,
                colors=["#1a5fa8", "#4a90d9", "#a8c8f0"],
                linewidths=1.8,
            )
            legend_handles = [
                Line2D(
                    [], [], marker="*", linestyle="None",
                    color="k", markersize=7,
                    label="Best fit"
                ),
                Patch(facecolor=to_rgba("#1a5fa8", 0.35), edgecolor=to_rgba("#1a5fa8", 1.0), linewidth=1.8, label=r"$1\sigma$"),
                Patch(facecolor=to_rgba("#4a90d9", 0.35), edgecolor=to_rgba("#4a90d9", 1.0), linewidth=1.8, label=r"$2\sigma$"),
                Patch(facecolor=to_rgba("#a8c8f0", 0.35), edgecolor=to_rgba("#a8c8f0", 1.0), linewidth=1.8, label=r"$3\sigma$"),
            ]

            ax.scatter(
                xbest,
                ybest,
                marker="*",
                color="k",
                s=70,
                zorder=10,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            ax.minorticks_on()
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(handles=legend_handles, loc="upper right")

            fig.tight_layout()
            return fig, ax

        def plot_li9_he8_contour(
            exposure: float, 
            br9li: float, br8he: float,
            r9min: float, r9max: float,
            r8min: float, r8max: float,
        ) -> tuple[plt.Figure, plt.Axes]:
            N_       = result.popt[0]
            fbkg_    = result.popt[1]
            f9li_    = result.popt[2]
            # li_    = result.popt[3]
            # he_    = result.popt[4]

            R9_best = N_ * (1.0 - fbkg_) * f9li_            / (br9li * exposure)
            R8_best = N_ * (1.0 - fbkg_) * (1.0 - f9li_)    / (br8he * exposure)

            r9 = np.linspace(r9min, r9max, 101)
            r8 = np.linspace(r8min, r8max, 101)

            R9, R8 = np.meshgrid(r9, r8)
            chi2 = np.empty_like(R9)

            for iy in range(R9.shape[0]):
                for ix in range(R9.shape[1]):

                    _N = exposure * (br9li * R9[iy, ix] + br8he * R8[iy, ix]) / (1.0 - fbkg_)

                    if _N <= 0:
                        chi2[iy, ix] = np.nan
                        continue

                    _f9li = br9li * R9[iy, ix] / (br9li * R9[iy, ix] + br8he * R8[iy, ix])

                    _, _, chi2[iy, ix] = chi2_from_params(
                        _N,
                        fbkg_,
                        _f9li,
                        # t9li_,
                        # t8he_,
                    )

            chi2 -= np.nanmin(chi2)

            fig, ax = plt.subplots(figsize=(7,6))

            sigma_levels = [2.30, 6.18, 11.83]

            cf = ax.contourf(
                R9,
                R8,
                chi2,
                levels=[0.0] + sigma_levels,
                colors=["#1a5fa8", "#4a90d9", "#a8c8f0"],
                alpha=0.35,
            )
            cl = ax.contour(
                R9,
                R8,
                chi2,
                levels=sigma_levels,
                colors=["#1a5fa8", "#4a90d9", "#a8c8f0"],
                linewidths=1.8,
            )
            ax.scatter(
                R9_best,
                R8_best,
                marker="*",
                color="k",
                s=70,
                zorder=10,
            )
            legend_handles = [
                Line2D(
                    [], [], marker="*", linestyle="None",
                    color="k", markersize=7,
                    label="Best fit"
                ),
                Patch(facecolor=to_rgba("#1a5fa8", 0.35), edgecolor=to_rgba("#1a5fa8", 1.0), linewidth=1.8, label=r"$1\sigma$"),
                Patch(facecolor=to_rgba("#4a90d9", 0.35), edgecolor=to_rgba("#4a90d9", 1.0), linewidth=1.8, label=r"$2\sigma$"),
                Patch(facecolor=to_rgba("#a8c8f0", 0.35), edgecolor=to_rgba("#a8c8f0", 1.0), linewidth=1.8, label=r"$3\sigma$"),
            ]

            ax.set_xlabel(r"$^{9}$Li rate (kton$^{-1}$ day$^{-1}$)")
            ax.set_ylabel(r"$^{8}$He rate (kton$^{-1}$ day$^{-1}$)")

            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))

            ax.minorticks_on()
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(handles=legend_handles, loc="lower left")

            fig.tight_layout()
            return fig, ax

        fig, _ = plot_1d_contour(N,    Nerr,    0, r"$N$")
        save_figure(fig, self.stem, "_ratio_contour_N", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_1d_contour(fbkg, fbkgerr, 1, r"$f_{\mathrm{bkg}}$")
        save_figure(fig, self.stem, "_ratio_contour_fbkg", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_1d_contour(f9li, f9lierr, 2, r"$f_{^9\mathrm{Li}}$")
        save_figure(fig, self.stem, "_ratio_contour_f9li", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_2d_contour(
            N,    Nerr,    0, r"$N$",
            fbkg, fbkgerr, 1, r"$f_{\mathrm{bkg}}$", 
        )
        save_figure(fig, self.stem, "_ratio_contour_N_fbkg", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_2d_contour(
            N,    Nerr,    0, r"$N$",
            f9li, f9lierr, 2, r"$f_{^9\mathrm{Li}}$", 
        )
        save_figure(fig, self.stem, "_ratio_contour_N_f9li", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_2d_contour(
            fbkg, fbkgerr, 1, r"$f_{\mathrm{bkg}}$", 
            f9li, f9lierr, 2, r"$f_{^9\mathrm{Li}}$", 
        )
        save_figure(fig, self.stem, "_ratio_contour_fbkg_f9li", output_dir=self.output_dir)
        plt.close(fig)

        br9li = 0.505
        br8he = 0.16
        fig, _ = plot_li9_he8_contour(
            208.1 * 20.0 * (17.2 / 17.7)**3, 
            1.0, 1.0,
            1.0, 1.5,
            0.0, 0.25,
        )
        save_figure(fig, self.stem, "_ratio_contour_li9_he8", output_dir=self.output_dir)
        plt.close(fig)

        fig, _ = plot_li9_he8_contour(
            208.1 * 20.0 * (17.2 / 17.7)**3, 
            br9li, br8he,
            2.0, 3.0,
            0.0, 1.5,
        )
        save_figure(fig, self.stem, "_ratio_contour_li9_he8_corrected", output_dir=self.output_dir)
        plt.close(fig)

        plt.show()