import argparse
import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import uproot

from jrafhead.config import (
    CUSTOM_BLUE,
    CUSTOM_GREEN,
    CUSTOM_RED,
    setup_style,
)
from jrafhead.loader import load_li9he8_shape
from jrafhead.plotters import PromptEnergyPlotter

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input path")
args = parser.parse_args()

setup_style()

method_optimization        = "li9he8_shape_muon__changing_veto__analysis__cdwpttchi2_1_55m_1_13s__omilrec_jvertex"
method_subtraction         = "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"
method_subtraction_neutron = "li9he8_shape_muon__with_neutron__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"

data_optimization          = load_li9he8_shape(args.input, method_optimization)
data_subtraction           = load_li9he8_shape(args.input, method_subtraction)
data_subtraction_neutron   = load_li9he8_shape(args.input, method_subtraction_neutron)

plotter = PromptEnergyPlotter(
    binmode="normal",
    ylabel=r"P.D.F.",
    ylim=(0, 0.25),
    show_errors=False,
    show_steps=True,
    legend_loc="upper right",
    legend_ncol=1,
)
histsig, _ = np.histogram(data_optimization.signal.e_p, bins=plotter.bins)
# histbkg, _ = np.histogram(data_optimization.background.e_p, bins=plotter.bins)
histtot    = histsig # - histbkg
errtot     = np.sqrt(histtot) / np.sum(histtot * plotter.widths)
histtot    = histtot / np.sum(histtot * plotter.widths)
plotter.add_histogram(
    histtot,
    errtot,
    CUSTOM_BLUE,
    linestyle="-",
    label="Optimization",
)
histsig, _ = np.histogram(data_subtraction.signal.e_p, bins=plotter.bins)
histbkg, _ = np.histogram(data_subtraction.background.e_p, bins=plotter.bins)
histtot    = histsig - histbkg
errtot     = np.sqrt(histsig + histbkg) / np.sum(histtot * plotter.widths)
histtot    = histtot / np.sum(histtot * plotter.widths)
plotter.add_histogram(
    histtot,
    errtot,
    CUSTOM_GREEN,
    linestyle="-",
    label="Subtraction",
)
histsig, _ = np.histogram(data_subtraction_neutron.signal.e_p, bins=plotter.bins)
histbkg, _ = np.histogram(data_subtraction_neutron.background.e_p, bins=plotter.bins)
histtot    = histsig - histbkg
errtot     = np.sqrt(histsig + histbkg) / np.sum(histtot * plotter.widths)
histtot    = histtot / np.sum(histtot * plotter.widths)
plotter.add_histogram(
    histtot,
    errtot,
    CUSTOM_RED,
    linestyle="-",
    label="Subtraction (neutron)",
)

fig, _ = plotter.plot()
plt.show()