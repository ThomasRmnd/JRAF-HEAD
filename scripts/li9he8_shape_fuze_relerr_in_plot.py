import argparse
import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from jrafhead.config import CUSTOM_BLUE, CUSTOM_GREEN, CUSTOM_RED, setup_style
from jrafhead.plotters import RelativeUncertaintyPromptEnergyPlotter

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", type=str, help="Input path")
args = parser.parse_args()

setup_style()

method_optimization        = "li9he8_shape_muon__changing_veto__analysis__cdwpttchi2_1_55m_1_13s__omilrec_jvertex"
method_subtraction         = "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"
method_subtraction_neutron = "li9he8_shape_muon__with_neutron__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"

path_optimization          = f"{args.input_path}/{method_optimization}/data/{method_optimization}_relative_uncertainty_e_p.json"
path_subtraction           = f"{args.input_path}/{method_subtraction}/data/{method_subtraction}_relative_uncertainty_e_p.json"
path_subtraction_neutron   = f"{args.input_path}/{method_subtraction_neutron}/data/{method_subtraction_neutron}_relative_uncertainty_e_p.json"

with open(path_optimization, 'r') as f:
    data_optimization          = json.load(f)
with open(path_subtraction, 'r') as f:
    data_subtraction           = json.load(f)
with open(path_subtraction_neutron, 'r') as f:
    data_subtraction_neutron   = json.load(f)

relerr_208days_optimization         = data_optimization["normal"]["208 days"]
relerr_208days_subtraction          = data_subtraction["normal"]["208 days"]
relerr_208days_subtraction_neutron  = data_subtraction_neutron["normal"]["208 days"]

relerr_6years_optimization          = data_optimization["normal"]["6 years"]
relerr_6years_subtraction           = data_subtraction["normal"]["6 years"]
relerr_6years_subtraction_neutron   = data_subtraction_neutron["normal"]["6 years"]

COLOR_OPTIMIZATION        = CUSTOM_BLUE
COLOR_SUBTRACTION         = "#479900"
COLOR_SUBTRACTION_NEUTRON = CUSTOM_RED

LINESTYLE_208DAYS = "--"
LINESTYLE_6YEARS  = "-"

plotter = RelativeUncertaintyPromptEnergyPlotter(
    binmode="normal",
    bins=np.linspace(0.0, 12.0, 121),
    show_steps=True,
    legend_loc="upper center",
    legend_ncol=1
)

plotter.add_histogram(
    relerr_208days_optimization,
    np.zeros_like(relerr_208days_optimization),
    COLOR_OPTIMIZATION,
    linestyle=LINESTYLE_208DAYS,
    label=None,
)
plotter.add_histogram(
    relerr_208days_subtraction,
    np.zeros_like(relerr_208days_subtraction),
    COLOR_SUBTRACTION,
    linestyle=LINESTYLE_208DAYS,
    label=None,
)
plotter.add_histogram(
    relerr_208days_subtraction_neutron,
    np.zeros_like(relerr_208days_subtraction_neutron),
    COLOR_SUBTRACTION_NEUTRON,
    linestyle=LINESTYLE_208DAYS,
    label=None,
)
plotter.add_histogram(
    relerr_6years_optimization,
    np.zeros_like(relerr_6years_optimization),
    COLOR_OPTIMIZATION,
    linestyle=LINESTYLE_6YEARS,
    label=None,
)
plotter.add_histogram(
    relerr_6years_subtraction,
    np.zeros_like(relerr_6years_subtraction),
    COLOR_SUBTRACTION,
    linestyle=LINESTYLE_6YEARS,
    label=None,
)
plotter.add_histogram(
    relerr_6years_subtraction_neutron,
    np.zeros_like(relerr_6years_subtraction_neutron),
    COLOR_SUBTRACTION_NEUTRON,
    linestyle=LINESTYLE_6YEARS,
    label=None,
)

plotter.add([], COLOR_OPTIMIZATION,        linestyle="-",               label="Optimization")
plotter.add([], COLOR_SUBTRACTION,         linestyle="-",               label="Subtraction")
plotter.add([], COLOR_SUBTRACTION_NEUTRON, linestyle="-",               label="Subtraction (neutron)")
plotter.add([], "#868686",               linestyle=LINESTYLE_208DAYS, label="208 days")
plotter.add([], "#868686",               linestyle=LINESTYLE_6YEARS,  label="6 years")

fig, ax = plotter.plot()
handles, labels = ax.get_legend_handles_labels()
for i in range(len(handles[:3])):
    handles[i] = Line2D([], [], color=handles[i].get_color(), marker="o", linestyle="None", markersize=6)
ax.legend(handles, labels, loc="upper center")
plt.show()