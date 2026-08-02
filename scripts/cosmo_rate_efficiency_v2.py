import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from jrafhead.config import BLACK, CUSTOM_BLUE, set_latex_style
from jrafhead.utils import extract_window

# from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input JSON file contaning rate")
parser.add_argument("--input-neutron", type=str, help="Input JSON file containing rate with neutron")
args = parser.parse_args()

set_latex_style()

with open(args.input, 'r') as f:
    data = json.load(f)

with open(args.input_neutron, 'r') as f:
    data_neutron = json.load(f)

radius_dict             = {}
N9li8he_dict            = {}
N9li8heerr_dict         = {}
N9li8he_neutron_dict    = {}
N9li8heerr_neutron_dict = {}
Nbkg_dict               = {}
Nbkgerr_dict            = {}
Nbkg_neutron_dict       = {}
Nbkgerr_neutron_dict    = {}
Rmu_dict                = {}
Rmuerr_dict             = {}
Rmu_neutron_dict        = {}
Rmuerr_neutron_dict     = {}

for key, values in data.items():
    r = extract_window("_" + key + "_", 'm')
    if r == 0:
        continue

    radius_dict[key]     = r
    N9li8he_dict[key]    = values["N9li8he"]
    N9li8heerr_dict[key] = values["N9li8heerr"]
    Nbkg_dict[key]       = values["Nbkg"]
    Nbkgerr_dict[key]    = values["Nbkgerr"]
    Rmu_dict[key]        = values["Rmu"]
    Rmuerr_dict[key]     = values["Rmuerr"]

for key, values in data_neutron.items():
    r = extract_window("_" + key + "_", 'm')
    if r == 0:
        continue

    N9li8he_neutron_dict[key]    = values["N9li8he"]
    N9li8heerr_neutron_dict[key] = values["N9li8heerr"]
    Nbkg_neutron_dict[key]       = values["Nbkg"]
    Nbkgerr_neutron_dict[key]    = values["Nbkgerr"]
    Rmu_neutron_dict[key]        = values["Rmu"]
    Rmuerr_neutron_dict[key]     = values["Rmuerr"]

common_keys        = sorted(set(N9li8he_dict).intersection(N9li8he_neutron_dict))
radius             = np.array([radius_dict[key] for key in common_keys])
N9li8he            = np.array([N9li8he_dict[key] for key in common_keys])
N9li8heerr         = np.array([N9li8heerr_dict[key] for key in common_keys])
N9li8he_neutron    = np.array([N9li8he_neutron_dict[key] for key in common_keys])
N9li8heerr_neutron = np.array([N9li8heerr_neutron_dict[key] for key in common_keys])
Nbkg               = np.array([Nbkg_dict[key] for key in common_keys])
Nbkgerr            = np.array([Nbkgerr_dict[key] for key in common_keys])
Nbkg_neutron       = np.array([Nbkg_neutron_dict[key] for key in common_keys])
Nbkgerr_neutron    = np.array([Nbkgerr_neutron_dict[key] for key in common_keys])
Rmu                = np.array([Rmu_dict[key] for key in common_keys])
Rmuerr             = np.array([Rmuerr_dict[key] for key in common_keys])
Rmu_neutron        = np.array([Rmu_neutron_dict[key] for key in common_keys])
Rmuerr_neutron     = np.array([Rmuerr_neutron_dict[key] for key in common_keys])

indices            = np.argsort(radius)
radius             = radius[indices]
N9li8he            = N9li8he[indices]
N9li8heerr         = N9li8heerr[indices]
N9li8he_neutron    = N9li8he_neutron[indices]
N9li8heerr_neutron = N9li8heerr_neutron[indices]
Nbkg               = Nbkg[indices]
Nbkgerr            = Nbkgerr[indices]
Nbkg_neutron       = Nbkg_neutron[indices]
Nbkgerr_neutron    = Nbkgerr_neutron[indices]
Rmu                = Rmu[indices]
Rmuerr             = Rmuerr[indices]
Rmu_neutron        = Rmu_neutron[indices]
Rmuerr_neutron     = Rmuerr_neutron[indices]

for r, n9li8he, n9li8heerr, n9li8heneu, n9li8heneuerr, nbkg, nbkgerr, nbkgneu, nbkgneuerr, rmu, rmuerr, rmuneu, rmuneuerr in zip(radius, N9li8he, N9li8heerr, N9li8he_neutron, N9li8heerr_neutron, Nbkg, Nbkgerr, Nbkg_neutron, Nbkgerr_neutron, Rmu, Rmuerr, Rmu_neutron, Rmuerr_neutron):
    print(f"Radius: {r}")
    lamb    = rmu    + 1.0 / 0.257
    lambneu = rmuneu + 1.0 / 0.257
    print(f"Number of events (cosmogenics) (without neutron): {- n9li8he *    (np.exp(-lamb    * 10.0) - np.exp(-lamb    * 0.0))}")
    print(f"Number of events (backgrounds) (without neutron): {- nbkg    *    (np.exp(-rmu     * 10.0) - np.exp(-rmu     * 0.0))}")
    print(f"Number of events (cosmogenics) (with    neutron): {- n9li8heneu * (np.exp(-lambneu * 10.0) - np.exp(-lambneu * 0.0))}")
    print(f"Number of events (backgrounds) (with    neutron): {- nbkgneu *    (np.exp(-rmuneu  * 10.0) - np.exp(-rmuneu  * 0.0))}")
    print(f"Number of events (cosmogenics) (without neutron): {n9li8he}")
    print(f"Number of events (backgrounds) (without neutron): {nbkg}")
    print(f"Number of events (total      ) (without neutron): {n9li8he + nbkg}")
    print(f"Number of events (cosmogenics) (with    neutron): {n9li8heneu}")
    print(f"Number of events (backgrounds) (with    neutron): {nbkgneu}")
    print(f"Number of events (total      ) (with    neutron): {n9li8heneu + nbkgneu}")

x         = radius
y         = N9li8he_neutron / N9li8he
yerr      = y * np.sqrt(
    (N9li8heerr_neutron / N9li8he_neutron)**2
    + (N9li8heerr / N9li8he)**2
)

for xx, yy, yyerr in zip(x, y, yerr):
    print(f"Radius {xx} ==> Efficiency = {yy} +/- {yyerr}")

mean = np.mean(y)
stat  = np.sqrt(np.sum(yerr**2)) / len(y)
syst  = np.std(y, ddof=1)

print(f"Global ==> Efficiency = {mean} +/- {stat} stat +/- {syst} syst")

xlabel = r"$d_{\mu-p}$ cut (m)"
ylabel = r"$\epsilon_{n}$ (\%)"
xlim   = (0.0, 11.0)
ylim   = (90.0, 100.0)
xscale = "linear"
yscale = "linear"

fig, ax = plt.subplots(figsize=(7, 6))

ax.errorbar(
    radius, y * 100.0, yerr * 100.0, 
    fmt="o", color=CUSTOM_BLUE, 
    markersize=4.5, zorder=3
)
ax.axhline(
    mean * 100.0,
    label=(
        rf"$\bar{{\epsilon}} = {100.0*mean:.1f}"
        rf"\pm{100.0*stat:.1f}_{{\rm stat}}"
        rf"\pm{100.0*syst:.1f}_{{\rm syst}}\,\%$"
    ),
    color=BLACK, linestyle="--", linewidth=2.0, zorder=2,
)
# ax.add_patch(Rectangle(
#     (0.0, (mean - std) * 100.0), 11.0, 2.0 * std * 100.0,
#     color=CUSTOM_BLUE, alpha=0.2, zorder=1,
# ))

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

if xlim:
    ax.set_xlim(*xlim)
if ylim:
    ax.set_ylim(*ylim)

if xscale == "linear":
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
if yscale == "linear":
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.set_xscale(xscale)
ax.set_yscale(yscale)

ax.minorticks_on()
ax.tick_params(direction="in", which="both", top=True, right=True)
ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend(loc="lower right")

fig.tight_layout()
fig.show()

plt.show()