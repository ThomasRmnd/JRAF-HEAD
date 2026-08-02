import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jrafhead.utils import (
    extract_window,
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input JSON file")
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

radius = np.arange(1.0, 10.1, 1.0)
time   = np.arange(0.2, 2.01, 0.2)

R, T = np.meshgrid(radius, time, indexing="ij")

N = np.full(R.shape, np.nan)
Nerr = np.full(R.shape, np.nan)

for key, values in data.items():
    r_str, t_str = key.split("m_")

    r = extract_window(r_str + "m_", 'm')
    t = extract_window("_" + t_str, 's')

    ir = np.argmin(np.abs(radius - r))
    it = np.argmin(np.abs(time - t))

    N[ir, it] = values["N"]["value"]
    Nerr[ir, it] = values["N"]["error"]

mask = np.isfinite(N)

def model(x, N0, a, b):
    r, t = x
    return N0 * (1.0 - np.exp(-a * t)) * (1.0 - np.exp(-b * r))

xdata = np.vstack((
    R[mask],
    T[mask],
))

ydata = N[mask]
sigma = Nerr[mask]

popt, pcov = curve_fit(
    model,
    xdata,
    ydata,
    sigma=sigma,
    absolute_sigma=True,
    p0=[5600.0, 4.0, 0.7],
)

perr = np.sqrt(np.diag(pcov))

print()
print("Fit parameters")
print("---------------------------")
print(f"N0 = {popt[0]:.2f} ± {perr[0]:.2f}")
print(f"a  = {popt[1]:.3f} ± {perr[1]:.3f}")
print(f"b  = {popt[2]:.3f} ± {perr[2]:.3f}")

Nfit = model((R, T), *popt)

chi2 = np.sum(((N[mask] - Nfit[mask]) / Nerr[mask]) ** 2)
ndf = np.count_nonzero(mask) - len(popt)

print(f"chi2 / ndf = {chi2:.2f} / {ndf} = {chi2/ndf:.3f}")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    R,
    T,
    Nfit,
    alpha=0.6,
)

ax.scatter(
    R[mask],
    T[mask],
    N[mask],
    color="k",
    s=40,
)

ax.set_xlabel("Radius (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("Fitted N")

plt.tight_layout()

residual = np.full_like(N, np.nan)
residual[mask] = (N[mask] - Nfit[mask]) / Nerr[mask]

fig2, ax2 = plt.subplots(figsize=(7, 6))

im = ax2.imshow(
    residual.T,
    origin="lower",
    extent=[
        radius[0] - 0.5,
        radius[-1] + 0.5,
        time[0] - 0.1,
        time[-1] + 0.1,
    ],
    aspect="auto",
    vmin=-3,
    vmax=3,
    cmap="coolwarm",
)

ax2.set_xlabel("Radius (m)")
ax2.set_ylabel("Time (s)")

cb = fig2.colorbar(im)
cb.set_label(r"$(N-N_{\rm fit})/\sigma$")

plt.tight_layout()
plt.show()