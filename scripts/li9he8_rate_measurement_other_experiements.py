import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
from scipy.stats import chi2

from jrafhead.config import (
    BLACK,
    CUSTOM_BLUE,
    CUSTOM_RED,
    GOOGLE_BLUE,
    GOOGLE_GREEN,
    GOOGLE_PURPLE,
    GOOGLE_RED,
    GOOGLE_YELLOW,
    setup_style,
)

setup_style()

#############################
### Yield helper function ###
#############################

def calculate_juno_yield(
        N9li:         float, 
        N9li_stat:    float,
        N9li_syst:    float,
        eff_neu:      float,
        eff_neu_stat: float,
        eff_neu_syst: float, 
        mult_eff:     float = 1.0, 
        mult_eff_unc: float = 0.0,
        Lmu:          float = 2203.0,
        Lmu_err:      float = 7.0
    ):
    """Calculates 9Li yield and uncertainties for a specific configuration."""
    print(f"Fitted number of beta-n decay 9Li: {N9li:.1f} +/- {N9li_stat:.1f}")
    print(f"Neutron-accompanying efficiency: {eff_neu:.3f} +/- {eff_neu_stat:.3f} (stat) +/- {eff_neu_syst:.3f} (syst)")
    # Neutron-accompanying efficiency propagation
    Ncorr          = N9li / eff_neu
    # Stat
    rel_N_fit_stat = N9li_stat / N9li
    rel_eff_stat   = eff_neu_stat / eff_neu
    N_corr_stat    = Ncorr * np.sqrt(rel_N_fit_stat**2 + rel_eff_stat**2)
    # Syst
    rel_N_fit_syst = N9li_syst / N9li
    rel_eff_syst   = eff_neu_syst / eff_neu
    N_corr_syst    = Ncorr * np.sqrt(rel_N_fit_syst**2 + rel_eff_syst**2)

    # Rate
    sec_per_day = 86400.0
    time_sec    = 208.08 * sec_per_day
    R9li      = Ncorr / time_sec
    R9li_stat = N_corr_stat / time_sec
    R9li_syst = N_corr_syst / time_sec
    print(f"Corrected rate of beta-n decay 9Li: {R9li * sec_per_day:.1f} +/- {R9li_stat * sec_per_day:.1f} (stat) +/- {R9li_syst * sec_per_day:.1f} (syst)")

    eff = np.array([
        0.914,   # Fiducial volume
        0.9955,  # Prompt energy
        0.9994,  # Delayed energy
        0.96636, # Prompt-delayed time
        0.99,    # Prompt-delayed distance
        0.9378,  # Muon veto
        mult_eff # Multiplicity veto
    ])
    eff_rel_unc = np.array([
        1.9,     # Fiducial volume
        0.0,     # Prompt energy
        0.1,     # Delayed energy
        0.03,    # Prompt-delayed
        0.2,     # Prompt-delayed
        0.0,     # Muon veto
        mult_eff_unc # Multiplicity veto uncertainty (%)
    ]) / 100.0 # relative, "-" treated as 0

    eff_sel         = np.prod(eff)
    eff_sel_rel_unc = np.sqrt(np.sum(eff_rel_unc**2))
    eff_sel_unc     = eff_sel * eff_sel_rel_unc
    print(f"Full efficiency: {eff_sel} +/- {eff_sel_unc}")

    Bbn, Bbn_err     = 0.508,  0.009

    R_prod      = (R9li * sec_per_day) / (eff_sel * Bbn)
    R_prod_stat = (R9li_stat * sec_per_day) / (eff_sel * Bbn)
    rel_R_prod_syst = np.sqrt(
        (R9li_syst / R9li)**2 +
        (eff_sel_unc / eff_sel)**2 +
        (Bbn_err / Bbn)**2
    )
    R_prod_syst = R_prod * rel_R_prod_syst
    print(f"Total production rate of 9Li: {R_prod:.1f} +/- {R_prod_stat:.1f} (stat) +/- {R_prod_syst:.1f} (syst) day^-1")

    Rmu, Rmu_err     = 4.27,   0.03
    ntrk, ntrk_err   = 1.148,  0.003
    
    rhoLS, rhoLS_err = 0.856,  0.001

    # Value
    Y = R9li / (Rmu * ntrk * Lmu * rhoLS * Bbn * eff_sel)
    # Stat
    rel_Y_stat = R9li_stat / R9li
    Y_stat = Y * rel_Y_stat
    # Syst
    rel_Y_syst = np.sqrt(
        (R9li_syst / R9li)**2 +
        (eff_sel_unc / eff_sel)**2 +
        (Rmu_err / Rmu)**2 +
        (ntrk_err / ntrk)**2 +
        (Lmu_err / Lmu)**2 +
        (rhoLS_err / rhoLS)**2 +
        (Bbn_err / Bbn)**2
    )
    Y_syst = Y * rel_Y_syst
    # Total
    Y_tot = np.sqrt(Y_stat**2 + Y_syst**2)

    return Y / 1e-8, Y_stat / 1e-8, Y_syst / 1e-8, Y_tot / 1e-8

# Y_{^9\mathrm{Li}} = \frac{ R_{^9\mathrm{Li}} }{ R_{\mu} n_{\mathrm{trk}} L_{\mu} \rho_{\mathrm{LS}} B_{\beta n} \epsilon_{^9\mathrm{Li}} } 
# with:
# - R_{^9\mathrm{Li}}         the measured rate of lithium
# - R_{\mu}                 the muon rate 
# - n_{\mathrm{trk}}          the mean muon multiplicity
# - L_{\mu}                 the mean muon length in the LS
# - \rho_{\mathrm{LS}}        the LS density
# - B_{\beta n}             the \beta-n branching ratio
# - \epsilong_{^9\mathrm{Li}} the selection efficiency

#########################
### Yield Cases Setup ###
#########################

from itertools import product

# 1. Track Configurations: (track_id, Lmu, Lmu_err, tick_offset, markerstyle)
tracks = [
    ("cdwpttchi2_wpclassify",  2203.0, 7.0, -0.2, "^"),
    ("cdwpttchi2",             2336.0, 7.0,  0.0, "s"),
    ("mc",                     2320,   19.0, 0.2, "d"),
]

# 2. Multiplicity Regimes: (regime_id, base_tick, color, calculate_yield_kwargs)
fit_50_ms_scale_factor = (
    (np.exp(-(0.55 + 1.0 / 0.257) * 0.007) - np.exp(-(0.55 + 1.0 / 0.257) * 10.007)) /
    (np.exp(-(0.55 + 1.0 / 0.257) * 0.050) - np.exp(-(0.55 + 1.0 / 0.257) * 10.050))
) # Extend the 50 ms lower bound as if it was a 7 ms lower bound
print(fit_50_ms_scale_factor)
regimes = [
    (
        "no_multiplicity", 1.0, GOOGLE_GREEN, "No mult. veto",
        {
            "N9li":     12702.9, "N9li_stat":    146.2,  "N9li_syst":    90.9, # Binning systematic
            "eff_neu":  0.9402,  "eff_neu_stat": 0.0053, "eff_neu_syst": np.sqrt(0.0119**2 + 0.0038**2), # Dispersion + Binning systematic
            "mult_eff": 1.0,     "mult_eff_unc": 0.0,
        }
    ),
    (
        "no_multiplicity_b12", 2.0, GOOGLE_RED, "No mult. veto\nwith " + r"$^{12}\mathrm{B}$",
        {
            "N9li":     12002.1, "N9li_stat":    190.2,  "N9li_syst":    43.0, # Binning systematic
            "eff_neu":  0.9287,  "eff_neu_stat": 0.0062, "eff_neu_syst": np.sqrt(0.0120**2 + 0.0026**2), # Dispersion + Binning systematic
            "mult_eff": 1.0,     "mult_eff_unc": 0.0,
        }
    ),
    (
        "no_multiplicity_50_ms", 3.0, GOOGLE_BLUE, "No mult. veto\n" + r"$> 50$~ms",
        {
            "N9li":     10094.0 * fit_50_ms_scale_factor, "N9li_stat":    134.4 * fit_50_ms_scale_factor,  "N9li_syst":    27.7, # Binning systematic
            "eff_neu":  0.9496,  "eff_neu_stat": 0.0055, "eff_neu_syst": np.sqrt(0.0146**2 + 0.0023**2), # Dispersion + Binning systematic
            "mult_eff": 1.0,     "mult_eff_unc": 0.0,
        }
    ),
    (
        "multiplicity_efficiency_ibd", 4.0, GOOGLE_PURPLE, "IBD mult.\nefficiency",
        {
            "N9li":     12129.4, "N9li_stat":    143.7,  "N9li_syst":    35.3, # Binning systematic
            "eff_neu":  0.9314,  "eff_neu_stat": 0.0049, "eff_neu_syst": np.sqrt(0.0137**2 + 0.0016**2), # Dispersion + Binning systematic
            "mult_eff": 0.975,   "mult_eff_unc": 0.5,
        }
    ),
    (
        "multiplicity_efficiency_cosmo", 5.0, GOOGLE_YELLOW, "Data driven\nmult. efficiency",
        {
            "N9li":     12129.4, "N9li_stat":    143.7,  "N9li_syst":    35.3, # Binning systematic
            "eff_neu":  0.9314,  "eff_neu_stat": 0.0049, "eff_neu_syst": np.sqrt(0.0137**2 + 0.0016**2), # Dispersion + Binning systematic
            "mult_eff": 0.9744,  "mult_eff_unc": 0.23,
        }
    ),
]

# 3. Generate studies map
studies = {}
for (reg_id, base_tick, color, label, kwargs), (trk_id, Lmu, Lmu_err, offset, style) in product(regimes, tracks):
    print(f"Yield for {reg_id}__{trk_id} configuration")
    Y, Y_stat, Y_syst, Y_tot = calculate_juno_yield(**kwargs, Lmu=Lmu, Lmu_err=Lmu_err)
    
    studies[f"{reg_id}__{trk_id}"] = {
        "Y": Y, "Y_stat": Y_stat, "Y_syst": Y_syst, "Y_tot": Y_tot,
        "ticks": base_tick + offset, "markerstyle": style, "markercolor": color,
    }

# Adopt the Multiplicity with 9Li/8He cosmogenic efficiency estimation as a baseline
# - no_multiplicity__cdwpttchi2_wpclassify
# - no_multiplicity__mc
Y_juno     = studies["no_multiplicity__mc"]["Y"]
Y_juno_err = studies["no_multiplicity__mc"]["Y_tot"]

#########################
### Other experiments ###
#########################

# Double Chooz - https://arxiv.org/abs/1802.08048
# RENO         - https://arxiv.org/abs/2204.09215
# KamLAND      - https://arxiv.org/abs/0907.0066
# Borexino     - https://arxiv.org/abs/1304.7381
# Daya Bay     - https://arxiv.org/abs/2402.05383

other_experiments = [
    {"markerstyle": "^", "markersize": 7, "label": r"Double Chooz", "Y9li": 5.51, "Y9li_err": 0.51, "Emu": 32.1, "Emu_err": 2.0},
    {"markerstyle": "^", "markersize": 7, "label": r"Double Chooz", "Y9li": 7.90, "Y9li_err": 0.51, "Emu": 63.7, "Emu_err": 5.5},
    {"markerstyle": "s", "markersize": 6, "label": r"RENO",         "Y9li": 4.80, "Y9li_err": 0.36, "Emu": 33.1, "Emu_err": 2.3},
    {"markerstyle": "s", "markersize": 6, "label": r"RENO",         "Y9li": 9.9,  "Y9li_err": 1.1,  "Emu": 73.6, "Emu_err": 4.4},
    {"markerstyle": "d", "markersize": 7, "label": r"KamLAND",      "Y9li": 22.0, "Y9li_err": 2.0,  "Emu": 260.0,"Emu_err": 8.0},
    {"markerstyle": "v", "markersize": 7, "label": r"Borexino",     "Y9li": 29.0, "Y9li_err": 3.0,  "Emu": 283.0,"Emu_err": 19.0},
    {"markerstyle": "*", "markersize": 10,"label": r"Daya Bay",     "Y9li": 6.73, "Y9li_err": 0.73, "Emu": 63.9, "Emu_err": 3.8},
    {"markerstyle": "*", "markersize": 10,"label": r"Daya Bay",     "Y9li": 6.75, "Y9li_err": 0.70, "Emu": 64.7, "Emu_err": 3.9},
    {"markerstyle": "*", "markersize": 10,"label": r"Daya Bay",     "Y9li": 13.74,"Y9li_err": 0.82, "Emu": 143.0,"Emu_err": 8.6},
]

#########################
### Predictions & Fit ###
#########################

Emu_juno  = 207.0

# Old Power Law parameters (Daya Bay)
Y0_9li_old    = 0.33e-8
alpha_9li_old = 0.76
Y0_8he_old    = 2.14e-10
alpha_8he_old = 0.65

def power_law(Emu, Y0, alpha):
    return Y0 * (Emu ** alpha)

# Prepare combined data for weighted power law fitting
all_Emu = np.array([exp["Emu"] for exp in other_experiments] + [Emu_juno])
all_Y   = np.array([exp["Y9li"] for exp in other_experiments] + [Y_juno]) * 1e-8
all_Y_err = np.array([exp["Y9li_err"] for exp in other_experiments] + [Y_juno_err]) * 1e-8

# Perform weighted fit in original scale using scipy.optimize.curve_fit
popt, pcov = curve_fit(
    power_law, all_Emu, all_Y, sigma=all_Y_err, absolute_sigma=True, p0=[0.33e-8, 0.76]
)

Y0_9li_new, alpha_9li_new = popt
perr = np.sqrt(np.diag(pcov))

print(f"Old fit: Y0 = {Y0_9li_old / 1e-8:.3f}, alpha = {alpha_9li_old:.3f}")
print(f"New fit: Y0 = {Y0_9li_new / 1e-8:.3f} +/- {perr[0] / 1e-8:.3f}, alpha = {alpha_9li_new:.3f} +/- {perr[1]:.3f}")

y_pred = power_law(all_Emu, Y0_9li_new, alpha_9li_new)
chi2_val = np.sum(((all_Y - y_pred) / all_Y_err) ** 2)
ndf = len(all_Y) - len(popt)  # N points - 2 parameters (Y0, alpha)
pvalue = chi2.sf(chi2_val, ndf)
print(f"P(chi2 / ndf = {chi2_val:.1f} / {ndf}) = {pvalue:.3f}")
chi2_str = rf"$P(\chi^{{2}} / \mathrm{{ndf}} = {chi2_val:.1f} / {ndf}) = {pvalue:.3f}$"

# Curves for plotting
Emu_curve       = np.logspace(np.log10(25.0), np.log10(350.0), 200)
Y_9li_curve_old = power_law(Emu_curve, Y0_9li_old, alpha_9li_old) / 1e-8
Y_9li_curve_new = power_law(Emu_curve, Y0_9li_new, alpha_9li_new) / 1e-8
Y_8he_curve_old = power_law(Emu_curve, Y0_8he_old, alpha_8he_old) / 1e-8

print(f"Predicted (Daya Bay) Y(9Li) = {power_law(Emu_juno, Y0_9li_old, alpha_9li_old) / 1e-8:.2f} * 10^-8 mu^-1 g^-1 cm^2")
print(f"Predicted (Daya Bay) Y(8He) = {power_law(Emu_juno, Y0_8he_old, alpha_8he_old) / 1e-8:.2f} * 10^-8 mu^-1 g^-1 cm^2")

########################################################
### FIGURE 1: World Data & Residuals vs. Muon Energy ###
########################################################

fig1, (ax_main, ax_resi) = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
)

# 1. Main Yield Spectrum
line_old = ax_main.plot(
    Emu_curve, Y_9li_curve_old, color="gray", linestyle=":", linewidth=1.8,
    zorder=2, # label=r"Old Fit ($\alpha = 0.76$)",
)
line_new = ax_main.plot(
    Emu_curve, Y_9li_curve_new, color=CUSTOM_BLUE, linestyle="--", linewidth=2.0,
    zorder=2, # label=rf"New Fit ($\alpha = {alpha_9li_new:.2f}$)",
)
line_8he = ax_main.plot(
    Emu_curve, Y_8he_curve_old, color=CUSTOM_RED, linestyle="--", linewidth=1.8,
    zorder=2, # label=r"$^{8}\mathrm{He}$ Prediction",
)

# Other experiments data points
seen_labels = set()
for exp in other_experiments:
    label = exp["label"] if exp["label"] not in seen_labels else None
    seen_labels.add(exp["label"])
    ax_main.errorbar(
        exp["Emu"], exp["Y9li"], xerr=exp["Emu_err"], yerr=exp["Y9li_err"],
        fmt=exp["markerstyle"], markerfacecolor="white", markeredgecolor=CUSTOM_BLUE,
        ecolor=CUSTOM_BLUE, markersize=exp["markersize"], capsize=3, elinewidth=1.5,
        markeredgewidth=1.5, label=label, zorder=3,
    )

# JUNO data point
ax_main.errorbar(
    Emu_juno, Y_juno, yerr=Y_juno_err, fmt="o",
    markerfacecolor=CUSTOM_BLUE, markeredgecolor=CUSTOM_BLUE, ecolor=CUSTOM_BLUE,
    markersize=7, capsize=3, elinewidth=1.5, label="This study", zorder=4,
)

ax_main.set_ylabel(r"Yield ($\times 10^{-8} \mu^{-1} \mathrm{g}^{-1} \mathrm{cm}^{2}$)")
ax_main.set_xlim(25.0, 350.0)
ax_main.set_ylim(3.0, 55.0)
ax_main.set_xscale("log")
ax_main.set_yscale("log")
ax_main.minorticks_on()
ax_main.tick_params(direction="in", which="both", top=True, right=True, labelbottom=False)
ax_main.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax_main.grid(which="minor", linestyle="--", linewidth=0.25, alpha=0.7)

leg_exp = ax_main.legend(loc="upper left", frameon=False)
fit_handles = line_old + line_new + line_8he
fit_labels = [
    r"Old Fit ($\alpha = 0.76$)",
    rf"New Fit ($\alpha = {alpha_9li_new:.2f}$)" #  + chi2_str,
    # r"$^{8}\mathrm{He}$ Prediction",
]
leg_fits = ax_main.legend(
    fit_handles, fit_labels, loc="lower right", frameon=False,
)
ax_main.add_artist(leg_exp)

# 2. Residuals Plot
# Ratio = 1 line for New Fit (Blue)
ax_resi.axhline(1.0, color=CUSTOM_BLUE, linestyle="-", linewidth=1.8, zorder=2)

# Ratio of Old Fit / New Fit (Gray)
ax_resi.plot(
    Emu_curve, Y_9li_curve_old / Y_9li_curve_new, color="gray", linestyle="--",
    linewidth=1.8, zorder=2
)

for exp in other_experiments:
    y_fit_new = power_law(exp["Emu"], Y0_9li_new, alpha_9li_new) / 1e-8
    ax_resi.errorbar(
        exp["Emu"], exp["Y9li"] / y_fit_new, xerr=exp["Emu_err"], yerr=exp["Y9li_err"] / y_fit_new,
        fmt=exp["markerstyle"], markerfacecolor="white", markeredgecolor=CUSTOM_BLUE,
        ecolor=CUSTOM_BLUE, markersize=exp["markersize"], capsize=3, elinewidth=1.5,
        markeredgewidth=1.5, zorder=3,
    )

y_fit_juno_new = power_law(Emu_juno, Y0_9li_new, alpha_9li_new) / 1e-8
ax_resi.errorbar(
    Emu_juno, Y_juno / y_fit_juno_new, yerr=Y_juno_err / y_fit_juno_new, fmt="o",
    markerfacecolor=CUSTOM_BLUE, markeredgecolor=CUSTOM_BLUE, ecolor=CUSTOM_BLUE,
    markersize=8, capsize=3, elinewidth=1.5, zorder=4,
)

ax_resi.set_xlabel(r"$\langle E_{\mu} \rangle$ (GeV)")
ax_resi.set_ylabel(r"Ratio $^{9}\mathrm{Li}$")
ax_resi.set_ylim(0.65, 1.35)
ax_resi.set_xscale("log")
ax_resi.minorticks_on()
ax_resi.tick_params(direction="in", which="both", top=True, right=True)
ax_resi.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
ax_resi.grid(which="minor", linestyle="--", linewidth=0.25, alpha=0.7)

# fig1.tight_layout()

########################################################
### FIGURE 2: Multiplicity Veto Cases Comparison     ###
########################################################

fig2, ax_comp = plt.subplots(figsize=(10, 6))

for key, study in studies.items():
    color = study["markercolor"]
    marker = study["markerstyle"]
    x = study["ticks"]
    y = study["Y"]
    print(f"{key}: {y:.2f} +/- {study['Y_stat']:.2f} (stat) +/- {study['Y_syst']:.2f} (syst) * 10^-8 mu^-1 g^-1 cm^2")

    # Total uncertainty bar (outer)
    ax_comp.errorbar(
        x, y, yerr=study["Y_tot"], fmt="none",
        ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0, zorder=2
    )
    # Statistical uncertainty bar (inner)
    ax_comp.errorbar(
        x, y, yerr=study["Y_stat"], fmt="none",
        ecolor=BLACK, elinewidth=4.0, capsize=0, capthick=0, zorder=3
    )
    # Data marker
    ax_comp.plot(
        x, y, marker=marker,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=2.0,
        markersize=12, zorder=4
    )

# Reference dotted line to nominal case
ax_comp.axhline(power_law(Emu_juno, Y0_9li_old, alpha_9li_old) / 1e-8, color="gray", linestyle="--", linewidth=1.5, alpha=0.8, zorder=1)

ax_comp.set_xticks([base_tick for (_, base_tick, _, _, _) in regimes])
ax_comp.set_xticklabels([label for (_, _, _, label, _) in regimes])
ax_comp.set_xlim(0.5, 5.5)
ax_comp.set_ylim(16.0, 21.0)
ax_comp.set_ylabel(r"Yield ($\times 10^{-8} \mu^{-1} \mathrm{g}^{-1} \mathrm{cm}^{2}$)")
ax_comp.tick_params(direction="in", which="both", top=True, right=True)
ax_comp.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)

ax_comp.xaxis.set_minor_locator(AutoMinorLocator(5))
ax_comp.yaxis.set_minor_locator(AutoMinorLocator(5))

legend_handles = [
    mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                  markerfacecolor="white", markeredgecolor="gray", markeredgewidth=1.8,
                  markersize=12, label=r"$\langle L_{\mu} \rangle = 2203~\mathrm{m}$ (CdWpTtChi2 + WpClassify)"),
    mlines.Line2D([], [], color="gray", marker="s", linestyle="None",
                  markerfacecolor="white", markeredgecolor="gray", markeredgewidth=1.8,
                  markersize=12, label=r"$\langle L_{\mu} \rangle = 2322~\mathrm{m}$ (CdWpTtChi2)"),
    mlines.Line2D([], [], color="gray", marker="d", linestyle="None",
                  markerfacecolor="white", markeredgecolor="gray", markeredgewidth=1.8,
                  markersize=12, label=r"$\langle L_{\mu} \rangle = 2320~\mathrm{m}$ (MC)"),
    mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5,
                  label=r"Predicted $^{9}\mathrm{Li}$ yield ($\langle E_{\mu} \rangle = 207~\mathrm{GeV}$)"),
]

# title=r"\textbf{Muon Track Length}",
ax_comp.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=1,
    frameon=False,
    columnspacing=1.2,
    handletextpad=0.4
)

fig2.tight_layout()

plt.show()