import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs=2, type=float, default=(12608.3, 238.3), help="Input total number of comosgenic with its uncertainty")
args = parser.parse_args()

# 0.9592225549614767 +/- 0.006068434955699611 stat +/- 0.013600886811613979 syst  |  neutron without flasher cut, without helium, with multipliticy
# 0.9301847594467259 +/- 0.005580037072882779 stat +/- 0.01456172201634703  syst  |  neutron with flasher cut,    with helium,    with multiplicity
# 0.9300745298317897 +/- 0.005495252568151527 stat +/- 0.014854065891313968 syst  |  neutron with flasher cut,    with helium,    without multiplicity

neutron_accompanying_effiency      = 0.9300745298317897
neutron_accompanying_effiency_stat = 0.005495252568151527
neutron_accompanying_effiency_syst = 0.014854065891313968

selection_table_efficiency = {
    "fiducial_volume":          91.40 / 100.0,
    "prompt_energy":            99.55 / 100.0,
    "delayed_energy":           99.94 / 100.0,
    "prompt_delayed_distance":  99.16 / 100.0,
    "prompt_delayed_time":      96.63 / 100.0,
#     "multiplicity":             97.50 / 100.0,
}

selection_table_relerr = {
    "fiducial_volume":          1.90 / 100.0,
    "delayed_energy":           0.10 / 100.0,
    "prompt_delayed_distance":  0.20 / 100.0,
    "prompt_delayed_time":      0.03 / 100.0,
}

selection_efficiency = np.prod(list(selection_table_efficiency.values()))
selection_relerr = np.sqrt(np.sum(np.square(list(selection_table_relerr.values()))))
selection_error = selection_efficiency * selection_relerr

lifetime = 208.08473379629631

ncosmo_fitted       = args.input[0]
ncosmo_fitted_error = args.input[1]

rate_fitted       = ncosmo_fitted / lifetime
rate_fitted_error = ncosmo_fitted_error / lifetime

rate_corr       = rate_fitted / neutron_accompanying_effiency
rate_corr_stat = rate_corr * np.sqrt( (rate_fitted_error / rate_fitted)**2 + (neutron_accompanying_effiency_stat / neutron_accompanying_effiency)**2 )
rate_corr_syst = rate_corr * neutron_accompanying_effiency_syst / neutron_accompanying_effiency

print(f"Corrected rate (before selection) = {rate_corr} +/- {rate_corr_stat} +/- {rate_corr_syst} cpd")

rate_corr = rate_fitted / neutron_accompanying_effiency / selection_efficiency
rate_corr_stat = rate_corr * np.sqrt((rate_fitted_error / rate_fitted)**2 + (neutron_accompanying_effiency_stat / neutron_accompanying_effiency)**2)
rate_corr_syst = rate_corr * np.sqrt((neutron_accompanying_effiency_syst / neutron_accompanying_effiency)**2 + selection_relerr**2)

print(f"Corrected rate (after selection)= {rate_corr} +/- {rate_corr_stat} +/- {rate_corr_syst} cpd")