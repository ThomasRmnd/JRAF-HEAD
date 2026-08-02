import numpy as np

neutron_accompanying_effiency      = 0.9572125786132055
neutron_accompanying_effiency_stat = 0.005804800228724907
neutron_accompanying_effiency_syst = 0.015271661539917995

selection_table_efficiency = {
    "fiducial_volume":          89.62 / 100.0,
    "prompt_energy":            100.0 / 100.0,
    "delayed_energy":           99.90 / 100.0,
    "prompt_delayed_distance":  99.16 / 100.0,
    "prompt_delayed_time":      96.95 / 100.0,
    "multiplicity":             98.58 / 100.0,
}

selection_table_relerr = {
    "fiducial_volume":          1.80 / 100.0,
    "delayed_energy":           0.28 / 100.0,
    "prompt_delayed_distance":  0.30 / 100.0,
    "prompt_delayed_time":      0.05 / 100.0,
}

selection_efficiency = np.prod(list(selection_table_efficiency.values()))
selection_relerr = np.sqrt(np.sum(np.square(list(selection_table_relerr.values()))))
selection_error = selection_efficiency * selection_relerr

lifetime = 208.08473379629631

ncosmo_fitted       = 12608.3
ncosmo_fitted_error = 238.3

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