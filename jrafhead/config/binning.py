import numpy as np

from jrafhead.utils import (
    nmo_analysis_bins,
    uniform_bins,
)

# ---------------------------------------------------------------------------
# Named standard binnings used across multiple plotters
# ---------------------------------------------------------------------------

#: Prompt energy binnings
PROMPT_ENERGY_BINS_UNIFORM:         np.ndarray = uniform_bins(0.0, 12.0, 100)
PROMPT_ENERGY_BINS_207DAYS:         np.ndarray = np.array([
    0.7000, 0.9400, 0.9800, 1.0200, 1.0600, 1.1000, 1.1400, 
    1.1800, 1.2200, 1.2600, 1.3000, 1.3400, 1.3800, 1.4200, 
    1.4600, 1.5000, 1.5400, 1.5800, 1.6200, 1.6600, 1.7000, 
    1.7400, 1.7800, 1.8200, 1.8600, 1.9000, 1.9400, 1.9800, 
    2.0200, 2.0600, 2.1000, 2.1400, 2.1800, 2.2200, 2.2600, 
    2.3000, 2.3400, 2.3800, 2.4200, 2.4600, 2.5000, 2.5400, 
    2.5800, 2.6200, 2.6600, 2.7000, 2.7400, 2.7800, 2.8200, 
    2.8600, 2.9000, 2.9400, 2.9800, 3.0200, 3.0600, 3.1000, 
    3.1400, 3.1800, 3.2200, 3.2600, 3.3000, 3.3400, 3.3800, 
    3.4200, 3.4600, 3.5000, 3.5400, 3.5800, 3.6200, 3.6600, 
    3.7000, 3.7400, 3.7800, 3.8200, 3.8600, 3.9000, 3.9400, 
    3.9800, 4.0200, 4.0600, 4.1000, 4.1400, 4.1800, 4.2200, 
    4.2600, 4.3000, 4.3400, 4.3800, 4.4200, 4.4600, 4.5000, 
    4.5400, 4.5800, 4.6200, 4.6600, 4.7000, 4.7400, 4.7800, 
    4.8200, 4.8600, 4.9000, 4.9400, 4.9800, 5.0200, 5.0600, 
    5.1000, 5.1400, 5.1800, 5.2200, 5.2600, 5.3000, 5.3400, 
    5.3800, 5.4200, 5.4600, 5.5000, 5.5400, 5.5800, 5.6200, 
    5.6600, 5.7000, 5.7400, 5.7800, 5.8200, 5.8600, 5.9000, 
    5.9400, 5.9800, 6.0200, 6.0600, 6.1000, 6.1400, 6.1800, 
    6.2200, 6.2600, 6.3000, 6.3400, 6.3800, 6.4200, 6.4600, 
    6.5000, 6.5400, 6.5800, 6.6200, 6.6600, 6.7400, 6.8200, 
    6.9000, 7.0000, 7.1000, 7.2000, 7.4000, 7.6200, 7.8600, 
    8.1200, 8.4600, 12.000
])
PROMPT_ENERGY_BINS_NMO:             np.ndarray = nmo_analysis_bins(
    np.array([0.8, 0.94, 7.44, 7.8, 8.2, 12.0]),
    np.array([1,   325,  9,    4,   1])
)

#: Delayed energy region around the neutron capture peak on H (2.0-2.5 MeV, 50 bins)
DELAYED_ENERGY_HYDROGEN_BINS:       np.ndarray = uniform_bins(2.0, 2.5, 50)
DELAYED_ENERGY_CARBON_BINS:         np.ndarray = uniform_bins(4.5, 5.5, 50)
DELAYED_ENERGY_BINS:                np.ndarray = uniform_bins(2.0, 5.5, 350)

#: Prompt-delayed time difference
PROMPT_DELAYED_DT_BINS:             np.ndarray = uniform_bins(0.0, 1.0, 100)
PROMPT_DELAYED_DT_ACCIDENTAL_BINS:  np.ndarray = uniform_bins(2.0, 4.0, 100)

#: Prompt-delayed spatial distance
PROMPT_DELAYED_DR_BINS:             np.ndarray = uniform_bins(0.0, 1.5, 100)

#: Spatial distribution: transverse radius squared \rho^2
SPATIAL_RHO2_BINS:                  np.ndarray = uniform_bins(0.0, 17.7**2, 50)

#: Spatial distribution: z coordinate
SPATIAL_Z_BINS:                     np.ndarray = uniform_bins(-17.7, 17.7, 50)

#: Muon performance angle
MUON_PERFORMANCE_ANGLE_BINS:        np.ndarray = uniform_bins(0.0, 5.0, 100)

#: Muon performance distance
MUON_PERFORMANCE_DISTANCE_BINS:     np.ndarray = uniform_bins(0.0, 2.0, 100)