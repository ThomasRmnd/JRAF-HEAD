from dataclasses import dataclass

import numpy as np
import uproot

# -------------------------------------------------------------------------------------------------
# Muon WP tagging efficiency
# -------------------------------------------------------------------------------------------------

_MUON_EFFICIENCY_WP = [
    "run_id",
    "nb_cd_wp_30k",
    "nb_cd_only_10ms",
]

@dataclass
class MuonEfficiencyWP:
    """ Arrays of the muon effiency WP analysis. """
    run_id:                 np.ndarray  # (N,)  int     Run number
    nb_cd_wp_30k:           np.ndarray  # (N,)  int     Number of CD-WP muon over 30k PEs threshold
    nb_cd_only_10ms:        np.ndarray  # (N,)  int     Number of CD only muon over 10 ms threshold from previous muon

def load_muon_efficiency_wp(filepath: str, dirpath: str) -> MuonEfficiencyWP:
    """ Load and prepare all arrays for the muon efficiency WP analysis. """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/efficiency"].arrays(_MUON_EFFICIENCY_WP, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon comparisons")

    return MuonEfficiencyWP(
        run_id                  = raw["run_id"],
        nb_cd_wp_30k            = raw["nb_cd_wp_30k"],
        nb_cd_only_10ms         = raw["nb_cd_only_10ms"],
    )