from dataclasses import dataclass

import numpy as np
import uproot

from jrafhead.utils import (
    Histogram,
    load_histogram_from_tree,
)

# -------------------------------------------------------------------------------------------------
# Muon multiplicity
# -------------------------------------------------------------------------------------------------

_MUON_MULTIPLICITY_BRANCHES = [
    "run_id",
    "hist_multiplicity_edges",
    "hist_multiplicity_counts",
    "hist_multiplicity_errors",
    "hist_multiplicity_underflow",
    "hist_multiplicity_overflow",
]

@dataclass
class MuonMultiplicityData:
    """ Arrays of the muon multiplicity analysis. """
    run_id:             np.ndarray      # (N,)  int         Run number
    hist_multipliticy:  list[Histogram] # (N,)              Bin edges for muon multiplicity

def load_muon_multiplicity(filepath: str, dirpath: str) -> MuonMultiplicityData:
    """ Load and prepare all arrays for the muon multiplicity analysis. """
    file = uproot.open(filepath)
    raw = file[f"{dirpath}/multiplicity"].arrays(_MUON_MULTIPLICITY_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon multiplicities")

    return MuonMultiplicityData(
        run_id            = raw["run_id"],
        hist_multipliticy = load_histogram_from_tree(raw, "hist_multiplicity"), 
    )