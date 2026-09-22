from dataclasses import dataclass

import numpy as np
import uproot

from jrafhead.utils import (
    Histogram,
    load_histogram_from_tree,
)

# -------------------------------------------------------------------------------------------------
# Muon length
# -------------------------------------------------------------------------------------------------

_MUON_LENGTH_BRANCHES = [
    "run_id",
    "hist_length_edges",
    "hist_length_counts",
    "hist_length_errors",
    "hist_length_underflow",
    "hist_length_overflow",
    "total_length",
    "total_muon",
]

@dataclass
class MuonLengthData:
    """ Arrays of the muon length analysis. """
    run_id:             np.ndarray      # (N,)  int         Run number
    hist_length:        list[Histogram] # (N,)              Bin edges for muon length
    total_length:       np.ndarray      # (N,)  float       Total muon length
    total_muon:         np.ndarray      # (N,)  float       Total number of muon

def load_muon_length(filepath: str, dirpath: str) -> MuonLengthData:
    """ Load and prepare all arrays for the muon length analysis. """
    file = uproot.open(filepath)
    raw = file[f"{dirpath}/length"].arrays(_MUON_LENGTH_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon lengths")

    return MuonLengthData(
        run_id          = raw["run_id"],
        hist_length     = load_histogram_from_tree(raw, "hist_length"),
        total_length    = raw["total_length"],
        total_muon      = raw["total_muon"], 
    )