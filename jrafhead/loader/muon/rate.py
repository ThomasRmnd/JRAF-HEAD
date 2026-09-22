from dataclasses import dataclass

import numpy as np
import uproot

from jrafhead.utils import (
    Histogram,
    load_histogram_from_tree,
)

# -------------------------------------------------------------------------------------------------
# Muon rate
# -------------------------------------------------------------------------------------------------

_MUON_RATE_BRANCHES = [
    "run_id",
    "hist_cd_only_edges",
    "hist_cd_only_counts",
    "hist_cd_only_errors",
    "hist_cd_only_underflow",
    "hist_cd_only_overflow",
    "hist_wp_only_edges",
    "hist_wp_only_counts",
    "hist_wp_only_errors",
    "hist_wp_only_underflow",
    "hist_wp_only_overflow",
    "hist_cd_wp_edges",
    "hist_cd_wp_counts",
    "hist_cd_wp_errors",
    "hist_cd_wp_underflow",
    "hist_cd_wp_overflow",
]

@dataclass
class MuonRateData:
    """ Arrays of the muon rate analysis. """
    run_id:             np.ndarray      # (N,)  int         Run number
    hist_cd_only:       list[Histogram] # (N,)              Bin edges for CD only muons
    hist_wp_only:       list[Histogram] # (N,)              Bin counts for CD only muons
    hist_cd_wp:         list[Histogram] # (N,)              Bin uncertainties for CD only muons

def load_muon_rate(filepath: str, dirpath: str) -> MuonRateData:
    """ Load and prepare all arrays for the muon rate analysis. """
    file = uproot.open(filepath)
    raw = file[f"{dirpath}/rate"].arrays(_MUON_RATE_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon rates")

    return MuonRateData(
        run_id          = raw["run_id"],
        hist_cd_only    = load_histogram_from_tree(raw, "hist_cd_only"),
        hist_wp_only    = load_histogram_from_tree(raw, "hist_wp_only"),
        hist_cd_wp      = load_histogram_from_tree(raw, "hist_cd_wp"),
    )