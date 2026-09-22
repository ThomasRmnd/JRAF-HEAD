from dataclasses import dataclass

import numpy as np
import uproot

# -------------------------------------------------------------------------------------------------
# Muon performance
# -------------------------------------------------------------------------------------------------

_MUON_PERFORMANCE_BRANCHES = [
    "run_id",
    "sec", "nsec",
    "totq_cd", "totq_wp",
    "angle", "distance",
    "iposdist", "fposdist",
    "target_quality", "target_clippingness",
    "ref_quality", "ref_clippingness",
]

@dataclass
class MuonPerformanceData:
    """ Arrays of the muon performance analysis. """
    run_id:                 np.ndarray  # (N,)  int     Run number
    sec:                    np.ndarray  # (N,)  float   Timestamp second 
    nsec:                   np.ndarray  # (N,)  float   Timestamp nanosecond
    totq_cd:                np.ndarray  # (N,)  float   Total charge in the CD                          (PE)
    totq_wp:                np.ndarray  # (N,)  float   Total charge in the WP                          (PE)
    angle:                  np.ndarray  # (N,)  float   Angle between reference and target              (deg) 
    distance:               np.ndarray  # (N,)  float   Distance between reference and target           (m)
    iposdist:               np.ndarray  # (N,)  float   Entry distance between reference and target     (m) 
    fposdist:               np.ndarray  # (N,)  float   Exit distance between reference and target      (m)
    target_quality:         np.ndarray  # (N,)  float   Target quality 
    target_clippingness:    np.ndarray  # (N,)  float   Target clippingness                             (m)
    ref_quality:            np.ndarray  # (N,)  float   Reference clippingness 
    ref_clippingness:       np.ndarray  # (N,)  float   Reference clippingness                          (m)

def load_muon_performance(filepath: str, dirpath: str) -> MuonPerformanceData:
    """ Load and prepare all arrays for the muon performance analysis. """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/performance"].arrays(_MUON_PERFORMANCE_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon comparisons")

    return MuonPerformanceData(
        run_id                  = raw["run_id"],
        sec                     = raw["sec"],
        nsec                    = raw["nsec"],
        totq_cd                 = raw["totq_cd"],
        totq_wp                 = raw["totq_wp"],
        angle                   = raw["angle"],
        distance                = raw["distance"],
        iposdist                = raw["iposdist"],
        fposdist                = raw["fposdist"],
        target_quality          = raw["target_quality"],
        target_clippingness     = raw["target_clippingness"],
        ref_quality             = raw["ref_quality"],
        ref_clippingness        = raw["ref_clippingness"],
    )