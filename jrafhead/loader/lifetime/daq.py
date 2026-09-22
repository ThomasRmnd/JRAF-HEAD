from dataclasses import dataclass

import numpy as np
import uproot

# -------------------------------------------------------------------------------------------------
# DAQ
# -------------------------------------------------------------------------------------------------

_LIFETIME_DAQ_BRANCHES = [
    "run_id", 
    "start_sec", "start_nsec",
    "duration_sec", "duration_nsec"
]

@dataclass
class LifetimeDAQData:
    """ Arrays of the DAQ analysis. """
    run_id:         np.ndarray  # (N,)  int     Run number
    start_sec:      np.ndarray  # (N,)  int     Start timestamp second
    start_nsec:     np.ndarray  # (N,)  int     Start timestamp nanosecond
    duration_sec:   np.ndarray  # (N,)  int     Duration second
    duration_nsec:  np.ndarray  # (N,)  int     Duration nanosecond

def load_lifetime_daq(filepath: str, dirpath: str) -> LifetimeDAQData:
    """ Load and prepare all arrays for the DAQ analysis. """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/daq"].arrays(_LIFETIME_DAQ_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} DAQ informations")

    return LifetimeDAQData(
        run_id          = raw["run_id"],
        start_sec       = raw["start_sec"],
        start_nsec      = raw["start_nsec"],
        duration_sec    = raw["duration_sec"],
        duration_nsec   = raw["duration_nsec"],
    )