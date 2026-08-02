from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

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
    """
    Arrays of the DAQ analysis
    """
    run_id:         np.ndarray  # (N,)  int     Run number
    start_sec:      np.ndarray  # (N,)  int     Start timestamp second
    start_nsec:     np.ndarray  # (N,)  int     Start timestamp nanosecond
    duration_sec:   np.ndarray  # (N,)  int     Duration second
    duration_nsec:  np.ndarray  # (N,)  int     Duration nanosecond


def load_lifetime_daq(filepath: str, dirpath: str) -> LifetimeDAQData:
    """
    Load and prepare all arrays for the DAQ analysis.
    """
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

# -------------------------------------------------------------------------------------------------
# Veto
# -------------------------------------------------------------------------------------------------

_LIFETIME_VETO_BRANCHES = [
    "run_id",
    "sec", "nsec",
    "type", "entries",
]

class VetoType(IntEnum):
    _None           = 0
    BeginningOfJob  = 1
    MissingHeaders  = 2
    BigGaps         = 3
    Muon            = 4
    MuonCd          = 5
    MuonWp          = 6

@dataclass
class LifetimeVetoData:
    """
    Arrays of the veto analysis
    """
    run_id:     np.ndarray  # (N,)  int     Run number
    sec:        np.ndarray  # (N,)  int     Duration timestamp second
    nsec:       np.ndarray  # (N,)  int     Duration timestamp nanosecond
    type:       np.ndarray  # (N,)  int     Veto type
    entries:    np.ndarray  # (N,)  int     Number of veto


def load_lifetime_veto(filepath: str, dirpath: str) -> LifetimeVetoData:
    """
    Load and prepare all arrays for the veto analysis.
    """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/veto"].arrays(_LIFETIME_VETO_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} veto informations")

    return LifetimeVetoData(
        run_id      = raw["run_id"],
        sec         = raw["sec"],
        nsec        = raw["nsec"],
        type        = raw["type"],
        entries     = raw["entries"],
    )