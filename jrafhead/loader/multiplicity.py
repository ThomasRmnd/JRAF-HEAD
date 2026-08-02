from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import uproot

from ._common import _stack_timestamps, _stack_positions


# -------------------------------------------------------------------------------------------------
# Multiplicity
# -------------------------------------------------------------------------------------------------

_MULTIPLICITY_BRANCHES = [
    "run_id", 
    "sec", "nsec", 
    "posx", "posy", "posz", 
    "e", 
]


@dataclass
class MultiplicityData:
    """
    Arrays of the multiplicity analysis.
    """
    run_id: np.ndarray # (N,)   int     Run number
    sec:    np.ndarray # (N,)   int     Timestamp second
    nsec:   np.ndarray # (N,)   int     Timestamp nanosecond
    posx:   np.ndarray # (N,)   float   Position x
    posy:   np.ndarray # (N,)   float   Position y
    posz:   np.ndarray # (N,)   float   Position z
    e:      np.ndarray # (N,)   float   Energy


def load_multiplicity(filepath: str, dirpath: str) -> MultiplicityData:
    """
    Load and prepare all arrays for the multiplicity analysis.
    """
    file = uproot.open(filepath)
    raw = file[f"{dirpath}/events"].arrays(_MULTIPLICITY_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} multiplicity events")

    return MultiplicityData(
        run_id=raw["run_id"],
        sec=raw["sec"],
        nsec=raw["nsec"],
        posx=raw["posx"],
        posy=raw["posy"],
        posz=raw["posz"],
        e=raw["e"],
    )