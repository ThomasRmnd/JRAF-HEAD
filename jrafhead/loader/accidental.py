from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import uproot

from ._common import (
    _build_timestamps,
    _dt_ms,
    _stack_positions,
    _stack_timestamps,
)

# -------------------------------------------------------------------------------------------------
# Accidental
# -------------------------------------------------------------------------------------------------

_ACCIDENTAL_BRANCHES = [
    "run_id",
    "posx_p", "posy_p", "posz_p", "sec_p", "nsec_p", "e_p",
    "posx_d", "posy_d", "posz_d", "sec_d", "nsec_d", "e_d",
    "dt_last_mu_with_neu", "dt_mu2p", "dlat_mu2p",
]


@dataclass
class AccidentalData:
    """
    Arrays of the accidental analysis.
    """
    run_id:                 np.ndarray  # (N,)      int     Run number
    e_p:                    np.ndarray  # (N,)      float   Prompt energy                       (MeV)
    e_d:                    np.ndarray  # (N,)      float   Delayed energy                      (MeV)
    ts_p:                   np.ndarray  # (N, 2)    int     Prompt timestamp
    ts_d:                   np.ndarray  # (N, 2)    int     Delayed timestamp
    pos_p_mm:               np.ndarray  # (N, 3)    float   Prompt position                     (mm)
    pos_d_mm:               np.ndarray  # (N, 3)    float   Delayed position                    (mm)
    dt_p_d_ms:              np.ndarray  # (N,)      float   Prompt-delayed time coincidence     (ms)
    dt_last_mu_with_neu:    np.ndarray  # (N,)      float   Time since last muon with neutron   (s)
    dt_mu2p:                np.ndarray  # (N,)      float   Time muon-to-prompt                 (s)
    dlat_mu2p:              np.ndarray  # (N,)      float   Distance muon-to-prompt             (mm)


def load_accidental(filepath: str, dirpath: str) -> AccidentalData:
    """
    Load and prepare all arrays for the accidental analysis.
    """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/events"].arrays(_ACCIDENTAL_BRANCHES, library="np")

    pos_p_mm = _stack_positions(raw, "p")
    r_p = np.linalg.norm(pos_p_mm, axis=1)
    mask = (r_p <= 16500.0)
    raw = {k: v[mask] for k, v in raw.items()}

    ts_p = _build_timestamps(raw["sec_p"], raw["nsec_p"])
    ts_d = _build_timestamps(raw["sec_d"], raw["nsec_d"])

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} accidental events")

    return AccidentalData(
        run_id              = raw["run_id"],
        e_p                 = raw["e_p"],
        e_d                 = raw["e_d"],
        ts_p                = _stack_timestamps(raw, "p"),
        ts_d                = _stack_timestamps(raw, "d"),
        pos_p_mm            = _stack_positions(raw, "p"),
        pos_d_mm            = _stack_positions(raw, "d"),
        dt_p_d_ms           = _dt_ms(ts_p, ts_d),
        dt_last_mu_with_neu = raw["dt_last_mu_with_neu"],
        dt_mu2p             = raw["dt_mu2p"],
        dlat_mu2p           = raw["dlat_mu2p"],
    )