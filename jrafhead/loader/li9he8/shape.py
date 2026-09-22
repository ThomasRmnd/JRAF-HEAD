from dataclasses import dataclass

import numpy as np
import uproot

from .._common import (
    _build_timestamps,
    _dt_ms,
    _stack_positions,
    _stack_timestamps,
)

# -------------------------------------------------------------------------------------------------
# Li9He8 shape
# -------------------------------------------------------------------------------------------------

_LI9HE8_SHAPE_BRANCHES = [
    "run_id",
    "posx_p", "posy_p", "posz_p", "sec_p", "nsec_p", "e_p", "dlat_mu2p", "dt_mu2p",
    "posx_d", "posy_d", "posz_d", "sec_d", "nsec_d", "e_d", "dlat_mu2d", "dt_mu2d",
    "dt_last_mu_with_neu", "dt_last_mu",
]

@dataclass
class Li9He8Data:
    """ Arrays of the cosmogenic shape analysis. """
    run_id:                 np.ndarray  # (N,)      int     Run number
    e_p:                    np.ndarray  # (N,)      float   Prompt energy                               (MeV)
    e_d:                    np.ndarray  # (N,)      float   Delayed energy                              (MeV)
    pos_p_mm:               np.ndarray  # (N, 3)    float   Prompt position                             (mm)
    pos_d_mm:               np.ndarray  # (N, 3)    float   Delayed position                            (mm)
    ts_p:                   np.ndarray  # (N, 2)    int     Prompt timestamp
    ts_d:                   np.ndarray  # (N, 2)    int     Delayed timestamp
    dt_p_d_ms:              np.ndarray  # (N,)      float   Prompt-delayed time coincidence             (ms)
    dt_mu2p:                np.ndarray  # (N,)      float   Time muon-to-prompt                         (s)
    dlat_mu2p:              np.ndarray  # (N,)      float   Distance muon-to-prompt                     (mm)
    dt_mu2d:                np.ndarray  # (N,)      float   Time muon-to-delayed                        (s)
    dlat_mu2d:              np.ndarray  # (N,)      float   Distance muon-to-delayed                    (mm)
    dt_last_mu_with_neu:    np.ndarray  # (N,)      float   Time to last muon with spallation neutron   (s)
    dt_last_mu:             np.ndarray  # (N,)      float   Time to last muon                           (s)

@dataclass
class Li9He8ShapeData:
    """ Arrays of the cosmogenic shape analysis. """
    signal:     Li9He8Data
    background: Li9He8Data

def _load_li9he8_sample(raw: dict[str, np.ndarray], label: str, filepath: str) -> Li9He8Data:
    """
    Load and prepare one cosmo sample (signal or background).
    """

    # pos_p_mm = _stack_positions(raw, "p")
    # r_p = np.linalg.norm(pos_p_mm, axis=1)
    # mask = (r_p <= 16500.0)
    # raw = {k: v[mask] for k, v in raw.items()}

    ts_p = _build_timestamps(raw["sec_p"], raw["nsec_p"])
    ts_d = _build_timestamps(raw["sec_d"], raw["nsec_d"])

    n = len(raw["run_id"])
    print(f"  Loaded {n} {label} events")

    return Li9He8Data(
        run_id              = raw["run_id"],
        e_p                 = raw["e_p"],
        e_d                 = raw["e_d"],
        pos_p_mm            = _stack_positions(raw, "p"),
        pos_d_mm            = _stack_positions(raw, "d"),
        ts_p                = _stack_timestamps(raw, "p"),
        ts_d                = _stack_timestamps(raw, "d"),
        dt_p_d_ms           = _dt_ms(ts_p, ts_d),
        dt_mu2p             = raw["dt_mu2p"],
        dlat_mu2p           = raw["dlat_mu2p"],
        dt_mu2d             = raw["dt_mu2d"],
        dlat_mu2d           = raw["dlat_mu2d"],
        dt_last_mu_with_neu = raw["dt_last_mu_with_neu"],
        dt_last_mu          = raw["dt_last_mu"]
    )

def load_li9he8_shape(filepath: str, dirpath: str) -> Li9He8ShapeData:
    """ Load signal and background samples for the cosmogenic shape analysis. """
    file = uproot.open(filepath)

    raw_sig = file[f"{dirpath}/signal_events"].arrays(_LI9HE8_SHAPE_BRANCHES,     library="np")
    raw_bkg = file[f"{dirpath}/background_events"].arrays(_LI9HE8_SHAPE_BRANCHES, library="np")

    print(f"Loading from {filepath}/{dirpath}")

    return Li9He8ShapeData(
        signal     = _load_li9he8_sample(raw_sig, "signal",     filepath),
        background = _load_li9he8_sample(raw_bkg, "background", filepath),
    )