from dataclasses import dataclass

import numpy as np
import uproot

from .._common import (
    _build_timestamps,
    _dt_ms,
    _stack_positions,
)

# -------------------------------------------------------------------------------------------------
# Li9He8 shape
# -------------------------------------------------------------------------------------------------

_LI9HE8_RATE_BRANCHES = [
    "run_id",
    "posx_p", "posy_p", "posz_p", "sec_p", "nsec_p", "e_p",
    "posx_d", "posy_d", "posz_d", "sec_d", "nsec_d", "e_d",
    "dlat_mu2p", "dt_mu2p",
    "dlat_mu2d", "dt_mu2d",
    "dt_last_mu_with_neu", "dt_last_mu",
    "dt_last_mu_with_neu_1m", "dt_last_mu_1m",
    "dt_last_mu_with_neu_2m", "dt_last_mu_2m",
    "dt_last_mu_with_neu_3m", "dt_last_mu_3m",
    "dt_last_mu_with_neu_4m", "dt_last_mu_4m",
    "dt_last_mu_with_neu_5m", "dt_last_mu_5m",
    "dt_last_mu_with_neu_6m", "dt_last_mu_6m",
    "dt_last_mu_with_neu_7m", "dt_last_mu_7m",
    "dt_last_mu_with_neu_8m", "dt_last_mu_8m",
    "dt_last_mu_with_neu_9m", "dt_last_mu_9m",
    "dt_last_mu_with_neu_10m", "dt_last_mu_10m",
]

@dataclass
class Li9He8RateData:
    """ Arrays of the cosmogenic rate analysis. """
    run_id:                  np.ndarray  # (N,)      int     Run number
    e_p:                     np.ndarray  # (N,)      float   Prompt energy                                                  (MeV)
    e_d:                     np.ndarray  # (N,)      float   Delayed energy                                                 (MeV)
    pos_p_mm:                np.ndarray  # (N, 3)    float   Prompt position                                                (mm)
    pos_d_mm:                np.ndarray  # (N, 3)    float   Delayed position                                               (mm)
    dt_p_d_ms:               np.ndarray  # (N,)      float   Prompt-delayed time coincidence                                (ms)
    dt_mu2p:                 np.ndarray  # (N,)      float   Time muon-to-prompt                                            (s)
    dlat_mu2p:               np.ndarray  # (N,)      float   Distance muon-to-prompt                                        (mm)
    dt_mu2d:                 np.ndarray  # (N,)      float   Time muon-to-delayed                                           (s)
    dlat_mu2d:               np.ndarray  # (N,)      float   Distance muon-to-delayed                                       (mm)
    dt_last_mu_with_neu:     np.ndarray  # (N,)      float   Time to last muon with spallation neutron                      (s)
    dt_last_mu:              np.ndarray  # (N,)      float   Time to last muon                                              (s)
    dt_last_mu_with_neu_1m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 1m  (s)
    dt_last_mu_1m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 1m                          (s)
    dt_last_mu_with_neu_2m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 2m  (s)
    dt_last_mu_2m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 2m                          (s)
    dt_last_mu_with_neu_3m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 3m  (s)
    dt_last_mu_3m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 3m                          (s)
    dt_last_mu_with_neu_4m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 4m  (s)
    dt_last_mu_4m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 4m                          (s)
    dt_last_mu_with_neu_5m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 5m  (s)
    dt_last_mu_5m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 5m                          (s)
    dt_last_mu_with_neu_6m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 6m  (s)
    dt_last_mu_6m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 6m                          (s)
    dt_last_mu_with_neu_7m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 7m  (s)
    dt_last_mu_7m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 7m                          (s)
    dt_last_mu_with_neu_8m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 8m  (s)
    dt_last_mu_8m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 8m                          (s)
    dt_last_mu_with_neu_9m:  np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 9m  (s)
    dt_last_mu_9m:           np.ndarray  # (N,)      float   Time to last muon with a radius of 9m                          (s)
    dt_last_mu_with_neu_10m: np.ndarray  # (N,)      float   Time to last muon with spallation neutron with a radius of 10m (s)
    dt_last_mu_10m:          np.ndarray  # (N,)      float   Time to last muon with a radius of 10m                         (s)

def load_li9he8_rate(filepath: str, dirpath: str) -> Li9He8RateData:
    """ Load the sample for the cosmogenic rate analysis. """

    file = uproot.open(filepath)
    raw = file[f"{dirpath}/events"].arrays(_LI9HE8_RATE_BRANCHES, library="np")

    # pos_p_mm = _stack_positions(raw, "p")
    # r_p = np.linalg.norm(pos_p_mm, axis=1)
    # mask = (r_p <= 16500.0)
    # raw = {k: v[mask] for k, v in raw.items()}

    ts_p = _build_timestamps(raw["sec_p"], raw["nsec_p"])
    ts_d = _build_timestamps(raw["sec_d"], raw["nsec_d"])

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} events")

    return Li9He8RateData(
        run_id                  = raw["run_id"],
        e_p                     = raw["e_p"],
        e_d                     = raw["e_d"],
        pos_p_mm                = _stack_positions(raw, "p"),
        pos_d_mm                = _stack_positions(raw, "d"),
        dt_p_d_ms               = _dt_ms(ts_p, ts_d),
        dt_mu2p                 = raw["dt_mu2p"],
        dlat_mu2p               = raw["dlat_mu2p"],
        dt_mu2d                 = raw["dt_mu2d"],
        dlat_mu2d               = raw["dlat_mu2d"],
        dt_last_mu_with_neu     = raw["dt_last_mu_with_neu"],
        dt_last_mu              = raw["dt_last_mu"],
        dt_last_mu_with_neu_1m  = raw["dt_last_mu_with_neu_1m"],
        dt_last_mu_1m           = raw["dt_last_mu_1m"],
        dt_last_mu_with_neu_2m  = raw["dt_last_mu_with_neu_2m"],
        dt_last_mu_2m           = raw["dt_last_mu_2m"],
        dt_last_mu_with_neu_3m  = raw["dt_last_mu_with_neu_3m"],
        dt_last_mu_3m           = raw["dt_last_mu_3m"],
        dt_last_mu_with_neu_4m  = raw["dt_last_mu_with_neu_4m"],
        dt_last_mu_4m           = raw["dt_last_mu_4m"],
        dt_last_mu_with_neu_5m  = raw["dt_last_mu_with_neu_5m"],
        dt_last_mu_5m           = raw["dt_last_mu_5m"],
        dt_last_mu_with_neu_6m  = raw["dt_last_mu_with_neu_6m"],
        dt_last_mu_6m           = raw["dt_last_mu_6m"],
        dt_last_mu_with_neu_7m  = raw["dt_last_mu_with_neu_7m"],
        dt_last_mu_7m           = raw["dt_last_mu_7m"],
        dt_last_mu_with_neu_8m  = raw["dt_last_mu_with_neu_8m"],
        dt_last_mu_8m           = raw["dt_last_mu_8m"],
        dt_last_mu_with_neu_9m  = raw["dt_last_mu_with_neu_9m"],
        dt_last_mu_9m           = raw["dt_last_mu_9m"],
        dt_last_mu_with_neu_10m = raw["dt_last_mu_with_neu_10m"],
        dt_last_mu_10m          = raw["dt_last_mu_10m"],
    )