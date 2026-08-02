from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import uproot

from ._common import (
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
    """
    Arrays of the cosmogenic general analysis.
    """
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
    """
    Load the sample for the cosmogenic rate analysis.
    """

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
    """
    Arrays of the cosmogenic general analysis.
    """
    run_id:                 np.ndarray  # (N,)      int     Run number
    e_p:                    np.ndarray  # (N,)      float   Prompt energy                               (MeV)
    e_d:                    np.ndarray  # (N,)      float   Delayed energy                              (MeV)
    pos_p_mm:               np.ndarray  # (N, 3)    float   Prompt position                             (mm)
    pos_d_mm:               np.ndarray  # (N, 3)    float   Delayed position                            (mm)
    dt_p_d_ms:              np.ndarray  # (N,)      float   Prompt-delayed time coincidence             (ms)
    dt_mu2p:                np.ndarray  # (N,)      float   Time muon-to-prompt                         (s)
    dlat_mu2p:              np.ndarray  # (N,)      float   Distance muon-to-prompt                     (mm)
    dt_mu2d:                np.ndarray  # (N,)      float   Time muon-to-delayed                        (s)
    dlat_mu2d:              np.ndarray  # (N,)      float   Distance muon-to-delayed                    (mm)
    dt_last_mu_with_neu:    np.ndarray  # (N,)      float   Time to last muon with spallation neutron   (s)
    dt_last_mu:             np.ndarray  # (N,)      float   Time to last muon                           (s)


@dataclass
class Li9He8ShapeData:
    """
    Arrays of the cosmogenic shape analysis.
    """
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
        dt_p_d_ms           = _dt_ms(ts_p, ts_d),
        dt_mu2p             = raw["dt_mu2p"],
        dlat_mu2p           = raw["dlat_mu2p"],
        dt_mu2d             = raw["dt_mu2d"],
        dlat_mu2d           = raw["dlat_mu2d"],
        dt_last_mu_with_neu = raw["dt_last_mu_with_neu"],
        dt_last_mu          = raw["dt_last_mu"]
    )


def load_li9he8_shape(filepath: str, dirpath: str) -> Li9He8ShapeData:
    """
    Load signal and background samples for the cosmogenic shape analysis.
    """
    file = uproot.open(filepath)

    raw_sig = file[f"{dirpath}/signal_events"].arrays(_LI9HE8_SHAPE_BRANCHES,     library="np")
    raw_bkg = file[f"{dirpath}/background_events"].arrays(_LI9HE8_SHAPE_BRANCHES, library="np")

    print(f"Loading from {filepath}/{dirpath}")

    return Li9He8ShapeData(
        signal     = _load_li9he8_sample(raw_sig, "signal",     filepath),
        background = _load_li9he8_sample(raw_bkg, "background", filepath),
    )



# ---------------------------------------------------------------------------
# MC cosmogenics GroupC template loader
# ---------------------------------------------------------------------------


@dataclass
class MCGroupCTemplateHistogram:
    edges:  np.ndarray
    counts: np.ndarray
    errors: np.ndarray
    label:  str

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return self.edges[1:] - self.edges[:-1]


def load_mc_groupc_template(
    filepath: str,
    hist_path: str = "prefit/lihe",
    label: str = "Li9/He8 (MC)",
) -> MCGroupCTemplateHistogram:
    """Load a single TH1F prediction histogram directly (not a TTree branch)."""
    file = uproot.open(filepath)
    hist = file[hist_path]
    counts, edges = hist.to_numpy()
    errors = np.full_like(counts, 1.0e-5)
    # errors = np.sqrt(np.maximum(hist.variances(), 0.0))
    # if this is a P.D.F. the error bar are too big
    print(f"Loaded MC template '{hist_path}' from {filepath}: "
          f"{len(counts)} bins, [{edges[0]:.2f}, {edges[-1]:.2f}] MeV")
    return MCGroupCTemplateHistogram(
        edges=np.asarray(edges, dtype=float),
        counts=np.asarray(counts, dtype=float),
        errors=errors,
        label=label,
    )


# ---------------------------------------------------------------------------
# MC cosmogenics Chengzhuo template loader
# ---------------------------------------------------------------------------


@dataclass
class MCChengzhuoTemplateHistogramContribution:
    edges:  np.ndarray
    counts: np.ndarray
    errors: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return self.edges[1:] - self.edges[:-1]
    

@dataclass
class MCChengzhuoTemplateHistogram:
    branch0: MCChengzhuoTemplateHistogramContribution
    branch1: MCChengzhuoTemplateHistogramContribution
    branch2: MCChengzhuoTemplateHistogramContribution
    branch3: MCChengzhuoTemplateHistogramContribution
    branch4: MCChengzhuoTemplateHistogramContribution
    all:     MCChengzhuoTemplateHistogramContribution


def load_mc_chengzhuo_template(
    filepath: str, 
    hist_path: str = "SetB", 
    br0: float = 1.0, # 0.119, # 0.32, 
    br1: float = 1.0, # -0.041, # 0.12, 
    br2: float = 1.0, # 0.428, # 0.03, 
    br3: float = 1.0, # -0.145, # 0.01, 
    br4: float = 1.0, # 0.146, # 0.03, 
) -> MCChengzhuoTemplateHistogram:
    """Load a single TH1F prediction histogram directly (not a TTree branch)."""
    file = uproot.open(filepath)
    lhist: list[MCChengzhuoTemplateHistogramContribution] = []
    lbr  = np.array([br0, br1, br2, br3, br4], dtype=float)
    # tlbr = np.array([0.32, 0.12, 0.03, 0.01, 0.03], dtype=float)
    for k in range(5):
        hist = file[f"{hist_path}_branch{k}"]
        counts, edges = hist.to_numpy()
        errors = np.full_like(counts, 1.0e-5)
        lhist.append(MCChengzhuoTemplateHistogramContribution(
            edges=np.asarray(edges, dtype=float), 
            counts=np.asarray(counts, dtype=float), 
            errors=errors, 
        ))
    allhist = MCChengzhuoTemplateHistogramContribution(
        edges=lhist[0].edges, 
        counts=sum(hist.counts * br for hist, br in zip(lhist, lbr)),
        # counts=sum(hist.counts * br / tbr for hist, br, tbr in zip(lhist, lbr, tlbr)),
        errors=np.full_like(counts, 1.0e-5),
    )
    print(f"Loaded MC template '{hist_path}' from {filepath}")
    return MCChengzhuoTemplateHistogram(
        branch0=lhist[0], 
        branch1=lhist[1], 
        branch2=lhist[2], 
        branch3=lhist[3], 
        branch4=lhist[4], 
        all=allhist, 
    )


# ---------------------------------------------------------------------------
# MC cosmogenics 9Li/8He template loader
# ---------------------------------------------------------------------------

_MC_COSMO_BRANCHES = ["posx_p", "posy_p", "posz_p", "e_p", "element"]


@dataclass
class MC9Li8HeTemplateHistogram:
    """
    MC cosmogenic spectrum, filtered to a single isotope.

    The element branch is a string array (e.g. "Li9", "He8").
    Only the requested isotope's events are kept; the filter is applied
    at load time to avoid carrying the full (unfiltered) array around.
    """
    isotope:  str            # e.g. "Li9"
    e_p:      np.ndarray     # (N,) float  [MeV]
    pos_p_mm: np.ndarray     # (N, 3) float  [mm]


def load_mc_9li8he_cosmogenics(
    filepath: str,
    isotope: str = "Li9",
) -> MC9Li8HeTemplateHistogram:
    """
    Load MC cosmogenic events for a given isotope from a ROOT file.

    Parameters
    ----------
    filepath : str
        Path to the MC ROOT file (default: "mc/mc_cosmogenics.root").
    isotope : str, optional
        Element label to filter on (default: "Li9").

    Returns
    -------
    MC9Li8HeTemplateHistogram
    """
    file = uproot.open(filepath)
    raw  = file["cosmogenics"].arrays(_MC_COSMO_BRANCHES, library="np")

    mask = raw["element"] == isotope
    n_total    = len(raw["e_p"])
    n_filtered = int(np.sum(mask))
    print(f"Loaded {n_filtered}/{n_total} MC cosmogenic events ({isotope}) from {filepath}")

    return MC9Li8HeTemplateHistogram(
        isotope  = isotope,
        e_p      = raw["e_p"][mask],
        pos_p_mm = _stack_positions(raw, "p")[mask],
    )