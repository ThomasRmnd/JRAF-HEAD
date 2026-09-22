from dataclasses import dataclass

import numpy as np
import uproot

from jrafhead.utils import Histogram

from .._common import (
    _stack_positions,
)

# ---------------------------------------------------------------------------
# MC cosmogenics GroupC template loader
# ---------------------------------------------------------------------------

def load_mc_groupc_template(
    filepath: str,
    hist_path: str = "prefit/lihe",
) -> Histogram:
    """ Load a single TH1F prediction histogram directly (not a TTree branch). """
    file = uproot.open(filepath)
    hist = file[hist_path]
    counts, edges = hist.to_numpy()
    errors = np.full_like(counts, 1.0e-5)
    # errors = np.sqrt(np.maximum(hist.variances(), 0.0))
    # if this is a P.D.F. the error bar are too big
    print(f"Loaded MC template '{hist_path}' from {filepath}: "
          f"{len(counts)} bins, [{edges[0]:.2f}, {edges[-1]:.2f}] MeV")
    return Histogram(
        edges=np.asarray(edges, dtype=float),
        counts=np.asarray(counts, dtype=float),
        errors=errors,
        underflow=0, overflow=0,
    )

# ---------------------------------------------------------------------------
# MC cosmogenics Chengzhuo template loader
# ---------------------------------------------------------------------------

@dataclass
class MCChengzhuoTemplateHistogram:
    branch0: Histogram
    branch1: Histogram
    branch2: Histogram
    branch3: Histogram
    branch4: Histogram
    all:     Histogram

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
    lhist: list[Histogram] = []
    lbr  = np.array([br0, br1, br2, br3, br4], dtype=float)
    # tlbr = np.array([0.32, 0.12, 0.03, 0.01, 0.03], dtype=float)
    for k in range(5):
        hist = file[f"{hist_path}_branch{k}"]
        counts, edges = hist.to_numpy()
        # print(f"Loaded MC template '{hist_path}_branch{k}' from {filepath}. Integral = {np.sum(counts * np.diff(edges))}")
        errors = np.full_like(counts, 1.0e-5)
        lhist.append(Histogram(
            edges=np.asarray(edges, dtype=float), 
            counts=np.asarray(counts, dtype=float), 
            errors=errors, 
            underflow=0, overflow=0,
        ))
    allhist = Histogram(
        edges=lhist[0].edges, 
        counts=sum(hist.counts * br for hist, br in zip(lhist, lbr)),
        # counts=sum(hist.counts * br / tbr for hist, br, tbr in zip(lhist, lbr, tlbr)),
        errors=np.full_like(counts, 1.0e-5),
        underflow=0, overflow=0,
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
    e_p:      np.ndarray     # (N,)     float   [MeV]
    pos_p_mm: np.ndarray     # (N, 3)   float   [mm]


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