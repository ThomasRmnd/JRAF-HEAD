from __future__ import annotations

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
    """
    Arrays of the muon performance analysis.
    """
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
    """
    Load and prepare all arrays for the muon performance analysis.
    """
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
class MuonRateHistogramData:
    """
    Arrays of the muon rate histogram analysis.
    """
    edges:              np.ndarray  # (N, nbin + 1) float       Bin edges for CD only muons
    counts:             np.ndarray  # (N, nbin)     float       Bin counts for CD only muons
    errors:             np.ndarray  # (N, nbin)     float       Bin uncertainties for CD only muons
    underflow:          float       # (N,)          float       Underflow bin
    overflow:           float       # (N,)          float       Overflow bin


@dataclass
class MuonRateData:
    """
    Arrays of the muon rate analysis.
    """
    run_id:             np.ndarray                  # (N,)  int         Run number
    hist_cd_only:       list[MuonRateHistogramData] # (N,)              Bin edges for CD only muons
    hist_wp_only:       list[MuonRateHistogramData] # (N,)              Bin counts for CD only muons
    hist_cd_wp:         list[MuonRateHistogramData] # (N,)              Bin uncertainties for CD only muons


def _load_histograms(raw: np.ndarray, pattern: str) -> list[MuonRateHistogramData]:
    """
    Load and prepare all histograms for the muon rate analysis.
    """
    edges = raw[f"{pattern}_edges"]
    counts = raw[f"{pattern}_counts"]
    errors = raw[f"{pattern}_errors"]
    underflow = raw[f"{pattern}_underflow"]
    overflow = raw[f"{pattern}_overflow"]

    return [
        MuonRateHistogramData(
            edges       = e,
            counts      = c,
            errors      = err,
            underflow   = u,
            overflow    = o,
        )
        for e, c, err, u, o in zip(
            edges,
            counts,
            errors,
            underflow,
            overflow,
        )
    ]


def load_muon_rate(filepath: str, dirpath: str) -> MuonRateData:
    """
    Load and prepare all arrays for the muon rate analysis.
    """
    file = uproot.open(filepath)
    raw = file[f"{dirpath}/rate"].arrays(_MUON_RATE_BRANCHES, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon rates")

    return MuonRateData(
        run_id          = raw["run_id"],
        hist_cd_only    = _load_histograms(raw, "hist_cd_only"),
        hist_wp_only    = _load_histograms(raw, "hist_wp_only"),
        hist_cd_wp      = _load_histograms(raw, "hist_cd_wp"),
    )

# -------------------------------------------------------------------------------------------------
# Muon WP tagging efficiency
# -------------------------------------------------------------------------------------------------

_MUON_EFFICIENCY_WP = [
    "run_id",
    "nb_cd_wp_30k",
    "nb_cd_only_10ms",
]


@dataclass
class MuonEfficiencyWP:
    """
    Arrays of the muon effiency WP analysis.
    """
    run_id:                 np.ndarray  # (N,)  int     Run number
    nb_cd_wp_30k:           np.ndarray  # (N,)  int     Number of CD-WP muon over 30k PEs threshold
    nb_cd_only_10ms:        np.ndarray  # (N,)  int     Number of CD only muon over 10 ms threshold from previous muon


def load_muon_efficiency_wp(filepath: str, dirpath: str) -> MuonPerformanceData:
    """
    Load and prepare all arrays for the muon efficiency WP analysis.
    """
    file = uproot.open(filepath)
    raw  = file[f"{dirpath}/efficiency"].arrays(_MUON_EFFICIENCY_WP, library="np")

    n = len(raw["run_id"])
    print(f"Loading from {filepath}/{dirpath}")
    print(f"  Loaded {n} muon comparisons")

    return MuonPerformanceData(
        run_id                  = raw["run_id"],
        nb_cd_wp_30k            = raw["nb_cd_wp_30k"],
        nb_cd_only_10ms         = raw["nb_cd_only_10ms"],
    )