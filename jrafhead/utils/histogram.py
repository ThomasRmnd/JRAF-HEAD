from dataclasses import dataclass

import numpy as np


@dataclass
class Histogram:
    """
    Arrays defining an histogram.
    """
    edges:              np.ndarray  # (N + 1,)  Bin edges for CD only muons
    counts:             np.ndarray  # (N,)      Bin counts for CD only muons
    errors:             np.ndarray  # (N,)      Bin uncertainties for CD only muons
    underflow:          float       # float     Underflow bin
    overflow:           float       # float     Overflow bin

def load_histogram_from_tree(raw: np.ndarray, pattern: str) -> list[Histogram]:
    """
    Load and prepare all histograms in a given tree data.
    """
    edges = raw[f"{pattern}_edges"]
    counts = raw[f"{pattern}_counts"]
    errors = raw[f"{pattern}_errors"]
    underflow = raw[f"{pattern}_underflow"]
    overflow = raw[f"{pattern}_overflow"]

    return [
        Histogram(
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