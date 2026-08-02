from __future__ import annotations

import numpy as np

def rebin_histogram(
    src_edges: np.ndarray,
    src_counts: np.ndarray,
    dst_edges: np.ndarray,
) -> np.ndarray:
    """
    Rebin a count histogram onto new edges, assuming uniform density within
    each source bin and integrating that density over each destination bin's
    overlap. Destination bins outside the source range receive zero.
    """
    src_edges  = np.asarray(src_edges,  dtype=float)
    src_counts = np.asarray(src_counts, dtype=float)
    dst_edges  = np.asarray(dst_edges,  dtype=float)
    src_widths = np.diff(src_edges)
    dst_counts = np.zeros(len(dst_edges) - 1)

    for i in range(len(dst_edges) - 1):
        d_lo, d_hi = dst_edges[i], dst_edges[i + 1]
        total = 0.0
        for j in range(len(src_edges) - 1):
            s_lo, s_hi = src_edges[j], src_edges[j + 1]
            overlap = min(d_hi, s_hi) - max(d_lo, s_lo)
            if overlap > 0:
                total += src_counts[j] * (overlap / src_widths[j])
        dst_counts[i] = total
    return dst_counts