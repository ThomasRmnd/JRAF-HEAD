from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

_MM_TO_M = 1e-3  # reconstruction positions are in mm, plots are in m


# ---------------------------------------------------------------------------
# Individual calculators
# ---------------------------------------------------------------------------

def compute_rho2(pos_mm: np.ndarray) -> np.ndarray:
    """
    Transverse radius squared ρ² = x² + y² in metres².

    Parameters
    ----------
    pos_mm : np.ndarray, shape (N, 3)
        Positions [x, y, z] in millimetres.

    Returns
    -------
    np.ndarray, shape (N,)
        ρ² in metres².
    """
    xy_m = pos_mm[:, :2] * _MM_TO_M
    return np.sum(xy_m ** 2, axis=1)


def compute_z(pos_mm: np.ndarray) -> np.ndarray:
    """
    Axial coordinate z in metres.

    Parameters
    ----------
    pos_mm : np.ndarray, shape (N, 3)
        Positions [x, y, z] in millimetres.

    Returns
    -------
    np.ndarray, shape (N,)
        z in metres.
    """
    return pos_mm[:, 2] * _MM_TO_M


def compute_3d_distance(pos_a_mm: np.ndarray, pos_b_mm: np.ndarray) -> np.ndarray:
    """
    Euclidean distance between two sets of 3D positions, in metres.

    Typical use: prompt–delayed vertex separation |r_p - r_d|.

    Parameters
    ----------
    pos_a_mm : np.ndarray, shape (N, 3)
        First set of positions [x, y, z] in millimetres.
    pos_b_mm : np.ndarray, shape (N, 3)
        Second set of positions [x, y, z] in millimetres.

    Returns
    -------
    np.ndarray, shape (N,)
        |pos_a - pos_b| in metres.
    """
    delta_m = (pos_a_mm - pos_b_mm) * _MM_TO_M
    return np.linalg.norm(delta_m, axis=1)


# ---------------------------------------------------------------------------
# Composite result object
# ---------------------------------------------------------------------------

@dataclass
class PositionGeometry:
    """
    Pre-computed geometric quantities for one set of reconstructed vertices.

    Attributes
    ----------
    rho2_m2 : np.ndarray, shape (N,)
        Transverse radius squared ρ² [m²].
    z_m : np.ndarray, shape (N,)
        Axial coordinate z [m].
    """
    rho2_m2: np.ndarray
    z_m: np.ndarray


def compute_geometry(pos_mm: np.ndarray) -> PositionGeometry:
    """
    Compute all per-vertex geometric quantities from a position array.

    This is the main entry point for the spatial analysis. Call it once
    per vertex type (prompt, delayed) and unpack the result:

        geom_p = compute_geometry(pos_p_mm)
        geom_d = compute_geometry(pos_d_mm)
        distance = compute_3d_distance(pos_p_mm, pos_d_mm)

    Parameters
    ----------
    pos_mm : np.ndarray, shape (N, 3)
        Reconstructed vertex positions [x, y, z] in millimetres.

    Returns
    -------
    PositionGeometry
        Dataclass holding rho2_m2 and z_m arrays.
    """
    return PositionGeometry(
        rho2_m2=compute_rho2(pos_mm),
        z_m=compute_z(pos_mm),
    )