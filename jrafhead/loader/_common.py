from __future__ import annotations

import numpy as np

from utils import Timestamp


def _build_timestamps(sec: np.ndarray, nsec: np.ndarray) -> np.ndarray:
    """
    Convert parallel sec/nsec arrays into an array of Timestamp objects.
    """
    return np.array([Timestamp(s, ns) for s, ns in zip(sec, nsec)])


def _dt_ms(ts_p: np.ndarray, ts_d: np.ndarray) -> np.ndarray:
    """
    Prompt–delayed time difference (in ms).
    """
    return np.array([(td - tp).to_sec() * 1e3 for tp, td in zip(ts_p, ts_d)])


def _stack_timestamps(data: dict[str, np.ndarray], prefix: str | None = None) -> np.ndarray:
    """
    Stack sec/nsec branch arrays into a single (N, 2) timestamp matrix.
    """
    if prefix is not None:
        return np.column_stack((
            data[f"sec_{prefix}"],
            data[f"nsec_{prefix}"],
        ))
    else:
        return np.column_stack((
            data["sec"], 
            data["nsec"], 
        ))


def _stack_positions(data: dict[str, np.ndarray], prefix: str | None = None) -> np.ndarray:
    """
    Stack x/y/z branch arrays into a single (N, 3) position matrix (in mm).
    """
    if prefix is not None:
        return np.column_stack((
            data[f"posx_{prefix}"],
            data[f"posy_{prefix}"],
            data[f"posz_{prefix}"],
        ))
    else:
        return np.column_stack((
            data["posx"], 
            data["posy"], 
            data["posz"], 
        ))