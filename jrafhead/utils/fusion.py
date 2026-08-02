from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Literal

import numpy as np


FusionMode = Literal["mean", "sum"]

def fuze_by_day(
    timestamps: np.ndarray, 
    values:     np.ndarray, 
    errors:     np.ndarray,
    mode:       FusionMode = "mean"
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """
    Combine per-run measurements into one weighted point per calendar day.

    Parameters
    ----------
    timestamps : np.ndarray, shape (N,)
        Unix timestamps for each measurement (one per run).
    values : np.ndarray, shape (N,)
        Measured quantity for each run.
    errors : np.ndarray, shape (N,)
        Uncertainty on each measurement.
    mode : {"mean", "sum"}
        Fusion strategy. See module docstring.

    Returns
    -------
    dates : list of datetime
        One "datetime" per day (midnight UTC), sorted chronologically.
    fused_values : np.ndarray, shape (n_days,)
        Weighted mean value for each day.
    fused_errors : np.ndarray, shape (n_days,)
        Uncertainty on the weighted mean for each day.

    Examples
    --------
    >>> import numpy as np
    >>> ts = np.array([0.0, 3600.0, 86400.0])   # day 0, day 0, day 1
    >>> y  = np.array([1.0, 3.0,    5.0])
    >>> ye = np.array([1.0, 1.0,    1.0])
    >>> dates, yf, ef = fuze_by_day(ts, y, ye)
    >>> len(dates)   # two days
    2
    """
    if mode not in ("mean", "sum"):
        raise ValueError(
            f"Unknown fusion mode {mode!r}. "
            "Expected one of: 'mean', 'sum'."
        )

    # Group indices by calendar date (UTC)
    day_groups: dict[date, list[int]] = defaultdict(list)
    for i, ts in enumerate(timestamps):
        day = datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
        day_groups[day].append(i)

    # Fuse each day
    sorted_days                     = sorted(day_groups.keys())
    fused_dates:  list[datetime]    = []
    fused_values: list[float]       = []
    fused_errors: list[float]       = []

    for day in sorted_days:
        idx    = day_groups[day]
        y_day  = np.asarray(values)[idx]
        ye_day = np.asarray(errors)[idx]

        # x position: midnight UTC of that day
        fused_dates.append(
            datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        )

        if len(idx) == 1:
            fused_values.append(float(y_day[0]))
            fused_errors.append(float(ye_day[0]))
            continue

        if mode == "sum":
            fused_values.append(float(y_day.sum()))
            fused_errors.append(float(np.sqrt(np.sum(ye_day ** 2))))

        elif mode == "mean":
            if np.all(ye_day > 0):
                weights      = 1.0 / ye_day ** 2
                total_weight = weights.sum()
                fused_values.append(float(np.dot(weights, y_day) / total_weight))
                fused_errors.append(float(1.0 / np.sqrt(total_weight)))
            else:
                fused_values.append(float(y_day.mean()))
                fused_errors.append(float(y_day.std() / np.sqrt(len(y_day))))
        
        else:
            raise ValueError(
                f"Unknown fusion mode {mode!r}. "
                "Expected one of: 'mean', 'sum'."
            )

    return fused_dates, np.array(fused_values), np.array(fused_errors)

def sum_by_day(timestamps: np.ndarray, *arrays: np.ndarray):
    groups = defaultdict(list)

    for i, ts in enumerate(timestamps):
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        groups[day].append(i)

    dates = []
    summed = [[] for _ in arrays]

    for day in sorted(groups):
        idx = groups[day]

        dates.append(
            datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        )

        for out, array in zip(summed, arrays):
            out.append(array[idx].sum())

    return (
        dates,
        *(np.asarray(x) for x in summed),
    )