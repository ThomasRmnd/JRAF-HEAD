from __future__ import annotations

import math


class Timestamp:
    """
    Nanosecond-precision timestamp with arithmetic support.

    Parameters
    ----------
    sec : int
        Whole seconds component.
    nsec : int
        Nanosecond remainder (0 ≤ nsec < 1_000_000_000).
        Values outside this range are normalised automatically.
    """

    _NS_PER_S: int = 1_000_000_000

    def __init__(self, sec: int, nsec: int) -> None:
        self._sec = sec
        self._nsec = nsec
        self._normalize()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sec(self) -> int:
        """Whole-seconds component."""
        return self._sec

    @property
    def nsec(self) -> int:
        """Nanosecond remainder (0 ≤ nsec < 1_000_000_000)."""
        return self._nsec
    
    def _normalize(self) -> None:
        """Normalise so that 0 ≤ nsec < 1e9 (handles overflow from arithmetic)."""
        extra_sec, self._nsec = divmod(int(self.nsec), self._NS_PER_S)
        self._sec  = int(self._sec) + extra_sec

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Timestamp) -> Timestamp:
        """Sum of two timestamps, used to accumulate durations per run."""
        return Timestamp(self._sec + other._sec, self._nsec + other._nsec)

    def __sub__(self, other: Timestamp) -> Timestamp:
        """Difference of two timestamps."""
        return Timestamp(self._sec - other._sec, self._nsec - other._nsec)

    # ------------------------------------------------------------------
    # Comparison  (needed for min/max in analyze_run_info)
    # ------------------------------------------------------------------

    def _total_ns(self) -> int:
        return self._sec * self._NS_PER_S + self._nsec

    def __lt__(self, other: Timestamp) -> bool:
        return self._total_ns() < other._total_ns()

    def __le__(self, other: Timestamp) -> bool:
        return self._total_ns() <= other._total_ns()

    def __gt__(self, other: Timestamp) -> bool:
        return self._total_ns() > other._total_ns()

    def __ge__(self, other: Timestamp) -> bool:
        return self._total_ns() >= other._total_ns()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        return self._total_ns() == other._total_ns()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_sec(self) -> float:
        """
        Convert to a floating-point number of seconds.

        Used throughout the analysis for livetime and Δt calculations:
            dt_ms = (ts_d - ts_p).to_sec() * 1e3
        """
        return self._sec + self._nsec / self._NS_PER_S

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Timestamp(sec={self._sec}, nsec={self._nsec})"


# ---------------------------------------------------------------------------
# Legacy alias — keeps old "from timestamp import timestamp" imports working
# ---------------------------------------------------------------------------
timestamp = Timestamp