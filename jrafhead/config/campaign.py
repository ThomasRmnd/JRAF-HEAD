from dataclasses import dataclass

from .color import (
    GOOGLE_BLUE,
    GOOGLE_GREEN,
    GOOGLE_RED,
    GOOGLE_YELLOW,
)

# -------------------------------------------------------------------------------------------------
# Reprocessing campaigns
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    """
    Metadata for a single phase.
    """
    name:       str
    run_min:    int
    run_max:    int
    date_min:   str
    date_max:   str
    color:      str


@dataclass(frozen=True)
class Campaign:
    """
    All known reprocessing phases in the campaign, ordered chronologically by run ID.
    """
    phases: list[Phase]

# -------------------------------------------------------------------------------------------------
# Constant campaigns
# -------------------------------------------------------------------------------------------------

ReProd26B = Campaign(phases=[
    Phase("Phase 1",  9789, 11039, "2025-08-30", "2025-11-02", GOOGLE_BLUE),
    Phase("Phase 2", 11049, 12135, "2025-11-03", "2025-12-14", GOOGLE_RED),
    Phase("Phase 3", 12422, 13463, "2025-12-17", "2026-02-03", GOOGLE_YELLOW),
    Phase("Phase 4", 13479, 15612, "2026-02-03", "2026-05-17", GOOGLE_GREEN)
])