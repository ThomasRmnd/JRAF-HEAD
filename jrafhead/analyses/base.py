from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAnalysis(ABC):
    """
    Abstract base for an analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT input file.
    dirpath : str or Path
        Path to the directory inside the ROOT file.
    output_dir : str or Path
        Root directory for saved figures.
    """

    def __init__(
        self,
        filepath:   str | Path,
        dirpath:    str | Path,
        output_dir: str | Path = ".",
    ) -> None:
        self.filepath   = Path(filepath)
        self.dirpath    = Path(dirpath)
        self.output_dir = Path(output_dir)
        self.stem       = self.dirpath

    @abstractmethod
    def _load(self) -> None:
        """Load all data from "self.filepath" into instance attributes."""
        ...

    @abstractmethod
    def _plot(self) -> None:
        """Create all figures, saving them via "io.saving.save_figure"."""
        ...

    def run(self) -> None:
        """Execute the full analysis (load ==> plot)."""
        self._load()
        self._plot()