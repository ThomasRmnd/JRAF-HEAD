from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import matplotlib.figure

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SaveFormat(str, Enum):
    """Supported output formats. Values are the file extensions."""
    PDF = "pdf"
    PNG = "png"


_DEFAULT_FORMATS: list[SaveFormat] = [SaveFormat.PDF, SaveFormat.PNG]


def save_figure(
    fig: matplotlib.figure.Figure,
    stem: str,
    suffix: str = "",
    formats: list[str | SaveFormat] | None = None,
    output_dir: str | Path = ".",
    dpi: int | None = None,
) -> list[Path]:
    """
    Save a matplotlib figure to one or more formats under "output_dir".

    For each requested format a sub-directory named after the format is created
    inside "output_dir" (e.g. "output_dir/pdf/", "output_dir/png/").
    This mirrors the original directory layout ("pdf/", "png/" at the
    script root) while making it configurable per analysis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    stem : str
        Base name for the file, typically derived from the ROOT filename:
        "os.path.basename(filepath).replace('.root', '')".
    suffix : str, optional
        Plot-type suffix appended to "stem", e.g. "_e_p_nmo".
        Defaults to "" (no suffix).
    formats : list of str or SaveFormat, optional
        Output formats to write. Defaults to ["pdf", "png"].
    output_dir : str or Path, optional
        Root output directory. Sub-directories for each format are created
        inside it. Defaults to the current working directory.
    dpi : int, optional
        Resolution override. If None, uses the value from rcParams
        (set to 300 by set_latex_style()).

    Returns
    -------
    list of Path
        The paths of the files that were written.

    Examples
    --------
    Save the standard pdf + png pair from a filepath:

        stem = Path(filepath).stem          # e.g. "cosmo_shape_3m_1_2s"
        save_figure(fig, stem, "_e_p_nmo")  # → pdf/cosmo_shape_3m_1_2s_e_p_nmo.pdf
                                            # → png/cosmo_shape_3m_1_2s_e_p_nmo.png

    Save only PDF to a specific directory:

        save_figure(fig, stem, "_e_d", formats=["pdf"], output_dir="/results")
    """
    if formats is None:
        formats = _DEFAULT_FORMATS

    formats = [SaveFormat(f) for f in formats]
    output_dir = Path(output_dir)
    filename = f"{stem}{suffix}"
    written: list[Path] = []

    for fmt in formats:
        fmt_dir = output_dir / fmt.value
        fmt_dir.mkdir(parents=True, exist_ok=True)

        path = fmt_dir / f"{filename}.{fmt.value}"
        kwargs: dict = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi

        fig.savefig(path, **kwargs)
        written.append(path)

    return written


def stem_from_filepath(filepath: str | Path) -> str:
    """
    Derive the file stem from a ROOT filepath, stripping the ".root" extension.

    This replaces the repeated pattern:
        "os.path.basename(filepath).replace('.root', '')"

    Parameters
    ----------
    filepath : str or Path
        Path to the ROOT file, e.g. "/data/cosmo_shape_3m_1_2s.root".

    Returns
    -------
    str
        Bare filename without directory or extension, e.g. "cosmo_shape_3m_1_2s".
    """
    return Path(filepath).stem

def save_json(
    data: dict,
    key: str,
    stem: str,
    suffix: str = "",
    output_dir: str | Path = ".",
) -> Path:
    """
    Save a dictionary under a json file in "output_dir".

    Parameters
    ----------
    data : dict
        The dictionary to save.
    key : str
        The key of the dictionary entry.
    stem : str
        Base name for the file, typically derived from the ROOT filename:
        "os.path.basename(filepath).replace('.root', '')".
    suffix : str, optional
        Plot-type suffix appended to "stem", e.g. "_e_p_nmo".
        Defaults to "" (no suffix).
    output_dir : str or Path, optional
        Root output directory. Sub-directories for each format are created
        inside it. Defaults to the current working directory.

    Returns
    -------
    Path
        The paths of the file that was written.
    """
    json_dir = Path(output_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}{suffix}.json"

    json_file = json_dir / filename
    if json_file.exists():
        with open(json_file, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {}
            
    json_data[key] = data
            
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4, sort_keys=True)