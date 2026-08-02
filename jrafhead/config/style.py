import matplotlib as mpl

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_style(no_latex: bool = False) -> None:
    """
    Apply the publication matplotlib style.

    Parameters
    ----------
    no_latex : bool
        If True, use the mathtext fallback (no LaTeX required).
        If False (default), use the full LaTeX style.
    """
    if no_latex:
        set_mathtext_style()
    else:
        set_latex_style()


def set_latex_style() -> None:
    """
    Apply the LaTeX style to all matplotlib figures.

    Requires a working LaTeX installation with the Computer Modern font family.
    If LaTeX is not available, fall back to set_mathtext_style() instead.
    """
    mpl.rcParams.update({
        # --- Font & rendering ---
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
        "mathtext.fontset": "cm",

        # --- Font sizes ---
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,

        # --- Axes frame ---
        "axes.linewidth": 1.35,

        # --- Tick style: inward, on all four sides ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # --- Tick sizes ---
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,

        # --- Tick widths ---
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 0.75,
        "ytick.minor.width": 0.75,

        # --- Legend ---
        "legend.frameon": False,

        # --- Figure defaults ---
        "figure.figsize": (7, 6),
        "figure.dpi": 120,

        # --- Saving ---
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def set_mathtext_style() -> None:
    """
    Apply the same style as set_latex_style() but using matplotlib's built-in
    mathtext renderer instead of a full LaTeX installation.

    Useful for quick local runs or environments without LaTeX.
    Renders slightly differently from the publication style.
    """
    mpl.rcParams.update({
        # --- Font & rendering ---
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",

        # --- Font sizes ---
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,

        # --- Axes frame ---
        "axes.linewidth": 1.35,

        # --- Tick style ---
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # --- Tick sizes ---
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,

        # --- Tick widths ---
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 0.75,
        "ytick.minor.width": 0.75,

        # --- Legend ---
        "legend.frameon": False,

        # --- Figure defaults ---
        "figure.figsize": (7, 6),
        "figure.dpi": 120,

        # --- Saving ---
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })