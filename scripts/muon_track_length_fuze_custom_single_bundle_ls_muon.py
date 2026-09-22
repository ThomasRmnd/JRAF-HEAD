import argparse

import matplotlib.pyplot as plt
import numpy as np

from jrafhead.config import BLACK, CUSTOM_BLUE, CUSTOM_RED, set_latex_style
from jrafhead.loader import load_muon_length
from jrafhead.plotters import MuonLengthPlotter
from jrafhead.utils import rebin_histogram, save_figure, uniform_bins


def accumulated_length_hist(
    filepath: str,
    dirpath:  str,
    bins:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one muon-length analysis and return its accumulated (hist, err)
    over the given run range, rebinned onto `bins`.

    Mirrors MuonLengthAnalysis._plot_accumulated_length's own logic exactly.
    """
    data = load_muon_length(filepath, dirpath)

    # mask = np.logical_and(run_min <= data.run_id, data.run_id <= run_max)

    hist = np.sum(
        [h.counts for h in data.hist_length],
        # [h.counts for h, m in zip(data.hist_length, mask) if m],
        axis=0,
    )
    hist = rebin_histogram(data.hist_length[0].edges, hist, bins)
    err  = np.sqrt(hist)

    return hist, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse accumulated muon-length plots from three analyses")
    parser.add_argument("--input", required=True, help="ROOT file for the first analysis")

    # parser.add_argument("--output-dir", default=".", help="Output directory for the figure")
    parser.add_argument("--stem",       default="muon_length_fused", help="Output filename stem")

    args = parser.parse_args()

    set_latex_style()

    bins = uniform_bins(0.0, 40.0, 100)
    plotter = MuonLengthPlotter(bins=bins, show_steps=True, show_errors=False, legend_loc="upper left")

    sources = [
        (args.input, "muon_length__ls_custom_cdwpttchi2_wpclassify__analysis", r"Total",  BLACK),
        (args.input, "muon_length__ls_single__cdwpttchi2__analysis",           r"Single (CdWpTtChi2)", CUSTOM_BLUE),
        (args.input, "muon_length__ls_bundle__wpclassify__analysis",           r"Bundle (WpClassify)", CUSTOM_RED),
    ]

    for filepath, dirpath, label, color in sources:
        hist, err = accumulated_length_hist(filepath, dirpath, bins)
        plotter.add_histogram(
            hist, err,
            color, # fillcolor=color,
            label=label,
        )

    fig, _ = plotter.plot()
    # save_figure(fig, args.stem, "_accumulate", output_dir=args.output_dir)
    # plt.close(fig)

    plt.show()


if __name__ == "__main__":
    main()