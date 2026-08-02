from __future__ import annotations

import argparse
import pathlib
import sys

# Allow running as python main.py from the project root, or from any working
# directory by resolving relative to this file's location.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import uproot

from jrafhead.analyses import (
    AccidentalAnalysis,
    IBDAnalysis,
    Li9He8RateAnalysis,
    Li9He8ShapeAnalysis,
    LifetimeDAQAnalysis,
    LifetimeVetoAnalysis,
    MultiplicityAnalysis,
    MuonPerformanceAnalysis,
    MuonRateAnalysis,
)
from jrafhead.config import (
    setup_style,
)

__version__ = "1.0.6"

ANALYSES = {
    # "ibd_": (
    #     IBDAnalysis, (
    #         "lifetime_daq__total_time__analysis", 
    #         "lifetime_veto__total_time__analysis"
    #     )
    # ), 
    "li9he8_rate_": (
        Li9He8RateAnalysis, (
            "lifetime_daq__total_time__analysis", 
            "lifetime_veto__total_time__analysis"
        )
    ), 
    # "li9he8_shape_": (
    #     Li9He8ShapeAnalysis, (
    #         "lifetime_daq__total_time__analysis", 
    #         "lifetime_veto__total_time__analysis"
    #     )
    # ), 
    # "accidental_": (
    #     AccidentalAnalysis, (
    #         "lifetime_daq__total_time__analysis", 
    #         "lifetime_veto__total_time__analysis"
    #     )
    # ), 
    # "lifetime_daq_": (
    #     LifetimeDAQAnalysis, ()
    # ), 
    # "lifetime_veto_": (
    #     LifetimeVetoAnalysis, ()
    # ), 
    # # "muon_performance_": (
    # #     MuonPerformanceAnalysis, (
    # #         "lifetime_daq__total_time__analysis",
    # #     )
    # # ), 
    # "muon_rate_": (
    #     MuonRateAnalysis, (
    #         "lifetime_daq__total_time__analysis",
    #     )
    # ), 
    # "multiplicity_": (
    #     MultiplicityAnalysis, (
    #          "lifetime_daq__total_time__analysis", 
    #     )
    # ), 
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="main", description="JRAF-HEAD")

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--input", type=str, help="Input file")
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help=(
            "Root output directory. Sub-directories pdf/ and png/ are "
            "created automatically inside it. Default: current directory."
        ),
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        default=False,
        help=(
            "Use matplotlib's built-in mathtext renderer instead of a "
            "LaTeX installation. Produces slightly different output but "
            "requires no external dependencies."
        ),
    )

    # Cosmo-specific flag
    parser.add_argument(
        "--mc-groupc-path",
        type=str,
        default="data/GroupC_spectra_P26B_total_FV17.2_June11_clean.root",
        help=("MC cosmogenics ROOT file for the GroupC Li9 spectrum overlay in the cosmogenic shape analysis."),
    )
    parser.add_argument(
        "--mc-9li8he-path",
        type=str,
        default="mc/mc_cosmogenics.root",
        help=("MC cosmogenics ROOT file for the Li9/He8 spectrum overlay in the cosmogenic shape analysis."),
    )
    parser.add_argument(
        "--mc-chengzhuo-path",
        type=str,
        default="data/9Li_predictedSpec_byChengzhuoIHEP.root",
        help=("MC cosmogenics ROOT file for the Chengzhuo Li9/He8 spectrum overlay in the cosmogenic shape analysis."),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file = uproot.open(args.input)

    dirs = [
        name.split(";")[0]
        for name, obj in file.items()
        if isinstance(obj, uproot.reading.ReadOnlyDirectory)
    ]
    if not dirs:
        print("No analysis requested")
        sys.exit(1)

    setup_style(no_latex=args.no_latex)

    for directory in dirs:
        # if (
        #     directory.startswith("li9he8_shape_muon__standard__analysis__") and
        #     directory != "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex" and
        #     directory != "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_2s__omilrec_jvertex" and
        #     not directory.endswith("_10s__omilrec_jvertex")
        # ):
        # if "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_2s_" not in directory:
        #     continue
        for prefix, (analysis_cls, extra_args) in ANALYSES.items():
            if prefix in directory:
                kwargs = {"output_dir": f"{args.output_dir}/{directory}"}
                if analysis_cls is Li9He8ShapeAnalysis:
                    kwargs["mc_groupc_path"]    = args.mc_groupc_path
                    kwargs["mc_9li8he_path"]    = args.mc_9li8he_path
                    kwargs["mc_chengzhuo_path"] = args.mc_chengzhuo_path
                analysis_cls(args.input, directory, *extra_args, **kwargs).run()

if __name__ == "__main__":
    main()