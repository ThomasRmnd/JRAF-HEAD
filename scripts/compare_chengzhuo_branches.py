import argparse

import numpy as np
import uproot
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from jrafhead.config import (
    BLACK,
    GOOGLE_BLUE,
    GOOGLE_GREEN,
    GOOGLE_PURPLE,
    GOOGLE_RED,
    GOOGLE_YELLOW,
    setup_style,
)
from jrafhead.loader import (
    MCChengzhuoTemplateHistogram,
    MCChengzhuoTemplateHistogramContribution,
    load_mc_chengzhuo_template,
)
from jrafhead.plotters import Histogram1DPlotter

NEW_BR = {
    "SetB_branch0": 0.288,
    "SetB_branch1": 0.102,
    "SetB_branch2": 0.095,
    "SetB_branch3": 0.012,
    "SetB_branch4": 0.011,
}

NAMES  = ["SetB", "SetB_branch0", "SetB_branch1", "SetB_branch2", "SetB_branch3", "SetB_branch4"]
COLORS = [BLACK,  GOOGLE_RED,     GOOGLE_BLUE,    GOOGLE_GREEN,   GOOGLE_PURPLE,  GOOGLE_YELLOW ]

def modify_branching_ratio_chengzhuo(
    template:   MCChengzhuoTemplateHistogram,
    names:      list[str],
    new_br:     dict[str, float]
) -> MCChengzhuoTemplateHistogram:
    new_branches = []
    for name, branch in zip(names[1:], [template.branch0, template.branch1, template.branch2, template.branch3, template.branch4]):
        scale = new_br[name] / np.sum(branch.counts * np.diff(branch.edges))
        hist = branch.counts * scale
        print(f"BR {name} integral = {np.sum(branch.counts * np.diff(branch.edges)):.4f} before and {np.sum(hist * np.diff(branch.edges)):.4f} after")
        new_branches.append(hist)
    new_all = np.sum(new_branches, axis=0)
    return MCChengzhuoTemplateHistogram(
        branch0=MCChengzhuoTemplateHistogramContribution(template.branch0.edges, new_branches[0], np.zeros_like(new_branches[0])),
        branch1=MCChengzhuoTemplateHistogramContribution(template.branch1.edges, new_branches[1], np.zeros_like(new_branches[1])),
        branch2=MCChengzhuoTemplateHistogramContribution(template.branch2.edges, new_branches[2], np.zeros_like(new_branches[2])),
        branch3=MCChengzhuoTemplateHistogramContribution(template.branch3.edges, new_branches[3], np.zeros_like(new_branches[3])),
        branch4=MCChengzhuoTemplateHistogramContribution(template.branch4.edges, new_branches[4], np.zeros_like(new_branches[4])),
        all=MCChengzhuoTemplateHistogramContribution(template.all.edges, new_all, np.zeros_like(new_all))
    )

def compare_chengzhuo_shape(
    new_template: MCChengzhuoTemplateHistogram,
    template:     MCChengzhuoTemplateHistogram,
) -> None:
    plotter = Histogram1DPlotter(
        bins=new_template.all.edges,
        xlabel=r"$E_{p}$~(MeV)",
        ylabel=r"P.D.F.",
        xlim=(0.0, 15.0),
        ylim=(0.0, 0.09),
        xscale="linear",
        yscale="linear",
        legend_ncol=1,
        legend_loc="upper right",
        density=False,
        show_steps=True,
        show_errors=False,
    )

    plotter.add_histogram(
        np.full_like(template.all.counts, -10.0),
        np.zeros_like(template.all.errors),
        "#868686",
        linestyle="-",
        label="Smoothed",
    )
    plotter.add_histogram(
        np.full_like(template.branch0.counts, -10.0),
        np.zeros_like(template.branch0.errors),
        "#868686",
        linestyle="--",
        label="Reference",
    )
    for index, (new_branch, color) in enumerate(zip(
        [new_template.all, new_template.branch0, new_template.branch1, new_template.branch2, new_template.branch3, new_template.branch4],
        COLORS,
    )):
        if index == 0:
            plotter.add_histogram(
                np.full_like(new_branch.counts, -10.0),
                np.zeros_like(new_branch.errors),
                color,
                linestyle="-",
                label="Total",
            )
        else:
            plotter.add_histogram(
                np.full_like(new_branch.counts, -10.0),
                np.zeros_like(new_branch.errors),
                color,
                linestyle="-",
                label=f"Branch {index}",
            )

    for index, (new_branch, old_branch, color) in enumerate(zip(
        [new_template.all, new_template.branch0, new_template.branch1, new_template.branch2, new_template.branch3, new_template.branch4],
        [template.all, template.branch0, template.branch1, template.branch2, template.branch3, template.branch4],
        COLORS,
    )):
        if index == 0:
            plotter.add_histogram(
                new_branch.counts,
                new_branch.errors,
                color,
                linestyle="-",
            )
            plotter.add_histogram(
                old_branch.counts,
                old_branch.errors,
                color,
                linestyle="--",
            )
        else:
            plotter.add_histogram(
                new_branch.counts,
                new_branch.errors,
                color,
                linestyle="-",
            )
            plotter.add_histogram(
                old_branch.counts,
                old_branch.errors,
                color,
                linestyle="--",
            )

    fig, ax = plotter.plot()
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(handles[2:])):
        handles[i + 2] = Line2D([], [], color=handles[i + 2].get_color(), marker="o", linestyle="None", markersize=6)
    ax.legend(handles, labels, loc="upper right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input filepath")
    args = parser.parse_args()

    setup_style()

    template = load_mc_chengzhuo_template(args.input)
    new_template = modify_branching_ratio_chengzhuo(template, NAMES, NEW_BR)
    compare_chengzhuo_shape(new_template, template)