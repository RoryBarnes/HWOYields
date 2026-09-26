#!/usr/bin/env python3
"""Publication figure: the realized EEC yield distribution for the habitable zone and the narrow Earth zone.

The paper version of the right panel of plotYieldPrediction.py, stripped to what the text argues:
one realized-yield curve per selection box, no grid, no title, no 25-EEC goal line, and a legend
of two plain horizontal rules. The realized yield is the Poisson draw per posterior sample, which
is the quantity Stark et al. (2024) Fig. 10 plots.
"""

import argparse
import sys

import matplotlib
import matplotlib.ticker
import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
from yieldlib.yielddistribution import faPoissonMixturePmf  # noqa: E402
import vplot  # noqa: E402,F401  -- VPLanet house style; sets fonts on import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DICT_ZONE_LABEL = {"canonical": "Habitable Zone", "redefined": "Earth Zone"}
DICT_ZONE_COLOUR = {"canonical": vplot.colors.pale_blue, "redefined": vplot.colors.dark_blue}
LIST_FONT_KEYS = ["font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                  "ytick.labelsize", "legend.fontsize"]


def fnScaleFonts(fFactor):
    """Multiply every font size in the active style, resolving named sizes to points first."""
    fBase = plt.rcParams["font.size"]
    for sKey in LIST_FONT_KEYS:
        fValue = plt.rcParams[sKey]
        if isinstance(fValue, str):
            fValue = fBase * matplotlib.font_manager.font_scalings.get(fValue, 1.0)
        plt.rcParams[sKey] = fValue * fFactor


def fnBoxAxes(oAxes):
    """Close the frame on all four sides; plotstyle leaves the top and right open."""
    for sSide in ("top", "right", "bottom", "left"):
        oAxes.spines[sSide].set_visible(True)
        oAxes.spines[sSide].set_color(ps.S_INK_SECONDARY)
    oAxes.tick_params(direction="in", top=True, right=True)


def fnDrawZone(oAxes, faExpected, sBox, iMaxYield):
    """One box's EEC yield distribution, as the exact Poisson mixture over expected yields."""
    faPmf = faPoissonMixturePmf(faExpected, iMax=iMaxYield + 1)
    oAxes.plot(np.arange(faPmf.size), faPmf, linewidth=2.0, color=DICT_ZONE_COLOUR[sBox])


def fnSetFrequencyAxis(oAxes, fMaxFrequency, fTickStep):
    """Fix the frequency axis to 0..fMaxFrequency with a tick every fTickStep, as Stark's Fig. 10 does."""
    oAxes.set_ylim(0.0, fMaxFrequency)
    oAxes.set_yticks(np.arange(0.0, fMaxFrequency + 0.5 * fTickStep, fTickStep))
    oAxes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))


def flistLegendHandles(listBoxes):
    """Plain horizontal rules, one per zone, in the order the boxes are drawn."""
    return [Line2D([0], [0], color=DICT_ZONE_COLOUR[s], linewidth=2.0,
                   label=DICT_ZONE_LABEL[s]) for s in listBoxes]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--boxes", default="canonical,redefined")
    p.add_argument("--max-yield", type=int, default=50,
                   help="x-axis limit; 50 matches Stark et al. (2024) Fig. 10")
    p.add_argument("--max-frequency", type=float, default=0.14,
                   help="y-axis limit; tall enough for the Earth-zone peak")
    p.add_argument("--frequency-step", type=float, default=0.02,
                   help="y tick spacing; 0.02 matches Stark et al. (2024) Fig. 10")
    p.add_argument("--font-scale", type=float, default=1.25)
    p.add_argument("sPlotPath")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    fnScaleFonts(dictArgs["font_scale"])
    dictSamples = np.load(dictArgs["samples"])
    listBoxes = [s.strip() for s in dictArgs["boxes"].split(",") if s.strip()]
    oFigure, oAxes = plt.subplots(figsize=(4.4, 3.2))
    for sBox in listBoxes:
        fnDrawZone(oAxes, dictSamples[f"faExpected_{sBox}"], sBox, dictArgs["max_yield"])
    oAxes.set_xlim(0, dictArgs["max_yield"])
    ps.fnFinishAxes(oAxes, "Expected Yield", "Frequency")
    oAxes.grid(False)
    fnSetFrequencyAxis(oAxes, dictArgs["max_frequency"], dictArgs["frequency_step"])
    fnBoxAxes(oAxes)
    oAxes.legend(handles=flistLegendHandles(listBoxes), loc="upper right")
    oFigure.tight_layout()
    oFigure.savefig(dictArgs["sPlotPath"])
    plt.close(oFigure)


if __name__ == "__main__":
    main()
