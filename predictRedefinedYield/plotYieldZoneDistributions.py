#!/usr/bin/env python3
"""Publication figure: the realized EEC yield distribution for the habitable zone and the narrow Earth zone.

The paper version of the right panel of plotYieldPrediction.py, stripped to what the text argues:
one realized-yield curve per selection box, no grid, no title, no 25-EEC goal line, and a legend
of two plain horizontal rules. The realized yield is the Poisson draw per posterior sample, which
is the quantity Stark et al. (2024) Fig. 10 plots.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
from yieldlib.yielddistribution import faPoissonMixturePmf  # noqa: E402
import vplot  # noqa: E402,F401  -- VPLanet house style; sets fonts on import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DICT_ZONE_LABEL = {"canonical": "Habitable Zone", "redefined": "Earth Zone"}


def fnBoxAxes(oAxes):
    """Close the frame on all four sides; plotstyle leaves the top and right open."""
    for sSide in ("top", "right", "bottom", "left"):
        oAxes.spines[sSide].set_visible(True)
        oAxes.spines[sSide].set_color(ps.S_INK_SECONDARY)
    oAxes.tick_params(direction="in", top=True, right=True)


def fnDrawZone(oAxes, faExpected, sBox, iMaxYield):
    """One box's EEC yield distribution, as the exact Poisson mixture over expected yields."""
    faPmf = faPoissonMixturePmf(faExpected, iMax=iMaxYield + 1)
    oAxes.plot(np.arange(faPmf.size), faPmf, linewidth=2.0, color=ps.DICT_BOX_COLOUR[sBox])


def flistLegendHandles(listBoxes):
    """Plain horizontal rules, one per zone, in the order the boxes are drawn."""
    return [Line2D([0], [0], color=ps.DICT_BOX_COLOUR[s], linewidth=2.0,
                   label=DICT_ZONE_LABEL[s]) for s in listBoxes]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True)
    p.add_argument("--boxes", default="canonical,redefined")
    p.add_argument("--max-yield", type=int, default=50,
                   help="x-axis limit; 50 matches Stark et al. (2024) Fig. 10")
    p.add_argument("sPlotPath")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    dictSamples = np.load(dictArgs["samples"])
    listBoxes = [s.strip() for s in dictArgs["boxes"].split(",") if s.strip()]
    oFigure, oAxes = plt.subplots(figsize=(4.4, 3.2))
    for sBox in listBoxes:
        fnDrawZone(oAxes, dictSamples[f"faExpected_{sBox}"], sBox, dictArgs["max_yield"])
    oAxes.set_xlim(0, dictArgs["max_yield"])
    ps.fnFinishAxes(oAxes, "EEC Yield", "Frequency")
    oAxes.grid(False)
    fnBoxAxes(oAxes)
    oAxes.legend(handles=flistLegendHandles(listBoxes), loc="upper right")
    oFigure.tight_layout()
    oFigure.savefig(dictArgs["sPlotPath"])
    plt.close(oFigure)


if __name__ == "__main__":
    main()
