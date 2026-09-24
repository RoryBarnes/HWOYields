#!/usr/bin/env python3
"""Plot per-target completeness against the published Fig. 11 values, and its distance trend."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the per-target completeness figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["comparison"]) as oFile:
        listRows = json.load(oFile)["listRows"]
    faPub = np.array([d["fCompletenessPublished"] for d in listRows])
    faMod = np.array([d["fCompletenessModel"] for d in listRows])
    faDist = np.array([d["fDistancePc"] for d in listRows])
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 3.1))
    oScatter = oAxesPair[0].scatter(faPub, faMod, c=faDist, cmap="viridis", s=18, lw=0)
    oAxesPair[0].plot([0, 1], [0, 1], color=ps.S_INK_SECONDARY, lw=1.0, ls="--")
    oFig.colorbar(oScatter, ax=oAxesPair[0], label="distance (pc)")
    ps.fnFinishAxes(oAxesPair[0], "published completeness", "this model",
                    "Per target (dashed = agreement)")
    for faValues, sColour, sLabel in ((faPub, ps.LIST_SERIES[0], "Stark+2024 Fig. 11"),
                                      (faMod, ps.LIST_SERIES[1], "this model")):
        faOrder = np.argsort(faDist)
        faBinned = [np.median(faValues[faOrder][i:i + 15])
                    for i in range(0, len(faOrder) - 14, 15)]
        faCentres = [np.median(faDist[faOrder][i:i + 15])
                     for i in range(0, len(faOrder) - 14, 15)]
        oAxesPair[1].plot(faCentres, faBinned, "-o", ms=4, color=sColour, label=sLabel)
    ps.fnFinishAxes(oAxesPair[1], "distance (pc)", "median completeness",
                    "Median completeness by distance")
    oAxesPair[1].legend(loc="upper right", fontsize=8)
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
