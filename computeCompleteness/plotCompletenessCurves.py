#!/usr/bin/env python3
"""Plot completeness against exposure time, and the spread of reachable completeness per box."""

import argparse
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the completeness figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completeness", required=True)
    p.add_argument("--boxes", default="canonical,redefined")
    p.add_argument("--num-stars", type=int, default=3)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    listBoxes = [b.strip() for b in dictArgs["boxes"].split(",")]
    faTauDays = dictNpz["faTauGridS"] / 86400.0
    faRank = np.argsort(-dictNpz[f"faComp_{listBoxes[0]}"][:, -1])[:dictArgs["num_stars"]]
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 3.0))
    listStyles = ["-", "--", ":"]
    for sBox in listBoxes:
        for iStyle, iStar in enumerate(faRank):
            oAxesPair[0].plot(faTauDays, dictNpz[f"faComp_{sBox}"][iStar],
                              color=ps.DICT_BOX_COLOUR[sBox], ls=listStyles[iStyle % 3],
                              label=ps.DICT_BOX_LABEL_LONG[sBox] if iStyle == 0 else None)
    oAxesPair[0].set_xscale("log")
    ps.fnFinishAxes(oAxesPair[0], "Exposure time (days)", "Completeness $C(\\tau)$",
                    f"C($\\tau$), top {dictArgs['num_stars']} targets")
    oAxesPair[0].legend(loc="lower right")
    for sBox in listBoxes:
        faMax = dictNpz[f"faComp_{sBox}"][:, -1]
        oAxesPair[1].hist(faMax[faMax > 0], bins=30, histtype="step", linewidth=2.0,
                          color=ps.DICT_BOX_COLOUR[sBox],
                          label=f"{sBox}: {int(np.sum(faMax > 0))} stars reachable")
    ps.fnFinishAxes(oAxesPair[1], "Maximum completeness reachable", "Number of stars",
                    "Reachable targets per box")
    oAxesPair[1].legend(loc="upper right")
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
