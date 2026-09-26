#!/usr/bin/env python3
"""Show every design point of the throughput-parameter study against Stark's published observables.

Each panel plots one published observable against the 6 m planning yield for all design points,
colored by the T_sky shape parameter s_sky, with the published value marked and the band of
points that reproduce the published yield to 10% shaded. A setting that reconciles the yield with
the characterization times and the target mix would put a point on the published star in every
panel at once; the figure shows directly whether any point comes close.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

F_PUBLISHED_YIELD = 22.5
F_YIELD_TOLERANCE = 0.10
LIST_PANELS = [("fChar18Days6", "Mean char. time, first 18 EECs, 6 m (d)", 22.0),
               ("fChar18Days9", "Mean char. time, first 18 EECs, 9 m (d)", 3.5),
               ("10-15", "Median completeness, 10-15 pc (Fig. 11)", None),
               ("15-20", "Median completeness, 15-20 pc (Fig. 11)", None)]


def ffObservable(dictPoint, sKey):
    """One design point's value of a panel's observable."""
    if sKey in dictPoint:
        return dictPoint[sKey]
    return dictPoint["dictFigElevenMedianByDistance"][sKey]


def fnPanel(oAxes, listPoints, sKey, sLabel, fPublished):
    """Scatter one observable against the 6 m yield, with the published point and yield band."""
    faYield = np.array([d["fYield6"] for d in listPoints])
    faValue = np.array([ffObservable(d, sKey) for d in listPoints])
    faSky = np.array([d["dictTheta"]["s_sky"] for d in listPoints])
    oAxes.axvspan(F_PUBLISHED_YIELD * (1 - F_YIELD_TOLERANCE),
                  F_PUBLISHED_YIELD * (1 + F_YIELD_TOLERANCE), color=ps.S_GRID, alpha=0.6, lw=0)
    oScatter = oAxes.scatter(faYield, faValue, c=faSky, cmap="viridis", vmin=0, vmax=1, s=18,
                             edgecolor=ps.S_INK_SECONDARY, linewidth=0.3, zorder=3)
    oAxes.plot([F_PUBLISHED_YIELD], [fPublished], marker="*", ms=14, color=ps.LIST_SERIES[1],
               markeredgecolor=ps.S_INK_PRIMARY, zorder=4, ls="none")
    ps.fnFinishAxes(oAxes, "6 m planning yield (EECs)", sLabel)
    return oScatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", default="calibrationDegeneracyDesign.json")
    p.add_argument("--fig11-published", default="reference/figureElevenPublishedBins.json")
    p.add_argument("--out-pdf", default="../Plot/figCalibrationDegeneracyObservables.pdf")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    listPoints = json.load(open(dictArgs["design"]))["listPoints"]
    dictBins = json.load(open(dictArgs["fig11_published"]))["dictFigureEleven"]["dictByDistance"]
    oFig, oaAxes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for oAxes, (sKey, sLabel, fPub) in zip(oaAxes.ravel(), LIST_PANELS):
        fPub = dictBins[sKey]["fMedianPublished"] if fPub is None else fPub
        oScatter = fnPanel(oAxes, listPoints, sKey, sLabel, fPub)
    oFig.tight_layout(rect=(0, 0, 0.9, 1))
    oBar = oFig.colorbar(oScatter, ax=oaAxes, fraction=0.03, pad=0.02)
    oBar.set_label(r"$T_{\rm sky}$ shape $s_{\rm sky}$ (0 = this model, 1 = AYO-like)")
    oFig.savefig(dictArgs["out_pdf"])
    plt.close(oFig)


if __name__ == "__main__":
    main()
