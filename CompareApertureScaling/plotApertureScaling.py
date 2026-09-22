#!/usr/bin/env python3
"""Overlay the predicted P25-versus-aperture curves on the published Stark et al. (2024) points."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fnPlotProbabilities(oAxes, dictData):
    """Left panel: modelled P25 curves against the published points."""
    faDiameters = np.array([float(s) for s in dictData["dictByDiameter"]])
    dictPub = dictData["dictPublished"]
    for sKey, sPubKey, sColour, sLabel in (
            ("fProbability25IncludingSigmaEta", "dictIncludingSigmaEta", ps.LIST_SERIES[0],
             "including $\\sigma_{\\eta_\\oplus}$"),
            ("fProbability25ExcludingSigmaEta", "dictExcludingSigmaEta", ps.LIST_SERIES[1],
             "excluding $\\sigma_{\\eta_\\oplus}$")):
        faMine = np.array([100 * dictData["dictByDiameter"][s][sKey]
                           for s in dictData["dictByDiameter"]])
        faPub = np.array([100 * dictPub[sPubKey][s] for s in dictData["dictByDiameter"]])
        oAxes.plot(faDiameters, faMine, "-o", ms=5, color=sColour,
                   label=f"this work, {sLabel}")
        oAxes.plot(faDiameters, faPub, "--s", ms=6, mfc="none", color=sColour,
                   label=f"Stark+2024, {sLabel}")
    oAxes.set_ylim(0, 105)
    ps.fnFinishAxes(oAxes, "Inscribed diameter (m)", "$P_{25}$ (%)",
                    "Calibrated at 6 m only")
    oAxes.legend(loc="upper left", fontsize=7)


DICT_STARK2019_DMVC_BAND = {  # read off Stark et al. (2019) yield-vs-diameter figure, red band
    "faDiameters": [6.0, 7.0, 8.0, 9.0],
    "faLower": [20.0, 26.0, 33.0, 43.0],
    "faUpper": [26.0, 34.0, 42.0, 55.0],
}


def fnPlotYields(oAxes, dictData):
    """Right panel: expected yield against aperture, against the published DMVC band.

    The published band is the segmented off-axis DM-assisted vortex case from Stark et al.
    (2019) -- the same configuration modelled here -- read off their yield-versus-diameter
    figure. It scales as roughly D^1.9; this model scales as D^1.2, which is the clearest
    statement of where the rederivation fails.
    """
    faDiameters = np.array([float(s) for s in dictData["dictByDiameter"]])
    faYield = np.array([dictData["dictByDiameter"][s]["fExpectedYieldAtBaselineEta"]
                        for s in dictData["dictByDiameter"]])
    dictBand = DICT_STARK2019_DMVC_BAND
    oAxes.fill_between(dictBand["faDiameters"], dictBand["faLower"], dictBand["faUpper"],
                       color=ps.LIST_SERIES[1], alpha=0.22, lw=0,
                       label="Stark+2019 DMVC band ($\\sim D^{1.9}$)")
    oAxes.plot(faDiameters, faYield, "-o", ms=5, color=ps.LIST_SERIES[2],
               label="this work, $\\eta = 0.24$ ($\\sim D^{1.2}$)")
    oAxes.axhline(25.0, color=ps.S_INK_SECONDARY, lw=1.0, ls="--")
    oAxes.annotate("25 EEC goal", xy=(faDiameters[0], 25.0), xytext=(4, 4),
                   textcoords="offset points", color=ps.S_INK_SECONDARY, fontsize=8)
    ps.fnFinishAxes(oAxes, "Inscribed diameter (m)", "Expected EEC yield",
                    "Aperture scaling is too weak")
    oAxes.legend(loc="upper left", fontsize=7)


def fdictParseArgs():
    """Command-line configuration for the aperture-scaling figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["comparison"]) as oFile:
        dictData = json.load(oFile)
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fnPlotProbabilities(oAxesPair[0], dictData)
    fnPlotYields(oAxesPair[1], dictData)
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
