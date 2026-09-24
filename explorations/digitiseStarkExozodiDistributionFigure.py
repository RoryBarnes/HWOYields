#!/usr/bin/env python3
"""Digitize the LBTI HOSTS maximum-likelihood exozodi distribution from Stark et al. (2024) Fig. 9.

Fig. 9 (left) histograms 10k exozodi draws from each of 500 randomly drawn distributions for
four fitting approaches; the red solid "Max. Likelihood" curve is the one Stark adopts for the
exozodi sampling and distribution uncertainties behind Fig. 10. The panel is vector: x is linear
in zodis (0-1000, one polyline vertex per 2-zodi bin) and y is log10 of the count with a floor at
10^2, where empty bins are drawn. This recovers bin probabilities so the pipeline can draw from
the published distribution instead of a lognormal stand-in; mass off the right edge of the axis
is reported, not invented.
"""

import argparse
import json
import re
import zlib

import numpy as np

S_RED_STROKE = "0 1 1 0 K"


def fsInflateContent(sPdfPath):
    """Return the inflated content stream that draws the panels."""
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for baStream in re.findall(rb"stream\r?\n(.*?)endstream", baRaw, re.S):
        try:
            sOut = zlib.decompress(baStream).decode("latin-1")
        except zlib.error:
            continue
        if "Max. Likelihood" in sOut:
            return sOut
    raise RuntimeError("no content stream with the Max. Likelihood legend")


def fdictAxes(sContent):
    """Linear x (0 and 1000 zodi major ticks) and log-y (10^2 .. 10^7 major ticks) mapping."""
    faY = sorted(float(a) for a in re.findall(r"\n432 ([\d.]+) m\n493\.797 [\d.]+ l", sContent))
    return {"fX0": 432.0, "fXPerZodi": (3528.0 - 432.0) / 1000.0, "fY0": faY[0],
            "fYPerDecade": (faY[-1] - faY[0]) / (len(faY) - 1), "fLogAtY0": 2.0}


def faRedMainPolyline(sContent):
    """Vertices of the first red polyline: the main-panel maximum-likelihood histogram."""
    iStart = sContent.index(S_RED_STROKE) + len(S_RED_STROKE)
    sBlock = sContent[iStart:sContent.index("\nS", iStart)]
    return np.array([(float(a), float(b)) for a, b in
                     re.findall(r"([-\d.]+) ([-\d.]+) [ml]", sBlock)])


def fdictHistogram(faXY, dictAxes, iDraws):
    """Bin centres (zodi) and probabilities; floor-level vertices are empty bins."""
    faZodi = (faXY[:, 0] - dictAxes["fX0"]) / dictAxes["fXPerZodi"]
    faLog = dictAxes["fLogAtY0"] + (faXY[:, 1] - dictAxes["fY0"]) / dictAxes["fYPerDecade"]
    faCount = np.where(faXY[:, 1] > dictAxes["fY0"] + 1.0, 10.0 ** faLog, 0.0)
    faWidth = float(np.median(np.diff(faZodi)))
    faEdges = np.arange(0.0, faZodi.max() + faWidth, faWidth)
    faBinned, _ = np.histogram(faZodi, bins=faEdges, weights=faCount)
    return {"faEdges": faEdges, "faProbability": faBinned / float(iDraws),
            "fBinWidthZodi": faWidth, "fMassOnAxis": float(faBinned.sum() / iDraws)}


def fdictSummary(dictHist):
    """Median, fractions below/above key levels, and the mean of the on-axis part."""
    faP, faEdges = dictHist["faProbability"], dictHist["faEdges"]
    faCentre = 0.5 * (faEdges[:-1] + faEdges[1:])
    faCdf = np.cumsum(faP) / faP.sum()
    return {"fMedianZodi": float(np.interp(0.5, faCdf, faEdges[1:])),
            "fFractionBelow2": float(faCdf[0]),
            "fFractionAbove10": float(faP[faEdges[:-1] >= 10.0].sum() / faP.sum()),
            "fFractionAbove30": float(faP[faEdges[:-1] >= 30.0].sum() / faP.sum()),
            "fFractionAbove100": float(faP[faEdges[:-1] >= 100.0].sum() / faP.sum()),
            "fMeanZodiOnAxis": float(np.sum(faCentre * faP) / faP.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", default="reference/starkFigure9.pdf")
    p.add_argument("--draws", type=int, default=5000000, help="10k draws x 500 distributions")
    p.add_argument("--out-json", default="starkFigure9ExozodiDigitised.json")
    dictArgs = vars(p.parse_args())
    sContent = fsInflateContent(dictArgs["pdf"])
    dictAxes = fdictAxes(sContent)
    dictHist = fdictHistogram(faRedMainPolyline(sContent), dictAxes, dictArgs["draws"])
    dictOut = {"sSource": "Stark et al. (2024) arXiv:2405.19418v1 Fig. 9 left, red solid "
                          "(LBTI HOSTS maximum likelihood)",
               "dictAxes": dictAxes, "fBinWidthZodi": dictHist["fBinWidthZodi"],
               "fMassOnAxis": dictHist["fMassOnAxis"],
               "faEdgesZodi": dictHist["faEdges"].tolist(),
               "faProbability": dictHist["faProbability"].tolist(),
               "dictSummary": fdictSummary(dictHist)}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"bin width {dictOut['fBinWidthZodi']:.3f} zodi; mass on axis {dictOut['fMassOnAxis']:.3f}")
    print(json.dumps(dictOut["dictSummary"], indent=2))


if __name__ == "__main__":
    main()
