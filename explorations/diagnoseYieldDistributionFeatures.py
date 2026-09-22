#!/usr/bin/env python3
"""Test whether features in the marginalized yield distribution are physical or interpolation.

The yield curve is evaluated on a coarse log-spaced eta grid and interpolated for each
posterior draw. Linear interpolation between grid nodes puts kinks in the mapping eta -> yield,
which can print as steps or drops in the histogram of yields. This locates the grid nodes in
yield space and checks whether the visible features sit on them.
"""

import argparse
import json

import numpy as np


def faHistogramDensity(faValues, fLo, fHi, iBins):
    """Histogram density and bin centres over a fixed range."""
    faCounts, faEdges = np.histogram(faValues, bins=iBins, range=(fLo, fHi), density=True)
    return faCounts, 0.5 * (faEdges[:-1] + faEdges[1:])


def fdictGridNodesInYield(dictBox):
    """Map each eta grid node onto its yield, the locations where interpolation kinks."""
    return {"faEtaGrid": np.array(dictBox["faEtaGrid"]),
            "faYieldGrid": np.array(dictBox["faYieldGrid"])}


def faLargestDrops(faDensity, faCentres, iCount):
    """Bin centres with the most negative bin-to-bin change in density."""
    faDelta = np.diff(faDensity)
    faIndex = np.argsort(faDelta)[:iCount]
    return [(float(faCentres[i]), float(faDelta[i])) for i in sorted(faIndex)]


def fdictParseArgs():
    """Command-line configuration for the distribution diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--range-lo", type=float, default=0.0)
    p.add_argument("--range-hi", type=float, default=75.0)
    p.add_argument("--out-json", default="yieldDistributionFeatures.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["prediction"]) as oFile:
        dictPrediction = json.load(oFile)
    dictSamples = np.load(dictArgs["samples"])
    faValues = dictSamples[f"faExpected_{dictArgs['box']}"]
    faDensity, faCentres = faHistogramDensity(faValues, dictArgs["range_lo"],
                                              dictArgs["range_hi"], dictArgs["bins"])
    dictGrid = fdictGridNodesInYield(dictPrediction["dictByBox"][dictArgs["box"]])
    faYieldNodes = dictGrid["faYieldGrid"]
    listDrops = faLargestDrops(faDensity, faCentres, 5)
    dictOut = {
        "sBox": dictArgs["box"],
        "iSamples": int(faValues.size),
        "fBinWidth": float((dictArgs["range_hi"] - dictArgs["range_lo"]) / dictArgs["bins"]),
        "listYieldGridNodes": [float(f) for f in faYieldNodes],
        "listYieldNodesInRange": [float(f) for f in faYieldNodes
                                  if dictArgs["range_lo"] < f < dictArgs["range_hi"]],
        "listLargestDensityDrops": [{"fYield": y, "fDeltaDensity": d} for y, d in listDrops],
        "listNearestNodeToEachDrop": [
            {"fDropYield": y,
             "fNearestGridNodeYield": float(faYieldNodes[int(np.argmin(np.abs(faYieldNodes - y)))]),
             "fSeparation": float(np.min(np.abs(faYieldNodes - y)))}
            for y, _ in listDrops],
        "fProbabilityAtLeast25": float(np.mean(dictSamples[f"faObserved_{dictArgs['box']}"] >= 25)),
        "fFractionExpectedAtLeast25": float(np.mean(faValues >= 25.0)),
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "listYieldGridNodes"}, indent=2))


if __name__ == "__main__":
    main()
