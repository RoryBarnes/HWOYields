#!/usr/bin/env python3
"""Bin A10's per-target completeness comparison with Stark (2024) Fig. 11 by luminosity and distance.

A10 reports a median model/published completeness ratio of 0.38 over 141 matched targets while
the summed completeness is higher in the model (88 vs 77). This locates where the per-target
shortfall sits -- which luminosities and distances -- and how many matched targets the model gives
essentially no completeness (< 0.05), which would indicate targets scrubbed by the exozodi draw or
blocked by the noise floor rather than merely under-allocated.
"""

import argparse
import json

import numpy as np


def fdictBin(listRows):
    """Counts, medians, sums and the near-zero fraction for one bin."""
    if not listRows:
        return {"iTargets": 0}
    faPub = np.array([r["fCompletenessPublished"] for r in listRows])
    faMod = np.array([r["fCompletenessModel"] for r in listRows])
    return {"iTargets": len(listRows),
            "fMedianPublished": float(np.median(faPub)), "fMedianModel": float(np.median(faMod)),
            "fSumPublished": float(faPub.sum()), "fSumModel": float(faMod.sum()),
            "fFractionModelNearZero": float(np.mean(faMod < 0.05))}


def fdictByEdges(listRows, sKey, faEdges):
    """Bin rows on one field."""
    return {f"{faEdges[i]:g}-{faEdges[i + 1]:g}": fdictBin(
        [r for r in listRows if faEdges[i] <= r[sKey] < faEdges[i + 1]])
        for i in range(len(faEdges) - 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison",
                   default="../CompareTargetCompleteness/targetCompletenessComparison.json")
    p.add_argument("--rows", default="listRows",
                   help="listRows (representative exozodi draw) or dictDrawMean (mean over draws)")
    p.add_argument("--out-json", default="output/targetCompletenessBinned.json")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["comparison"]) as oFile:
        dictIn = json.load(oFile)
    listRows = dictIn["listRows"] if dictArgs["rows"] == "listRows" else \
        dictIn[dictArgs["rows"]]["listRows"]
    dictOut = {"dictAll": fdictBin(listRows),
               "dictByLuminosity": fdictByEdges(listRows, "fLuminosityLsun",
                                                [0.0, 0.3, 0.6, 1.0, 2.0, 5.0, 100.0]),
               "dictByDistance": fdictByEdges(listRows, "fDistancePc",
                                              [0.0, 6.0, 10.0, 15.0, 20.0, 50.0])}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sGroup in ("dictByLuminosity", "dictByDistance"):
        print(sGroup)
        for sBin, d in dictOut[sGroup].items():
            if d["iTargets"]:
                print(f"  {sBin:<9} n={d['iTargets']:>3}  median pub {d['fMedianPublished']:.2f} "
                      f"model {d['fMedianModel']:.2f}  sum pub {d['fSumPublished']:5.1f} "
                      f"model {d['fSumModel']:5.1f}  model<0.05: {d['fFractionModelNearZero']:.2f}")
    print("all", dictOut["dictAll"])


if __name__ == "__main__":
    main()
