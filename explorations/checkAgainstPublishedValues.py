#!/usr/bin/env python3
"""Compare this pipeline's results against values published by Stark et al. (2024), pass/fail.

Only anchors stated in the PAPER TEXT are checked here, not values read off figures, and each
is tagged with whether the pipeline was tuned to it. The calibration gate is tuned by
construction and is therefore reported but never counted as evidence; everything else is an
independent comparison. Values read off figures are listed as targets for checks that this
pipeline has not yet run.
"""

import argparse
import json
import os

LIST_ANCHORS = [
    {"sKey": "etaCanonicalSag13", "sLabel": "eta_Earth, SAG13 over canonical EEC box",
     "fPublished": 0.24, "sSource": "Stark+2024 Table 1", "fRelTol": 0.03, "bTuned": False,
     "sNote": "Population layer only; no telescope model involved."},
    {"sKey": "yieldUncalibrated", "sLabel": "Expected EEC yield, 6 m baseline (uncalibrated)",
     "fPublished": 22.5, "sSource": "Stark+2024 Sec. 3.1", "fRelTol": 0.20, "bTuned": False,
     "sNote": "The pipeline's prediction BEFORE the throughput factor is fitted."},
    {"sKey": "yieldCalibrated", "sLabel": "Expected EEC yield, 6 m baseline (calibrated)",
     "fPublished": 22.5, "sSource": "Stark+2024 Sec. 3.1", "fRelTol": 0.02, "bTuned": True,
     "sNote": "TUNED to this value by construction. Not evidence."},
    {"sKey": "probability25", "sLabel": "P25 including eta_Earth uncertainty, 6 m",
     "fPublished": 0.32, "sSource": "Stark+2024 Sec. 3.5", "fRelTol": 0.25, "bTuned": False,
     "sNote": "Independent: nothing in the pipeline was fitted to this."},
]

LIST_NOT_YET_RUN = [
    {"sLabel": "P25 vs aperture, 7/8/9 m (including sigma_eta)",
     "faPublished": [0.53, 0.67, 0.78], "sSource": "Stark+2024 Fig. 15 (read off figure)",
     "sHow": "Re-run A04/A05/A07 at --diameter-m 7, 8, 9 with the 6 m calibration frozen."},
    {"sLabel": "P25 vs aperture, 6/7/8/9 m (excluding sigma_eta)",
     "faPublished": [0.056, 0.49, 0.89, 0.995], "sSource": "Stark+2024 Fig. 15 (read off figure)",
     "sHow": "Same runs, with eta fixed at 0.24 and only Poisson sampling applied."},
    {"sLabel": "Per-target HZ completeness vs distance",
     "faPublished": [], "sSource": "Stark+2024 Fig. 11",
     "sHow": "Plot faComp[:, -1] against faDistancePc from completeness.npz."},
]


def fnReadValue(sRepoRoot, sKey):
    """Pull one comparable quantity out of the pipeline's step outputs."""
    def fnLoad(sRel):
        with open(os.path.join(sRepoRoot, sRel)) as oFile:
            return json.load(oFile)
    if sKey == "etaCanonicalSag13":
        import sys
        sys.path.insert(0, sRepoRoot)
        from yieldlib import occurrence as oc
        dictBox = fnLoad("modelCoronagraph/missionParameters.json")["dictBoxes"]["canonical"]
        return oc.fnOccurrenceAnalytic(dictBox, 0.38, -0.19, 0.26)
    if sKey == "yieldUncalibrated":
        return fnLoad("calibrateToStark/calibration.json")["fUncalibratedYield"]
    if sKey == "yieldCalibrated":
        return fnLoad("calibrateToStark/calibration.json")["fCalibratedYield"]
    if sKey == "probability25":
        return fnLoad("predictRedefinedYield/yieldPrediction.json")[
            "dictByBox"]["canonical"]["dictDistribution"]["fProbabilityAtLeastGoal"]
    raise KeyError(sKey)


def fdictCheckAnchor(sRepoRoot, dictAnchor):
    """Evaluate one published anchor and report agreement."""
    fValue = fnReadValue(sRepoRoot, dictAnchor["sKey"])
    fRelative = abs(fValue - dictAnchor["fPublished"]) / dictAnchor["fPublished"]
    return {"sLabel": dictAnchor["sLabel"], "sSource": dictAnchor["sSource"],
            "fPublished": dictAnchor["fPublished"], "fThisPipeline": float(fValue),
            "fRelativeDifference": float(fRelative), "fTolerance": dictAnchor["fRelTol"],
            "bWithinTolerance": bool(fRelative <= dictAnchor["fRelTol"]),
            "bTunedToThisValue": dictAnchor["bTuned"], "sNote": dictAnchor["sNote"]}


def fnPrintTable(listResults):
    """Print the comparison as a fixed-width table."""
    print(f"{'quantity':<46}{'published':>10}{'here':>10}{'diff':>8}  verdict")
    print("-" * 92)
    for d in listResults:
        sVerdict = "TUNED (not evidence)" if d["bTunedToThisValue"] else (
            "agrees" if d["bWithinTolerance"] else "OUTSIDE TOLERANCE")
        print(f"{d['sLabel']:<46}{d['fPublished']:>10.4g}{d['fThisPipeline']:>10.4g}"
              f"{100*d['fRelativeDifference']:>7.1f}%  {sVerdict}")


def fdictParseArgs():
    """Command-line configuration for the published-value comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="..")
    p.add_argument("--out-json", default="publishedValueComparison.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    sRepoRoot = os.path.abspath(dictArgs["repo_root"])
    listResults = [fdictCheckAnchor(sRepoRoot, a) for a in LIST_ANCHORS]
    listIndependent = [d for d in listResults if not d["bTunedToThisValue"]]
    dictOut = {
        "listChecked": listResults,
        "iIndependentChecks": len(listIndependent),
        "iIndependentAgreeing": sum(d["bWithinTolerance"] for d in listIndependent),
        "bAllIndependentAgree": all(d["bWithinTolerance"] for d in listIndependent),
        "listNotYetRun": LIST_NOT_YET_RUN,
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    fnPrintTable(listResults)
    print()
    print(f"independent checks agreeing: {dictOut['iIndependentAgreeing']}"
          f"/{dictOut['iIndependentChecks']}")
    print(f"stronger checks available but NOT yet run: {len(LIST_NOT_YET_RUN)}")
    for d in LIST_NOT_YET_RUN:
        print(f"  - {d['sLabel']}  [{d['sSource']}]")


if __name__ == "__main__":
    main()
