#!/usr/bin/env python3
"""Redo A10's Fig. 11 comparison with DETECTION completeness at the same optimized allocation.

A10 compares each target's allocated completeness from the table's faComp, which counts a planet
only if its characterization also fits under the two-month cap. Stark et al. (2024) Fig. 11 is
captioned "HZ completeness of selected targets", which this project previously concluded means
detection completeness (handoff, "Comparing the wrong quantity"). Comparing a characterization-
gated number against a detection-only one would manufacture a shortfall on exactly the distant
and luminous stars where characterization binds. The allocation is kept identical -- optimized
on faComp as the survey is -- and the detection-only curve is carried through the cost-curve
hull in the faCompYield slot, so both numbers are read at the same (visits, exposure) vertex.
Output rows use A10's format for each variant, so binTargetCompletenessComparison.py bins them.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

S_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S_ROOT)
sys.path.insert(0, os.path.join(S_ROOT, "CompareTargetCompleteness"))

from dataCompareTargetCompleteness import fdictMatch  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def faAllocatedPair(listStars, fSlope, fEta, dictMission):
    """Gated and detection-only completeness at each star's optimized vertex."""
    faGated, faDetection = np.zeros(len(listStars)), np.zeros(len(listStars))
    for i, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEta, dictMission)
        if len(dictCurve["faCost"]) < 2:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex > 0:
            faGated[i] = dictCurve["faComp"][iVertex]
            faDetection[i] = dictCurve["faCompYield"][iVertex]
    return faGated, faDetection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digitised", default="../CompareTargetCompleteness/starkFigure11Digitised.json")
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--num-planets", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-prefix", default="output/targetCompleteness")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox, dictMission = dictParams["dictBoxes"]["canonical"], dictParams["dictMission"]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    0.5, 1500, 2500.0, 7500.0, dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictMission["fExposureLimitS"]), 150)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    fSlope = opt.fdictOptimizeSurvey(listStars, 0.24, dictMission)["fSlope"]
    for i, dictStar in enumerate(listStars):
        dictStar["faCompYield"] = dictTable["faCompDetectionOnly"][i]
    faGated, faDetection = faAllocatedPair(listStars, fSlope, 0.24, dictMission)
    with open(dictArgs["digitised"]) as oFile:
        listStark = json.load(oFile)["listPoints"]
    for sName, faC in (("Gated", faGated), ("Detection", faDetection)):
        listRows = fdictMatch(dfTargets["fDistancePc"].to_numpy(),
                              dfTargets["fLuminosityLsun"].to_numpy(), faC, listStark, 0.12, 1.2)
        with open(f"{dictArgs['out_prefix']}{sName}.json", "w") as oFile:
            json.dump({"listRows": listRows, "fModelSummed": float(faC.sum())}, oFile, indent=2)
        print(f"{sName}: summed model completeness {faC.sum():.1f} over "
              f"{int(np.sum(faC > 0))} stars; matched {len(listRows)}")


if __name__ == "__main__":
    main()
