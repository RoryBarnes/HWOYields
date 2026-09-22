#!/usr/bin/env python3
"""Run the equal-slope AYO survey optimization over the precomputed completeness curves.

Allocates the two-year exoplanet science budget across targets so that dC/dt is equal for every
observation, charging each star for its detection time, its overheads, and the characterization
time its expected detections will demand.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402


def flistLoadStars(sCompletenessPath, sBox):
    """Rehydrate one box's per-star completeness curves for the optimizer."""
    dictNpz = np.load(sCompletenessPath, allow_pickle=True)
    faTauGridS = dictNpz["faTauGridS"]
    faComp, faTauChar = dictNpz[f"faComp_{sBox}"], dictNpz[f"faTauChar_{sBox}"]
    faCharMean, faCompAlb = dictNpz[f"faTauCharMean_{sBox}"], dictNpz[f"faCompAlbedo_{sBox}"]
    return [dict(faTauGridS=faTauGridS, faComp=faComp[i], faCompYield=faCompAlb[i],
                 faTauCharMeanS=faCharMean[i], fTauCharS=float(faTauChar[i]))
            for i in range(faComp.shape[0])], dictNpz


def fdictRunOptimization(listStars, faEtaValues, dictMission):
    """Optimize the survey once per eta_Earth value, exposing the non-linear eta dependence."""
    listResults = []
    for fEta in faEtaValues:
        dictResult = opt.fdictOptimizeSurvey(listStars, float(fEta), dictMission)
        listResults.append({
            "fEtaEarth": float(fEta),
            "fYield": float(dictResult["fYield"]),
            "fSummedCompleteness": float(dictResult["fSummedCompleteness"]),
            "iStarsUsed": int(dictResult["iStarsUsed"]),
            "fTimeUsedFraction": float(dictResult["fTotalTimeS"] /
                                       dictMission["fTotalScienceTimeS"]),
        })
    return listResults


def fdictParseArgs():
    """Command-line configuration for the survey optimization."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completeness", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--eta-scan", type=str, default="0.02,0.06,0.12,0.24,0.40,0.60")
    p.add_argument("--calibration-json", default=None)
    p.add_argument("--throughput-calibration", type=float, default=1.0)
    p.add_argument("--out-survey", default="surveyResult.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictMission = json.load(oFile)["dictMission"]
    fCalibration = dictArgs["throughput_calibration"]
    if dictArgs["calibration_json"]:
        with open(dictArgs["calibration_json"]) as oCal:
            fCalibration = json.load(oCal)["fCalibratedThroughputFactor"]
    dictMission["fThroughputCalibration"] = fCalibration
    listStars, dictNpz = flistLoadStars(dictArgs["completeness"], dictArgs["box"])
    dictBaseline = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    faScan = np.array([float(s) for s in dictArgs["eta_scan"].split(",")])
    dictOut = {
        "sBox": dictArgs["box"],
        "iStarsAvailable": len(listStars),
        "fThroughputCalibration": fCalibration,
        "fEtaEarthBaseline": dictArgs["eta_earth"],
        "fYieldBaseline": float(dictBaseline["fYield"]),
        "fSummedCompletenessBaseline": float(dictBaseline["fSummedCompleteness"]),
        "iStarsUsedBaseline": int(dictBaseline["iStarsUsed"]),
        "fTimeUsedFractionBaseline": float(dictBaseline["fTotalTimeS"] /
                                           dictMission["fTotalScienceTimeS"]),
        "listEtaScan": fdictRunOptimization(listStars, faScan, dictMission),
    }
    with open(dictArgs["out_survey"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
