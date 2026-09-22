#!/usr/bin/env python3
"""Compare this pipeline's per-target completeness with Stark et al. (2024) Fig. 11, star by star.

The digitized figure is the only published per-target completeness data, and it constrains the
HEIGHT of C at the operating point rather than its slope. Matching targets by distance and
luminosity and comparing completeness therefore answers a specific question: is this model's
completeness systematically wrong, or only its response to perturbation? A flat offset would
point at the height, a trend with distance or luminosity at the shape.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def faAllocatedCompleteness(listStars, dictTable, fSlope, fEtaEarth, dictMission):
    """Completeness each star actually reaches at the optimized allocation."""
    faOut = np.zeros(len(listStars))
    for i, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        faOut[i] = dictCurve["faComp"][iVertex] if iVertex > 0 else 0.0
    return faOut


def fdictMatch(faMineD, faMineL, faMineC, listStark, fTolDex, fTolDistance):
    """Pair each published target with the nearest modelled star in distance and luminosity."""
    listRows = []
    for dictPoint in listStark:
        faDistOk = np.abs(faMineD - dictPoint["fDistancePc"]) < fTolDistance
        faLumOk = np.abs(np.log10(np.maximum(faMineL, 1e-6)) -
                         np.log10(max(dictPoint["fLuminosityLsun"], 1e-6))) < fTolDex
        faBoth = np.where(faDistOk & faLumOk)[0]
        if not faBoth.size:
            continue
        faScore = (np.abs(faMineD[faBoth] - dictPoint["fDistancePc"]) / fTolDistance) ** 2 + \
            (np.abs(np.log10(np.maximum(faMineL[faBoth], 1e-6)) -
                    np.log10(max(dictPoint["fLuminosityLsun"], 1e-6))) / fTolDex) ** 2
        i = int(faBoth[int(np.argmin(faScore))])
        listRows.append({"fDistancePc": dictPoint["fDistancePc"],
                         "fLuminosityLsun": dictPoint["fLuminosityLsun"],
                         "fCompletenessPublished": dictPoint["fCompleteness"],
                         "fCompletenessModel": float(faMineC[i])})
    return listRows


def fdictParseArgs():
    """Command-line configuration for the per-target completeness comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digitised", required=True)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--num-planets", type=int, default=1500)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--tolerance-dex", type=float, default=0.12)
    p.add_argument("--tolerance-distance-pc", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="targetCompletenessComparison.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"]["canonical"]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    0.5, dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]), 150)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictOpt = opt.fdictOptimizeSurvey(listStars, 0.24, dictParams["dictMission"])
    faMineC = faAllocatedCompleteness(listStars, dictTable, dictOpt["fSlope"], 0.24,
                                      dictParams["dictMission"])
    with open(dictArgs["digitised"]) as oFile:
        listStark = json.load(oFile)["listPoints"]
    listRows = fdictMatch(dfTargets["fDistancePc"].to_numpy(),
                          dfTargets["fLuminosityLsun"].to_numpy(), faMineC, listStark,
                          dictArgs["tolerance_dex"], dictArgs["tolerance_distance_pc"])
    faPub = np.array([d["fCompletenessPublished"] for d in listRows])
    faMod = np.array([d["fCompletenessModel"] for d in listRows])
    faDist = np.array([d["fDistancePc"] for d in listRows])
    dictOut = {
        "iMatched": len(listRows),
        "iPublishedTargets": len(listStark),
        "iModelStarsUsed": int(dictOpt["iStarsUsed"]),
        "fPublishedSummedCompleteness": float(sum(d["fCompleteness"] for d in listStark)),
        "fModelSummedCompleteness": float(faMineC.sum()),
        "fMedianPublished": float(np.median(faPub)) if faPub.size else 0.0,
        "fMedianModel": float(np.median(faMod)) if faMod.size else 0.0,
        "fMedianRatio": float(np.median(faMod[faPub > 0.05] / faPub[faPub > 0.05]))
        if np.any(faPub > 0.05) else 0.0,
        "fTrendWithDistance": float(np.polyfit(faDist, faMod - faPub, 1)[0])
        if len(listRows) > 3 else 0.0,
        "listRows": listRows,
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"matched targets              : {dictOut['iMatched']} of "
          f"{dictOut['iPublishedTargets']} published")
    print(f"summed completeness  published {dictOut['fPublishedSummedCompleteness']:.1f}  "
          f"model {dictOut['fModelSummedCompleteness']:.1f}")
    print(f"stars used           published {dictOut['iPublishedTargets']}  "
          f"model {dictOut['iModelStarsUsed']}")
    print(f"median completeness  published {dictOut['fMedianPublished']:.3f}  "
          f"model {dictOut['fMedianModel']:.3f}")
    print(f"median model/published ratio : {dictOut['fMedianRatio']:.2f}")
    print(f"trend of (model - published) with distance : "
          f"{dictOut['fTrendWithDistance']:+.4f} per pc")


if __name__ == "__main__":
    main()
