#!/usr/bin/env python3
"""Attribute the uncalibrated-yield change between two target catalogs to specific stars.

Swapping the Gaia reconstruction for the real HPIC dropped the uncalibrated 6 m yield from 19.5
to 14.6. The screened target lists are nearly identical at the top -- the same alpha Cen A and B,
Procyon and eta Boo, with stellar parameters agreeing to better than a percent -- so composition
does not obviously explain it, and an explanation inferred from summary statistics would be a
guess. This script computes the yield for both catalogs under identical parameters and seeds, and
then reports where the difference actually sits: the per-star completeness and allocated time of
the highest-priority targets, and the cumulative yield as a function of priority rank.

Run it with --calibration 1.0 to compare the uncalibrated physics, which is the number the
calibration gate judges.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdfScreen(dfCatalog, dictParams, dictBox, dictArgs):
    """Screen a catalog with the calibration step's own bounds."""
    return sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                               dictParams["dictBands"]["listBandsDetection"][0],
                               dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                               dictArgs["teff_min"], dictArgs["teff_max"], dictBox)


def flistPerStarAllocation(listStars, dictParams, fEtaEarth, fSlope):
    """Rebuild each star's chosen envelope vertex at the optimizer's final slope.

    fdictOptimizeSurvey returns only survey totals, so the per-star allocation is recomputed here
    from the same cost curves at the same slope. This mirrors fdictAllocateAtSlope exactly and is
    a read-out of the optimizer's answer, not a second optimization.
    """
    listRows = []
    for iStar, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictParams["dictMission"])
        if len(dictCurve["faCost"]) <= 1:
            listRows.append({"iStar": iStar, "fCompleteness": 0.0, "fTimeS": 0.0})
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        listRows.append({
            "iStar": iStar,
            "fCompleteness": float(dictCurve["faComp"][iVertex]) if iVertex > 0 else 0.0,
            "fTimeS": float(dictCurve["faCost"][iVertex]) if iVertex > 0 else 0.0,
        })
    return listRows


def fdictOptimiseOne(dfTargets, dictParams, dictBox, faTauGridS, dictArgs):
    """Run the completeness table and the survey optimizer, returning the per-star allocation."""
    dictParams["dictMission"]["fThroughputCalibration"] = dictArgs["calibration"]
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"],
                                         dictParams["dictMission"])
    dictResult["listAllocation"] = flistPerStarAllocation(
        listStars, dictParams, dictArgs["eta_earth"], dictResult["fSlope"])
    return dictTable, dictResult


def flistPerStarRows(dfTargets, dictResult, iTop):
    """Per-star completeness, allocated time and yield contribution for the top targets."""
    dictByIndex = {int(d["iStar"]): d for d in dictResult["listAllocation"]}
    listRows = []
    for i in range(min(iTop, len(dfTargets))):
        dictAlloc = dictByIndex.get(i, {})
        listRows.append({
            "iRank": i,
            "sName": str(dfTargets["sSourceId"].iloc[i]),
            "fDistancePc": round(float(dfTargets["fDistancePc"].iloc[i]), 2),
            "fTeffK": round(float(dfTargets["fTeffK"].iloc[i]), 0),
            "fLuminosityLsun": round(float(dfTargets["fLuminosityLsun"].iloc[i]), 3),
            "fEeidLamD": round(float(dfTargets["fEeidLamD"].iloc[i]), 2),
            "fCompleteness": round(float(dictAlloc.get("fCompleteness", 0.0)), 4),
            "fTimeDays": round(float(dictAlloc.get("fTimeS", 0.0)) / 86400.0, 2),
        })
    return listRows


def faCumulativeYieldByRank(dfTargets, dictResult, fEtaEarth, listEdges):
    """Cumulative expected yield contributed by the first N targets in priority order."""
    dictByIndex = {int(d["iStar"]): d for d in dictResult["listAllocation"]}
    faComp = np.array([dictByIndex.get(i, {}).get("fCompleteness", 0.0)
                       for i in range(len(dfTargets))])
    return {str(i): round(float(fEtaEarth * np.sum(faComp[:i])), 3) for i in listEdges
            if i <= len(faComp)}


def fdictDistanceProfile(dfTargets, dictResult, fEtaEarth, faEdges):
    """Yield and observed-star count in bins of target distance."""
    dictByIndex = {int(d["iStar"]): d for d in dictResult["listAllocation"]}
    faComp = np.array([dictByIndex.get(i, {}).get("fCompleteness", 0.0)
                       for i in range(len(dfTargets))])
    faTime = np.array([dictByIndex.get(i, {}).get("fTimeS", 0.0) for i in range(len(dfTargets))])
    faDist = dfTargets["fDistancePc"].to_numpy()
    dictOut = {}
    for fLo, fHi in zip(faEdges[:-1], faEdges[1:]):
        bMask = (faDist >= fLo) & (faDist < fHi)
        dictOut[f"{int(fLo)}-{int(fHi)}"] = {
            "iScreened": int(np.sum(bMask)),
            "iObserved": int(np.sum(bMask & (faTime > 0))),
            "fYield": round(float(fEtaEarth * np.sum(faComp[bMask])), 3),
        }
    return dictOut


def fdictSummarise(sLabel, dfTargets, dictResult, iTop, fEtaEarth):
    """Yield, observed-star count and time budget for one catalog."""
    faComp = np.array([d["fCompleteness"] for d in dictResult["listAllocation"]])
    faTime = np.array([d["fTimeS"] for d in dictResult["listAllocation"]])
    return {
        "sLabel": sLabel,
        "iScreened": int(len(dfTargets)),
        "fYieldPlanning": float(dictResult["fYieldPlanning"]),
        "fYieldAlbedoDrawn": float(dictResult.get("fYield", np.nan)),
        "iStarsObserved": int(np.sum(faTime > 0)),
        "fTotalTimeDays": float(np.sum(faTime) / 86400.0),
        "fSumCompleteness": float(np.sum(faComp)),
        "dictCumulativeYieldByRank": faCumulativeYieldByRank(
            dfTargets, dictResult, fEtaEarth, [10, 25, 50, 100, 200, 400, 800, 1500]),
        "dictYieldByDistance": fdictDistanceProfile(
            dfTargets, dictResult, fEtaEarth,
            np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0])),
        "listTopStars": flistPerStarRows(dfTargets, dictResult, iTop),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", action="append", nargs=2, metavar=("LABEL", "PATH"),
                   required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--calibration", type=float, default=1.0)
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    listOut = []
    for sLabel, sPath in dictArgs["catalog"]:
        dfTargets = fdfScreen(pd.read_csv(sPath), dictParams, dictBox, dictArgs)
        _, dictResult = fdictOptimiseOne(dfTargets, dictParams, dictBox, faTauGridS, dictArgs)
        listOut.append(fdictSummarise(sLabel, dfTargets, dictResult, dictArgs["top"],
                                      dictArgs["eta_earth"]))
    dictOut = {"fCalibration": dictArgs["calibration"], "listCatalogs": listOut}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps([{k: v for k, v in d.items() if k != "listTopStars"} for d in listOut],
                     indent=2))


if __name__ == "__main__":
    main()
