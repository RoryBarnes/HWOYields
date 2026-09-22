#!/usr/bin/env python3
"""Split the optimized survey's time budget into search and characterization, against aperture.

If characterization consumes most of the budget, the survey is characterization-limited rather
than search-limited, and adding aperture buys far less yield than it should. That would explain
a yield growing as D^1.2 where the published result grows as D^1.9.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdictDecomposeAllocation(listStars, fSlope, fEtaEarth, dictMission):
    """Re-walk each star's cost curve at the optimal slope and separate the two time sinks."""
    fMult = dictMission["fWavefrontMultiplier"]
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fSearch, fChar, fComp, iUsed = 0.0, 0.0, 0.0, 0
    for dictStar in listStars:
        faTau = np.concatenate(([0.0], dictStar["faTauGridS"]))
        faComp = np.concatenate(([0.0], dictStar["faComp"]))
        faCost = np.where(faTau > 0.0, fMult * faTau + fOverhead, 0.0)
        if np.isfinite(dictStar["fTauCharS"]):
            faCost = faCost + fEtaEarth * faComp * (fMult * dictStar["fTauCharS"] + fOverhead)
        faHull = opt.faUpperConcaveHull(faCost, faComp)
        faHullCost, faHullComp = faCost[faHull], faComp[faHull]
        faSlopes = np.diff(faHullComp) / np.diff(faHullCost)
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex == 0:
            continue
        iOriginal = int(faHull[iVertex])
        fTau, fC = faTau[iOriginal], faComp[iOriginal]
        fSearch += fMult * fTau + fOverhead
        if np.isfinite(dictStar["fTauCharS"]):
            fChar += fEtaEarth * fC * (fMult * dictStar["fTauCharS"] + fOverhead)
        fComp += fC
        iUsed += 1
    return {"fSearchTimeS": fSearch, "fCharTimeS": fChar, "fSummedCompleteness": fComp,
            "iStarsUsed": iUsed}


def fdictAtDiameter(dfCatalog, dictParams, fDiameterM, dictArgs):
    """Optimize at one aperture and decompose the resulting time budget."""
    dictParams["dictMission"]["fDiameterM"] = fDiameterM
    dictBox = dictParams["dictBoxes"]["canonical"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    0.5, dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]), 150)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    if dictArgs.get("no_characterization"):
        for dictStar in listStars:
            dictStar["fTauCharS"] = np.inf
    dictOpt = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"],
                                      dictParams["dictMission"])
    dictSplit = fdictDecomposeAllocation(listStars, dictOpt["fSlope"], dictArgs["eta_earth"],
                                         dictParams["dictMission"])
    fTotal = dictParams["dictMission"]["fTotalScienceTimeS"]
    faTauChar = dictTable["faTauChar"]
    return {
        "fDiameterM": fDiameterM,
        "iStarsScreened": int(len(dfTargets)),
        "iStarsUsed": dictSplit["iStarsUsed"],
        "fYield": dictOpt["fYield"],
        "fSearchFraction": dictSplit["fSearchTimeS"] / fTotal,
        "fCharFraction": dictSplit["fCharTimeS"] / fTotal,
        "fMedianTauCharDaysUsed": float(np.nanmedian(
            np.where(np.isfinite(faTauChar), faTauChar, np.nan)) / 86400.0),
        "fMeanSearchTimePerStarHours": dictSplit["fSearchTimeS"] / max(dictSplit["iStarsUsed"], 1)
        / 3600.0,
    }


def fdictParseArgs():
    """Command-line configuration for the time-budget diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--diameters", default="6,9")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=2000)
    p.add_argument("--max-stars", type=int, default=7000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--no-characterization", action="store_true",
                   help="drop the characterization burden entirely, to isolate its effect")
    p.add_argument("--out-json", default="timeBudgetSplit.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    listRows = [fdictAtDiameter(dfCatalog, dictParams, float(s), dictArgs)
                for s in dictArgs["diameters"].split(",")]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"listRows": listRows}, oFile, indent=2)
    print(f"{'D':>4}{'used':>7}{'yield':>8}{'search %':>10}{'char %':>9}"
          f"{'tau_char d':>12}{'search h/star':>14}")
    for d in listRows:
        print(f"{d['fDiameterM']:>4.0f}{d['iStarsUsed']:>7}{d['fYield']:>8.2f}"
              f"{100*d['fSearchFraction']:>9.1f}%{100*d['fCharFraction']:>8.1f}%"
              f"{d['fMedianTauCharDaysUsed']:>12.1f}{d['fMeanSearchTimePerStarHours']:>14.1f}")


if __name__ == "__main__":
    main()
