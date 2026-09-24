#!/usr/bin/env python3
"""Test the characterization-time model against the two figures Stark et al. (2024) publish for it.

Sec. 4.1: "the mean spectral characterization time of the first 18 EECs is 22 days for a 6 m ID
telescope, but can be shortened to just 3.5 days for a 9 m ID telescope" -- a value and a ratio,
both unused by this pipeline until now, and both bearing directly on the checks that still fail.

The case for looking here is that the characterization gate has turned out to control everything
downstream. Running the survey with the gate off raises the yield from 23.8 to 33.3 and the albedo
penalty from 5% to 43%, against a published 12%; running it on gives 5%. The gate admits only
planets whose spectra can be taken inside two months, and those are the bright, well-separated
ones that also have detection time to spare -- so a survey defined by that gate is robust to a
factor of 2.5 in planet albedo in a way Stark's is not. That is consistent with this model's
characterization times being too SHORT for the best targets, which would make the surviving set
too comfortable, while being too long for distant ones, which is the trend already seen.

This measures the mean characterization time of the brightest N EECs, ordered as the survey would
acquire them, at both published diameters.
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


def faStarCharacterizationTimes(listStars, dictMission, fEtaEarth, fSlope):
    """Mean characterization time and expected detections for each observed star.

    faTauCharMean is already the mean characterization cost per counted detection at each point on
    the exposure grid, so reading it at the vertex the optimizer chose gives the per-star figure
    the published statistic is built from.
    """
    listRows = []
    for iStar, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
        if len(dictCurve["faCost"]) <= 1:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex <= 0:
            continue
        listRows.append({"iStar": iStar,
                         "fCompleteness": float(dictCurve["faComp"][iVertex]),
                         "fCharSeconds": float(dictCurve["faTauCharMeanS"][iVertex]),
                         "fPriority": float(dictCurve["faComp"][iVertex] /
                                            dictCurve["faCost"][iVertex])})
    return listRows


def fnMeanCharOfFirstN(listRows, fEtaEarth, iFirstN, sOrder="cheapestChar"):
    """Mean characterization time over the first N expected EECs under a stated target ordering.

    Each star contributes eta_Earth * C expected detections at its own mean characterization cost,
    and stars are taken in order until N detections have accumulated. sOrder chooses the order:
    "cheapestChar" (the original choice) sorts by characterization time, which is the most
    favourable subset possible and biases the mean LOW by construction; "priority" sorts by
    completeness per unit total survey cost, the benefit-to-cost ranking the optimizer uses;
    "all" takes every counted EEC. Stark (2024) Sec. 4.1 take "the first 18 EECs of any
    simulation" so that harder targets added by larger telescopes do not dominate, which is a
    priority ordering, not a characterization-cost ordering.
    """
    listOk = [d for d in listRows if np.isfinite(d["fCharSeconds"]) and d["fCharSeconds"] > 0]
    if sOrder == "all":
        iFirstN = 10 ** 9
    listOk.sort(key=(lambda d: d["fCharSeconds"]) if sOrder == "cheapestChar"
                else (lambda d: -d["fPriority"]))
    fAccrued, fWeighted = 0.0, 0.0
    for dictRow in listOk:
        fCount = fEtaEarth * dictRow["fCompleteness"]
        fTake = min(fCount, iFirstN - fAccrued)
        if fTake <= 0:
            break
        fWeighted += fTake * dictRow["fCharSeconds"]
        fAccrued += fTake
    return (fWeighted / fAccrued / 86400.0) if fAccrued > 0 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--diameters", default="6,9")
    p.add_argument("--first-n", type=int, default=18)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParamsBase = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        fCalibration = json.load(oFile)["fCalibratedThroughputFactor"]
    dictParamsBase["fAlpha"], dictParamsBase["fBeta"] = -0.19, 0.26
    dictBox = dictParamsBase["dictBoxes"][dictArgs["box"]]
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])

    dictPublished = {"6": 22.0, "9": 3.5}
    listOut = []
    for sDiam in dictArgs["diameters"].split(","):
        dictParams = json.loads(json.dumps(dictParamsBase))
        dictMission = dictParams["dictMission"]
        dictMission["fDiameterM"] = float(sDiam)
        dictMission["fThroughputCalibration"] = fCalibration
        faTauGridS = np.logspace(1.0, np.log10(dictMission["fExposureLimitS"]),
                                 dictArgs["num_tau_points"])
        dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                        dictParams["dictBands"]["listBandsDetection"][0],
                                        dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                        dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
        dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                              dictArgs["num_planets"], dictArgs["seed"])
        listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
        dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
        listRows = faStarCharacterizationTimes(listStars, dictMission, dictArgs["eta_earth"],
                                               dictResult["fSlope"])
        fMean = fnMeanCharOfFirstN(listRows, dictArgs["eta_earth"], dictArgs["first_n"])
        listOut.append({
            "fDiameterM": float(sDiam),
            "fMeanCharDaysFirstN": round(float(fMean), 2),
            "fMeanCharDaysFirstNPriority": round(float(fnMeanCharOfFirstN(
                listRows, dictArgs["eta_earth"], dictArgs["first_n"], "priority")), 2),
            "fMeanCharDaysAll": round(float(fnMeanCharOfFirstN(
                listRows, dictArgs["eta_earth"], dictArgs["first_n"], "all")), 2),
            "fPublishedDays": dictPublished.get(sDiam),
            "iFirstN": dictArgs["first_n"],
            "fYieldPlanning": round(float(dictResult["fYieldPlanning"]), 2),
            "iStarsUsed": int(dictResult["iStarsUsed"]),
        })
        print(json.dumps(listOut[-1]))
    if len(listOut) == 2 and listOut[1]["fMeanCharDaysFirstN"] > 0:
        fRatio = listOut[0]["fMeanCharDaysFirstN"] / listOut[1]["fMeanCharDaysFirstN"]
        print(json.dumps({"fRatio6over9": round(fRatio, 2), "fRatioPublished": 6.2}))
        listOut.append({"fRatio6over9": round(fRatio, 2), "fRatioPublished": 6.2})
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"listRuns": listOut}, oFile, indent=2)


if __name__ == "__main__":
    main()
