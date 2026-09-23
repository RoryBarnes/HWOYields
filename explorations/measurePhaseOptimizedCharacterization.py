#!/usr/bin/env python3
"""Measure what scheduling characterization at the best orbital phase does to the survey.

Stark et al. (2019) Sec. 7.2 assume the orbit is determined before spectral characterization,
"such that the phase of the planet could be optimized". This pipeline instead charged the
characterization at whatever phase the planet occupied when it was detected. Because a planet
detected near crescent phase is both faint and close in, and because the two-month cap converts an
expensive observation into one that does not count toward the yield at all, the old treatment
turned marginal targets into hard zeros: 41 of the 154 targets in Stark's Fig. 11 come out of this
model with a maximum achievable completeness of exactly zero, not merely a low one.

The three published checks that still disagree -- the albedo penalty, P_25, and the mode of the
realized-yield distribution -- are one symptom seen three ways, namely that this survey is spent
on too few, too nearby stars whose completeness is saturated and therefore insensitive to any
perturbation. If the characterization phase is what excludes the distant targets, fixing it should
de-concentrate the survey and move all three together.

This script runs the survey both ways at a fixed throughput calibration and reports the yield,
the number of stars observed, the spread of per-target completeness, and the albedo penalty.
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


def faAllocatedCompleteness(listStars, dictMission, fEtaEarth, fSlope, iStars):
    """Each star's completeness at the optimizer's final slope, zero where it is not observed."""
    faOut = np.zeros(iStars)
    for iStar, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
        if len(dictCurve["faCost"]) <= 1:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex > 0:
            faOut[iStar] = dictCurve["faComp"][iVertex]
    return faOut


def fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, dictOverrides):
    """One survey optimization under a set of mission overrides."""
    dictParams = json.loads(json.dumps(dictParams))
    dictMission = dictParams["dictMission"]
    dictMission.update(dictOverrides)
    dictMission["fThroughputCalibration"] = dictArgs["calibration"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    faComp = faAllocatedCompleteness(listStars, dictMission, dictArgs["eta_earth"],
                                     dictResult["fSlope"], len(dfTargets))
    return dfTargets, dictResult, faComp


def fdictSummarise(sLabel, dfTargets, dictResult, faComp):
    """Yield, concentration and albedo sensitivity for one configuration."""
    faObs = faComp[faComp > 0]
    faDist = dfTargets["fDistancePc"].to_numpy()
    dictByDistance = {}
    for fLo, fHi in ((0, 10), (10, 15), (15, 20), (20, 30), (30, 50)):
        bMask = (faDist >= fLo) & (faDist < fHi) & (faComp > 0)
        dictByDistance[f"{fLo}-{fHi}"] = {
            "iObserved": int(bMask.sum()),
            "fMeanCompleteness": round(float(faComp[bMask].mean()) if bMask.any() else 0.0, 3),
        }
    return {
        "sLabel": sLabel,
        "fYieldPlanning": round(float(dictResult["fYieldPlanning"]), 3),
        "fYieldAlbedoDrawn": round(float(dictResult["fYield"]), 3),
        "fAlbedoPenalty": round(1.0 - float(dictResult["fYield"]) /
                                float(dictResult["fYieldPlanning"]), 4),
        "iStarsObserved": int((faComp > 0).sum()),
        "fMeanCompleteness": round(float(faObs.mean()) if faObs.size else 0.0, 3),
        "fMedianCompleteness": round(float(np.median(faObs)) if faObs.size else 0.0, 3),
        "fMedianOverMean": round(float(np.median(faObs) / faObs.mean()) if faObs.size else 0.0, 3),
        "dictByDistance": dictByDistance,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
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
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictArgs["calibration"] = json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    listOut = []
    for sLabel, dictOv in (("charGateOn", {"bOptimizeCharacterizationPhase": True}),
                           ("charGateOff", {"bOptimizeCharacterizationPhase": True,
                                            "bYieldRequiresCharacterization": False})):
        dfTargets, dictResult, faComp = fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS,
                                                    dictArgs, dictOv)
        listOut.append(fdictSummarise(sLabel, dfTargets, dictResult, faComp))
        print(json.dumps({k: v for k, v in listOut[-1].items() if k != "dictByDistance"}))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"fCalibration": dictArgs["calibration"], "listRuns": listOut}, oFile, indent=2)


if __name__ == "__main__":
    main()
