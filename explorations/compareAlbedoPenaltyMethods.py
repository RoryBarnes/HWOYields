#!/usr/bin/env python3
"""Measure the albedo penalty on expected yield under each of the two implemented methods.

Stark et al. (2024) Sec. 3.2 adopt a uniform 0.08 < A_G < 0.32 as the fiducial albedo
distribution and report that budgeting for it lowers the mean yield from 22.5 to 19.8 -- a 12%
penalty. This pipeline reproduces only 5%, which is one of three remaining disagreements with the
published numbers, all of them in the direction of this model being too optimistic.

Two methods are implemented and this script runs both against the same survey:

  * "recompute" re-derives each drawn planet's required exposure time at its own albedo. It is
    the more physical calculation, and it is NOT what Stark does.
  * "perVisitThreshold" is Stark's: the observation plan is fixed at A_G = 0.2 and a drawn planet
    counts only if its adjusted flux exceeds the faintest flux actually detected on that visit.
    It is blunter, because it cannot reward a dark planet that happens to sit where the
    background is low, and it should therefore penalise dark planets harder.

If the published 12% is recovered by the second method, the disagreement is a methods difference
rather than a physics error, and the fiducial method should change to match the paper being
reproduced.
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


def fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, sMethod):
    """Optimize the survey once and return both the planning and albedo-drawn yields."""
    dictParams = json.loads(json.dumps(dictParams))
    dictMission = dictParams["dictMission"]
    dictMission["sAlbedoMethod"] = sMethod
    dictMission["fThroughputCalibration"] = dictArgs["calibration"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    return opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)


def fdictPenalty(sMethod, dictResult, fPublishedPenalty):
    """Fractional drop from the planning yield to the albedo-drawn yield."""
    fPlanning = float(dictResult["fYieldPlanning"])
    fDrawn = float(dictResult["fYield"])
    fPenalty = 1.0 - fDrawn / fPlanning if fPlanning else float("nan")
    return {
        "sMethod": sMethod,
        "fYieldPlanning": round(fPlanning, 3),
        "fYieldAlbedoDrawn": round(fDrawn, 3),
        "fAlbedoPenalty": round(fPenalty, 4),
        "fPublishedPenalty": fPublishedPenalty,
        "fRatioToPublished": round(fPenalty / fPublishedPenalty, 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-file")
    p.add_argument("--calibration", type=float, default=1.0)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--published-penalty", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    if dictArgs["calibration_file"]:
        with open(dictArgs["calibration_file"]) as oFile:
            dictArgs["calibration"] = json.load(oFile)["fCalibratedThroughputFactor"]
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    listOut = []
    for sMethod in ("recompute", "perVisitThreshold"):
        dictResult = fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, sMethod)
        listOut.append(fdictPenalty(sMethod, dictResult, dictArgs["published_penalty"]))
        print(json.dumps(listOut[-1]))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"fCalibration": dictArgs["calibration"], "listMethods": listOut}, oFile,
                  indent=2)


if __name__ == "__main__":
    main()
