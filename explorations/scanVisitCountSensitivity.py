#!/usr/bin/env python3
"""Test whether the visit model is what makes the albedo and exozodi penalties too weak.

Two published checks still disagree in the same direction and by a similar factor: Stark et al.
(2024) report that budgeting for the geometric-albedo distribution costs 12% of the expected
yield and that exozodi sampling costs a further ~11%, where this pipeline produces about 5% and
1.5%. Both are the same kind of quantity -- fix the observing plan, then perturb the planets or
the backgrounds and ask how much yield survives -- so a single mechanism that makes this model
insensitive to perturbation would explain both.

Visit count is the obvious candidate. Each visit is an independent chance to catch a planet at a
favourable phase, so a planet detectable at a fraction f of its orbit is caught with probability
1-(1-f)^N over N visits. At N=6 a planet whose detectable fraction is halved by a dark albedo
still has a 74% chance of being caught, against 98% for the bright case: the multi-visit average
washes out precisely the sensitivity the published penalties measure. Modelling visits was also
what closed the calibration gate earlier in this project, taking the fitted factor from 4.23 to
1.06, so if the visit count is too generous it has been compensating for the aperture-geometry
error that has since been fixed -- which is the pattern this project has hit four times already.

This scans the maximum visit count and reports, at each, the yield, the albedo penalty and the
exozodi penalty, against the published 12% and 11%.
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


def fdictOptimise(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, dictOverrides):
    """Screen, build the completeness table and optimize, under a set of mission overrides."""
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
    return opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)


def fnExozodiPenalty(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, dictOverrides,
                     fBaseline):
    """Fractional yield lost when per-star exozodi levels are drawn rather than held at median.

    Stark et al. (2024) Sec. 3.3 report the mean yield falling from 19.8 to 17.6 when exozodi is
    sampled per star, because a high draw on a high-priority target lengthens its exposure enough
    that the optimizer must substitute a less productive star from a limited pool.
    """
    dictDraw = dict(dictOverrides)
    dictParamsLocal = json.loads(json.dumps(dictParams))
    dictParamsLocal["dictMission"].update(dictDraw)
    dictParamsLocal["dictMission"]["fThroughputCalibration"] = dictArgs["calibration"]
    dictMission = dictParamsLocal["dictMission"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                    dictParamsLocal["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    faLevels = sv.faDrawExozodiLevels(dictMission, len(dfTargets), dictArgs["seed"])
    dfTargets = dfTargets.copy()
    dfTargets["fExozodiLevel"] = faLevels
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParamsLocal, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    fDrawn = float(opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"],
                                           dictMission)["fYieldPlanning"])
    return 1.0 - fDrawn / fBaseline if fBaseline else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--visits", default="1,2,3,4,6")
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
    for iVisits in [int(s) for s in dictArgs["visits"].split(",")]:
        dictOverrides = {"iMaxVisits": iVisits}
        dictResult = fdictOptimise(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs,
                                   dictOverrides)
        fPlanning = float(dictResult["fYieldPlanning"])
        fExozodi = fnExozodiPenalty(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs,
                                    dictOverrides, fPlanning)
        listOut.append({
            "iMaxVisits": iVisits,
            "fYieldPlanning": round(fPlanning, 3),
            "fYieldAlbedoDrawn": round(float(dictResult["fYield"]), 3),
            "fAlbedoPenalty": round(1.0 - float(dictResult["fYield"]) / fPlanning, 4),
            "fAlbedoPenaltyPublished": 0.12,
            "fExozodiPenalty": round(fExozodi, 4),
            "fExozodiPenaltyPublished": 0.11,
            "iStarsUsed": int(dictResult["iStarsUsed"]),
        })
        print(json.dumps(listOut[-1]))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"fCalibration": dictArgs["calibration"], "listVisits": listOut}, oFile,
                  indent=2)


if __name__ == "__main__":
    main()
