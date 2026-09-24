#!/usr/bin/env python3
"""Check that completeness interpolated on the exozodi grid reproduces direct evaluation.

yieldlib.exozodi tabulates each star's completeness on a grid of exozodi levels and interpolates
between nodes for each draw, rather than recomputing every star at every drawn level. This runs
both on the same screened stars and the same draws and compares the optimized yields (planning
and albedo-drawn, at eta = 0.24), which are the quantities the pipeline reports. It also times
the grid, since the point of the grid is to make many draws affordable.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import completeness as cp  # noqa: E402
from yieldlib import exozodi as ez  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def flistStarsDirect(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed, faZodi):
    """Star dicts from a direct single-level evaluation of every star at its drawn level."""
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    listStars = []
    for i, dictRow in enumerate(dfTargets.to_dict("records")):
        dictRow["fExozodiLevel"] = float(faZodi[i])
        d = cp.fdictStarCompleteness(dictRow, dictBox, dictParams["dictBands"]["listBandsDetection"],
                                     dictParams["dictBands"]["dictBandCharacterization"],
                                     dictMission, faTauGridS, iNumPlanets, dictParams["fAlpha"],
                                     dictParams["fBeta"], iSeed + i)
        listStars.append(dict(faTauGridS=faTauGridS, faComp=d["faComp"],
                              faCompYield=d["faCompAlbedo"], faTauCharMeanS=d["faTauCharMeanS"]))
    return listStars


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--num-planets", type=int, default=1000)
    p.add_argument("--num-draws", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/exozodiGridValidation.json")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"]["canonical"]
    dictMission = dictParams["dictMission"]
    faTauGridS = np.logspace(1.0, np.log10(dictMission["fExposureLimitS"]), 150)
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBox)
    fStart = time.time()
    dictGrid = ez.fdictCompletenessZodiGrid(dfTargets, dictParams, dictBox, faTauGridS,
                                            dictArgs["num_planets"], dictArgs["seed"])
    fGridSeconds = time.time() - fStart
    faDraws = ez.faDrawZodiLevels(dictMission, len(dfTargets), dictArgs["num_draws"],
                                  dictArgs["seed"] + 977, sv.flistHipNumbers(dfTargets))
    listRows = []
    for j in range(dictArgs["num_draws"]):
        dictGridRun = opt.fdictOptimizeSurvey(
            ez.flistStarsAtZodi(dictGrid, faDraws[j], faTauGridS), 0.24, dictMission)
        dictDirect = opt.fdictOptimizeSurvey(
            flistStarsDirect(dfTargets, dictParams, dictBox, faTauGridS, dictArgs["num_planets"],
                             dictArgs["seed"], faDraws[j]), 0.24, dictMission)
        listRows.append({sKey: {"fGrid": float(dictGridRun[sKey]),
                                "fDirect": float(dictDirect[sKey])}
                         for sKey in ("fYieldPlanning", "fYield", "iStarsUsed")})
        print(json.dumps(listRows[-1]), flush=True)
    dictOut = {"fGridSeconds": fGridSeconds, "iStarsStored": int(dictGrid["iaStar"].size),
               "iStarsScreened": int(len(dfTargets)), "listDraws": listRows,
               "fMaxRelativeYieldError": float(max(
                   abs(r[k]["fGrid"] / r[k]["fDirect"] - 1.0)
                   for r in listRows for k in ("fYieldPlanning", "fYield")))}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "listDraws"}, indent=2))


if __name__ == "__main__":
    main()
