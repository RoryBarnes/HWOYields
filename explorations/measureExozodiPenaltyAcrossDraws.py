#!/usr/bin/env python3
"""Measure the exozodi-sampling penalty over many independent exozodi draws, not just one.

Stark et al. (2024) Sec. 3.3 repeat the exozodi draw 500 times and report the mean yield. The
pipeline draws once (a fixed seed), so its penalty carries the luck of which high-priority stars
that one draw buries under hundreds of zodis. This holds the planet population fixed, varies only
the exozodi draw (mission key iExozodiSeed), and reports the mean and scatter of the planning and
albedo-drawn yields against the fixed-exozodi run the calibration targets.
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


def fdictRun(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, dictOverrides):
    """Build the completeness table and optimize under a set of mission overrides."""
    dictParams = json.loads(json.dumps(dictParams))
    dictParams["dictMission"].update(dictOverrides)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    dictResult = opt.fdictOptimizeSurvey(sv.flistStarsFromTable(dictTable, faTauGridS),
                                         dictArgs["eta_earth"], dictParams["dictMission"])
    return {"fPlanning": float(dictResult["fYieldPlanning"]),
            "fAlbedoDrawn": float(dictResult["fYield"]),
            "iStarsUsed": int(dictResult["iStarsUsed"])}


def fdictSummary(listDraws, dictFixed):
    """Mean, scatter and implied penalties over the exozodi draws."""
    faPlan = np.array([d["fPlanning"] for d in listDraws])
    faAlb = np.array([d["fAlbedoDrawn"] for d in listDraws])
    return {"iDraws": len(listDraws),
            "fPlanningMean": float(faPlan.mean()), "fPlanningStd": float(faPlan.std(ddof=1)),
            "fAlbedoDrawnMean": float(faAlb.mean()), "fAlbedoDrawnStd": float(faAlb.std(ddof=1)),
            "fExozodiPenaltyMean": float(1.0 - faPlan.mean() / dictFixed["fPlanning"]),
            "fExozodiPenaltyPublished": round(1.0 - 17.6 / 19.8, 4),
            "fCombinedPenaltyMean": float(1.0 - faAlb.mean() / dictFixed["fPlanning"]),
            "fCombinedPenaltyPublished": round(1.0 - 17.35 / 22.5, 4)}


def fdictParseArgs():
    """Command-line configuration, matching the pipeline's A03/A05 survey settings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--num-draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/exozodiPenaltyAcrossDraws.json")
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
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0], 0.5,
                                    dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    dictFixed = fdictRun(dfTargets, dictParams, dictBox, faTauGridS, dictArgs,
                         {"bDrawExozodiLevels": False})
    listDraws = []
    for i in range(dictArgs["num_draws"]):
        listDraws.append(fdictRun(dfTargets, dictParams, dictBox, faTauGridS, dictArgs,
                                  {"bDrawExozodiLevels": True,
                                   "iExozodiSeed": dictArgs["seed"] + 977 + 1000 * i}))
        print(f"draw {i}: {listDraws[-1]}", flush=True)
    dictOut = {"dictFixed": dictFixed, "listDraws": listDraws,
               "dictSummary": fdictSummary(listDraws, dictFixed)}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut["dictSummary"], indent=2))


if __name__ == "__main__":
    main()
