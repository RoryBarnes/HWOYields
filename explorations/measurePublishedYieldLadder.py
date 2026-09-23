#!/usr/bin/env python3
"""Reproduce Stark's three-rung yield ladder and measure each published penalty separately.

Stark et al. (2024) report three yields for the same baseline mission, and the differences between
them are two of the checks this pipeline still fails:

    22.5   a single AYO run, every EEC at A_G = 0.2 and every star at the median exozodi level
    19.8   the same plan, with geometric albedos drawn from U(0.08, 0.32)   -- a 12% penalty
    17.6   and with per-star exozodi levels drawn from the HOSTS fit        -- a further 11%

This pipeline drew exozodi levels unconditionally, so its "22.5" already sat on the third rung
while being calibrated against the first. Two things follow. The throughput calibration was
fitted against a yield that already carried the exozodi cost, and the exozodi penalty was
structurally unmeasurable, because both sides of the comparison contained it -- which is why it
came out as exactly 0.0000 at every visit count rather than merely small.

This script walks the ladder properly, holding the observing plan fixed and perturbing one thing
at a time, and reports each penalty against its published value.
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
    """Screen, build the completeness table and optimize under a set of mission overrides."""
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
    return {"fPlanning": float(dictResult["fYieldPlanning"]),
            "fAlbedoDrawn": float(dictResult["fYield"]),
            "iStarsUsed": int(dictResult["iStarsUsed"])}


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

    dictFixed = fdictOptimise(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs,
                              {"bDrawExozodiLevels": False})
    dictDrawn = fdictOptimise(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs,
                              {"bDrawExozodiLevels": True})
    fRung1 = dictFixed["fPlanning"]
    fRung2 = dictFixed["fAlbedoDrawn"]
    fRung3 = dictDrawn["fAlbedoDrawn"]
    dictOut = {
        "fCalibration": dictArgs["calibration"],
        "dictRungs": {
            "fFixedExozodiPlanningAlbedo": round(fRung1, 3),
            "fFixedExozodiDrawnAlbedo": round(fRung2, 3),
            "fDrawnExozodiDrawnAlbedo": round(fRung3, 3),
            "fDrawnExozodiPlanningAlbedo": round(dictDrawn["fPlanning"], 3),
        },
        "dictPublishedRungs": {"fBaseline": 22.5, "fWithAlbedo": 19.8, "fWithExozodi": 17.6},
        "dictPenalties": {
            "fAlbedo": round(1.0 - fRung2 / fRung1, 4),
            "fAlbedoPublished": round(1.0 - 19.8 / 22.5, 4),
            "fExozodi": round(1.0 - fRung3 / fRung2, 4),
            "fExozodiPublished": round(1.0 - 17.6 / 19.8, 4),
            "fCombined": round(1.0 - fRung3 / fRung1, 4),
            "fCombinedPublished": round(1.0 - 17.6 / 22.5, 4),
        },
        "iStarsFixedExozodi": dictFixed["iStarsUsed"],
        "iStarsDrawnExozodi": dictDrawn["iStarsUsed"],
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
