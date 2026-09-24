#!/usr/bin/env python3
"""Measure how strongly the stars that carry the yield respond to their exozodi level.

The re-measured yield ladder (output/publishedYieldLadderCurrent.json) gives an exozodi-sampling
penalty of 0.7 percent against Stark et al. (2024)'s 11 percent. A per-star exozodi draw can only
cost yield through the exposure times of the stars the survey actually spends time on, and it
moves those times only in proportion to exozodi's share of the background. This rebuilds the
baseline survey at a fixed 3 zodis, recovers each star's allocated completeness from the final
equal-slope level, and reports, weighted by each star's expected EEC contribution:
  - the share of the background (leakage, local zodi, exozodi, detector) for detectable planets
  - the elasticity d ln(tau) / d ln(exozodi), which for tau ~ (P + 2B)/P^2 is 2 X / (P + 2 B)
An elasticity near zero means the per-star draw cannot produce Stark's penalty in this model,
whatever the draw's distribution.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import completeness as cp  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import physics as ph  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdictParams(dictArgs):
    """Mission parameters at the current calibration with exozodi fixed at its median."""
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        fCal = json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["dictMission"].update({"fThroughputCalibration": fCal,
                                      "bDrawExozodiLevels": False})
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    return dictParams


def faAllocatedCompleteness(listStars, dictMission, fEta):
    """Each star's completeness at the vertex the final equal-slope level selects."""
    dictResult = opt.fdictOptimizeSurvey(listStars, fEta, dictMission)
    faOut = np.zeros(len(listStars))
    for i, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEta, dictMission)
        if len(dictCurve["faCost"]) < 2:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -dictResult["fSlope"], side="right"))
        faOut[i] = dictCurve["faComp"][iVertex] if iVertex > 0 else 0.0
    return faOut, float(dictResult["fYieldPlanning"])


def fdictStarNoise(dictStar, dictParams, dictBox, iNumPlanets, iSeed):
    """Median background shares and exozodi elasticity over one star's detectable planets."""
    dictMission = dictParams["dictMission"]
    dictBand = dictParams["dictBands"]["listBandsDetection"][0]
    dictPlanets = cp.fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                        iNumPlanets, dictParams["fAlpha"],
                                        dictParams["fBeta"], np.random.default_rng(iSeed))
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, cp.fdictProjectOrbits(dictPlanets),
                                   dictBand, dictMission)
    faP = dictRates["faPlanet"].ravel()
    faLeak = dictRates["faLeak"].ravel()
    faAstro = faLeak + dictRates["fZodi"] + dictRates["fExozodi"]
    faDet = ph.faDetectorCountRate((faAstro + faP) / dictBand["iNumPixels"],
                                   dictBand["iNumPixels"], dictMission["fDarkCurrent"],
                                   dictMission["fReadNoise"], None,
                                   dictMission["fClockInducedCharge"])
    return fdictShares(faP, faLeak, dictRates["fZodi"], dictRates["fExozodi"], faDet,
                       dictRates["faUpsilon"].ravel() > 0.0)


def fdictShares(faP, faLeak, fZodi, fExo, faDet, bOk):
    """Background shares and elasticity, median over planets with signal."""
    bOk = bOk & (faP > 0.0)
    if not bOk.any():
        return {}
    faB = faLeak[bOk] + fZodi + fExo + faDet[bOk]
    return {"fLeakShare": float(np.median(faLeak[bOk] / faB)),
            "fZodiShare": float(np.median(fZodi / faB)),
            "fExozodiShare": float(np.median(fExo / faB)),
            "fDetectorShare": float(np.median(faDet[bOk] / faB)),
            "fPlanetOverBackground": float(np.median(faP[bOk] / faB)),
            "fElasticity": float(np.median(2.0 * fExo / (faP[bOk] + 2.0 * faB)))}


def fdictWeighted(listRows):
    """Yield-weighted means of every per-star quantity."""
    faW = np.array([r["fEec"] for r in listRows])
    return {sKey: float(np.average([r[sKey] for r in listRows], weights=faW))
            for sKey in ("fLeakShare", "fZodiShare", "fExozodiShare", "fDetectorShare",
                         "fPlanetOverBackground", "fElasticity")}


def fdictParseArgs():
    """Command-line configuration, matching measurePublishedYieldLadder.py's survey."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/exozodiSensitivity.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    dictParams = fdictParams(dictArgs)
    dictBox, dictMission = dictParams["dictBoxes"]["canonical"], dictParams["dictMission"]
    faTauGridS = np.logspace(1.0, np.log10(dictMission["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0], 0.5,
                                    dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    faComp, fYield = faAllocatedCompleteness(sv.flistStarsFromTable(dictTable, faTauGridS),
                                             dictMission, dictArgs["eta_earth"])
    listRows = []
    for i, dictStar in enumerate(dfTargets.to_dict("records")):
        if faComp[i] <= 0.0:
            continue
        dictNoise = fdictStarNoise(dictStar, dictParams, dictBox, 400, dictArgs["seed"] + i)
        if dictNoise:
            listRows.append({"sName": str(dictStar.get("sSimbadName")),
                             "fDistancePc": float(dictStar["fDistancePc"]),
                             "fEec": float(dictArgs["eta_earth"] * faComp[i]), **dictNoise})
    listRows.sort(key=lambda r: -r["fEec"])
    dictOut = {"fYieldPlanning": fYield, "iStarsUsed": len(listRows),
               "dictYieldWeighted": fdictWeighted(listRows), "listTopStars": listRows[:15],
               "dictByDistance": {s: fdictWeighted([r for r in listRows if f(r["fDistancePc"])])
                                  for s, f in (("<8 pc", lambda d: d < 8.0),
                                               ("8-15 pc", lambda d: 8.0 <= d < 15.0),
                                               (">=15 pc", lambda d: d >= 15.0))}}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: dictOut[k] for k in ("fYieldPlanning", "iStarsUsed",
                                              "dictYieldWeighted", "dictByDistance")}, indent=2))
    for r in listRows[:10]:
        print(f"{r['sName']:<16}{r['fDistancePc']:>6.1f} pc  EEC {r['fEec']:.3f}  "
              f"leak {r['fLeakShare']:.2f} exo {r['fExozodiShare']:.2f} "
              f"det {r['fDetectorShare']:.2f}  elasticity {r['fElasticity']:.3f}")


if __name__ == "__main__":
    main()
