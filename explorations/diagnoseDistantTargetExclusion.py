#!/usr/bin/env python3
"""Ask why the optimizer gives distant stars nothing, in the currency the optimizer actually uses.

Completeness here collapses beyond about 10 pc while Stark et al. (2024) Fig. 11 sustains a
median of 0.25 out to 30 pc. The equal-slope rule admits a star only if some point on its
cost/completeness envelope has marginal return dC/dT above the survey-wide slope lambda, so the
question is not whether a distant star is detectable but whether it can ever clear that bar. This
reports lambda, and for representative stars the best marginal return they can offer, the
completeness they would reach, and what it would cost.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdictStarEconomics(dictStar, fEtaEarth, dictMission, fLambda):
    """Best marginal return a star can offer, and what the equal-slope rule does with it."""
    dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
    faCost, faComp = dictCurve["faCost"], dictCurve["faComp"]
    if faCost.size < 2:
        return {"fBestSlope": 0.0, "bAdmitted": False, "fAllocatedCompleteness": 0.0}
    faSlopes = np.diff(faComp) / np.diff(faCost)
    iVertex = int(np.searchsorted(-faSlopes, -fLambda, side="right"))
    return {
        "fBestSlopePerDay": float(np.max(faSlopes) * 86400.0),
        "fLambdaPerDay": float(fLambda * 86400.0),
        "bAdmitted": bool(iVertex > 0),
        "fAllocatedCompleteness": float(faComp[iVertex]) if iVertex > 0 else 0.0,
        "fAllocatedCostDays": float(faCost[iVertex] / 86400.0) if iVertex > 0 else 0.0,
        "fMaxCompleteness": float(faComp[-1]),
        "fCostForQuarterCompletenessDays": float(
            np.interp(0.25, faComp, faCost) / 86400.0) if faComp[-1] >= 0.25 else float("nan"),
    }


def fdictParseArgs():
    """Command-line configuration for the distant-target diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--num-planets", type=int, default=1500)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="distantTargetExclusion.json")
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
    dfAll = pd.read_csv(dictArgs["target_catalog"])
    dfTargets = sv.fdfScreenTargets(dfAll, dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    0.5, dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]), 150)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictOpt = opt.fdictOptimizeSurvey(listStars, 0.24, dictParams["dictMission"])
    faDistance = dfTargets["fDistancePc"].to_numpy()
    faLum = dfTargets["fLuminosityLsun"].to_numpy()
    dictRows = {}
    for fTarget in (5.0, 10.0, 15.0, 20.0, 25.0):
        bNear = (np.abs(faDistance - fTarget) < 2.0) & (faLum > 0.4) & (faLum < 2.0)
        if not bNear.any():
            continue
        i = int(np.where(bNear)[0][int(np.argmax(dictTable["faComp"][bNear][:, -1, -1]))])
        dictRows[f"{fTarget:.0f} pc"] = {
            "fDistancePc": float(faDistance[i]), "fLuminosityLsun": float(faLum[i]),
            **fdictStarEconomics(listStars[i], 0.24, dictParams["dictMission"],
                                 dictOpt["fSlope"])}
    dictOut = {"fLambdaPerDay": float(dictOpt["fSlope"] * 86400.0),
               "iStarsUsed": int(dictOpt["iStarsUsed"]),
               "fYield": float(dictOpt["fYield"]), "dictRows": dictRows}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"survey-wide slope lambda : {dictOut['fLambdaPerDay']:.4g} completeness per day")
    print(f"stars admitted           : {dictOut['iStarsUsed']}\n")
    print(f"{'star':>8}{'bestSlope/day':>15}{'admitted':>10}{'C alloc':>9}"
          f"{'cost d':>9}{'C max':>8}{'days for C=0.25':>17}")
    for sLabel, d in dictRows.items():
        print(f"{sLabel:>8}{d['fBestSlopePerDay']:>15.4g}{str(d['bAdmitted']):>10}"
              f"{d['fAllocatedCompleteness']:>9.3f}{d['fAllocatedCostDays']:>9.2f}"
              f"{d['fMaxCompleteness']:>8.3f}{d['fCostForQuarterCompletenessDays']:>17.2f}")


if __name__ == "__main__":
    main()
