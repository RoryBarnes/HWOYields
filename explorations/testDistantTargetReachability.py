#!/usr/bin/env python3
"""Can the model reach Stark's completeness on his 10-20 pc targets at all, and would it pay to?

Against Stark et al. (2024) Fig. 11 the model matches the nearest stars but gives the 10-20 pc
targets far less completeness (median 0.35 vs 0.73 at 10-15 pc). Changing single physics terms
(exozodi law, characterization phase, albedo method) barely moves this, because the calibration
refit absorbs each one. This asks the question directly, target by target, for every published
10-20 pc target with completeness >= --min-published, matched to the model star:

  reachability  the completeness the model can reach at all (six visits, 60-day cap, 3 zodis),
                and why the remaining planets are lost: never detectable within the cap, or
                detectable but with a spectrum beyond the cap;
  metric        detection-only completeness at the survey's own allocation, the like quantity if
                Fig. 11 plots detection completeness while the survey counts characterized planets;
  economics     the cheapest allocation (visits x exposure, characterization budgeted as the
                optimizer charges it) reaching Stark's completeness, and its benefit per unit time
                C / cost against the survey's marginal rate lambda at fixed exozodi. Ratio < 1
                means the optimizer would not pay for it.

Stars are re-derived with the pipeline's own seeds and calibration, so the planets are the ones
A04 used.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))

from yieldlib import completeness as cp  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fiMatch(faD, faL, dictPoint, fTolDex, fTolPc):
    """Index of the model star nearest a published target in distance and log luminosity."""
    faScore = ((faD - dictPoint["fDistancePc"]) / fTolPc) ** 2 + \
        ((np.log10(faL) - np.log10(dictPoint["fLuminosityLsun"])) / fTolDex) ** 2
    i = int(np.argmin(faScore))
    return i if faScore[i] <= 2.0 else -1


def fdictLossBreakdown(dictRates, dictParams, dictMission):
    """Fractions of planets counted, lost to detection, and lost to characterization only."""
    listBands = dictParams["dictBands"]["listBandsDetection"]
    fZodi = dictMission["fExozodiLevel"]
    faDet = cp.faRequiredExposureTime(dictRates["listDetRates"], listBands,
                                      listBands[0]["fSignalToNoise"], dictMission, fZodi)
    faChar = cp.faCharacterizationTimeFromRates(dictRates["listCharRates"],
                                                dictRates["listCharOptions"], dictMission,
                                                dictRates["bBestPhase"], dictRates["iVisits"],
                                                fZodi)
    fCap = dictMission["fExposureLimitS"]
    bDet = np.min(faDet, axis=1) <= fCap
    bChar = np.min(faChar, axis=1) <= fCap
    faCharDays = np.min(faChar, axis=1)[bDet] / 86400.0
    return {"fCounted": float(np.mean(bDet & bChar)),
            "fLostDetection": float(np.mean(~bDet)),
            "fLostCharacterizationOnly": float(np.mean(bDet & ~bChar)),
            "fMedianCharDaysOfDetectable": float(np.median(faCharDays)) if faCharDays.size
            else float("nan")}


def fdictCheapestReaching(dictStar, fTarget, fEta, dictMission):
    """Cheapest (visits, exposure) whose completeness reaches fTarget, with its cost and C/cost."""
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    faTau = dictStar["faTauGridS"]
    fBest = None
    for k in range(dictStar["faComp"].shape[0]):
        faC, faChar = dictStar["faComp"][k], dictStar["faTauCharMeanS"][k]
        faCost = (k + 1) * (fMult * faTau + fOverhead) + \
            fEta * faC * np.where(faChar > 0, fMult * faChar + fOverhead, 0.0)
        bOk = faC >= fTarget
        if bOk.any():
            j = int(np.argmin(np.where(bOk, faCost, np.inf)))
            if fBest is None or faCost[j] < fBest["fCostDays"] * 86400.0:
                fBest = {"iVisits": k + 1, "fExposureDays": float(faTau[j] / 86400.0),
                         "fCostDays": float(faCost[j] / 86400.0), "fComp": float(faC[j])}
    return fBest


def ffDetectionAtAllocation(dictCurves, dictSurvey, i, faTauGridS):
    """Detection-only completeness at the (visits, exposure) the survey actually allocated.

    If Stark's Fig. 11 colours stars by detection completeness while the survey itself is still
    driven by characterization-gated completeness, this, not the gated value, is the like
    quantity to compare with the published points.
    """
    iVisits = int(dictSurvey["faStarVisits"][i])
    if iVisits == 0:
        return 0.0
    iTau = int(np.argmin(np.abs(faTauGridS - dictSurvey["faStarTauS"][i])))
    return float(dictCurves["faCompDetectionOnly"][iVisits - 1][iTau])


def fdictStarRow(i, dfTargets, dictParams, dictMission, faTauGridS, dictArgs, fSlope):
    """Reachability and economics for one matched model star."""
    dictRow = dfTargets.iloc[i].to_dict()
    dictRow["fExozodiLevel"] = dictMission["fExozodiLevel"]
    dictBands = dictParams["dictBands"]
    dictRates = cp.fdictStarRates(dictRow, dictParams["dictBoxes"]["canonical"],
                                  dictBands["listBandsDetection"],
                                  dictBands["dictBandCharacterization"], dictMission,
                                  dictArgs["num_planets"], -0.19, 0.26, dictArgs["seed"] + i)
    dictCurves = cp.fdictCompletenessAtZodi(dictRates, dictMission["fExozodiLevel"],
                                            dictBands["listBandsDetection"], dictMission,
                                            faTauGridS, dictArgs["num_planets"])
    return dictRow, dictRates, dictCurves


def fdictParseArgs():
    """Command-line configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--completeness", default="../computeCompleteness/completeness.npz")
    p.add_argument("--digitised",
                   default="../CompareTargetCompleteness/starkFigure11Digitised.json")
    p.add_argument("--distance-range", default="10,20")
    p.add_argument("--min-published", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/distantTargetReachability.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = dictParams["dictBands"]["listBandsCharacterization"]
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    faTauGridS = dictNpz["faTauGridS"]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictParams["dictBoxes"]["canonical"])
    assert np.allclose(dfTargets["fDistancePc"].to_numpy(), dictNpz["faDistancePc"])
    faComp, faCompAlb = dictNpz["faComp_canonical"], dictNpz["faCompAlbedo_canonical"]
    faCharMean = dictNpz["faTauCharMean_canonical"]
    listStars = [dict(faTauGridS=faTauGridS, faComp=faComp[i], faCompYield=faCompAlb[i],
                      faTauCharMeanS=faCharMean[i]) for i in range(len(dfTargets))]
    dictSurvey = opt.fdictOptimizeSurvey(listStars, dictArgs["eta"], dictMission, bPerStar=True)
    fSlope = dictSurvey["fSlope"]
    fLo, fHi = (float(s) for s in dictArgs["distance_range"].split(","))
    with open(dictArgs["digitised"]) as oFile:
        listStark = [d for d in json.load(oFile)["listPoints"]
                     if fLo <= d["fDistancePc"] < fHi
                     and d["fCompleteness"] >= dictArgs["min_published"]]
    faD, faL = dictNpz["faDistancePc"], dictNpz["faLuminosityLsun"]
    listRows = []
    for dictPoint in listStark:
        i = fiMatch(faD, faL, dictPoint, 0.12, 1.2)
        if i < 0:
            continue
        dictRow, dictRates, dictCurves = fdictStarRow(i, dfTargets, dictParams, dictMission,
                                                      faTauGridS, dictArgs, fSlope)
        dictStar = dict(faTauGridS=faTauGridS, faComp=dictCurves["faComp"],
                        faTauCharMeanS=dictCurves["faTauCharMeanS"])
        dictCheap = fdictCheapestReaching(dictStar, dictPoint["fCompleteness"], dictArgs["eta"],
                                          dictMission)
        listRows.append({
            "sName": str(dictRow.get("sSimbadName", "")), "fDistancePc": float(faD[i]),
            "fLuminosityLsun": float(faL[i]),
            "fCompletenessPublished": dictPoint["fCompleteness"],
            "fCompletenessAllocated": float(dictSurvey["faStarComp"][i]),
            "fCompletenessMaxReachable": float(dictCurves["faComp"].max()),
            "fCompletenessMaxDetectionOnly": float(dictCurves["faCompDetectionOnly"].max()),
            "fCompletenessDetectionAtAllocation": ffDetectionAtAllocation(
                dictCurves, dictSurvey, i, faTauGridS),
            **fdictLossBreakdown(dictRates, dictParams, dictMission),
            "dictCheapestReachingPublished": dictCheap,
            "fBenefitOverMarginalRate": (dictCheap["fComp"] / (dictCheap["fCostDays"] * 86400.0)
                                         / fSlope) if dictCheap else None})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in listRows[-1].items()
                          if k != "dictCheapestReachingPublished"}), flush=True)
    dictOut = {"fMarginalRatePerDay": fSlope * 86400.0, "iTargets": len(listRows),
               "fYieldPlanningFixedExozodi": dictSurvey["fYieldPlanning"],
               "dictSummary": fdictSummary(listRows), "listRows": listRows}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut["dictSummary"], indent=2))


def fdictSummary(listRows):
    """How many targets are unreachable, reachable but unaffordable, or reachable and paid for."""
    iUnreach = sum(r["dictCheapestReachingPublished"] is None for r in listRows)
    faRatio = np.array([r["fBenefitOverMarginalRate"] for r in listRows
                        if r["fBenefitOverMarginalRate"] is not None])
    return {"iTargets": len(listRows), "iCannotReachPublished": iUnreach,
            "iReachableButBelowMarginalRate": int(np.sum(faRatio < 1.0)),
            "iReachableAndAffordable": int(np.sum(faRatio >= 1.0)),
            "fMedianBenefitOverMarginalRate": float(np.median(faRatio)) if faRatio.size else None,
            "fMedianPublished": float(np.median([r["fCompletenessPublished"] for r in listRows])),
            "fMedianAllocated": float(np.median([r["fCompletenessAllocated"] for r in listRows])),
            "fMedianDetectionAtAllocation": float(np.median(
                [r["fCompletenessDetectionAtAllocation"] for r in listRows])),
            "fMedianMaxReachable": float(np.median([r["fCompletenessMaxReachable"]
                                                    for r in listRows])),
            "fMedianLostDetection": float(np.median([r["fLostDetection"] for r in listRows])),
            "fMedianLostCharacterizationOnly": float(np.median(
                [r["fLostCharacterizationOnly"] for r in listRows]))}


if __name__ == "__main__":
    main()
