#!/usr/bin/env python3
"""What if spectra were not boosted by the throughput calibration, and/or taken at the detection phase?

The throughput calibration (1.62) multiplies every band, spectra included. Against Stark et al.
(2024) Sec. 5.2 the model's characterization times are too SHORT for the easiest targets
(priority-first 18 EECs: 12.6 d vs 22 d at 6 m) and far too long for distant ones, and removing
the calibration from spectra alone accounts for much of the first gap (17.4 d uncalibrated). If
the factor is really compensating a detection-side deficit, applying it to spectra is a
compensating error that also prices distant targets out of the survey. For each variant this
re-fits the calibration to the published 22.5 at fixed exozodi (as A03 does), then reports: the
factor; the drawn-exozodi and albedo-drawn yields at eta = 0.24 (the fixed-eta level Stark's
Fig. 10 red curve has at 17.35); the exozodi penalty; the mean characterization time of the
priority-first 18 EECs at 6 m and, with the factor frozen, at 9 m (Stark's 22 d and 3.5 d); and
the allocated completeness by luminosity, for comparison with Fig. 11.

The skyFollowsCore variants make the extended-source throughput T_sky follow the core-throughput
curve instead of holding its large-separation value (Stark et al. 2019 Eqs. 5-6 define it as a
map). If the calibration is compensating for background overcharged near the inner working angle,
these variants should need a factor nearer unity AND characterize distant targets faster.
exozodiRadial adds the exozodi surface brightness at each planet's separation (fExozodiRadialIndex);
detectionPhase takes each spectrum at the detection epoch instead of the best orbital phase.
albedoPerVisit* use Stark (2024) Sec. 3.2's per-visit detectability test for drawn albedos.
starGate applies the two-month limit to the star's probabilistic characterization time
eta * C * <t_c> (Stark 2015) instead of to each planet.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))
sys.path.insert(0, S_HERE)

from checkCharacterizationTimeAgainstPublished import (faStarCharacterizationTimes,  # noqa: E402
                                                       fnMeanCharOfFirstN)
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402

DICT_VARIANTS = {
    "current": {"bCalibrateSpectra": True, "bBestPhase": True},
    "skyFollowsCore": {"bCalibrateSpectra": True, "bBestPhase": True, "bSkyFollowsCore": True},
    "exozodiRadial": {"bCalibrateSpectra": True, "bBestPhase": True, "bSkyFollowsCore": True,
                      "fExozodiRadialIndex": 2.34},
    "albedoPerVisit": {"bCalibrateSpectra": True, "bBestPhase": True, "bSkyFollowsCore": True,
                       "sAlbedoMethod": "perVisitThreshold"},
    "albedoPerVisitBrightBound": {"bCalibrateSpectra": True, "bBestPhase": True,
                                  "bSkyFollowsCore": True, "sAlbedoMethod": "perVisitThreshold",
                                  "bAlbedoBrightBound": True},
    "starGate": {"bCalibrateSpectra": True, "bBestPhase": True, "bSkyFollowsCore": True,
                  "sCharacterizationGate": "star"},
    "detectionPhase": {"bCalibrateSpectra": True, "bBestPhase": False, "bSkyFollowsCore": True},
    "skyFollowsCoreSpectraUncalibrated": {"bCalibrateSpectra": False, "bBestPhase": True,
                                          "bSkyFollowsCore": True},
    "spectraUncalibrated": {"bCalibrateSpectra": False, "bBestPhase": True},
    "spectraUncalibratedDetectionPhase": {"bCalibrateSpectra": False, "bBestPhase": False},
}


def fdictVariantParams(dictBase, dictVariant):
    """Deep copy with the variant's spectrum calibration and characterization phase applied."""
    dictParams = json.loads(json.dumps(dictBase))
    dictBands = dictParams["dictBands"]
    for dictBand in [dictBands["dictBandCharacterization"]] + dictBands["listBandsCharacterization"]:
        dictBand["bApplyThroughputCalibration"] = dictVariant["bCalibrateSpectra"]
    dictParams["dictMission"]["bOptimizeCharacterizationPhase"] = dictVariant["bBestPhase"]
    dictParams["dictMission"]["bSkyThroughputFollowsCore"] = dictVariant.get("bSkyFollowsCore",
                                                                            False)
    dictParams["dictMission"]["fExozodiRadialIndex"] = dictVariant.get("fExozodiRadialIndex", 0.0)
    dictParams["dictMission"]["sAlbedoMethod"] = dictVariant.get("sAlbedoMethod", "recompute")
    dictParams["dictMission"]["sCharacterizationGate"] = dictVariant.get("sCharacterizationGate",
                                                                         "planet")
    dictParams["dictMission"]["bAlbedoBrightBound"] = dictVariant.get("bAlbedoBrightBound", False)
    return dictParams


def fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, bDrawExozodi):
    """Build the table, optimize, and return the optimizer result with its star list."""
    dictParams = json.loads(json.dumps(dictParams))
    dictParams["dictMission"]["bDrawExozodiLevels"] = bDrawExozodi
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    return opt.fdictOptimizeSurvey(listStars, 0.24, dictParams["dictMission"]), listStars


def ffCalibrate(dfTargets, dictParams, dictBox, faTauGridS, dictArgs):
    """Bisect the throughput factor in log space to the target planning yield at fixed exozodi."""
    fLo, fHi = 0.3, 6.0
    for _ in range(dictArgs["bisection_steps"]):
        fMid = float(np.sqrt(fLo * fHi))
        dictParams["dictMission"]["fThroughputCalibration"] = fMid
        fYield = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs,
                             False)[0]["fYieldPlanning"]
        fLo, fHi = (fMid, fHi) if fYield < dictArgs["target_yield"] else (fLo, fMid)
        print(f"    factor {fMid:.3f} -> planning yield {fYield:.2f}", flush=True)
    return float(np.sqrt(fLo * fHi))


def fdictCompletenessByLuminosity(dfTargets, listStars, dictResult, dictMission):
    """Summed allocated completeness and stars used, by stellar luminosity."""
    faL = dfTargets["fLuminosityLsun"].to_numpy()
    listRows = faStarCharacterizationTimes(listStars, dictMission, 0.24, dictResult["fSlope"])
    faC = np.zeros(len(listStars))
    for dictRow in listRows:
        faC[dictRow["iStar"]] = dictRow["fCompleteness"]
    return {f"{fLo:g}-{fHi:g}": {"fSummed": float(faC[(faL >= fLo) & (faL < fHi)].sum()),
                                 "iUsed": int(np.sum((faC > 0) & (faL >= fLo) & (faL < fHi)))}
            for fLo, fHi in ((0, 0.6), (0.6, 1), (1, 2), (2, 5), (5, 100))}


def fdictAtNineMetres(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs):
    """Planning yield and first-18 characterization time at 9 m with the 6 m factor frozen."""
    dictParams = json.loads(json.dumps(dictParams))
    dictParams["dictMission"]["fDiameterM"] = 9.0
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBox)
    dictFixed, listStars = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, False)
    listRows = faStarCharacterizationTimes(listStars, dictParams["dictMission"], 0.24,
                                           dictFixed["fSlope"])
    return {"fPlanningFixedExozodi9m": float(dictFixed["fYieldPlanning"]),
            "fMeanCharDaysPriorityFirst18_9m": float(fnMeanCharOfFirstN(listRows, 0.24, 18,
                                                                         "priority"))}


def fdictRunVariant(dfTargets, dictBase, dictBox, faTauGridS, dictArgs, dictVariant):
    """Calibrate one variant and measure everything the docstring lists."""
    dictParams = fdictVariantParams(dictBase, dictVariant)
    dictParams["dictMission"]["fThroughputCalibration"] = 1.0
    fUncalibrated = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs,
                                False)[0]["fYieldPlanning"]
    fCal = ffCalibrate(dfTargets, dictParams, dictBox, faTauGridS, dictArgs)
    dictParams["dictMission"]["fThroughputCalibration"] = fCal
    dictFixed, _ = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, False)
    dictDrawn, listStars = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, True)
    listRows = faStarCharacterizationTimes(listStars, dictParams["dictMission"], 0.24,
                                           dictDrawn["fSlope"])
    dictNine = fdictAtNineMetres(dictArgs["catalog"], dictParams, dictBox, faTauGridS, dictArgs)
    return {**dictNine, "fUncalibratedPlanning": float(fUncalibrated),
            "fCalibration": fCal, "fPlanningFixedExozodi": float(dictFixed["fYieldPlanning"]),
            "fPlanningDrawnExozodi": float(dictDrawn["fYieldPlanning"]),
            "fFixedEtaYield": float(dictDrawn["fYield"]),
            "fExozodiPenalty": float(1.0 - dictDrawn["fYieldPlanning"] / dictFixed["fYieldPlanning"]),
            "fMeanCharDaysPriorityFirst18": float(fnMeanCharOfFirstN(listRows, 0.24, 18, "priority")),
            "iStarsUsed": int(dictDrawn["iStarsUsed"]),
            "dictByLuminosity": fdictCompletenessByLuminosity(dfTargets, listStars, dictDrawn,
                                                              dictParams["dictMission"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--variants", default=",".join(DICT_VARIANTS))
    p.add_argument("--target-yield", type=float, default=22.5)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--bisection-steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/whatIfCharacterizationTreatment.json")
    p.add_argument("--merge", action="store_true", help="keep variants already in --out-json")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["mission_parameters"]) as oFile:
        dictBase = json.load(oFile)
    dictBase["fAlpha"], dictBase["fBeta"] = -0.19, 0.26
    dictBox = dictBase["dictBoxes"]["canonical"]
    faTauGridS = np.logspace(1.0, np.log10(dictBase["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dictArgs["catalog"] = pd.read_csv(dictArgs["target_catalog"])
    dfTargets = sv.fdfScreenTargets(dictArgs["catalog"], dictBase["dictMission"],
                                    dictBase["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBox)
    dictOut = {}
    if dictArgs["merge"] and os.path.exists(dictArgs["out_json"]):
        with open(dictArgs["out_json"]) as oFile:
            dictOut = json.load(oFile)
    for sName in dictArgs["variants"].split(","):
        print(f"== {sName}", flush=True)
        dictOut[sName] = fdictRunVariant(dfTargets, dictBase, dictBox, faTauGridS, dictArgs,
                                         DICT_VARIANTS[sName])
        print(json.dumps(dictOut[sName]), flush=True)
        with open(dictArgs["out_json"], "w") as oFile:
            json.dump(dictOut, oFile, indent=2)


if __name__ == "__main__":
    main()
