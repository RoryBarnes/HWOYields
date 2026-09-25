#!/usr/bin/env python3
"""Is each tested lever just a uniform exposure-time multiplier that the calibration kappa undoes?

κ multiplies the throughput of the planet and of every astrophysical background alike, so in the
background-limited regime it rescales every exposure time by 1/κ. A lever that also rescales
every time by a near-constant factor is then indistinguishable from κ once κ is refitted to the
published yield. For each variant of whatIfLeakAndSkyThroughput.py, at its OWN refitted κ, this
computes every planet's detection time (first visit) and best-phase characterization time on the
same injected planets as the base variant, and reports the ratio variant/base: its median, its
scatter, and its trend with stellar distance and luminosity. A flat ratio of 1 means the lever
and κ cancel exactly; only the trend and scatter can change which targets the survey picks.
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
from whatIfLeakAndSkyThroughput import LIST_VARIANTS, fdictVariantMission  # noqa: E402
from yieldlib import completeness as cp  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdictMissionFor(dictBase, sName, dictShape, fKappa):
    """The variant's mission dict at its refitted calibration, with the characterization options."""
    dictParams = json.loads(json.dumps(dictBase))
    dictMission = dictParams["dictMission"]
    dictMission.update(fdictVariantMission(sName, dictShape))
    dictMission["fThroughputCalibration"] = fKappa
    dictMission["listBandsCharacterization"] = dictParams["dictBands"]["listBandsCharacterization"]
    return dictParams, dictMission


def ftStarTimes(dictStar, dictParams, dictMission, dictArgs, iSeed):
    """Detection (first visit) and characterization times for one star's injected planets."""
    dictBands = dictParams["dictBands"]
    dictRates = cp.fdictStarRates(dictStar, dictParams["dictBoxes"]["canonical"],
                                  dictBands["listBandsDetection"],
                                  dictBands["dictBandCharacterization"], dictMission,
                                  dictArgs["num_planets"], -0.19, 0.26, iSeed)
    fZodi = dictMission["fExozodiLevel"]
    faDet = cp.faRequiredExposureTime(dictRates["listDetRates"], dictBands["listBandsDetection"],
                                      7.0, dictMission, fZodi)[:, 0]
    faChar = cp.faCharacterizationTimeFromRates(dictRates["listCharRates"],
                                                dictRates["listCharOptions"], dictMission,
                                                dictRates["bBestPhase"], 1, fZodi)[:, 0]
    return faDet, faChar


def fdictAllTimes(dfTargets, dictParams, dictMission, dictArgs):
    """Times for every screened star, concatenated, with each planet's star distance and L."""
    listDet, listChar, listD, listL = [], [], [], []
    for i, dictStar in enumerate(dfTargets.to_dict("records")):
        faDet, faChar = ftStarTimes(dictStar, dictParams, dictMission, dictArgs,
                                    dictArgs["seed"] + i)
        listDet.append(faDet)
        listChar.append(faChar)
        listD.append(np.full(faDet.size, dictStar["fDistancePc"]))
        listL.append(np.full(faDet.size, dictStar["fLuminosityLsun"]))
    return {k: np.concatenate(v) for k, v in
            (("faDet", listDet), ("faChar", listChar), ("faD", listD), ("faL", listL))}


def fdictRatioStats(faVar, faBase, faD, faL, fCap):
    """log10(variant/base) over planets finite in both and under the cap in the base."""
    bUse = np.isfinite(faVar) & np.isfinite(faBase) & (faBase <= fCap)
    faLog = np.log10(faVar[bUse] / faBase[bUse])
    return {"iPlanets": int(bUse.sum()), "fMedianRatio": float(10 ** np.median(faLog)),
            "fLogStd": float(np.std(faLog)),
            "fSlopePerDexDistance": float(np.polyfit(np.log10(faD[bUse]), faLog, 1)[0]),
            "fSlopePerDexLuminosity": float(np.polyfit(np.log10(faL[bUse]), faLog, 1)[0]),
            "faPercentiles5_95": [float(10 ** np.percentile(faLog, q)) for q in (5, 95)]}


def fdictCapCrossings(faVar, faBase, fCap):
    """Planets whose characterization crosses the two-month cap in either direction."""
    bBase, bVar = faBase <= fCap, faVar <= fCap
    return {"iLost": int(np.sum(bBase & ~bVar)), "iGained": int(np.sum(~bBase & bVar)),
            "iUnderCapBase": int(bBase.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--sky-shape", default="output/skyThroughputConvolutionTest.json")
    p.add_argument("--whatif-dir", default="output/whatIfLeakAndSkyThroughput")
    p.add_argument("--num-planets", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/leverTimeMultipliers.json")
    dictArgs = vars(p.parse_args())
    dictBase = json.load(open(dictArgs["mission_parameters"]))
    dictShape = json.load(open(dictArgs["sky_shape"]))["dictEmpiricalShape"]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictBase["dictMission"],
                                    dictBase["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBase["dictBoxes"]["canonical"])
    dictTimes = {}
    for sName in LIST_VARIANTS:
        fKappa = json.load(open(os.path.join(dictArgs["whatif_dir"], f"{sName}.json")))[
            "fCalibration"]
        dictParams, dictMission = fdictMissionFor(dictBase, sName, dictShape, fKappa)
        dictTimes[sName] = fdictAllTimes(dfTargets, dictParams, dictMission, dictArgs)
        print(f"computed {sName} (kappa {fKappa:.3f})", flush=True)
    fCap = cp.ffScienceTimeCap(dictBase["dictMission"])
    dictB = dictTimes["base"]
    dictOut = {s: {"dictDetection": fdictRatioStats(d["faDet"], dictB["faDet"], dictB["faD"],
                                                     dictB["faL"], fCap),
                   "dictCharacterization": fdictRatioStats(d["faChar"], dictB["faChar"],
                                                           dictB["faD"], dictB["faL"], fCap),
                   "dictCharCapCrossings": fdictCapCrossings(d["faChar"], dictB["faChar"], fCap)}
               for s, d in dictTimes.items() if s != "base"}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=1))


if __name__ == "__main__":
    main()
