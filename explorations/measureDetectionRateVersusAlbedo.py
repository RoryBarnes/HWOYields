#!/usr/bin/env python3
"""Compare the detection rate as a function of drawn albedo against Stark et al. (2024) Fig. 6.

The albedo penalty on expected yield is 4.7% here against a published 12%, and it has now
survived four distinct fixes: the visit model, phase-optimized characterization, Stark's own
per-visit-threshold draw method, and replacing the parametric coronagraph with the digitized
published curves. Attacking it through the survey total has stopped being informative, because a
single number cannot distinguish "this model's planets are too easy to detect" from "this model's
counted set is selected differently".

Stark publishes the shape that separates them. The right-hand panels of Fig. 6 give the
distribution of DETECTED planets against albedo alongside the distribution of INJECTED ones, and
Sec. 3.2 describes it: "The detection rate of planets with A_G > 0.2 is relatively flat, but
decreases linearly with albedo for A_G < 0.2. This explains the shift in the yield distribution:
many planets at the faint end of the albedo distribution will go undetected."

So the test is whether this model's detection rate falls off below A_G = 0.2 the way his does. A
flat curve would mean the planets that survive the characterization gate have so much detection
margin that halving their brightness changes nothing -- which is what the survey-level number
hints at but cannot establish. The allocation is read from the optimizer at the same exposure
times the survey actually bought, and the planets are re-injected with the seed the completeness
table used, so the tally is over the same population.
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
from yieldlib import survey as sv  # noqa: E402


def flistLoadStars(sCompletenessPath, sBox):
    """Rehydrate one box's per-star completeness curves, as the optimizer step does."""
    dictNpz = np.load(sCompletenessPath, allow_pickle=True)
    faTauGridS = dictNpz["faTauGridS"]
    faComp, faTauChar = dictNpz[f"faComp_{sBox}"], dictNpz[f"faTauChar_{sBox}"]
    faCharMean, faCompAlb = dictNpz[f"faTauCharMean_{sBox}"], dictNpz[f"faCompAlbedo_{sBox}"]
    return [dict(faTauGridS=faTauGridS, faComp=faComp[i], faCompYield=faCompAlb[i],
                 faTauCharMeanS=faCharMean[i], fTauCharS=float(faTauChar[i]))
            for i in range(faComp.shape[0])], faTauGridS


def flistAllocation(listStars, dictMission, fEtaEarth, fSlope, faTauGridS):
    """For each observed star, the visit count and exposure time the optimizer bought.

    The envelope pools every (visits, tau) option, so the chosen vertex is recovered by matching
    its cost back to the option that produced it rather than by carrying an index through.
    """
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    listOut = []
    for iStar, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
        if len(dictCurve["faCost"]) <= 1:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex <= 0:
            continue
        fCost = float(dictCurve["faCost"][iVertex])
        faComp2 = np.atleast_2d(dictStar["faComp"])
        faChar2 = np.atleast_2d(dictStar["faTauCharMeanS"])
        fBest, iBestK, iBestT = np.inf, 0, 0
        for k in range(faComp2.shape[0]):
            faCostK = (k + 1) * (fMult * faTauGridS + fOverhead)
            faCharTerm = np.where(faChar2[k] > 0.0, fMult * faChar2[k] + fOverhead, 0.0)
            faTotal = faCostK + fEtaEarth * faComp2[k] * faCharTerm
            iT = int(np.argmin(np.abs(faTotal - fCost)))
            if abs(faTotal[iT] - fCost) < fBest:
                fBest, iBestK, iBestT = abs(faTotal[iT] - fCost), k, iT
        listOut.append({"iStar": iStar, "iVisits": iBestK + 1,
                        "fTauS": float(faTauGridS[iBestT]),
                        "fCompleteness": float(dictCurve["faComp"][iVertex])})
    return listOut


def fdictTallyOneStar(dictRow, dictBox, dictParams, dictMission, iNumPlanets, iSeed, faEdges):
    """Counts of injected and detected planets per albedo bin for one star at its bought exposure."""
    rng = np.random.default_rng(iSeed)
    dictPlanets = cp.fdictInjectPlanets(dictBox, dictRow["fEeidAu"], iNumPlanets,
                                        dictParams["fAlpha"], dictParams["fBeta"], rng,
                                        dictRow["iVisits"])
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    dictRange = dictMission["dictAlbedoDistribution"]
    faDrawn = dictRange["fMin"] + dictPlanets["faAlbedo"] * (dictRange["fMax"] -
                                                            dictRange["fMin"])
    dictPlanets["faAlbedoDrawn"] = faDrawn
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    listChar = dictParams["dictBands"]["listBandsCharacterization"]
    faTauDet, faTauChar = cp.faDetectionTimes(dictRow, dictPlanets, dictGeom, listBandsDet,
                                              listChar, dictMission, iNumPlanets)
    fCap = dictMission["fExposureLimitS"]
    faBest, _, _ = cp.faCountedTimes(faTauDet, faTauChar, fCap, 1)
    faAtVisits = np.atleast_2d(faBest)[:, min(dictRow["iVisits"], faBest.shape[1]) - 1]
    bDetected = np.isfinite(faAtVisits) & (faAtVisits <= dictRow["fTauS"])
    iaInjected, _ = np.histogram(faDrawn, bins=faEdges)
    iaDetected, _ = np.histogram(faDrawn[bDetected], bins=faEdges)
    return iaInjected, iaDetected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--completeness-npz", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--top-stars", type=int, default=0,
                   help="0 tallies every observed star; a positive value takes the highest-"
                        "completeness ones, which biases the rate upward because those are the "
                        "saturated targets")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = dictParams["dictBands"]["listBandsCharacterization"]
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]

    listStars, faTauGridS = flistLoadStars(dictArgs["completeness_npz"], dictArgs["box"])
    dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    listAlloc = flistAllocation(listStars, dictMission, dictArgs["eta_earth"],
                                dictResult["fSlope"], faTauGridS)
    listAlloc.sort(key=lambda d: -d["fCompleteness"])
    if dictArgs["top_stars"] > 0:
        listAlloc = listAlloc[:dictArgs["top_stars"]]

    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    faEdges = np.linspace(dictMission["dictAlbedoDistribution"]["fMin"],
                          dictMission["dictAlbedoDistribution"]["fMax"], 13)
    iaInjected = np.zeros(len(faEdges) - 1, dtype=int)
    iaDetected = np.zeros(len(faEdges) - 1, dtype=int)
    for dictAlloc in listAlloc:
        dictRow = dfTargets.iloc[dictAlloc["iStar"]].to_dict()
        dictRow.update(dictAlloc)
        iaI, iaD = fdictTallyOneStar(dictRow, dictBox, dictParams, dictMission,
                                     dictArgs["num_planets"],
                                     dictArgs["seed"] + dictAlloc["iStar"], faEdges)
        iaInjected += iaI
        iaDetected += iaD

    faCentres = 0.5 * (faEdges[:-1] + faEdges[1:])
    faRate = np.where(iaInjected > 0, iaDetected / np.maximum(iaInjected, 1), np.nan)
    faAbove = faRate[faCentres > 0.2]
    faBelow = faRate[faCentres < 0.2]
    dictOut = {
        "iStarsTallied": len(listAlloc),
        "listBins": [{"fAlbedo": round(float(a), 4), "iInjected": int(b), "iDetected": int(c),
                      "fDetectionRate": round(float(d), 4)}
                     for a, b, c, d in zip(faCentres, iaInjected, iaDetected, faRate)],
        "fMeanRateAboveTwoTenths": round(float(np.nanmean(faAbove)), 4),
        "fMeanRateBelowTwoTenths": round(float(np.nanmean(faBelow)), 4),
        "fRatioBelowOverAbove": round(float(np.nanmean(faBelow) / np.nanmean(faAbove)), 4),
        "sPublished": "Stark et al. (2024) Sec. 3.2: flat above A_G = 0.2, falling linearly below",
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'A_G':>7}{'injected':>10}{'detected':>10}{'rate':>8}")
    for dictBin in dictOut["listBins"]:
        print(f"{dictBin['fAlbedo']:7.3f}{dictBin['iInjected']:10d}{dictBin['iDetected']:10d}"
              f"{dictBin['fDetectionRate']:8.3f}")
    print(f"\nmean rate above 0.2: {dictOut['fMeanRateAboveTwoTenths']}")
    print(f"mean rate below 0.2: {dictOut['fMeanRateBelowTwoTenths']}")
    print(f"ratio below/above  : {dictOut['fRatioBelowOverAbove']}")


if __name__ == "__main__":
    main()
