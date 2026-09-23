#!/usr/bin/env python3
"""Profile where the completeness calculation spends its time, and whether it needs to.

The pipeline went from minutes to just over two hours when characterization began to be scheduled
at the best orbital phase. Three steps are 93% of that -- the calibration bisection, the
completeness table and the aperture scan -- and all three are the same inner loop, so the question
is what that loop costs and which part of it is avoidable.

Two candidates, both measured here rather than argued:

  * The albedo re-evaluation calls faDetectionTimes, which computes characterization times as well
    as detection times. Since Ref. stark2024 Sec. 3.2 fixes the characterization budget at the
    planning albedo -- and this pipeline now does too -- that second characterization calculation
    is discarded. If characterization dominates the cost, skipping it is close to a free halving.
  * Characterization is minimised over iCharacterizationPhaseSamples points around the orbit. The
    quantity being minimised is smooth in phase, so the minimum should converge well before the
    100 samples currently used. This measures the convergence directly, because guessing at it
    would trade accuracy for speed without knowing the exchange rate.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import completeness as cp  # noqa: E402


def fdictTimeOneStar(dictStar, dictBox, dictParams, dictMission, iNumPlanets, iSeed):
    """Wall-clock cost of one star's completeness, split into its detection and characterization parts."""
    rng = np.random.default_rng(iSeed)
    iVisits = int(dictMission.get("iMaxVisits", 1))
    dictPlanets = cp.fdictInjectPlanets(dictBox, dictStar["fEeidAu"], iNumPlanets,
                                        dictParams["fAlpha"], dictParams["fBeta"], rng, iVisits)
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    listDet = dictParams["dictBands"]["listBandsDetection"]
    listChar = dictParams["dictBands"]["listBandsCharacterization"]

    fStart = time.perf_counter()
    for dictBand in listDet:
        cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    fDetection = time.perf_counter() - fStart

    fStart = time.perf_counter()
    cp.faCharacterizationTimeAtBestPhase(dictStar, dictPlanets, listChar, dictMission,
                                         int(dictMission.get("iCharacterizationPhaseSamples", 100)))
    fCharacterization = time.perf_counter() - fStart
    return {"fDetectionSeconds": round(fDetection, 4),
            "fCharacterizationSeconds": round(fCharacterization, 4),
            "fCharacterizationShare": round(fCharacterization / (fDetection + fCharacterization),
                                            4)}


def fdictPhaseConvergence(dictStar, dictBox, dictParams, dictMission, iNumPlanets, iSeed,
                          listSamples):
    """Characterization time against the number of orbital phases sampled, relative to the finest."""
    rng = np.random.default_rng(iSeed)
    iVisits = int(dictMission.get("iMaxVisits", 1))
    dictPlanets = cp.fdictInjectPlanets(dictBox, dictStar["fEeidAu"], iNumPlanets,
                                        dictParams["fAlpha"], dictParams["fBeta"], rng, iVisits)
    listChar = dictParams["dictBands"]["listBandsCharacterization"]
    dictOut = {}
    faReference = None
    for iSamples in sorted(listSamples, reverse=True):
        fStart = time.perf_counter()
        faTau = cp.faCharacterizationTimeAtBestPhase(dictStar, dictPlanets, listChar,
                                                     dictMission, iSamples)
        fElapsed = time.perf_counter() - fStart
        if faReference is None:
            faReference = faTau
        bBoth = np.isfinite(faTau) & np.isfinite(faReference)
        faRatio = faTau[bBoth] / faReference[bBoth]
        fCap = dictMission["fExposureLimitS"]
        dictOut[str(iSamples)] = {
            "fSeconds": round(fElapsed, 4),
            "fMedianRatioToFinest": round(float(np.median(faRatio)), 5) if faRatio.size else None,
            "fMaxRatioToFinest": round(float(np.max(faRatio)), 5) if faRatio.size else None,
            "iCountedUnderCap": int(np.sum(np.isfinite(faTau) & (faTau <= fCap))),
        }
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--distance-pc", type=float, default=10.0)
    p.add_argument("--luminosity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = dictParams["dictBands"]["listBandsCharacterization"]
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    fLum = dictArgs["luminosity"]
    fTeff = 5772.0 * fLum ** 0.11
    dictStar = {"fTeffK": fTeff, "fLuminosityLsun": fLum,
                "fRadiusRsun": np.sqrt(fLum) / (fTeff / 5772.0) ** 2,
                "fDistancePc": dictArgs["distance_pc"], "fEeidAu": np.sqrt(fLum)}

    dictOut = {
        "dictCostSplit": fdictTimeOneStar(dictStar, dictBox, dictParams, dictMission,
                                          dictArgs["num_planets"], dictArgs["seed"]),
        "dictPhaseConvergence": fdictPhaseConvergence(
            dictStar, dictBox, dictParams, dictMission, dictArgs["num_planets"],
            dictArgs["seed"], [200, 100, 60, 40, 30, 24, 16, 12, 8]),
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut["dictCostSplit"], indent=2))
    print(f"\n{'phases':>8}{'seconds':>10}{'median/finest':>16}{'max/finest':>13}{'counted':>9}")
    for sKey in sorted(dictOut["dictPhaseConvergence"], key=int, reverse=True):
        dictRow = dictOut["dictPhaseConvergence"][sKey]
        print(f"{sKey:>8}{dictRow['fSeconds']:10.3f}{dictRow['fMedianRatioToFinest']:16.5f}"
              f"{dictRow['fMaxRatioToFinest']:13.4f}{dictRow['iCountedUnderCap']:9d}")


if __name__ == "__main__":
    main()
