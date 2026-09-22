#!/usr/bin/env python3
"""Measure the slope of C(tau) at the operating point and decompose the spread in exposure time.

Both observational biases modelled here -- albedo and exozodi -- come out three to four times
weaker than Stark et al. report. Both act by lengthening a target's required exposure, so their
strength is set by how fast completeness falls when exposure requirements rise, which is
dC/dln(tau) at the allocation. A uniform albedo draw on 0.08 < A_G < 0.32 shifts E[ln tau] by
only +0.067, so reproducing Stark's 12 percent yield loss at C ~ 0.5 needs dC/dln(tau) ~ 0.9.
This measures the actual slope, and decomposes the scatter in ln(tau) into the phase function,
the coronagraph throughput at the planet's separation, and the planet radius, to show which one
is flattening the curve.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import completeness as cp  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fnSlopeAtAllocation(faTauGridS, faComp, fTauAllocated):
    """dC/dln(tau) evaluated at the allocated exposure time."""
    faLn = np.log(faTauGridS)
    faSlope = np.gradient(faComp, faLn)
    return float(np.interp(np.log(fTauAllocated), faLn, faSlope))


def fdictSpreadDecomposition(dictStar, dictBox, dictParams, iNumPlanets, iSeed):
    """Standard deviation of ln(tau) with each source of scatter frozen in turn."""
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    listBands = dictParams["dictBands"]["listBandsDetection"]
    rng = np.random.default_rng(iSeed)
    dictPlanets = cp.fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                        iNumPlanets, dictParams["fAlpha"], dictParams["fBeta"],
                                        rng, iVisits=1)
    dictOut = {}
    for sLabel, dictFrozen in (
            ("full", {}),
            ("phase frozen at quadrature", {"faPhase": 1.0 / np.pi}),
            ("radius frozen at 1 Rearth", {"faRadiusEarth": 1.0}),
            ("separation frozen at EEID", {"faAxisScaled": 1.0})):
        dictP = dict(dictPlanets)
        if "faRadiusEarth" in dictFrozen:
            dictP["faRadiusEarth"] = np.full(iNumPlanets, dictFrozen["faRadiusEarth"])
        if "faAxisScaled" in dictFrozen:
            dictP["faAxisScaled"] = np.full(iNumPlanets, 1.0)
            dictP["faAxisAu"] = np.full(iNumPlanets, np.sqrt(dictStar["fLuminosityLsun"]))
        dictGeom = cp.fdictProjectOrbits(dictP)
        if "faPhase" in dictFrozen:
            dictGeom["faPhase"] = np.full_like(dictGeom["faPhase"], dictFrozen["faPhase"])
        faTau, _ = cp.faDetectionTimes(dictStar, dictP, dictGeom, listBands,
                                       [dictParams["dictBands"]["dictBandCharacterization"]],
                                       dictMission, iNumPlanets)
        faFinite = faTau[np.isfinite(faTau)]
        dictOut[sLabel] = {"fStdLnTau": float(np.std(np.log(faFinite))) if faFinite.size else 0.0,
                           "fDetectableFraction": float(faFinite.size / faTau.size)}
    return dictOut


def fdictNoiseRegime(dictStar, dictBox, dictParams, iNumPlanets, iSeed):
    """Planet counts against background counts for the planets that are actually detectable.

    Exposure time goes as CR_p / CR_p^2 = 1/CR_p when the planet's own photons dominate, and as
    CR_b / CR_p^2 when the background does. The second doubles the scatter in ln(tau) for a given
    scatter in flux, which halves dC/dln(tau) and so halves the strength of every observational
    bias built on top of it. Which regime the model sits in is therefore not a detail.
    """
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    listBands = dictParams["dictBands"]["listBandsDetection"]
    rng = np.random.default_rng(iSeed)
    dictPlanets = cp.fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                        iNumPlanets, dictParams["fAlpha"], dictParams["fBeta"],
                                        rng, iVisits=1)
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, listBands[0], dictMission)
    faTau, _ = cp.faDetectionTimes(dictStar, dictPlanets, dictGeom, listBands,
                                   [dictParams["dictBands"]["dictBandCharacterization"]],
                                   dictMission, iNumPlanets)
    bOk = np.isfinite(faTau)
    if not bOk.any():
        return {}
    faPlanet = dictRates["faPlanet"][bOk]
    faLeak = dictRates["faLeak"][bOk]
    faZodi = np.full_like(faPlanet, dictRates["fZodi"])
    faExo = dictRates["faExozodi"][bOk]
    faBack = faLeak + faZodi + faExo
    return {"fMedianPlanetOverBackground": float(np.median(faPlanet / faBack)),
            "fFractionPhotonLimited": float(np.mean(faPlanet > 2.0 * faBack)),
            "fMedianLeakShare": float(np.median(faLeak / faBack)),
            "fMedianZodiShare": float(np.median(faZodi / faBack)),
            "fMedianExozodiShare": float(np.median(faExo / faBack))}


def fdictParseArgs():
    """Command-line configuration for the completeness-slope diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--num-stars", type=int, default=120)
    p.add_argument("--num-planets", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="completenessSlope.json")
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
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    0.5, dictArgs["num_stars"], 2500.0, 7500.0, dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]), 200)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictOpt = opt.fdictOptimizeSurvey(listStars, 0.24, dictParams["dictMission"])
    faSlopes, faRelative, faCompAt, faWeight = [], [], [], []
    for i, dictStar in enumerate(listStars):
        faComp = dictTable["faComp"][i, -1]
        if faComp[-1] <= 0.02:
            continue
        fTauAt = float(np.interp(0.5 * faComp[-1], faComp, faTauGridS))
        fSlope = fnSlopeAtAllocation(faTauGridS, faComp, fTauAt)
        fCompHere = 0.5 * faComp[-1]
        faSlopes.append(fSlope)
        faRelative.append(fSlope / fCompHere if fCompHere > 0 else 0.0)
        faCompAt.append(fCompHere)
        faWeight.append(fCompHere)
    dictOut = {
        "iStars": len(faSlopes),
        "fMedianSlopeAtHalfCompleteness": float(np.median(faSlopes)),
        "fMedianRelativeSlope": float(np.median(faRelative)),
        "fCompletenessWeightedRelativeSlope": float(
            np.average(faRelative, weights=faWeight)),
        "fRelativeSlopeNeededForStarkAlbedo": 0.12 / 0.067,
        "sRelativeSlopeNote": "Stark's albedo draw shifts E[ln tau] by +0.067 and costs 12 "
                              "percent of the yield, so the completeness-weighted relative "
                              "slope (dC/dln tau)/C must be about 1.8 per ln unit. The relative "
                              "form is the comparable one because the published loss is "
                              "relative; comparing absolute slopes across stars of very "
                              "different maximum completeness is not meaningful.",
        "fSlopeNeededForStarkAlbedo": 0.9,
        "fMedianCompletenessThere": float(np.median(faCompAt)),
        "dictSpread": fdictSpreadDecomposition(dfTargets.to_dict("records")[0], dictBox,
                                               dictParams, dictArgs["num_planets"],
                                               dictArgs["seed"]),
        "fOptimizedYield": float(dictOpt["fYield"]),
        "dictNoiseRegime": {
            sLabel: fdictNoiseRegime(dfTargets.to_dict("records")[i], dictBox, dictParams,
                                     dictArgs["num_planets"], dictArgs["seed"])
            for sLabel, i in (("widest HZ star", 0),
                              ("best-completeness star",
                               int(np.argmax(dictTable["faComp"][:, -1, -1]))))},
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"median dC/dln(tau) at half-completeness : "
          f"{dictOut['fMedianSlopeAtHalfCompleteness']:.3f}")
    print(f"median (dC/dln tau)/C                    : "
          f"{dictOut['fMedianRelativeSlope']:.2f}")
    print(f"completeness-weighted (dC/dln tau)/C     : "
          f"{dictOut['fCompletenessWeightedRelativeSlope']:.2f}")
    print(f"relative slope implied by Stark's albedo : "
          f"{dictOut['fRelativeSlopeNeededForStarkAlbedo']:.2f}")
    print("\nnoise regime (detectable planets):")
    for sLabel, d in dictOut["dictNoiseRegime"].items():
        if not d:
            continue
        print(f"  {sLabel:24} CR_p/CR_b median {d['fMedianPlanetOverBackground']:7.3f}   "
              f"photon-limited {100*d['fFractionPhotonLimited']:5.1f}%   "
              f"leak/zodi/exo {d['fMedianLeakShare']:.2f}/{d['fMedianZodiShare']:.2f}/"
              f"{d['fMedianExozodiShare']:.2f}")
    print("\nspread in ln(tau), one source frozen at a time (widest-HZ star):")
    for sLabel, d in dictOut["dictSpread"].items():
        print(f"  {sLabel:32} sd {d['fStdLnTau']:6.2f}   detectable "
              f"{100*d['fDetectableFraction']:5.1f}%")


if __name__ == "__main__":
    main()
