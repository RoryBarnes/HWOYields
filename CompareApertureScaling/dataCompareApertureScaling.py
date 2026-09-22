#!/usr/bin/env python3
"""Predict P25 against telescope aperture and compare with Stark et al. (2024) Fig. 15.

The coronagraph throughput calibration is fitted at 6 m and held FROZEN across every diameter,
so the 7, 8 and 9 m points are out-of-sample predictions rather than fits. Two curves are
produced: including eta_Earth uncertainty (marginalized over the occurrence posterior) and
excluding it (eta fixed at the baseline 0.24, Poisson counting only).

A caution recorded before the comparison was run: the excluding-sigma curve is NOT expected to
match. Stark's excluding-sigma distribution still carries albedo and exozodi uncertainty, which
together pull the expected yield from 22.5 down to 17.3 and are why their 6 m point sits at
~6 percent rather than the ~30 percent that Poisson counting on 22.5 alone would give. This
pipeline models neither source, so it should overshoot that curve. The including-sigma curve is
the meaningful test, because eta_Earth uncertainty dominates it.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402

DICT_PUBLISHED = {
    "sSource": "Stark et al. (2024) Fig. 15; the 6 m pair is also stated in Sec. 3.5 text",
    "dictIncludingSigmaEta": {"6": 0.32, "7": 0.53, "8": 0.67, "9": 0.78},
    "dictExcludingSigmaEta": {"6": 0.06, "7": 0.49, "8": 0.89, "9": 0.995},
    "saTextStated": ["6"],
    "sReadingCaveat": "Values at 7, 8 and 9 m were read off the published figure and carry "
                      "roughly +/-2-3 percentage points of reading error.",
}


def fdictYieldCurveAtDiameter(dfCatalog, dictParams, fDiameterM, faTauGridS, faEtaGrid,
                              dictArgs):
    """Screen, compute completeness and optimize the survey at one aperture."""
    dictParams["dictMission"]["fDiameterM"] = fDiameterM
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"],
                                    dictParams["dictBoxes"]["canonical"])
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams,
                                          dictParams["dictBoxes"]["canonical"], faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    faYields = np.array([opt.fdictOptimizeSurvey(listStars, float(f),
                                                 dictParams["dictMission"])["fYield"]
                         for f in faEtaGrid])
    return {"iStarsScreened": int(len(dfTargets)), "faYieldGrid": faYields}


def fdictProbabilities(faEtaGrid, faYields, faEtaPosterior, fEtaBaseline, fGoal, rng):
    """P25 with and without eta_Earth uncertainty, both including Poisson counting."""
    faExpectedMarginal = np.interp(np.log(np.clip(faEtaPosterior, faEtaGrid[0], faEtaGrid[-1])),
                                   np.log(faEtaGrid), faYields)
    faObservedMarginal = rng.poisson(np.maximum(faExpectedMarginal, 0.0))
    fExpectedFixed = float(np.interp(np.log(fEtaBaseline), np.log(faEtaGrid), faYields))
    faObservedFixed = rng.poisson(fExpectedFixed, size=faEtaPosterior.size)
    return {
        "fExpectedYieldAtBaselineEta": fExpectedFixed,
        "fMeanExpectedMarginal": float(np.mean(faExpectedMarginal)),
        "fProbability25IncludingSigmaEta": float(np.mean(faObservedMarginal >= fGoal)),
        "fProbability25ExcludingSigmaEta": float(np.mean(faObservedFixed >= fGoal)),
    }


def fnCrossingDiameter(faDiameters, faIncluding, faExcluding):
    """Diameter where the excluding-sigma curve overtakes the including-sigma curve.

    Stark et al. (2024) Fig. 15 shows this crossing near 7.1 m. It is set by how
    characterization time trades against search time as yields rise, so reproducing it tests the
    mechanism rather than a normalisation.
    """
    faDifference = np.asarray(faExcluding) - np.asarray(faIncluding)
    faSignChange = np.where(np.diff(np.sign(faDifference)) != 0)[0]
    if faSignChange.size == 0:
        return None
    i = int(faSignChange[0])
    fSlope = faDifference[i + 1] - faDifference[i]
    if fSlope == 0:
        return float(faDiameters[i])
    return float(faDiameters[i] - faDifference[i] * (faDiameters[i + 1] - faDiameters[i])
                 / fSlope)


def fdictParseArgs():
    """Command-line configuration for the aperture-scaling comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--occurrence-posterior", required=True)
    p.add_argument("--diameters", default="6,7,8,9")
    p.add_argument("--eta-baseline", type=float, default=0.24)
    p.add_argument("--yield-goal", type=float, default=25.0)
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--num-tau-points", type=int, default=150)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--eta-grid-min", type=float, default=0.003)
    p.add_argument("--eta-grid-max", type=float, default=1.2)
    p.add_argument("--eta-grid-points", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-comparison", default="apertureScaling.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        fCalibration = json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["dictMission"]["fThroughputCalibration"] = fCalibration
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    faEtaPosterior = np.load(dictArgs["occurrence_posterior"])["faEta_canonical"]
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    faEtaGrid = np.logspace(np.log10(dictArgs["eta_grid_min"]),
                            np.log10(dictArgs["eta_grid_max"]), dictArgs["eta_grid_points"])
    rng = np.random.default_rng(dictArgs["seed"])
    dictByDiameter = {}
    for sDiameter in [d.strip() for d in dictArgs["diameters"].split(",")]:
        dictCurve = fdictYieldCurveAtDiameter(dfCatalog, dictParams, float(sDiameter),
                                              faTauGridS, faEtaGrid, dictArgs)
        dictProb = fdictProbabilities(faEtaGrid, dictCurve["faYieldGrid"], faEtaPosterior,
                                      dictArgs["eta_baseline"], dictArgs["yield_goal"], rng)
        dictByDiameter[sDiameter] = {**dictProb, "iStarsScreened": dictCurve["iStarsScreened"],
                                     "faYieldGrid": dictCurve["faYieldGrid"].tolist()}
    faDiameters = [float(s) for s in dictByDiameter]
    fCrossing = fnCrossingDiameter(
        faDiameters,
        [dictByDiameter[s]["fProbability25IncludingSigmaEta"] for s in dictByDiameter],
        [dictByDiameter[s]["fProbability25ExcludingSigmaEta"] for s in dictByDiameter])
    dictOut = {
        "fFrozenCalibrationFactor": fCalibration,
        "fCrossingDiameterM": fCrossing,
        "fCrossingDiameterPublishedM": 7.1,
        "sCrossingNote": "Diameter at which the excluding-sigma curve overtakes the "
                         "including-sigma curve. Published value read off Fig. 15.",
        "sCalibrationNote": "Fitted at 6 m only; 7, 8 and 9 m are out-of-sample predictions.",
        "faEtaGrid": faEtaGrid.tolist(),
        "dictPublished": DICT_PUBLISHED,
        "dictByDiameter": dictByDiameter,
        "dictResiduals": {
            s: {"fIncludingSigmaEta": dictByDiameter[s]["fProbability25IncludingSigmaEta"]
                - DICT_PUBLISHED["dictIncludingSigmaEta"][s],
                "fExcludingSigmaEta": dictByDiameter[s]["fProbability25ExcludingSigmaEta"]
                - DICT_PUBLISHED["dictExcludingSigmaEta"][s]}
            for s in dictByDiameter},
    }
    with open(dictArgs["out_comparison"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'D (m)':>6}{'stars':>8}{'E[N] @0.24':>12}{'P25 incl':>10}{'pub':>8}"
          f"{'P25 excl':>10}{'pub':>8}")
    for s, d in dictByDiameter.items():
        print(f"{s:>6}{d['iStarsScreened']:>8}{d['fExpectedYieldAtBaselineEta']:>12.2f}"
              f"{100*d['fProbability25IncludingSigmaEta']:>9.1f}%"
              f"{100*DICT_PUBLISHED['dictIncludingSigmaEta'][s]:>7.1f}%"
              f"{100*d['fProbability25ExcludingSigmaEta']:>9.1f}%"
              f"{100*DICT_PUBLISHED['dictExcludingSigmaEta'][s]:>7.1f}%")


if __name__ == "__main__":
    main()
