#!/usr/bin/env python3
"""Predict P25 against telescope aperture and compare with Stark et al. (2024) Fig. 15.

The coronagraph throughput calibration is fitted at 6 m and held FROZEN across every diameter,
so the 7, 8 and 9 m points are out-of-sample predictions rather than fits. Two curves are
produced: including eta_Earth uncertainty (marginalized over the occurrence posterior) and
excluding it (eta fixed at the baseline 0.24, Poisson counting only).

Both curves carry albedo (through the albedo-drawn completeness) and exozodi: at every diameter
the survey is re-optimized over the same exozodi draws A04 uses (the per-star levels are redrawn
with the same seeds over that diameter's own screened list), which is how Stark's 498 runs per
diameter are built. The excluding-sigma curve is therefore comparable with his red curve, not
just the including-sigma one.

It also writes, per diameter, the realized-yield histograms (Stark Fig. 12: including and
excluding sigma_eta) and the characterization times of the first 18 expected EECs over the draws
(Stark Fig. 14, whose means are the 22 d and 3.5 d of Sec. 4.1). The histogram is over individual
planets, as Stark's is, re-derived for a subset of draws; its mean is also computed from star
means over all draws, and the two are reported side by side.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import exozodi as ez  # noqa: E402
from yieldlib import survey as sv  # noqa: E402
from yieldlib import yielddistribution as yd  # noqa: E402

DICT_PUBLISHED = {
    "sSource": "Stark et al. (2024) Fig. 15; the 6 m pair is also stated in Sec. 3.5 text",
    "dictIncludingSigmaEta": {"6": 0.32, "7": 0.53, "8": 0.67, "9": 0.78},
    "dictExcludingSigmaEta": {"6": 0.06, "7": 0.49, "8": 0.89, "9": 0.995},
    "saTextStated": ["6"],
    "sReadingCaveat": "Values at 7, 8 and 9 m were read off the published figure and carry "
                      "roughly +/-2-3 percentage points of reading error.",
}


def fdictGridAtDiameter(dfCatalog, dictParams, fDiameterM, faTauGridS, dictArgs):
    """Screen at one aperture, tabulate completeness over exozodi, and draw the levels."""
    dictParams["dictMission"]["fDiameterM"] = fDiameterM
    dictBox = dictParams["dictBoxes"]["canonical"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictGrid = ez.fdictCompletenessZodiGrid(dfTargets, dictParams, dictBox, faTauGridS,
                                            dictArgs["num_planets"], dictArgs["seed"])
    faDraws = ez.faDrawZodiLevels(dictParams["dictMission"], len(dfTargets),
                                  dictArgs["num_draws"], dictArgs["seed"] + 977,
                                  sv.flistHipNumbers(dfTargets))
    return dfTargets, dictGrid, faDraws


def fdictAtDiameter(dfCatalog, dictParams, fDiameterM, faTauGridS, faEtaGrid, faEtaPosterior,
                    dictArgs, rng):
    """Yield curves over the draws, P25 both ways, histograms and first-18 char times."""
    dfTargets, dictGrid, faDraws = fdictGridAtDiameter(dfCatalog, dictParams, fDiameterM,
                                                       faTauGridS, dictArgs)
    dictMission = dictParams["dictMission"]
    faByDraw = ez.fdictSurveyOverDraws(dictGrid, faDraws, faTauGridS, faEtaGrid, dictMission,
                                       dictArgs["processes"])["faYield"]
    dictProb = fdictProbabilities(faEtaGrid, faByDraw, faEtaPosterior, dictArgs["eta_baseline"],
                                  dictArgs["yield_goal"], rng)
    dictPerStar = ez.fdictPerStarOverDraws(dictGrid, faDraws, faTauGridS,
                                           dictArgs["eta_baseline"], dictMission,
                                           dictArgs["processes"])
    iaSubset = np.arange(min(dictArgs["planet_draws"], faDraws.shape[0]))
    faT, faW = ez.fdictPlanetCharacterizationFirstN(
        dfTargets, dictParams, dictParams["dictBoxes"]["canonical"], dictGrid, faDraws,
        dictPerStar, iaSubset, dictArgs["eta_baseline"], dictArgs["first_n"], dictArgs["seed"])
    return {**dictProb, **fdictFirstNCharacterization(dictPerStar, dictArgs),
            **fdictPlanetHistogram(faT, faW),
            "iStarsScreened": int(len(dfTargets)), "faYieldGrid": faByDraw.mean(axis=0).tolist(),
            "fStarsUsedMean": float(np.mean(np.sum(dictPerStar["faStarComp"] > 0, axis=1)))}


def fdictFirstNCharacterization(dictPerStar, dictArgs):
    """Characterization times of the first N EECs pooled over draws, and their mean."""
    listTimes, listWeights = [], []
    for j in range(dictPerStar["faStarComp"].shape[0]):
        faComp, faTime = dictPerStar["faStarComp"][j], dictPerStar["faStarTimeS"][j]
        faPriority = np.where(faTime > 0, faComp / np.maximum(faTime, 1e-30), 0.0)
        faT, faW = ez.faFirstNCharacterizationTimes(
            faComp, dictPerStar["faStarTauCharMeanS"][j], dictArgs["eta_baseline"],
            dictArgs["first_n"], faPriority)
        listTimes.append(faT / 86400.0)
        listWeights.append(faW)
    faT, faW = np.concatenate(listTimes), np.concatenate(listWeights)
    return {"fMeanCharDaysFirstN": float(np.sum(faT * faW) / np.sum(faW))}


def fdictPlanetHistogram(faT, faW):
    """Fig. 14's quantity: individual EEC characterization times in 1-day bins, and their mean.

    The mean should agree with fMeanCharDaysFirstN (star means over all draws) to within the
    sampling of the draw subset; it is reported so the two routes can be compared.
    """
    faEdges = np.linspace(0.0, 60.0, 61)
    return {"fMeanCharDaysFirstNPlanets": float(np.sum(faT * faW) / np.sum(faW)),
            "fMedianCharDaysFirstNPlanets": float(np.interp(0.5, np.cumsum(faW[np.argsort(faT)]) /
                                                            np.sum(faW), np.sort(faT))),
            "faCharDaysEdges": faEdges.tolist(),
            "faCharDaysHistogram": (np.histogram(faT, faEdges, weights=faW)[0] /
                                    np.sum(faW)).tolist()}


def fdictProbabilities(faEtaGrid, faByDraw, faEtaPosterior, fEtaBaseline, fGoal, rng):
    """P25 with and without eta_Earth uncertainty, over the exozodi draws and Poisson counting."""
    iaDraw = rng.permutation(np.arange(faEtaPosterior.size) % faByDraw.shape[0])
    faLog = np.log(np.clip(faEtaPosterior, faEtaGrid[0], faEtaGrid[-1]))
    faExpectedMarginal = np.empty(faEtaPosterior.size)
    for j in range(faByDraw.shape[0]):
        faExpectedMarginal[iaDraw == j] = np.interp(faLog[iaDraw == j], np.log(faEtaGrid),
                                                    faByDraw[j])
    faFixedByDraw = np.array([np.interp(np.log(fEtaBaseline), np.log(faEtaGrid), f)
                              for f in faByDraw])
    faObservedMarginal = rng.poisson(np.maximum(faExpectedMarginal, 0.0))
    faObservedFixed = rng.poisson(np.maximum(faFixedByDraw[iaDraw], 0.0))
    return {
        "fExpectedYieldAtBaselineEta": float(np.mean(faFixedByDraw)),
        "fMeanExpectedMarginal": float(np.mean(faExpectedMarginal)),
        "fProbability25IncludingSigmaEta": float(np.mean(faObservedMarginal >= fGoal)),
        "fProbability25ExcludingSigmaEta": float(np.mean(faObservedFixed >= fGoal)),
        "faPmfIncludingSigmaEta": yd.faPoissonMixturePmf(faExpectedMarginal).tolist(),
        "faPmfExcludingSigmaEta": yd.faPoissonMixturePmf(faFixedByDraw).tolist(),
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
    p.add_argument("--num-draws", type=int, default=100)
    p.add_argument("--first-n", type=int, default=18)
    p.add_argument("--planet-draws", type=int, default=20,
                   help="draws re-derived planet by planet for the Fig. 14 distribution")
    p.add_argument("--processes", type=int, default=8)
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
        dictByDiameter[sDiameter] = fdictAtDiameter(dfCatalog, dictParams, float(sDiameter),
                                                    faTauGridS, faEtaGrid, faEtaPosterior,
                                                    dictArgs, rng)
        print(sDiameter, {k: v for k, v in dictByDiameter[sDiameter].items()
                          if not k.startswith("fa")}, flush=True)
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
        "dictPublishedCharDaysFirst18": {"6": 22.0, "9": 3.5,
                                         "sSource": "Stark+2024 Sec. 4.1 (means of Fig. 14)"},
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
