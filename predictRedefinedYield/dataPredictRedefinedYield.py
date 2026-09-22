#!/usr/bin/env python3
"""Marginalize the calibrated AYO survey over the occurrence posterior for both selection boxes.

The survey optimization is run on a grid of eta_Earth values and interpolated, rather than once
per posterior draw: yield(eta) is smooth and monotonic, and re-optimizing thousands of times
would buy no accuracy. Because characterization time is charged against the same budget, the
grid is genuinely curved -- yield is not proportional to eta -- which is the whole reason the
redefined box cannot be evaluated by rescaling Stark's published number.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402


def flistStarsForBox(dictNpz, sBox):
    """Per-star completeness dicts for one selection box."""
    faTauGridS = dictNpz["faTauGridS"]
    faComp, faTauChar = dictNpz[f"faComp_{sBox}"], dictNpz[f"faTauChar_{sBox}"]
    return [dict(faTauGridS=faTauGridS, faComp=faComp[i], fTauCharS=float(faTauChar[i]))
            for i in range(faComp.shape[0])]


def faYieldGrid(listStars, faEtaGrid, dictMission):
    """Expected yield at each eta_Earth on the grid."""
    return np.array([opt.fdictOptimizeSurvey(listStars, float(f), dictMission)["fYield"]
                     for f in faEtaGrid])


def faInterpolateYield(faEtaGrid, faYields, faEtaSamples):
    """Interpolate the yield curve in log-eta, clipping samples to the grid's span."""
    faClipped = np.clip(faEtaSamples, faEtaGrid[0], faEtaGrid[-1])
    return np.interp(np.log(faClipped), np.log(faEtaGrid), faYields)


def fdictDistributionSummary(faExpected, faObserved, fGoal):
    """Percentiles of the expected-yield distribution and the probability of meeting a goal."""
    return {
        "fMeanExpected": float(np.mean(faExpected)),
        "fMedianExpected": float(np.median(faExpected)),
        "fP16Expected": float(np.percentile(faExpected, 16)),
        "fP84Expected": float(np.percentile(faExpected, 84)),
        "fP05Expected": float(np.percentile(faExpected, 5)),
        "fP95Expected": float(np.percentile(faExpected, 95)),
        "fProbabilityAtLeastGoal": float(np.mean(faObserved >= fGoal)),
        "fProbabilityAtLeastOne": float(np.mean(faObserved >= 1)),
    }


def fdictParseArgs():
    """Command-line configuration for the marginalized yield prediction."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completeness", required=True)
    p.add_argument("--occurrence-posterior", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--boxes", default="canonical,redefined")
    p.add_argument("--eta-grid-min", type=float, default=0.003)
    p.add_argument("--eta-grid-max", type=float, default=1.2)
    p.add_argument("--eta-grid-points", type=int, default=24)
    p.add_argument("--yield-goal", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-prediction", default="yieldPrediction.json")
    p.add_argument("--out-samples", default="yieldSamples.npz")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictMission = json.load(oFile)["dictMission"]
    with open(dictArgs["calibration_json"]) as oFile:
        dictMission["fThroughputCalibration"] = json.load(oFile)["fCalibratedThroughputFactor"]
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    dictPosterior = np.load(dictArgs["occurrence_posterior"])
    rng = np.random.default_rng(dictArgs["seed"])
    faEtaGrid = np.logspace(np.log10(dictArgs["eta_grid_min"]),
                            np.log10(dictArgs["eta_grid_max"]), dictArgs["eta_grid_points"])
    dictOut, dictSamples = {"dictByBox": {}, "fYieldGoal": dictArgs["yield_goal"]}, {}
    for sBox in [b.strip() for b in dictArgs["boxes"].split(",")]:
        faYields = faYieldGrid(flistStarsForBox(dictNpz, sBox), faEtaGrid, dictMission)
        faEtaSamples = dictPosterior[f"faEta_{sBox}"]
        faExpected = faInterpolateYield(faEtaGrid, faYields, faEtaSamples)
        faObserved = rng.poisson(np.maximum(faExpected, 0.0))
        dictOut["dictByBox"][sBox] = {
            "faEtaGrid": faEtaGrid.tolist(), "faYieldGrid": faYields.tolist(),
            "dictDistribution": fdictDistributionSummary(faExpected, faObserved,
                                                         dictArgs["yield_goal"]),
            "fEtaMedian": float(np.median(faEtaSamples)),
        }
        dictSamples[f"faExpected_{sBox}"] = faExpected
        dictSamples[f"faObserved_{sBox}"] = faObserved
    faRatio = dictSamples["faExpected_redefined"] / dictSamples["faExpected_canonical"]
    dictOut["dictYieldRatio"] = {
        "fMedian": float(np.median(faRatio)), "fP16": float(np.percentile(faRatio, 16)),
        "fP84": float(np.percentile(faRatio, 84)),
        "fOccurrenceRatioMedian": float(np.median(dictPosterior["faRatio"])),
    }
    np.savez_compressed(dictArgs["out_samples"], faEtaGrid=faEtaGrid, **dictSamples)
    with open(dictArgs["out_prediction"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({"dictYieldRatio": dictOut["dictYieldRatio"],
                      "dictByBox": {k: v["dictDistribution"]
                                    for k, v in dictOut["dictByBox"].items()}}, indent=2))


if __name__ == "__main__":
    main()
