#!/usr/bin/env python3
"""Marginalize the calibrated AYO survey over the occurrence posterior AND the exozodi draws.

The survey optimization is run on a grid of eta_Earth values and interpolated, rather than once
per posterior draw: yield(eta) is smooth and monotonic, and re-optimizing thousands of times
would buy no accuracy. Because characterization time is charged against the same budget, the
grid is genuinely curved -- yield is not proportional to eta -- which is the whole reason the
redefined box cannot be evaluated by rescaling Stark's published number.

The grid is computed once per exozodi draw from A04, since each draw re-optimizes the survey.
Each posterior sample of eta is paired with one draw (cycling through them in a shuffled order),
so the expected yield carries both uncertainties, and the realized yield adds Poisson counting on
top. The same pairing at eta fixed to the baseline gives the fixed-eta distribution, the analogue
of Stark et al. (2024) Fig. 10's red curve.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import exozodi as ez  # noqa: E402


def faYieldGridByDraw(dictNpz, sBox, faEtaGrid, dictMission, iProcesses):
    """Expected (albedo-drawn) yield on the eta grid for every exozodi draw, (draws, eta)."""
    return ez.fdictSurveyOverDraws(ez.fdictLoadGrid(dictNpz, sBox), dictNpz["faZodiDraws"],
                                   dictNpz["faTauGridS"], faEtaGrid, dictMission,
                                   iProcesses)["faYield"]


def faExpectedPerSample(faEtaGrid, faYieldByDraw, faEtaSamples, iaDraw):
    """Expected yield for each posterior sample, read off its paired draw's eta curve."""
    faOut = np.empty(faEtaSamples.size)
    for j in np.unique(iaDraw):
        bThis = iaDraw == j
        faOut[bThis] = faInterpolateYield(faEtaGrid, faYieldByDraw[j], faEtaSamples[bThis])
    return faOut


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


def fdictPredictBox(dictNpz, dictPosterior, sBox, faEtaGrid, dictMission, iaDraw, rng,
                    dictArgs):
    """Summary and samples for one box: eta-marginalized and eta-fixed, both over the draws."""
    faByDraw = faYieldGridByDraw(dictNpz, sBox, faEtaGrid, dictMission, dictArgs["processes"])
    faEtaSamples = dictPosterior[f"faEta_{sBox}"]
    faExpected = faExpectedPerSample(faEtaGrid, faByDraw, faEtaSamples, iaDraw)
    fEtaFixed = dictArgs["eta_fixed"] * float(np.median(faEtaSamples) /
                                              np.median(dictPosterior["faEta_canonical"]))
    faExpectedFixed = np.array([faInterpolateYield(faEtaGrid, f, np.array([fEtaFixed]))[0]
                                for f in faByDraw])[iaDraw]
    dictSamples = {"faExpected": faExpected, "faObserved": rng.poisson(np.maximum(faExpected, 0)),
                   "faExpectedFixedEta": faExpectedFixed,
                   "faObservedFixedEta": rng.poisson(np.maximum(faExpectedFixed, 0)),
                   "faYieldGridByDraw": faByDraw}
    fGoal = dictArgs["yield_goal"]
    return {"faEtaGrid": faEtaGrid.tolist(), "faYieldGrid": faByDraw.mean(axis=0).tolist(),
            "faYieldGridP16": np.percentile(faByDraw, 16, axis=0).tolist(),
            "faYieldGridP84": np.percentile(faByDraw, 84, axis=0).tolist(),
            "dictDistribution": fdictDistributionSummary(faExpected, dictSamples["faObserved"],
                                                         fGoal),
            "fEtaFixed": fEtaFixed,
            "dictDistributionFixedEta": fdictDistributionSummary(
                faExpectedFixed, dictSamples["faObservedFixedEta"], fGoal),
            "fEtaMedian": float(np.median(faEtaSamples))}, dictSamples


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
    p.add_argument("--eta-fixed", type=float, default=0.24,
                   help="canonical-box eta for the fixed-eta distribution (Fig. 10 red curve); "
                        "other boxes scale it by their posterior median ratio")
    p.add_argument("--processes", type=int, default=8)
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
    iDraws = int(dictNpz["faZodiDraws"].shape[0])
    iSamples = int(dictPosterior["faEta_canonical"].size)
    iaDraw = rng.permutation(np.arange(iSamples) % iDraws)
    dictOut = {"dictByBox": {}, "fYieldGoal": dictArgs["yield_goal"], "iExozodiDraws": iDraws,
               "fEtaFixed": dictArgs["eta_fixed"]}
    dictSamples = {"iaDraw": iaDraw}
    for sBox in [b.strip() for b in dictArgs["boxes"].split(",")]:
        dictOut["dictByBox"][sBox], dictBoxSamples = fdictPredictBox(
            dictNpz, dictPosterior, sBox, faEtaGrid, dictMission, iaDraw, rng, dictArgs)
        dictSamples.update({f"{k}_{sBox}": v for k, v in dictBoxSamples.items()})
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
