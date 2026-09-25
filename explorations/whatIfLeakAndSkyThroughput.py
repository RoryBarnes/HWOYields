#!/usr/bin/env python3
"""What do the two unverifiable exposure-time forms (#1 leak, #2 sky throughput) do to the survey?

The term-by-term comparison with the Stark et al. (2025) ETC benchmark
(compareEtcBenchmarkTermByTerm.py) left two forms that differ from AYO's but cannot be set from
published DMVC6 data:

  #1 leaked starlight, zeta * Upsilon_c in the model vs zeta * PSF_peak * Omega in Stark et al.
     (2019) Eq. 4. sLeakNormalization "airyPeak" multiplies the leak by 1.78, the Airy value,
     an upper bound for a PSF that is flatter near the inner working angle.
  #2 extended-source throughput T_sky. "tskyConvolved" adds the PSF convolution of Stark's
     definition (the physically certain part); "tskyOvcShape" applies the factor by which AYO's
     own T_sky for the benchmark vortex exceeds the unconvolved reconstruction, as a function of
     Upsilon_c / Upsilon_max (validateSkyThroughputConvolution.py). The second is a bracket
     transferred from a different coronagraph, not a DMVC6 property.

Each variant starts from the current mission parameters (which carry corrections #3-5), refits
the throughput calibration to the published 22.5, and reports what the characterization-time
problem is measured by: the calibration, the fixed-eta level (Stark 17.35), the exozodi penalty
(11.1%), the mean characterization time of the priority-first 18 EECs at 6 and 9 m (22 d, 3.5 d),
and the per-target completeness against Stark (2024) Fig. 11 binned by distance and luminosity.
Nothing here changes the pipeline.
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
sys.path.insert(0, os.path.join(os.path.dirname(S_HERE), "CompareTargetCompleteness"))
from binTargetCompletenessComparison import fdictByEdges  # noqa: E402
from checkCharacterizationTimeAgainstPublished import (faStarCharacterizationTimes,  # noqa: E402
                                                       fnMeanCharOfFirstN)
from dataCompareTargetCompleteness import fdictMatch  # noqa: E402
from whatIfCharacterizationTreatment import (fdictAtNineMetres, fdictSurvey,  # noqa: E402
                                             ffCalibrate)
from yieldlib import survey as sv  # noqa: E402

LIST_VARIANTS = ["base", "leakPeak", "tskyConvolved", "tskyOvcShape", "leakPeakTskyOvcShape"]


def fdictVariantMission(sName, dictShape):
    """Mission keys each variant sets on top of the current parameters."""
    dictLeak = {"sLeakNormalization": "airyPeak"}
    dictOvc = {"dictSkyThroughputShape": dictShape}
    return {"base": {}, "leakPeak": dictLeak,
            "tskyConvolved": {"bSkyThroughputConvolved": True},
            "tskyOvcShape": dictOvc, "leakPeakTskyOvcShape": {**dictLeak, **dictOvc}}[sName]


def faAllocatedCompleteness(listStars, dictResult, dictMission):
    """Each screened star's allocated completeness, zero where unobserved."""
    faC = np.zeros(len(listStars))
    for dictRow in faStarCharacterizationTimes(listStars, dictMission, 0.24, dictResult["fSlope"]):
        faC[dictRow["iStar"]] = dictRow["fCompleteness"]
    return faC


def fdictFigureElevenBins(dfTargets, faC, listStark):
    """Per-target completeness matched to Stark Fig. 11, binned by distance and luminosity."""
    listRows = fdictMatch(dfTargets["fDistancePc"].to_numpy(),
                          dfTargets["fLuminosityLsun"].to_numpy(), faC, listStark, 0.12, 1.2)
    return {"iMatched": len(listRows), "fSummedModel": float(faC.sum()),
            "dictByDistance": fdictByEdges(listRows, "fDistancePc", [0.0, 6.0, 10.0, 15.0, 20.0,
                                                                     50.0]),
            "dictByLuminosity": fdictByEdges(listRows, "fLuminosityLsun",
                                             [0.0, 0.3, 0.6, 1.0, 2.0, 5.0, 100.0])}


def fdictRunVariant(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, listStark):
    """Refit the calibration for one variant and measure everything the docstring lists."""
    dictParams["dictMission"]["fThroughputCalibration"] = 1.0
    fUncal = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs,
                         False)[0]["fYieldPlanning"]
    fCal = ffCalibrate(dfTargets, dictParams, dictBox, faTauGridS, dictArgs)
    dictParams["dictMission"]["fThroughputCalibration"] = fCal
    dictFixed, _ = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, False)
    dictDrawn, listStars = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, True)
    dictMission = dictParams["dictMission"]
    listRows = faStarCharacterizationTimes(listStars, dictMission, 0.24, dictDrawn["fSlope"])
    faC = faAllocatedCompleteness(listStars, dictDrawn, dictMission)
    return {**fdictAtNineMetres(dictArgs["catalog"], dictParams, dictBox, faTauGridS, dictArgs),
            "fUncalibratedPlanning": float(fUncal), "fCalibration": fCal,
            "fPlanningFixedExozodi": float(dictFixed["fYieldPlanning"]),
            "fPlanningDrawnExozodi": float(dictDrawn["fYieldPlanning"]),
            "fFixedEtaYield": float(dictDrawn["fYield"]),
            "fExozodiPenalty": float(1 - dictDrawn["fYieldPlanning"] /
                                     dictFixed["fYieldPlanning"]),
            "fMeanCharDaysPriorityFirst18": float(fnMeanCharOfFirstN(listRows, 0.24, 18,
                                                                     "priority")),
            "iStarsUsed": int(dictDrawn["iStarsUsed"]),
            "dictFigureEleven": fdictFigureElevenBins(dfTargets, faC, listStark)}


def fdictParseArgs():
    """Command-line configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--sky-shape", default="output/skyThroughputConvolutionTest.json")
    p.add_argument("--figure-eleven",
                   default="../CompareTargetCompleteness/starkFigure11Digitised.json")
    p.add_argument("--variants", default=",".join(LIST_VARIANTS))
    p.add_argument("--target-yield", type=float, default=22.5)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--bisection-steps", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-dir", default="output/whatIfLeakAndSkyThroughput")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    dictBase = json.load(open(dictArgs["mission_parameters"]))
    dictBase["fAlpha"], dictBase["fBeta"] = -0.19, 0.26
    dictShape = json.load(open(dictArgs["sky_shape"]))["dictEmpiricalShape"]
    listStark = json.load(open(dictArgs["figure_eleven"]))["listPoints"]
    dictBox = dictBase["dictBoxes"]["canonical"]
    faTauGridS = np.logspace(1.0, np.log10(dictBase["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dictArgs["catalog"] = pd.read_csv(dictArgs["target_catalog"])
    dfTargets = sv.fdfScreenTargets(dictArgs["catalog"], dictBase["dictMission"],
                                    dictBase["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBox)
    os.makedirs(dictArgs["out_dir"], exist_ok=True)
    for sName in dictArgs["variants"].split(","):
        dictParams = json.loads(json.dumps(dictBase))
        dictParams["dictMission"].update(fdictVariantMission(sName, dictShape))
        print(f"== {sName}", flush=True)
        dictOut = fdictRunVariant(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, listStark)
        with open(os.path.join(dictArgs["out_dir"], f"{sName}.json"), "w") as oFile:
            json.dump(dictOut, oFile, indent=2)
        print(json.dumps({k: v for k, v in dictOut.items() if k != "dictFigureEleven"}),
              flush=True)


if __name__ == "__main__":
    main()
