#!/usr/bin/env python3
"""Evaluate the survey over a Latin-hypercube design in the parameters kappa is suspected to absorb.

Parameters (box priors):
  ln_kappa    throughput calibration on every band            [ln 0.6, ln 3.0]
  ln_kappa_c  extra throughput factor on spectra only         [ln 0.4, ln 2.5]
  f_leak      leak normalization, 1 = zeta*Upsilon_c, 1.78 = Airy PSF_peak*Omega
  s_sky       T_sky shape strength, 0 = unconvolved Upsilon/EE, 1 = AYO-OVC-shaped
              (factor 1 + s (F(u) - 1), F from validateSkyThroughputConvolution.py)

Each point runs the survey at fixed exozodi (3 zodis, no draw noise) at 6 m and, with the same
parameters, at 9 m, and records the observables Stark et al. (2024) publish: the 6 m planning
yield (22.5), the mean characterization time of the priority-first 18 EECs at 6 and 9 m (22 d,
3.5 d), and the median per-target completeness against Fig. 11 by distance bin. Points are
written one JSON each, so the run can be resumed; fitCalibrationDegeneracy.py consumes them.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

S_HERE = os.path.dirname(os.path.abspath(__file__))
S_REPO = os.path.dirname(S_HERE)
sys.path.insert(0, S_REPO)
sys.path.insert(0, os.path.join(S_REPO, "explorations"))
from checkCharacterizationTimeAgainstPublished import (faStarCharacterizationTimes,  # noqa: E402
                                                       fnMeanCharOfFirstN)
from whatIfCharacterizationTreatment import fdictAtNineMetres, fdictSurvey  # noqa: E402
from whatIfLeakAndSkyThroughput import (faAllocatedCompleteness,  # noqa: E402
                                        fdictFigureElevenBins)
from yieldlib import survey as sv  # noqa: E402

LIST_PARAMS = ["ln_kappa", "ln_kappa_c", "f_leak", "s_sky"]
FA_LO = np.array([np.log(0.6), np.log(0.4), 1.0, 0.0])
FA_HI = np.array([np.log(3.0), np.log(2.5), 1.782, 1.0])


def fdictApplyPoint(dictBase, faTheta, dictShape):
    """Mission parameters at one design point."""
    dictParams = json.loads(json.dumps(dictBase))
    dictMission, dictBands = dictParams["dictMission"], dictParams["dictBands"]
    fLnK, fLnKc, fLeak, fSky = faTheta
    dictMission["fThroughputCalibration"] = float(np.exp(fLnK))
    dictMission["fLeakFactor"] = float(fLeak)
    dictMission["dictSkyThroughputShape"] = {
        "faCoreFraction": dictShape["faCoreFraction"],
        "faFactor": [1.0 + fSky * (f - 1.0) for f in dictShape["faFactor"]]}
    for dictBand in [dictBands["dictBandCharacterization"]] + dictBands["listBandsCharacterization"]:
        dictBand["fThroughputScale"] = float(np.exp(fLnKc))
    return dictParams


def fdictObservables(dictParams, dfTargets, faTauGridS, dictArgs, listStark):
    """Everything recorded at one design point."""
    dictBox = dictParams["dictBoxes"]["canonical"]
    dictFixed, listStars = fdictSurvey(dfTargets, dictParams, dictBox, faTauGridS, dictArgs, False)
    dictMission = dictParams["dictMission"]
    listRows = faStarCharacterizationTimes(listStars, dictMission, 0.24, dictFixed["fSlope"])
    faC = faAllocatedCompleteness(listStars, dictFixed, dictMission)
    dictFig = fdictFigureElevenBins(dfTargets, faC, listStark)["dictByDistance"]
    dictNine = fdictAtNineMetres(dictArgs["catalog"], dictParams, dictBox, faTauGridS, dictArgs)
    return {"fYield6": float(dictFixed["fYieldPlanning"]),
            "fChar18Days6": float(fnMeanCharOfFirstN(listRows, 0.24, 18, "priority")),
            "fChar18Days9": dictNine["fMeanCharDaysPriorityFirst18_9m"],
            "fYield9": dictNine["fPlanningFixedExozodi9m"],
            "dictFigElevenMedianByDistance": {k: v.get("fMedianModel") for k, v in
                                              dictFig.items()}}


def fnRunPoint(tJob):
    """Worker: evaluate one design point unless its output already exists."""
    iPoint, faTheta, dictArgs = tJob
    sOut = os.path.join(dictArgs["out_dir"], f"point{iPoint:03d}.json")
    if os.path.exists(sOut):
        return
    dictBase = json.load(open(dictArgs["mission_parameters"]))
    dictBase["fAlpha"], dictBase["fBeta"] = -0.19, 0.26
    dictShape = json.load(open(dictArgs["sky_shape"]))["dictEmpiricalShape"]
    listStark = json.load(open(dictArgs["figure_eleven"]))["listPoints"]
    dictArgs = dict(dictArgs, catalog=pd.read_csv(dictArgs["target_catalog"]))
    dfTargets = sv.fdfScreenTargets(dictArgs["catalog"], dictBase["dictMission"],
                                    dictBase["dictBands"]["listBandsDetection"][0], 0.5, 1500,
                                    2500.0, 7500.0, dictBase["dictBoxes"]["canonical"])
    faTauGridS = np.logspace(1.0, np.log10(dictBase["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dictObs = fdictObservables(fdictApplyPoint(dictBase, faTheta, dictShape), dfTargets,
                               faTauGridS, dictArgs, listStark)
    with open(sOut, "w") as oFile:
        json.dump({"iPoint": iPoint, "dictTheta": dict(zip(LIST_PARAMS, map(float, faTheta))),
                   **dictObs}, oFile, indent=1)
    print(f"point {iPoint} done: {dictObs['fYield6']:.2f} {dictObs['fChar18Days6']:.1f}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--figure-eleven", required=True)
    p.add_argument("--sky-shape", default="reference/skyThroughputConvolutionTest.json")
    p.add_argument("--num-points", type=int, default=56)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--design-seed", type=int, default=7)
    p.add_argument("--out-dir", default="design")
    p.add_argument("--out-design", default="calibrationDegeneracyDesign.json")
    dictArgs = vars(p.parse_args())
    os.makedirs(dictArgs["out_dir"], exist_ok=True)
    faUnit = qmc.LatinHypercube(d=len(LIST_PARAMS), seed=dictArgs["design_seed"]).random(
        dictArgs["num_points"])
    faDesign = FA_LO + faUnit * (FA_HI - FA_LO)
    listJobs = [(i, faDesign[i], dictArgs) for i in range(len(faDesign))]
    with mp.Pool(dictArgs["workers"]) as oPool:
        oPool.map(fnRunPoint, listJobs, chunksize=1)
    listPoints = [json.load(open(os.path.join(dictArgs["out_dir"], f"point{i:03d}.json")))
                  for i in range(len(faDesign))]
    with open(dictArgs["out_design"], "w") as oFile:
        json.dump({"listParams": LIST_PARAMS, "faLow": FA_LO.tolist(),
                   "faHigh": FA_HI.tolist(), "listPoints": listPoints}, oFile, indent=1)


if __name__ == "__main__":
    main()
