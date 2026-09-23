#!/usr/bin/env python3
"""Measure the albedo penalty under each reading of Stark's method, against his published 12%.

Three things distinguish the readings, and they are separable:

  * whether the CHARACTERIZATION time is recomputed at the drawn albedo. Ref. stark2024 Sec. 3.2
    says it is not -- "characterization times are budgeted for by AYO under the assumption that
    the planets have a single A_G = 0.2" -- and this pipeline recomputed it, which is the likeliest
    cause of the shortfall, since characterization binds beyond about 10 pc and a brighter planet
    was therefore being allowed to buy a cheaper spectrum as well as an easier detection;
  * whether detection is re-derived from the drawn flux ("recompute") or tested against the range
    of fluxes the visit actually reached ("perVisitThreshold"), which is what Stark describes;
  * whether that range carries its upper bound as well as its lower one. A flux above what the
    orbit segment reached implies gibbous phase, hence small projected separation, hence inside
    the inner working angle -- an identification that is exact for edge-on orbits and irrelevant
    for face-on ones, and since cos(i) is uniform the edge-on case dominates the detections.

The diagnostic that motivated all three is the detection rate against drawn albedo: below
A_G = 0.2 this model already falls in proportion to albedo as Stark describes, but above 0.2 his is
flat while this model's keeps climbing from 0.53 to 0.68, and that rising tail cancels the loss
from dark planets.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import optimizer as opt  # noqa: E402
from yieldlib import survey as sv  # noqa: E402


def fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, dictOverrides):
    """One survey optimization, returning the planning and albedo-drawn yields."""
    dictParams = json.loads(json.dumps(dictParams))
    dictMission = dictParams["dictMission"]
    dictMission.update(dictOverrides)
    dictMission["fThroughputCalibration"] = dictArgs["calibration"]
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    return opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictArgs["calibration"] = json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])

    listVariants = [
        ("100 characterization phases (reference)", {"iCharacterizationPhaseSamples": 100}),
        ("40 characterization phases (new default)", {"iCharacterizationPhaseSamples": 40}),
        ("24 characterization phases", {"iCharacterizationPhaseSamples": 24}),
    ]
    listOut = []
    for sLabel, dictOverrides in listVariants:
        dictResult = fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs,
                                 dictOverrides)
        fPlanning = float(dictResult["fYieldPlanning"])
        fDrawn = float(dictResult["fYield"])
        listOut.append({
            "sVariant": sLabel,
            "dictOverrides": dictOverrides,
            "fYieldPlanning": round(fPlanning, 3),
            "fYieldAlbedoDrawn": round(fDrawn, 3),
            "fAlbedoPenalty": round(1.0 - fDrawn / fPlanning, 4),
            "fPublishedPenalty": 0.12,
        })
        print(f"{sLabel:52s} planning={fPlanning:6.2f} drawn={fDrawn:6.2f} "
              f"penalty={listOut[-1]['fAlbedoPenalty']:.4f}  (published 0.12)")
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"fCalibration": dictArgs["calibration"], "listVariants": listOut}, oFile,
                  indent=2)


if __name__ == "__main__":
    main()
