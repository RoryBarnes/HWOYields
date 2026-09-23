#!/usr/bin/env python3
"""Attribute the pipeline's slowdown to specific changes, by timing the inner loop under each.

The pipeline went from roughly half an hour to just over two hours, and three steps account for
93% of that: the calibration bisection at 44 minutes, the completeness table at 30 and the
aperture scan at 45. All three are repeated calls to fdictCompletenessTable, so the slowdown is
entirely in that one function and the question is which recent change caused it.

Two changes are plausible and a micro-benchmark of a single star could not separate them, because
its timings were dominated by allocation noise -- the 16-phase case came out slower than the
30-phase case, which cannot be real. So this times the actual function over a realistic block of
stars, varying one thing at a time:

  * characterization scheduled at the best orbital phase, which replaced 6 visit epochs with 100
    phase samples per planet;
  * the coronagraph curves read from an interpolated table rather than evaluated in closed form,
    which touches every count-rate evaluation rather than only the characterization ones.

It also measures the saving available from not computing characterization times during the albedo
draw at all, which Ref. stark2024 Sec. 3.2 says are not used and which this pipeline consequently
discards.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import survey as sv  # noqa: E402


def fnTimeTable(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed, dictOverrides):
    """Seconds to build the completeness table for this block of stars under one configuration."""
    dictLocal = json.loads(json.dumps(dictParams))
    dictLocal["dictMission"].update(dictOverrides)
    fStart = time.perf_counter()
    sv.fdictCompletenessTable(dfTargets, dictLocal, dictBox, faTauGridS, iNumPlanets, iSeed)
    return time.perf_counter() - fStart


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=150)
    p.add_argument("--block-stars", type=int, default=40)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfAll = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                dictParams["dictMission"],
                                dictParams["dictBands"]["listBandsDetection"][0],
                                dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    iBlock = dictArgs["block_stars"]
    dfBlock = dfAll.iloc[::max(1, len(dfAll) // iBlock)].head(iBlock).reset_index(drop=True)
    dictTable = dictParams["dictMission"].get("dictCoronagraphTable")

    listCases = [
        ("default (40 phases, albedo characterization skipped)", {}),
        ("table curves, 40 phases", {"iCharacterizationPhaseSamples": 40}),
        ("table curves, 24 phases", {"iCharacterizationPhaseSamples": 24}),
        ("table curves, phase optimization OFF", {"bOptimizeCharacterizationPhase": False}),
        ("parametric curves, 100 phases", {"dictCoronagraphTable": None}),
        ("parametric curves, phase optimization OFF",
         {"dictCoronagraphTable": None, "bOptimizeCharacterizationPhase": False}),
    ]
    listOut = []
    for sLabel, dictOverrides in listCases:
        fSeconds = fnTimeTable(dfBlock, dictParams, dictBox, faTauGridS,
                               dictArgs["num_planets"], dictArgs["seed"], dictOverrides)
        listOut.append({"sCase": sLabel, "fSecondsForBlock": round(fSeconds, 2),
                        "fSecondsPerStar": round(fSeconds / len(dfBlock), 4),
                        "fMinutesFor1500Stars": round(fSeconds / len(dfBlock) * 1500 / 60.0, 1)})
        print(f"{sLabel:44s}{listOut[-1]['fSecondsPerStar']:9.4f} s/star"
              f"{listOut[-1]['fMinutesFor1500Stars']:9.1f} min per 1500-star table")
    dictOut = {"iBlockStars": len(dfBlock), "bTablePresent": bool(dictTable),
               "listCases": listOut}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)


if __name__ == "__main__":
    main()
