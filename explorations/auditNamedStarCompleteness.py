#!/usr/bin/env python3
"""Explain the completeness of named stars, and compare them with Stark (2024) Fig. 11 neighbours.

Stark et al. (2024) Sec. 3.3 say pinning eps Eri, tet Boo, 72 Her and 110 Her to their LBTI levels
"reduces the expected EEC yield by one". At eta = 0.24 one star can contribute at most 0.24 EEC, so
that statement needs all four near full completeness in AYO. In this pipeline only eps Eri is;
tet Boo, 110 Her and 72 Her reach 0.62, 0.19 and 0.15. For each named star this reports its best
completeness in A04's table, the gate that stops its injected planets at the fixed median exozodi
(noise floor, zero throughput, detection over the cap, characterization over the cap; reusing
auditDetectabilityBlockers.fdictBlockerTally), and the completeness of the nearest Fig. 11 target
in (log distance, log luminosity), which is a single exozodi-drawn run and therefore a rough guide.
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

from auditDetectabilityBlockers import fdictBlockerTally  # noqa: E402

DICT_DEFAULT_HIP = {"16537": "eps Eri", "70497": "tet Boo", "84862": "72 Her", "92043": "110 Her"}


def fdictParams(dictArgs):
    """Mission parameters at the current calibration and fixed median exozodi."""
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["dictMission"]["bDrawExozodiLevels"] = False
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    return dictParams


def ffBestCompleteness(dictNpz, sSourceId):
    """Maximum completeness over visits and exposure in A04's table (nan if not screened)."""
    faIds = dictNpz["saSourceId"].astype(str)
    iaMatch = np.flatnonzero(faIds == str(sSourceId))
    return float(np.max(dictNpz["faComp_canonical"][iaMatch[0]])) if iaMatch.size else float("nan")


def fdictNearestStark(listPoints, fDistancePc, fLuminosityLsun):
    """The Fig. 11 target closest in (log d, log L), with its separation in dex."""
    faD = np.log10([p["fDistancePc"] for p in listPoints])
    faL = np.log10([p["fLuminosityLsun"] for p in listPoints])
    faSep = np.hypot(faD - np.log10(fDistancePc), faL - np.log10(fLuminosityLsun))
    i = int(np.argmin(faSep))
    return {**listPoints[i], "fSeparationDex": float(faSep[i])}


def fdictStarRow(dictStar, dictParams, dictNpz, listPoints, dictArgs):
    """Everything reported for one named star."""
    return {"fDistancePc": float(dictStar["fDistancePc"]),
            "fLuminosityLsun": float(dictStar["fLuminosityLsun"]),
            "fTeffK": float(dictStar["fTeffK"]), "fVmag": float(dictStar["fVmag"]),
            "fBestCompletenessA04": ffBestCompleteness(dictNpz, dictStar["sSourceId"]),
            "dictGates": fdictBlockerTally(dictStar, dictParams["dictBoxes"]["canonical"],
                                           dictParams, dictParams["dictMission"],
                                           dictArgs["num_planets"], dictArgs["seed"]),
            "dictNearestStarkFig11": fdictNearestStark(listPoints, dictStar["fDistancePc"],
                                                       dictStar["fLuminosityLsun"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--completeness", default="../computeCompleteness/completeness.npz")
    p.add_argument("--stark-fig11", default="../CompareTargetCompleteness/starkFigure11Digitised.json")
    p.add_argument("--hip", default=",".join(DICT_DEFAULT_HIP))
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="output/namedStarCompleteness.json")
    dictArgs = vars(p.parse_args())
    dictParams = fdictParams(dictArgs)
    dfCat = pd.read_csv(dictArgs["target_catalog"])
    dfCat["sHip"] = [str(int(h)) if h == h else "" for h in dfCat["sHipName"]]
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    with open(dictArgs["stark_fig11"]) as oFile:
        listPoints = json.load(oFile)["listPoints"]
    dictOut = {}
    for sHip in dictArgs["hip"].split(","):
        dictStar = dfCat[dfCat["sHip"] == sHip].iloc[0].to_dict()
        sName = DICT_DEFAULT_HIP.get(sHip, str(dictStar["sSimbadName"]))
        dictOut[sName] = fdictStarRow(dictStar, dictParams, dictNpz, listPoints, dictArgs)
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sName, d in dictOut.items():
        dictN = d["dictNearestStarkFig11"]
        print(f"{sName:<8} d={d['fDistancePc']:.1f} L={d['fLuminosityLsun']:.2f} "
              f"C_A04={d['fBestCompletenessA04']:.2f}  gates={d['dictGates']}\n"
              f"         nearest Fig.11: d={dictN['fDistancePc']:.1f} L={dictN['fLuminosityLsun']:.2f}"
              f" C={dictN['fCompleteness']:.2f} ({dictN['fSeparationDex']:.2f} dex)")


if __name__ == "__main__":
    main()
