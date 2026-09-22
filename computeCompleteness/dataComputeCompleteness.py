#!/usr/bin/env python3
"""Compute single-visit exoEarth-candidate completeness C(tau) for every screened target star.

For each star this injects EECs over the chosen selection box, propagates them through the
radiometric model of Stark et al. (2019), and records the fraction detectable as a function of
exposure time, together with the characterization time a detected planet would demand.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import survey as sv  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the completeness calculation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--boxes", default="canonical,redefined,hzOnly")
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--num-tau-points", type=int, default=150)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--alpha", type=float, default=-0.19)
    p.add_argument("--beta", type=float, default=0.26)
    p.add_argument("--calibration-json", default=None,
                   help="calibration.json whose fCalibratedThroughputFactor is applied")
    p.add_argument("--throughput-calibration", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-completeness", default="completeness.npz")
    p.add_argument("--out-summary", default="completenessSummary.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    fCalibration = dictArgs["throughput_calibration"]
    if dictArgs["calibration_json"]:
        with open(dictArgs["calibration_json"]) as oCal:
            fCalibration = json.load(oCal)["fCalibratedThroughputFactor"]
    dictParams["dictMission"]["fThroughputCalibration"] = fCalibration
    dictParams["fAlpha"], dictParams["fBeta"] = dictArgs["alpha"], dictArgs["beta"]
    listBoxNames = [b.strip() for b in dictArgs["boxes"].split(",")]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"],
                                    dictParams["dictBoxes"][listBoxNames[0]])
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dictArrays = {"faTauGridS": faTauGridS,
                  "faTeffK": dfTargets["fTeffK"].to_numpy(),
                  "faDistancePc": dfTargets["fDistancePc"].to_numpy(),
                  "faLuminosityLsun": dfTargets["fLuminosityLsun"].to_numpy(),
                  "faEeidLamD": dfTargets["fEeidLamD"].to_numpy(),
                  "saSourceId": dfTargets["sSourceId"].astype(str).to_numpy(),
                  "saBoxNames": np.array(listBoxNames)}
    dictSummary = {"iStarsScreened": int(len(dfTargets)),
                   "iNumPlanetsPerStar": dictArgs["num_planets"],
                   "fThroughputCalibration": fCalibration,
                   "dictByBox": {}}
    for sBoxName in listBoxNames:
        dictAll = sv.fdictCompletenessTable(dfTargets, dictParams,
                                            dictParams["dictBoxes"][sBoxName], faTauGridS,
                                            dictArgs["num_planets"], dictArgs["seed"])
        dictArrays[f"faComp_{sBoxName}"] = dictAll["faComp"]
        dictArrays[f"faTauChar_{sBoxName}"] = dictAll["faTauChar"]
        faMax = dictAll["faComp"][:, -1]
        dictSummary["dictByBox"][sBoxName] = {
            "fMaxCompletenessBest": float(np.max(faMax)) if len(faMax) else 0.0,
            "fSummedMaxCompleteness": float(np.sum(faMax)),
            "iStarsWithAnyCompleteness": int(np.sum(faMax > 0)),
            "fMedianTauCharDays": float(np.nanmedian(np.where(
                np.isfinite(dictAll["faTauChar"]), dictAll["faTauChar"], np.nan)) / 86400.0)}
    np.savez_compressed(dictArgs["out_completeness"], **dictArrays)
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps(dictSummary, indent=2))


if __name__ == "__main__":
    main()
