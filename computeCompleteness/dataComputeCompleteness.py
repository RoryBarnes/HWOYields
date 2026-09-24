#!/usr/bin/env python3
"""Compute exoEarth-candidate completeness C(tau, visits, exozodi) for every screened target star.

For each star this injects EECs over the chosen selection box, propagates them through the
radiometric model of Stark et al. (2019), and records the fraction detected and characterizable
as a function of exposure time and visit count, together with the characterization time a
counted planet demands.

Exozodi is a random variable (Stark et al. 2024 Sec. 3.3), so the curves are tabulated on a grid
of exozodi levels (yieldlib.exozodi.LIST_ZODI_GRID) rather than at one drawn level, and this step
also draws the per-star levels for every survey realization downstream steps average over. The
fixed-level arrays (faComp_<box> etc.) are the grid at the planning level, 3 zodis, which is the
survey Stark's 22.5 describes.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import exozodi as ez  # noqa: E402
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
    p.add_argument("--grid-boxes", default="canonical,redefined",
                   help="boxes tabulated over exozodi level; others only at the planning level")
    p.add_argument("--num-draws", type=int, default=200,
                   help="exozodi realizations drawn for downstream survey averaging")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-completeness", default="completeness.npz")
    p.add_argument("--out-summary", default="completenessSummary.json")
    return vars(p.parse_args())


def fdictBoxArrays(sBox, dictGrid, iStars, faTauGridS, fPlanningZodi, bKeepGrid):
    """Planning-level curves for every screened star, plus the exozodi grid when kept."""
    iLevel = int(np.argmin(np.abs(dictGrid["faZodiGrid"] - fPlanningZodi)))
    iVisits, iTau = dictGrid["iaCompCounts"].shape[2:]
    dictOut = {}
    for sKey, sGridKey, fScale in (("faComp", "iaCompCounts", 1.0 / dictGrid["iNumPlanets"]),
                                   ("faCompAlbedo", "iaCompAlbedoCounts",
                                    1.0 / dictGrid["iNumPlanets"]),
                                   ("faTauCharMean", "faTauCharMean", 1.0)):
        faFull = np.zeros((iStars, iVisits, iTau))
        faFull[dictGrid["iaStar"]] = dictGrid[sGridKey][:, iLevel].astype(float) * fScale
        dictOut[f"{sKey}_{sBox}"] = faFull
    if bKeepGrid:
        dictOut.update({f"iaGridStar_{sBox}": dictGrid["iaStar"],
                        f"iaCompCounts_{sBox}": dictGrid["iaCompCounts"],
                        f"iaCompAlbedoCounts_{sBox}": dictGrid["iaCompAlbedoCounts"],
                        f"faTauCharMeanGrid_{sBox}": dictGrid["faTauCharMean"],
                        "faZodiGrid": dictGrid["faZodiGrid"],
                        "iNumPlanets": dictGrid["iNumPlanets"]})
    return dictOut


def fdictBoxSummary(dictArrays, sBox, dictGrid):
    """Headline numbers for one box at the planning exozodi level."""
    faMax = dictArrays[f"faComp_{sBox}"][:, -1, -1]
    faChar = dictArrays[f"faTauCharMean_{sBox}"][:, -1, -1]
    return {"fMaxCompletenessBest": float(np.max(faMax)) if len(faMax) else 0.0,
            "fSummedMaxCompleteness": float(np.sum(faMax)),
            "iStarsWithAnyCompleteness": int(np.sum(faMax > 0)),
            "iStarsWithAnyCompletenessAtZeroZodi": int(dictGrid["iaStar"].size),
            "fMedianTauCharDays": float(np.median(faChar[faChar > 0]) / 86400.0)
            if np.any(faChar > 0) else 0.0}


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
    listGridBoxes = [b.strip() for b in dictArgs["grid_boxes"].split(",")]
    for sBoxName in listBoxNames:
        faZodiGrid = ez.LIST_ZODI_GRID if sBoxName in listGridBoxes else [
            dictParams["dictMission"]["fExozodiLevel"]]
        dictGrid = ez.fdictCompletenessZodiGrid(dfTargets, dictParams,
                                                dictParams["dictBoxes"][sBoxName], faTauGridS,
                                                dictArgs["num_planets"], dictArgs["seed"],
                                                faZodiGrid)
        dictArrays.update(fdictBoxArrays(sBoxName, dictGrid, len(dfTargets), faTauGridS,
                                         dictParams["dictMission"]["fExozodiLevel"],
                                         sBoxName in listGridBoxes))
        dictSummary["dictByBox"][sBoxName] = fdictBoxSummary(dictArrays, sBoxName, dictGrid)
    dictArrays["faZodiDraws"] = ez.faDrawZodiLevels(
        dictParams["dictMission"], len(dfTargets), dictArgs["num_draws"],
        int(dictParams["dictMission"].get("iExozodiSeed", dictArgs["seed"] + 977)),
        sv.flistHipNumbers(dfTargets))
    dictSummary["iExozodiDraws"] = dictArgs["num_draws"]
    dictSummary["fMedianDrawnZodi"] = float(np.median(dictArrays["faZodiDraws"]))
    dictSummary["fFractionDrawnAbove100Zodi"] = float(np.mean(dictArrays["faZodiDraws"] > 100))
    np.savez_compressed(dictArgs["out_completeness"], **dictArrays)
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps(dictSummary, indent=2))


if __name__ == "__main__":
    main()
