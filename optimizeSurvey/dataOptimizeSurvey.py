#!/usr/bin/env python3
"""Run the equal-slope AYO survey optimization at fixed exozodi and over every exozodi draw.

Allocates the two-year exoplanet science budget across targets so that dC/dt is equal for every
observation, charging each star for its detection time, its overheads, and the characterization
time its expected detections will demand.

The survey is optimized twice over. Once with every star at the planning exozodi level (3 zodis),
which is the single AYO run Stark et al. (2024) report as 22.5 EECs and the one A03 calibrates
against. Then once per exozodi draw from A04, re-optimizing around each draw's dust as Stark does
over 500 draws (Sec. 3.3), which is what the fixed-eta yield of Fig. 10's red curve averages
over. The two exozodi-dependent penalties are read from the difference: exozodi is the planning
yield lost to the draw, albedo is the drawn-albedo yield lost from the planning one.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import exozodi as ez  # noqa: E402
from yieldlib import optimizer as opt  # noqa: E402


def flistLoadStars(sCompletenessPath, sBox):
    """Rehydrate one box's per-star completeness curves at the planning exozodi level."""
    dictNpz = np.load(sCompletenessPath, allow_pickle=True)
    faTauGridS = dictNpz["faTauGridS"]
    faComp, faCompAlb = dictNpz[f"faComp_{sBox}"], dictNpz[f"faCompAlbedo_{sBox}"]
    faCharMean = dictNpz[f"faTauCharMean_{sBox}"]
    return [dict(faTauGridS=faTauGridS, faComp=faComp[i], faCompYield=faCompAlb[i],
                 faTauCharMeanS=faCharMean[i]) for i in range(faComp.shape[0])], dictNpz


def fdictDrawSummary(dictDraws, dictFixed):
    """Means, spreads and penalties over the draws at the baseline eta (column 0)."""
    faPlan, faAlb = dictDraws["faYieldPlanning"][:, 0], dictDraws["faYield"][:, 0]
    fPlanFixed = dictFixed["fYieldPlanning"]
    return {"fYieldPlanningDrawnMean": float(np.mean(faPlan)),
            "fYieldPlanningDrawnStd": float(np.std(faPlan)),
            "fYieldDrawnMean": float(np.mean(faAlb)), "fYieldDrawnStd": float(np.std(faAlb)),
            "fStarsUsedDrawnMean": float(np.mean(dictDraws["iaStarsUsed"][:, 0])),
            "fExozodiPenalty": float(1.0 - np.mean(faPlan) / fPlanFixed) if fPlanFixed else 0.0,
            "fAlbedoPenalty": float(1.0 - np.mean(faAlb) / np.mean(faPlan))
            if np.mean(faPlan) else 0.0,
            "iRepresentativeDraw": int(np.argsort(faAlb)[len(faAlb) // 2]),
            "faYieldPlanningByDraw": faPlan.tolist(), "faYieldByDraw": faAlb.tolist()}


def fdictParseArgs():
    """Command-line configuration for the survey optimization."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completeness", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--eta-scan", type=str, default="0.02,0.06,0.12,0.24,0.40,0.60")
    p.add_argument("--calibration-json", default=None)
    p.add_argument("--throughput-calibration", type=float, default=1.0)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--out-survey", default="surveyResult.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictMission = json.load(oFile)["dictMission"]
    fCalibration = dictArgs["throughput_calibration"]
    if dictArgs["calibration_json"]:
        with open(dictArgs["calibration_json"]) as oCal:
            fCalibration = json.load(oCal)["fCalibratedThroughputFactor"]
    dictMission["fThroughputCalibration"] = fCalibration
    listStars, dictNpz = flistLoadStars(dictArgs["completeness"], dictArgs["box"])
    dictFixed = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    faScan = np.array([float(s) for s in dictArgs["eta_scan"].split(",")])
    faEta = np.concatenate(([dictArgs["eta_earth"]], faScan))
    dictDraws = ez.fdictSurveyOverDraws(ez.fdictLoadGrid(dictNpz, dictArgs["box"]),
                                        dictNpz["faZodiDraws"], dictNpz["faTauGridS"], faEta,
                                        dictMission, dictArgs["processes"])
    dictSummary = fdictDrawSummary(dictDraws, dictFixed)
    dictOut = {
        "sBox": dictArgs["box"], "iStarsAvailable": len(listStars),
        "fThroughputCalibration": fCalibration, "fEtaEarthBaseline": dictArgs["eta_earth"],
        "iExozodiDraws": int(dictNpz["faZodiDraws"].shape[0]),
        "sBaselineNote": "fYieldBaseline and fYieldPlanningBaseline are means over the exozodi "
                         "draws (albedo-drawn and planning albedo respectively); the "
                         "fixed-exozodi survey is reported separately.",
        "fYieldPlanningFixedExozodi": float(dictFixed["fYieldPlanning"]),
        "fYieldFixedExozodi": float(dictFixed["fYield"]),
        "iStarsUsedFixedExozodi": int(dictFixed["iStarsUsed"]),
        "fYieldBaseline": dictSummary["fYieldDrawnMean"],
        "fYieldPlanningBaseline": dictSummary["fYieldPlanningDrawnMean"],
        "iStarsUsedBaseline": int(round(dictSummary["fStarsUsedDrawnMean"])),
        "fTimeUsedFractionBaseline": float(dictFixed["fTotalTimeS"] /
                                           dictMission["fTotalScienceTimeS"]),
        "dictDraws": dictSummary,
        "listEtaScan": [{"fEtaEarth": float(f),
                         "fYield": float(np.mean(dictDraws["faYield"][:, k + 1])),
                         "fYieldPlanning": float(np.mean(dictDraws["faYieldPlanning"][:, k + 1])),
                         "fStarsUsed": float(np.mean(dictDraws["iaStarsUsed"][:, k + 1]))}
                        for k, f in enumerate(faScan)],
    }
    with open(dictArgs["out_survey"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "dictDraws"}, indent=2))
    print(json.dumps({k: v for k, v in dictSummary.items() if not k.startswith("fa")}, indent=2))


if __name__ == "__main__":
    main()
