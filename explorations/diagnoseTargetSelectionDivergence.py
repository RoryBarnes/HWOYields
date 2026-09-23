#!/usr/bin/env python3
"""Compare WHICH stars this survey observes against the ones Stark et al. (2024) Fig. 11 selects.

The per-target completeness comparison reports a puzzle it does not resolve. Summed over the 141
targets matched to Fig. 11 this model reaches 53.6 against a published 74.2, yet its survey total
is 93.6 -- more completeness than Stark achieves, spent somewhere else. That arithmetic only
closes if a large number of observed stars are not in Stark's selected list at all, which would
mean the two surveys disagree about WHICH stars to point at, not merely about how well each one
does. That is a different defect from the one the completeness-versus-distance trend suggests,
and it calls for a different fix, so it is worth establishing before anything else is tried.

This script optimizes the survey, then partitions the outcome three ways in the plane Fig. 11 is
drawn in (distance against luminosity):

  * stars this model observes that have no Fig. 11 counterpart -- yield spent where Stark spends
    none;
  * Fig. 11 targets this model observes -- the shared core;
  * Fig. 11 targets this model gives nothing -- yield Stark collects that this model forgoes.

Each group is broken down by effective temperature, luminosity and distance, because a
disagreement concentrated in one stellar type is a different diagnosis from one spread evenly.
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


def faAllocatedCompleteness(listStars, dictMission, fEtaEarth, fSlope, iStars):
    """Each star's completeness at the optimizer's final slope, zero where it is not observed."""
    faOut = np.zeros(iStars)
    for iStar, dictStar in enumerate(listStars):
        dictCurve = opt.fdictStarCostCurve(dictStar, fEtaEarth, dictMission)
        if len(dictCurve["faCost"]) <= 1:
            continue
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex > 0:
            faOut[iStar] = dictCurve["faComp"][iVertex]
    return faOut


def faMatchIndex(dfTargets, listPublished, fDistanceTol, fLogLumTol):
    """Index of the Fig. 11 point each model star corresponds to, or -1.

    Fig. 11 carries no star names, only positions, so the match is made in the plane it is drawn
    in: fractional distance and log luminosity, both of which this model and the catalog agree on
    to well within the digitization error.
    """
    faPubD = np.array([d["fDistancePc"] for d in listPublished])
    faPubL = np.log10(np.array([d["fLuminosityLsun"] for d in listPublished]))
    faOut = np.full(len(dfTargets), -1, dtype=int)
    bTaken = np.zeros(len(listPublished), dtype=bool)
    faD = dfTargets["fDistancePc"].to_numpy()
    faL = np.log10(dfTargets["fLuminosityLsun"].to_numpy())
    for i in range(len(dfTargets)):
        faDd = np.abs(faPubD - faD[i]) / np.maximum(faD[i], 1e-6)
        faDl = np.abs(faPubL - faL[i])
        bOk = (faDd <= fDistanceTol) & (faDl <= fLogLumTol) & (~bTaken)
        if not bOk.any():
            continue
        faCost = np.where(bOk, faDd / fDistanceTol + faDl / fLogLumTol, np.inf)
        iBest = int(np.argmin(faCost))
        faOut[i], bTaken[iBest] = iBest, True
    return faOut


def fdictGroupProfile(dfTargets, faComp, bMask):
    """Counts and totals for one group of stars, split by temperature, luminosity and distance."""
    if not bMask.any():
        return {"iStars": 0}
    faTeff = dfTargets["fTeffK"].to_numpy()[bMask]
    faLum = dfTargets["fLuminosityLsun"].to_numpy()[bMask]
    faDist = dfTargets["fDistancePc"].to_numpy()[bMask]
    faC = faComp[bMask]
    dictOut = {
        "iStars": int(bMask.sum()),
        "fSummedCompleteness": round(float(faC.sum()), 2),
        "fMeanCompleteness": round(float(faC.mean()), 3),
        "fMedianDistancePc": round(float(np.median(faDist)), 1),
        "fMedianLuminosityLsun": round(float(np.median(faLum)), 3),
        "fMedianTeffK": round(float(np.median(faTeff)), 0),
    }
    for sLabel, bBin in (("M (<3900 K)", faTeff < 3900), ("K (3900-5300)",
                         (faTeff >= 3900) & (faTeff < 5300)),
                         ("G (5300-6000)", (faTeff >= 5300) & (faTeff < 6000)),
                         ("F (6000-7300)", (faTeff >= 6000) & (faTeff < 7300)),
                         ("A and hotter (>7300)", faTeff >= 7300)):
        dictOut[f"iType {sLabel}"] = int(bBin.sum())
        dictOut[f"fComp {sLabel}"] = round(float(faC[bBin].sum()), 2)
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--published-figure", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--distance-tol", type=float, default=0.06)
    p.add_argument("--log-lum-tol", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    with open(dictArgs["published_figure"]) as oFile:
        listPublished = json.load(oFile)["listPoints"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    dictMission = dictParams["dictMission"]
    faTauGridS = np.logspace(1.0, np.log10(dictMission["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]), dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    faComp = faAllocatedCompleteness(listStars, dictMission, dictArgs["eta_earth"],
                                     dictResult["fSlope"], len(dfTargets))
    faMatch = faMatchIndex(dfTargets, listPublished, dictArgs["distance_tol"],
                           dictArgs["log_lum_tol"])

    bObserved = faComp > 0
    bMatched = faMatch >= 0
    faPubComp = np.array([d["fCompleteness"] for d in listPublished])
    dictOut = {
        "iScreened": int(len(dfTargets)),
        "iObserved": int(bObserved.sum()),
        "iPublishedTargets": len(listPublished),
        "fSummedCompletenessModel": round(float(faComp.sum()), 2),
        "fSummedCompletenessPublished": round(float(faPubComp.sum()), 2),
        "dictObservedNotInFigure": fdictGroupProfile(dfTargets, faComp, bObserved & ~bMatched),
        "dictObservedAndInFigure": fdictGroupProfile(dfTargets, faComp, bObserved & bMatched),
        "dictInFigureNotObserved": fdictGroupProfile(dfTargets, faComp, ~bObserved & bMatched),
        "fPublishedCompletenessOnStarsWeSkip": round(float(sum(
            faPubComp[faMatch[i]] for i in range(len(dfTargets))
            if bMatched[i] and not bObserved[i])), 2),
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
