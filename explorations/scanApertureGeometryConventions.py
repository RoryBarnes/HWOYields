#!/usr/bin/env python3
"""Adjudicate the two aperture-geometry conventions in Stark's papers against his published yield.

Stark et al. (2024) label their scenarios by INSCRIBED diameter -- the baseline is "6 m ID" -- but
Sec. 6.1 states that the x-axis of the coronagraph performance figure is the CIRCUMSCRIBED
diameter, and that the DMVC6's apparent 3.5 lambda/D inner working angle is inflated for exactly
that reason. Ref. stark2019 Sec. 6.2 closes the loop: normalised to the inscribed pupil the DMVC
and the monolithic vortex "would look nearly identical", and the monolithic vortex sits at
~3 lambda/D. The two readings therefore differ by D_circ / D_inscribed.

The same section forces a second correction. Upsilon_c is normalised to the light entering the
coronagraph from the FULL obscured primary, including the jagged region outside the inscribed
diameter that the Lyot stop discards -- that discard is why Upsilon_c,max is 0.46 rather than the
0.69 an ideal system would give. Pairing that Upsilon_c with a collecting area taken as the
inscribed CIRCLE charges the same loss twice.

This pipeline used the inscribed diameter for both the collecting area and the coronagraph
curves, so it is wrong on both counts if the circumscribed reading is right. Rather than pick the
reading that flatters the result, this script scans both factors and reports, for each, the
uncalibrated yield against Stark's published 22.5 and the completeness achieved at 15-30 pc,
where this pipeline reaches zero against a published 0.248. The right convention should improve
BOTH without a fitted throughput factor; a convention that fixes one and breaks the other is not
the explanation.
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

F_LUVOIR_B_INSCRIBED_M = 6.7
F_LUVOIR_B_CIRCUMSCRIBED_M = 8.0


def fdictRunOne(dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, fScale, fAreaFactor):
    """One survey optimization at a given coronagraph angular scale and collecting-area factor."""
    dictParams = json.loads(json.dumps(dictParams))
    dictMission = dictParams["dictMission"]
    dictMission["fThroughputCalibration"] = 1.0
    dictMission["fCoronagraphScale"] = fScale
    dictMission["fApertureAreaM2"] = fAreaFactor * np.pi * (dictMission["fDiameterM"] / 2.0) ** 2
    dfTargets = sv.fdfScreenTargets(dfCatalog, dictMission,
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    dictTable = sv.fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"])
    listStars = sv.flistStarsFromTable(dictTable, faTauGridS)
    dictResult = opt.fdictOptimizeSurvey(listStars, dictArgs["eta_earth"], dictMission)
    return dfTargets, listStars, dictResult, dictMission


def faPerStarCompleteness(listStars, dictMission, fEtaEarth, fSlope, iStars):
    """Each star's chosen completeness at the optimizer's final slope."""
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


def fdictDistanceProfile(dfTargets, faComp, faEdges):
    """Mean completeness over observed stars, and total yield, in bins of distance."""
    faDist = dfTargets["fDistancePc"].to_numpy()
    dictOut = {}
    for fLo, fHi in zip(faEdges[:-1], faEdges[1:]):
        bMask = (faDist >= fLo) & (faDist < fHi)
        iObserved = int(np.sum(bMask & (faComp > 0)))
        dictOut[f"{int(fLo)}-{int(fHi)}"] = {
            "iScreened": int(np.sum(bMask)),
            "iObserved": iObserved,
            "fMeanCompletenessObserved": round(
                float(np.mean(faComp[bMask & (faComp > 0)])) if iObserved else 0.0, 4),
        }
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--published-yield", type=float, default=22.5)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    fRatio = F_LUVOIR_B_CIRCUMSCRIBED_M / F_LUVOIR_B_INSCRIBED_M

    listCases = [
        ("asBuilt-inscribedBoth", 1.0, 1.0),
        ("circumscribedCurvesOnly", fRatio, 1.0),
        ("fullPrimaryAreaOnly", 1.0, fRatio ** 2 * 0.785),
        ("bothCorrections", fRatio, fRatio ** 2 * 0.785),
    ]
    faEdges = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0])
    listOut = []
    for sLabel, fScale, fAreaFactor in listCases:
        dfTargets, listStars, dictResult, dictMission = fdictRunOne(
            dfCatalog, dictParams, dictBox, faTauGridS, dictArgs, fScale, fAreaFactor)
        faComp = faPerStarCompleteness(listStars, dictMission, dictArgs["eta_earth"],
                                       dictResult["fSlope"], len(dfTargets))
        listOut.append({
            "sLabel": sLabel,
            "fCoronagraphScale": fScale,
            "fAreaFactor": fAreaFactor,
            "fApertureAreaM2": dictMission["fApertureAreaM2"],
            "fUncalibratedYieldPlanning": round(float(dictResult["fYieldPlanning"]), 3),
            "fFractionOfPublished": round(float(dictResult["fYieldPlanning"]) /
                                          dictArgs["published_yield"], 3),
            "iStarsObserved": int(np.sum(faComp > 0)),
            "dictByDistance": fdictDistanceProfile(dfTargets, faComp, faEdges),
        })
        print(json.dumps({k: v for k, v in listOut[-1].items() if k != "dictByDistance"}))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"fInscribedToCircumscribed": fRatio, "listCases": listOut}, oFile, indent=2)


if __name__ == "__main__":
    main()
