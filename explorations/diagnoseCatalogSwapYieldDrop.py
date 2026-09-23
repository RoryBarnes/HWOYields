#!/usr/bin/env python3
"""Explain why swapping the Gaia reconstruction for the real HPIC lowered the uncalibrated yield.

Replacing the target list with the catalog Stark actually used added a third more FGK dwarfs
inside 30 pc, yet the uncalibrated 6 m yield fell from 19.5 to 14.6 and the throughput factor the
calibration step needs rose from 1.49 to 2.83, past its plausibility bound. More targets cannot
by themselves reduce a yield -- the optimizer is free to ignore the extra stars -- so the cause
must be a change in the parameters of the stars that were already there, or in which stars the
screen admits.

This script screens both catalogs through the identical screen and compares them where it
matters: the composition of the top of the priority ordering, the per-star stellar parameters for
the stars common to both, and the completeness actually achieved on the highest-priority targets.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import survey as sv  # noqa: E402


def fdfScreen(dfCatalog, dictParams, dictBox, iMaxStars):
    """Apply the pipeline's own target screen to a catalog, with the calibration step's bounds."""
    return sv.fdfScreenTargets(dfCatalog, dictParams["dictMission"],
                               dictParams["dictBands"]["listBandsDetection"][0],
                               0.5, iMaxStars, 2500.0, 7500.0, dictBox)


def fdictTopComposition(dfScreened, iTop):
    """Distance, temperature and luminosity distribution of the highest-priority targets."""
    dfTop = dfScreened.head(iTop)
    return {
        "iTop": int(iTop),
        "fMedianDistancePc": float(np.median(dfTop["fDistancePc"])),
        "fMedianTeffK": float(np.median(dfTop["fTeffK"])),
        "fMedianLuminosityLsun": float(np.median(dfTop["fLuminosityLsun"])),
        "fMedianEeidLamD": float(np.median(dfTop["fEeidLamD"])),
        "fMedianRadiusRsun": float(np.median(dfTop["fRadiusRsun"])),
        "fFractionInside15Pc": float(np.mean(dfTop["fDistancePc"] < 15.0)),
    }


def fdfCommonStars(dfA, dfB, fMatchArcsec):
    """Positionally match two screened catalogs and return their paired stellar parameters."""
    faRaB, faDecB = dfB["ra"].to_numpy(), dfB["dec"].to_numpy()
    listRows = []
    for dictRow in dfA.to_dict("records"):
        faDd = faDecB - dictRow["dec"]
        faDr = (faRaB - dictRow["ra"]) * np.cos(np.radians(dictRow["dec"]))
        faSep = np.hypot(faDr, faDd) * 3600.0
        iBest = int(np.argmin(faSep))
        if faSep[iBest] <= fMatchArcsec:
            listRows.append({
                "fTeffA": dictRow["fTeffK"], "fTeffB": float(dfB["fTeffK"].iloc[iBest]),
                "fLumA": dictRow["fLuminosityLsun"],
                "fLumB": float(dfB["fLuminosityLsun"].iloc[iBest]),
                "fRadA": dictRow["fRadiusRsun"],
                "fRadB": float(dfB["fRadiusRsun"].iloc[iBest]),
                "fDistA": dictRow["fDistancePc"],
                "fDistB": float(dfB["fDistancePc"].iloc[iBest]),
            })
    return pd.DataFrame(listRows)


def fdictParameterShifts(dfCommon):
    """Median ratios of the stellar parameters for stars present in both catalogs."""
    if dfCommon.empty:
        return {"iMatched": 0}
    return {
        "iMatched": int(len(dfCommon)),
        "fMedianTeffRatioNewOverOld": float(np.median(dfCommon["fTeffB"] / dfCommon["fTeffA"])),
        "fMedianLumRatioNewOverOld": float(np.median(dfCommon["fLumB"] / dfCommon["fLumA"])),
        "fMedianRadiusRatioNewOverOld": float(np.median(dfCommon["fRadB"] / dfCommon["fRadA"])),
        "fMedianDistRatioNewOverOld": float(np.median(dfCommon["fDistB"] / dfCommon["fDistA"])),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-catalog", required=True)
    p.add_argument("--hpic-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--match-arcsec", type=float, default=10.0)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictBox = dictParams["dictBoxes"]["canonical"]
    dfLegacy = fdfScreen(pd.read_csv(dictArgs["legacy_catalog"]), dictParams, dictBox,
                         dictArgs["max_stars"])
    dfHpic = fdfScreen(pd.read_csv(dictArgs["hpic_catalog"]), dictParams, dictBox,
                       dictArgs["max_stars"])
    dfCommonTop = fdfCommonStars(dfLegacy.head(dictArgs["top"]), dfHpic,
                                 dictArgs["match_arcsec"])
    dictOut = {
        "iScreenedLegacy": int(len(dfLegacy)),
        "iScreenedHpic": int(len(dfHpic)),
        "dictTopLegacy": fdictTopComposition(dfLegacy, dictArgs["top"]),
        "dictTopHpic": fdictTopComposition(dfHpic, dictArgs["top"]),
        "dictParameterShiftsTopStars": fdictParameterShifts(dfCommonTop),
        "dictDuplicatesLegacy": fdictDuplicateCheck(dfLegacy, dictArgs["match_arcsec"], 15.0),
        "dictDuplicatesHpic": fdictDuplicateCheck(dfHpic, dictArgs["match_arcsec"], 15.0),
        "listTopLegacy": [
            {"sName": str(r["sSourceId"]), "fDistancePc": round(float(r["fDistancePc"]), 2),
             "fTeffK": round(float(r["fTeffK"]), 0),
             "fLuminosityLsun": round(float(r["fLuminosityLsun"]), 3),
             "fEeidLamD": round(float(r["fEeidLamD"]), 2)}
            for _, r in dfLegacy.head(15).iterrows()],
        "listTopHpic": [
            {"sName": str(r["sSourceId"]), "fDistancePc": round(float(r["fDistancePc"]), 2),
             "fTeffK": round(float(r["fTeffK"]), 0),
             "fLuminosityLsun": round(float(r["fLuminosityLsun"]), 3),
             "fEeidLamD": round(float(r["fEeidLamD"]), 2)}
            for _, r in dfHpic.head(15).iterrows()],
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


def fdictDuplicateCheck(dfCatalog, fMatchArcsec, fMaxDistancePc):
    """Count entries that sit within fMatchArcsec of another entry in the SAME catalog.

    Gaia DR3 resolves a close binary into one source per component, each with its own
    astrophysical parameters, whereas HPIC carries one row per system. A yield model that treats
    both components as independent targets counts the system's habitable zone twice and inflates
    the yield -- so a catalog's internal duplicate rate is a direct check on whether its target
    count is honest.
    """
    dfNear = dfCatalog[dfCatalog["fDistancePc"] <= fMaxDistancePc]
    faRa, faDec = dfNear["ra"].to_numpy(), dfNear["dec"].to_numpy()
    iPairs = 0
    for i in range(len(faRa)):
        faDd = faDec[i + 1:] - faDec[i]
        faDr = (faRa[i + 1:] - faRa[i]) * np.cos(np.radians(faDec[i]))
        iPairs += int(np.sum(np.hypot(faDr, faDd) * 3600.0 <= fMatchArcsec))
    return {"iRowsInsideLimit": int(len(dfNear)), "fMaxDistancePc": float(fMaxDistancePc),
            "fMatchArcsec": float(fMatchArcsec), "iDuplicatePairs": iPairs}


if __name__ == "__main__":
    main()
