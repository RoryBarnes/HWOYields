#!/usr/bin/env python3
"""Tabulate every screened star's properties, its Earth zone and classic habitable zone, and the
survey's expected yield in each, with statistics comparing the two zones.

The Earth zone is this project's redefined exoEarth-candidate box: 0.96-1.20 AU for the Sun,
scaled as sqrt(L), holding 0.5-2 Earth-mass planets (R = M^0.27). The classic habitable zone is
Stark et al. (2024)'s canonical box, 0.95-1.67 AU scaled the same way. Both are given in AU and in
milliarcseconds, and in lambda/D of the 6 m telescope at 550 nm (detection) and 1000 nm
(characterization), with the coronagraph's half-throughput inner working angle (3.5 lambda/D in
the circumscribed-diameter units of Stark's coronagraph figure) for reference.

For the survey columns, each box's survey is re-optimized in every exozodi draw of A04 at that
box's posterior-median eta (A06), and the per-star completeness reached is averaged over draws;
the expected EEC yield of a star is eta times that mean. A star's classic-zone and Earth-zone
columns therefore come from two different optimized surveys, each the best survey for its own
definition of an exoEarth candidate, which is the comparison the headline ratio makes.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import coronagraph as cg  # noqa: E402
from yieldlib import exozodi as ez  # noqa: E402

F_MAS_PER_ARCSEC = 1000.0
DICT_ZONES = {"EarthZone": (0.96, 1.20), "ClassicHz": (0.95, 1.67)}
LIST_LUMINOSITY_BINS = [(0.0, 0.3, "L < 0.3"), (0.3, 0.6, "0.3-0.6"), (0.6, 1.0, "0.6-1"),
                        (1.0, 2.0, "1-2"), (2.0, 5.0, "2-5"), (5.0, 1000.0, "L > 5")]
LIST_DISTANCE_BINS = [(0.0, 5.0, "< 5 pc"), (5.0, 10.0, "5-10 pc"), (10.0, 15.0, "10-15 pc"),
                      (15.0, 20.0, "15-20 pc"), (20.0, 100.0, "> 20 pc")]


def fdfStarProperties(dfCatalog, saSourceId):
    """Catalog properties of the screened stars, in A04's screening order."""
    dfIndexed = dfCatalog.drop_duplicates("sSourceId").set_index("sSourceId")
    dfOut = dfIndexed.loc[list(saSourceId), ["sSimbadName", "sHipName", "sSpectralType", "fVmag",
                                             "fTeffK", "fRadiusRsun", "fMassMsun",
                                             "fLuminosityLsun", "fDistancePc"]].reset_index()
    dfOut["sName"] = [str(s).lstrip("* ").strip() if isinstance(s, str) else ""
                      for s in dfOut["sSimbadName"]]
    dfOut["sHip"] = [f"{int(h)}" if h == h else "" for h in dfOut["sHipName"]]
    return dfOut.drop(columns=["sSimbadName", "sHipName"])


def fnAddZoneGeometry(dfStars, dictMission):
    """Zone edges in AU, mas and lambda/D (inscribed D) at 550 and 1000 nm, in place."""
    faSqrtL = np.sqrt(dfStars["fLuminosityLsun"].to_numpy())
    faDistance = dfStars["fDistancePc"].to_numpy()
    dfStars["fEeidAu"] = faSqrtL
    dfStars["fEeidMas"] = F_MAS_PER_ARCSEC * faSqrtL / faDistance
    for sZone, (fInner, fOuter) in DICT_ZONES.items():
        for sEdge, fEdge in (("Inner", fInner), ("Outer", fOuter)):
            dfStars[f"f{sZone}{sEdge}Au"] = fEdge * faSqrtL
            dfStars[f"f{sZone}{sEdge}Mas"] = F_MAS_PER_ARCSEC * fEdge * faSqrtL / faDistance
            for iNm in (550, 1000):
                fLamDMas = F_MAS_PER_ARCSEC * cg.fnLambdaOverDArcsec(iNm * 1e-9,
                                                                    dictMission["fDiameterM"])
                dfStars[f"f{sZone}{sEdge}LamD{iNm}"] = dfStars[f"f{sZone}{sEdge}Mas"] / fLamDMas


def fnIwaInscribedLamD(dictMission, fIwaCircumscribedLamD=3.5):
    """Half-throughput IWA converted from circumscribed to inscribed lambda/D."""
    return fIwaCircumscribedLamD / dictMission.get("fCircumscribedRatio", 1.0)


def fdictSurveyPerStar(dictNpz, sBox, fEta, dictMission, iProcesses):
    """Per-star mean completeness, draw-observed fraction and mean char time, all stars."""
    dictGrid = ez.fdictLoadGrid(dictNpz, sBox)
    dictPer = ez.fdictPerStarOverDraws(dictGrid, dictNpz["faZodiDraws"], dictNpz["faTauGridS"],
                                       fEta, dictMission, iProcesses)
    iStars = dictNpz["faDistancePc"].size
    dictOut = {k: np.zeros(iStars) for k in ("fComp", "fCompAlbedo", "fObservedFraction",
                                             "fTimeDays", "fCharDays")}
    iaStar = dictGrid["iaStar"]
    faObserved = dictPer["faStarComp"] > 0
    dictOut["fComp"][iaStar] = dictPer["faStarComp"].mean(axis=0)
    dictOut["fCompAlbedo"][iaStar] = dictPer["faStarCompYield"].mean(axis=0)
    dictOut["fObservedFraction"][iaStar] = faObserved.mean(axis=0)
    dictOut["fTimeDays"][iaStar] = dictPer["faStarTimeS"].mean(axis=0) / 86400.0
    faCharSum = np.where(faObserved, dictPer["faStarTauCharMeanS"], 0.0).sum(axis=0)
    dictOut["fCharDays"][iaStar] = faCharSum / np.maximum(faObserved.sum(axis=0), 1) / 86400.0
    return dictOut


def fnAddSurveyColumns(dfStars, dictNpz, dictEta, dictMission, iProcesses):
    """Mean completeness and expected yield per star, for each zone's own optimized survey."""
    for sZone, sBox in (("ClassicHz", "canonical"), ("EarthZone", "redefined")):
        dictPer = fdictSurveyPerStar(dictNpz, sBox, dictEta[sBox], dictMission, iProcesses)
        for sKey, faValues in dictPer.items():
            dfStars[f"{sKey}{sZone}"] = faValues
        dfStars[f"fYield{sZone}"] = dictEta[sBox] * dfStars[f"fCompAlbedo{sZone}"]
        faMax = dictNpz[f"faComp_{sBox}"][:, -1, -1]
        dfStars[f"fCompMaxReachable{sZone}"] = faMax


def fdictBinnedComparison(dfStars, sColumn, listBins):
    """Stars observed and expected yield in each zone, and their ratio, per bin of sColumn."""
    dictOut = {}
    for fLo, fHi, sLabel in listBins:
        dfBin = dfStars[(dfStars[sColumn] >= fLo) & (dfStars[sColumn] < fHi)]
        fYieldHz, fYieldEz = dfBin["fYieldClassicHz"].sum(), dfBin["fYieldEarthZone"].sum()
        dictOut[sLabel] = {"iStars": int(len(dfBin)),
                           "iObservedClassicHz": int(np.sum(dfBin["fObservedFractionClassicHz"] >= 0.5)),
                           "iObservedEarthZone": int(np.sum(dfBin["fObservedFractionEarthZone"] >= 0.5)),
                           "fYieldClassicHz": float(fYieldHz), "fYieldEarthZone": float(fYieldEz),
                           "fYieldRatio": float(fYieldEz / fYieldHz) if fYieldHz > 0 else None}
    return dictOut


def fdictAngularComparison(dfStars, fIwaLamD):
    """How much of each zone clears the IWA, over stars observed in either survey."""
    dfUsed = dfStars[(dfStars["fObservedFractionClassicHz"] > 0) |
                     (dfStars["fObservedFractionEarthZone"] > 0)]
    dictOut = {"fIwaInscribedLamD": fIwaLamD, "iStarsConsidered": int(len(dfUsed))}
    for sZone in DICT_ZONES:
        for iNm in (550, 1000):
            faIn = dfUsed[f"f{sZone}InnerLamD{iNm}"].to_numpy()
            faOut = dfUsed[f"f{sZone}OuterLamD{iNm}"].to_numpy()
            faClear = np.clip((faOut - np.maximum(faIn, fIwaLamD)) / (faOut - faIn), 0.0, 1.0)
            dictOut[f"f{sZone}MeanFractionOutsideIwa{iNm}"] = float(np.mean(faClear))
            dictOut[f"f{sZone}FractionFullyOutsideIwa{iNm}"] = float(np.mean(faIn >= fIwaLamD))
    return dictOut


def fdictSummary(dfStars, dictEta, fIwaLamD):
    """Headline zone-versus-zone statistics."""
    fYieldHz, fYieldEz = dfStars["fYieldClassicHz"].sum(), dfStars["fYieldEarthZone"].sum()
    dfBoth = dfStars[(dfStars["fYieldClassicHz"] > 1e-3) & (dfStars["fYieldEarthZone"] > 1e-3)]
    faRatio = (dfBoth["fYieldEarthZone"] / dfBoth["fYieldClassicHz"]).to_numpy()
    return {
        "dictEta": dictEta, "fEtaRatio": dictEta["redefined"] / dictEta["canonical"],
        "fZoneWidthRatio": (1.20 - 0.96) / (1.67 - 0.95),
        "fYieldClassicHz": float(fYieldHz), "fYieldEarthZone": float(fYieldEz),
        "fYieldRatio": float(fYieldEz / fYieldHz),
        "iStarsObservedClassicHz": int(np.sum(dfStars["fObservedFractionClassicHz"] >= 0.5)),
        "iStarsObservedEarthZone": int(np.sum(dfStars["fObservedFractionEarthZone"] >= 0.5)),
        "fMeanCompletenessObservedClassicHz": float(dfStars.loc[
            dfStars["fObservedFractionClassicHz"] >= 0.5, "fCompClassicHz"].mean()),
        "fMeanCompletenessObservedEarthZone": float(dfStars.loc[
            dfStars["fObservedFractionEarthZone"] >= 0.5, "fCompEarthZone"].mean()),
        "dictPerStarYieldRatio": {"iStars": int(faRatio.size),
                                  "fMedian": float(np.median(faRatio)),
                                  "fP16": float(np.percentile(faRatio, 16)),
                                  "fP84": float(np.percentile(faRatio, 84))},
        "fShareOfYieldTop20ClassicHz": float(np.sort(dfStars["fYieldClassicHz"])[::-1][:20].sum()
                                             / fYieldHz),
        "fShareOfYieldTop20EarthZone": float(np.sort(dfStars["fYieldEarthZone"])[::-1][:20].sum()
                                             / fYieldEz),
        "dictByLuminosity": fdictBinnedComparison(dfStars, "fLuminosityLsun",
                                                  LIST_LUMINOSITY_BINS),
        "dictByDistance": fdictBinnedComparison(dfStars, "fDistancePc", LIST_DISTANCE_BINS),
        "dictAngular": fdictAngularComparison(dfStars, fIwaLamD),
    }


def fdictParseArgs():
    """Command-line configuration for the Earth-zone tables."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--occurrence-summary", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--out-table", default="earthZoneTable.csv")
    p.add_argument("--out-summary", default="earthZoneSummary.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictMission = json.load(oFile)["dictMission"]
    with open(dictArgs["occurrence_summary"]) as oFile:
        dictEtaByBox = json.load(oFile)["dictEtaByBox"]
    dictEta = {s: float(dictEtaByBox[s]["fMedian"]) for s in ("canonical", "redefined")}
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    dfStars = fdfStarProperties(pd.read_csv(dictArgs["target_catalog"]), dictNpz["saSourceId"])
    fnAddZoneGeometry(dfStars, dictMission)
    fnAddSurveyColumns(dfStars, dictNpz, dictEta, dictMission, dictArgs["processes"])
    dfStars = dfStars.sort_values("fYieldClassicHz", ascending=False).reset_index(drop=True)
    dfStars.to_csv(dictArgs["out_table"], index=False, float_format="%.6g")
    dictSummary = fdictSummary(dfStars, dictEta, fnIwaInscribedLamD(dictMission))
    dictSummary["iStarsTabulated"] = int(len(dfStars))
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictSummary.items() if not k.startswith("dictBy")},
                     indent=2))


if __name__ == "__main__":
    main()
