#!/usr/bin/env python3
"""Compare this pipeline's per-target completeness with Stark et al. (2024) Fig. 11, star by star.

The digitized figure is the only published per-target completeness data, and it constrains the
HEIGHT of C at the operating point rather than its slope. Matching targets by distance and
luminosity and comparing completeness therefore answers a specific question: is this model's
completeness systematically wrong, or only its response to perturbation? A flat offset would
point at the height, a trend with distance or luminosity at the shape.

Stark's figure is "one representative run", one exozodi draw (seed 10). The model side is
therefore read from A04's exozodi draws: the representative draw is the one whose expected yield
is the median over draws, and the per-star mean over all draws is reported alongside it so a
difference cannot be blamed on which draw was picked. Nothing is recomputed here; the curves
and draws are A04's.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import exozodi as ez  # noqa: E402


def fdictAllocatedOverDraws(dictNpz, dictMission, fEta, iProcesses):
    """Every screened star's allocated completeness in every draw, shape (draws, stars)."""
    dictGrid = ez.fdictLoadGrid(dictNpz, "canonical")
    dictPer = ez.fdictPerStarOverDraws(dictGrid, dictNpz["faZodiDraws"], dictNpz["faTauGridS"],
                                       fEta, dictMission, iProcesses)
    faComp = np.zeros((dictNpz["faZodiDraws"].shape[0], dictNpz["faDistancePc"].size))
    faComp[:, dictGrid["iaStar"]] = dictPer["faStarComp"]
    return faComp


def fdictMatch(faMineD, faMineL, faMineC, listStark, fTolDex, fTolDistance):
    """Pair each published target with the nearest modelled star in distance and luminosity."""
    listRows = []
    for dictPoint in listStark:
        faDistOk = np.abs(faMineD - dictPoint["fDistancePc"]) < fTolDistance
        faLumOk = np.abs(np.log10(np.maximum(faMineL, 1e-6)) -
                         np.log10(max(dictPoint["fLuminosityLsun"], 1e-6))) < fTolDex
        faBoth = np.where(faDistOk & faLumOk)[0]
        if not faBoth.size:
            continue
        faScore = (np.abs(faMineD[faBoth] - dictPoint["fDistancePc"]) / fTolDistance) ** 2 + \
            (np.abs(np.log10(np.maximum(faMineL[faBoth], 1e-6)) -
                    np.log10(max(dictPoint["fLuminosityLsun"], 1e-6))) / fTolDex) ** 2
        i = int(faBoth[int(np.argmin(faScore))])
        listRows.append({"fDistancePc": dictPoint["fDistancePc"],
                         "fLuminosityLsun": dictPoint["fLuminosityLsun"],
                         "fCompletenessPublished": dictPoint["fCompleteness"],
                         "fCompletenessModel": float(faMineC[i])})
    return listRows


def fdictSummary(listRows, listStark, faMineC):
    """Headline comparison numbers for one model completeness vector."""
    faPub = np.array([d["fCompletenessPublished"] for d in listRows])
    faMod = np.array([d["fCompletenessModel"] for d in listRows])
    faDist = np.array([d["fDistancePc"] for d in listRows])
    return {
        "iMatched": len(listRows), "iPublishedTargets": len(listStark),
        "iModelStarsUsed": int(np.sum(faMineC > 0)),
        "fPublishedSummedCompleteness": float(sum(d["fCompleteness"] for d in listStark)),
        "fModelSummedCompleteness": float(faMineC.sum()),
        "fMedianPublished": float(np.median(faPub)) if faPub.size else 0.0,
        "fMedianModel": float(np.median(faMod)) if faMod.size else 0.0,
        "fMedianRatio": float(np.median(faMod[faPub > 0.05] / faPub[faPub > 0.05]))
        if np.any(faPub > 0.05) else 0.0,
        "fTrendWithDistance": float(np.polyfit(faDist, faMod - faPub, 1)[0])
        if len(listRows) > 3 else 0.0,
        "listRows": listRows}


def fdictParseArgs():
    """Command-line configuration for the per-target completeness comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digitised", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--tolerance-dex", type=float, default=0.12)
    p.add_argument("--tolerance-distance-pc", type=float, default=1.2)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--out-json", default="targetCompletenessComparison.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictMission = json.load(oFile)["dictMission"]
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    faComp = fdictAllocatedOverDraws(dictNpz, dictMission, dictArgs["eta_earth"],
                                     dictArgs["processes"])
    iRep = int(np.argsort(faComp.sum(axis=1))[faComp.shape[0] // 2])
    with open(dictArgs["digitised"]) as oFile:
        listStark = json.load(oFile)["listPoints"]
    faD, faL = dictNpz["faDistancePc"], dictNpz["faLuminosityLsun"]
    dictOut = {"iRepresentativeDraw": iRep, "iExozodiDraws": int(faComp.shape[0]),
               "faDistancePc": faD.tolist(), "faLuminosityLsun": faL.tolist(),
               "faCompletenessRepresentative": faComp[iRep].tolist(),
               "faCompletenessDrawMean": faComp.mean(axis=0).tolist(),
               "faFractionOfDrawsObserved": np.mean(faComp > 0, axis=0).tolist()}
    for sKey, faC in (("dictRepresentative", faComp[iRep]), ("dictDrawMean", faComp.mean(axis=0))):
        dictOut[sKey] = fdictSummary(fdictMatch(faD, faL, faC, listStark, dictArgs["tolerance_dex"],
                                                dictArgs["tolerance_distance_pc"]), listStark, faC)
    dictOut.update({k: v for k, v in dictOut["dictRepresentative"].items() if k != "listRows"})
    dictOut["listRows"] = dictOut["dictRepresentative"]["listRows"]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sKey in ("dictRepresentative", "dictDrawMean"):
        print(sKey, json.dumps({k: v for k, v in dictOut[sKey].items() if k != "listRows"}))


if __name__ == "__main__":
    main()
