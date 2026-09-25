#!/usr/bin/env python3
"""Tabulate the calibration-degeneracy posteriors in natural units against the published targets.

Reads fitCalibrationDegeneracy.py's output and reports, for each posterior (yield-only and all
published constraints): the throughput factors kappa = exp(ln_kappa) and kappa_c, the
posterior-predicted observables in days and completeness rather than logs, and whether the
published value lies inside the 90% predictive interval. A target outside the interval means no
point in the four-parameter box reproduces it, which is the question the design was run to answer.
Also reports the singular values of the local sensitivity, i.e. how many parameter combinations
the published observables constrain at all.
"""

import argparse
import glob
import json
import os

import numpy as np

LIST_LOG_OBS = ["ln_yield6", "ln_char18_6", "ln_char18_9"]
DICT_OBS_LABEL = {"ln_yield6": "6 m planning yield [EEC]",
                  "ln_char18_6": "first-18 mean char time, 6 m [d]",
                  "ln_char18_9": "first-18 mean char time, 9 m [d]",
                  "fig11_6-10": "Fig. 11 median completeness, 6-10 pc",
                  "fig11_10-15": "Fig. 11 median completeness, 10-15 pc",
                  "fig11_15-20": "Fig. 11 median completeness, 15-20 pc"}


def ftNatural(sObs, faTriple):
    """One observable's (low, median, high) in natural units."""
    faOut = np.exp(faTriple) if sObs in LIST_LOG_OBS else np.asarray(faTriple)
    return tuple(float(f) for f in faOut)


def fdictObservableRow(sObs, faPredicted, fTarget):
    """Predicted interval, published target and whether the target is reachable."""
    fLo, fMid, fHi = ftNatural(sObs, faPredicted)
    fPub = float(np.exp(fTarget)) if sObs in LIST_LOG_OBS else float(fTarget)
    return {"sLabel": DICT_OBS_LABEL[sObs], "fLow": fLo, "fMedian": fMid, "fHigh": fHi,
            "fPublished": fPub, "bTargetInsideInterval": bool(fLo <= fPub <= fHi)}


def fdictPosteriorSummary(dictPost, listObs, faTargets, listParams):
    """Parameters in natural units plus the observable rows for one posterior."""
    dictParams = {s: {"fMean": float(f), "fStd": float(fSd)} for s, f, fSd
                  in zip(listParams, dictPost["faMean"], dictPost["faStd"])}
    for sName, sLog in [("kappa", "ln_kappa"), ("kappa_c", "ln_kappa_c")]:
        dictParams[sName] = {"fMedian": float(np.exp(dictParams[sLog]["fMean"])),
                             "fLnStd": dictParams[sLog]["fStd"]}
    iA, iB = listParams.index("ln_kappa"), listParams.index("ln_kappa_c")
    return {"dictParameters": dictParams,
            "fKappaKappaCCorrelation": float(dictPost["faCorrelation"][iA][iB]),
            "listObservables": [fdictObservableRow(s, dictPost["dictPredicted"][s], f)
                                for s, f in zip(listObs, faTargets)]}


def fnPrintSummary(dictOut):
    """Human-readable dump of the summary."""
    faSv = dictOut["faSensitivitySingularValues"]
    print(f"design points {dictOut['iDesignPoints']}; sensitivity singular values "
          + " ".join(f"{f:.2f}" for f in faSv) + f" (ratio {faSv[0] / faSv[1]:.1f})")
    dictGate = dictOut["dictYieldGatedEnvelope"]
    if dictGate["iPoints"]:
        print(f"\nof the points reproducing the yield to {dictGate['fFraction']:.0%} "
              f"({dictGate['iPoints']} of 56): char time 6 m "
              f"{dictGate['fChar6Min']:.1f}-{dictGate['fChar6Max']:.1f} d (Stark 22), 9 m "
              f"{dictGate['fChar9Min']:.2f}-{dictGate['fChar9Max']:.2f} d (Stark 3.5)")
    dictEnv = dictOut["dictDesignEnvelope"]
    print(f"\nrange over the {dictEnv['iPoints']} evaluated points (no emulator):")
    for sObs, dictR in dictEnv["dictRange"].items():
        print(f"  {DICT_OBS_LABEL[sObs]:<38} {dictR['fMin']:8.2f} to {dictR['fMax']:.2f}")
    for sKey, dictPost in dictOut["dictPosteriors"].items():
        dictP = dictPost["dictParameters"]
        print(f"\n[{sKey}] kappa {dictP['kappa']['fMedian']:.3f} "
              f"(ln sd {dictP['kappa']['fLnStd']:.3f}), kappa_c "
              f"{dictP['kappa_c']['fMedian']:.3f} (ln sd {dictP['kappa_c']['fLnStd']:.3f}), "
              f"corr(ln kappa, ln kappa_c) {dictPost['fKappaKappaCCorrelation']:+.2f}")
        for dictRow in dictPost["listObservables"]:
            sMark = "reachable" if dictRow["bTargetInsideInterval"] else "OUT OF REACH"
            print(f"  {dictRow['sLabel']:<38} {dictRow['fMedian']:8.2f} "
                  f"[{dictRow['fLow']:.2f}, {dictRow['fHigh']:.2f}]  "
                  f"published {dictRow['fPublished']:.2f}  {sMark}")


def fdictDesignEnvelope(sDir):
    """Range each observable actually took over the directly evaluated design points.

    Emulator-free: if a published value lies outside this range, no evaluated point reached it.
    """
    listPoints = [json.load(open(s)) for s in sorted(glob.glob(os.path.join(sDir, "point*.json")))]
    dictRaw = {"ln_yield6": [d["fYield6"] for d in listPoints],
               "ln_char18_6": [d["fChar18Days6"] for d in listPoints],
               "ln_char18_9": [d["fChar18Days9"] for d in listPoints]}
    for sBin in ["6-10", "10-15", "15-20"]:
        dictRaw[f"fig11_{sBin}"] = [d["dictFigElevenMedianByDistance"].get(sBin) for d in listPoints]
    return {"iPoints": len(listPoints),
            "dictRange": {s: {"fMin": float(np.nanmin(np.array(fa, dtype=float))),
                              "fMax": float(np.nanmax(np.array(fa, dtype=float)))}
                          for s, fa in dictRaw.items()}}


def fdictYieldGatedEnvelope(sDir, fYieldTarget=22.5, fFrac=0.10):
    """Char times and Fig. 11 medians among the points that DO reproduce the published yield.

    The posteriors push kappa_c to its lower prior bound trying to slow spectra down, so the
    "22 d is unreachable" statement has to be made at fixed yield: buying characterization time
    by losing yield is not a reproduction of Stark. Emulator-free.
    """
    listPoints = [json.load(open(s)) for s in sorted(glob.glob(os.path.join(sDir, "point*.json")))]
    listGated = [d for d in listPoints
                 if abs(d["fYield6"] - fYieldTarget) <= fFrac * fYieldTarget]
    if not listGated:
        return {"iPoints": 0}
    faChar6 = np.array([d["fChar18Days6"] for d in listGated], dtype=float)
    faChar9 = np.array([d["fChar18Days9"] for d in listGated], dtype=float)
    return {"iPoints": len(listGated), "fYieldTarget": fYieldTarget, "fFraction": fFrac,
            "fChar6Min": float(faChar6.min()), "fChar6Max": float(faChar6.max()),
            "fChar9Min": float(faChar9.min()), "fChar9Max": float(faChar9.max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-json", default="calibrationDegeneracy.json")
    p.add_argument("--design-dir", default="design")
    p.add_argument("--out-json", default="calibrationDegeneracySummary.json")
    dictArgs = vars(p.parse_args())
    dictFit = json.load(open(dictArgs["fit_json"]))
    dictOut = {"iDesignPoints": dictFit["iDesignPoints"],
               "faSensitivitySingularValues":
                   dictFit["dictSensitivityAtAdopted"]["faSingularValues"],
               "listParams": dictFit["listParams"],
               "faLeadingDirection": dictFit["dictSensitivityAtAdopted"]["faDirections"][0],
               "dictDesignEnvelope": fdictDesignEnvelope(dictArgs["design_dir"]),
               "dictYieldGatedEnvelope": fdictYieldGatedEnvelope(dictArgs["design_dir"]),
               "dictPosteriors": {s: fdictPosteriorSummary(d, dictFit["listObservables"],
                                                           dictFit["faTargets"],
                                                           dictFit["listParams"])
                                  for s, d in dictFit["dictPosteriors"].items()}}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    fnPrintSummary(dictOut)


if __name__ == "__main__":
    main()
