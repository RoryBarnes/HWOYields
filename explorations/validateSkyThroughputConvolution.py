#!/usr/bin/env python3
"""Test the PSF-convolution reconstruction of T_sky against AYO's own values for the USORT OVC.

Stark et al. (2019) define T_sky(x,y) as the PSF convolved with a uniform background. The model's
current reconstruction, T_sky(r) = Upsilon_c(r) / EE(X), is that map WITHOUT the convolution.
yieldlib.coronagraph.fdictConvolvedSkyThroughput adds it. The Stark et al. (2025) ETC benchmark
gives AYO's T_sky (skytrans / 16, verified in compareEtcBenchmarkTermByTerm.py) and core
throughput at 14 separations for the USORT vortex coronagraph. That coronagraph's full curve is not
published, so its Upsilon_c is fitted here with a logistic ramp in log separation to those same
14 points; both reconstructions are then evaluated at the benchmark separations and compared with
AYO's T_sky. The same two reconstructions are also tabulated for the DMVC6 the model uses.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))
sys.path.insert(0, S_HERE)
from compareEtcBenchmarkTermByTerm import (DICT_SCENARIOS, F_USORT_CIRC_M,  # noqa: E402
                                           F_USORT_INSCRIBED_M, fdictAyoInternal,
                                           fdictReadWorkbook)
from yieldlib import coronagraph as cg  # noqa: E402


def faRamp(faSep, fMax, fIwa, fIndex):
    """Logistic ramp in log separation (the form of yieldlib.coronagraph.faCoreThroughput)."""
    return fMax / (1.0 + (fIwa / faSep) ** fIndex)


def flistBenchmarkPoints(dictBooks):
    """Unique (separation, core throughput, T_sky) triples over every benchmark scenario."""
    dictPoints = {}
    for sBook, iCol, fLambdaM in DICT_SCENARIOS.values():
        for dictA in fdictReadWorkbook(dictBooks[sBook], iCol).values():
            fTsky = fdictAyoInternal(dictA, fLambdaM)["fTskyImplied"]
            dictPoints[round(dictA["sp"], 3)] = (dictA["sp"], dictA["T_core"], fTsky)
    return sorted(dictPoints.values())


def fdictReconstructions(faSepCurve, faUpsilon, fRatio, fApertureRadius, faAt):
    """Unconvolved and convolved T_sky evaluated at faAt (circumscribed lambda/D)."""
    fEe = cg.fnAiryEncircledEnergy(fApertureRadius)
    faTotal = np.asarray(faUpsilon) / fEe
    dictConv = cg.fdictConvolvedSkyThroughput(np.asarray(faSepCurve), faTotal, fRatio)
    return {"faUnconvolved": np.interp(faAt, faSepCurve, faTotal),
            "faConvolved": np.interp(faAt, dictConv["faSeparation"], dictConv["faSkyThroughput"])}


def fdictOvcTest(listPoints, fApertureRadius):
    """Fit the OVC ramp and compare both reconstructions with AYO's T_sky."""
    faSep, faCore, faTsky = (np.array(c) for c in zip(*listPoints))
    faPar, _ = curve_fit(faRamp, faSep, faCore, p0=(0.4, 3.0, 3.0), maxfev=20000)
    faCurveSep = np.linspace(0.05, 30.0, 3000)
    dictRec = fdictReconstructions(faCurveSep, faRamp(faCurveSep, *faPar),
                                   F_USORT_CIRC_M / F_USORT_INSCRIBED_M, fApertureRadius, faSep)
    return {"faRampParams": faPar.tolist(),
            "fCoreFitRmsFrac": float(np.sqrt(np.mean((faRamp(faSep, *faPar) / faCore - 1) ** 2))),
            "listRows": [{"fSepCirc": float(s), "fCoreAyo": float(c), "fTskyAyo": float(t),
                          "fUnconvolvedOverAyo": float(u / t), "fConvolvedOverAyo": float(v / t)}
                         for s, c, t, u, v in zip(faSep, faCore, faTsky, dictRec["faUnconvolved"],
                                                  dictRec["faConvolved"])]}


def fdictEmpiricalShape(dictOvc):
    """AYO T_sky over the unconvolved reconstruction, fitted linearly in Upsilon_c / Upsilon_max.

    Expressed against the fraction of maximum core throughput rather than separation, so that it
    can be carried to another coronagraph as a bracket. This is an empirical transfer from the
    OVC, not a property derived for the DMVC6.
    """
    fMax = dictOvc["faRampParams"][0]
    faU = np.array([r["fCoreAyo"] / fMax for r in dictOvc["listRows"]])
    faF = np.array([1.0 / r["fUnconvolvedOverAyo"] for r in dictOvc["listRows"]])
    faCoef = np.polyfit(faU, faF, 1)
    return {"faCoreFraction": [0.0, 1.0],
            "faFactor": [float(max(np.polyval(faCoef, 0.0), 1.0)),
                         float(max(np.polyval(faCoef, 1.0), 1.0))],
            "fRmsFrac": float(np.sqrt(np.mean((np.polyval(faCoef, faU) / faF - 1) ** 2))),
            "fMaxUsed": float(fMax), "fCoreFractionRange": [float(faU.min()), float(faU.max())]}


def fdictDmvc6Table(dictMission):
    """Both reconstructions for the model's DMVC6, as T_sky / Upsilon_c, at chosen separations."""
    dictTable = dictMission["dictCoronagraphTable"]
    faAt = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0])
    dictRec = fdictReconstructions(dictTable["faSeparationUpsilon"], dictTable["faUpsilon"],
                                   dictMission["fCircumscribedRatio"],
                                   dictMission["fApertureRadiusLamD"], faAt)
    faCore = cg.faCoreThroughputTable(faAt, dictTable)
    return [{"fSepCirc": float(s), "fUpsilon": float(c), "fTskyUnconvolved": float(u),
             "fTskyConvolved": float(v), "fConvolvedOverUnconvolved": float(v / u)}
            for s, c, u, v in zip(faAt, faCore, dictRec["faUnconvolved"], dictRec["faConvolved"])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--char-xlsx", default="reference/ETC_cal_char.xlsx")
    p.add_argument("--detect-xlsx", default="reference/ETC_cal_detect.xlsx")
    p.add_argument("--mission", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--out-json", default="output/skyThroughputConvolutionTest.json")
    dictArgs = vars(p.parse_args())
    listPoints = flistBenchmarkPoints({"char": dictArgs["char_xlsx"],
                                       "detect": dictArgs["detect_xlsx"]})
    dictMission = json.load(open(dictArgs["mission"]))["dictMission"]
    dictOut = {"dictOvc": fdictOvcTest(listPoints, dictMission["fApertureRadiusLamD"]),
               "listDmvc6": fdictDmvc6Table(dictMission)}
    dictOut["dictEmpiricalShape"] = fdictEmpiricalShape(dictOut["dictOvc"])
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    print("OVC ramp", dictOut["dictOvc"]["faRampParams"], "core fit rms",
          round(dictOut["dictOvc"]["fCoreFitRmsFrac"], 3))
    for r in dictOut["dictOvc"]["listRows"]:
        print(f"  sp {r['fSepCirc']:.2f}  unconv/AYO {r['fUnconvolvedOverAyo']:.3f}  "
              f"conv/AYO {r['fConvolvedOverAyo']:.3f}")
    print("empirical shape", dictOut["dictEmpiricalShape"])
    for r in dictOut["listDmvc6"]:
        print(f"  DMVC6 {r['fSepCirc']:5.1f}  Ups {r['fUpsilon']:.3f}  conv/unconv "
              f"{r['fConvolvedOverUnconvolved']:.2f}")


if __name__ == "__main__":
    main()
