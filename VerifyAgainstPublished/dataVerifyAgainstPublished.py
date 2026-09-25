#!/usr/bin/env python3
"""Check every reachable quantity in this pipeline against the published Stark et al. record.

This exists because three separate apparent agreements in this project turned out to be
compensating errors. A single end-to-end number agreeing proves very little; a dozen
intermediate quantities agreeing, each tagged with whether the pipeline was fitted to it, proves
considerably more. Every check records its published value, its source, the tolerance, and
whether the pipeline was tuned to it, so a pass on a tuned quantity can never be mistaken for
evidence.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import coronagraph as cg  # noqa: E402
from yieldlib import occurrence as oc  # noqa: E402

LIST_THROUGHPUT_READOFF = [(2.0, 0.05334), (3.0, 0.13374), (3.5, 0.16757), (5.0, 0.2352), (10.0, 0.32547), (20.0, 0.37316)]
LIST_CONTRAST_READOFF = [(2.0, 7.93e-09), (3.0, 5.888e-10), (4.0, 2.861e-10), (10.0, 1.17e-10)]


def fdictCheck(sName, fPublished, fModel, fRelTol, sSource, bTuned=False, sNote=""):
    """One comparison, with provenance and whether the pipeline was fitted to the value."""
    fRel = abs(fModel - fPublished) / abs(fPublished) if fPublished else float("inf")
    return {"sName": sName, "fPublished": fPublished, "fModel": fModel,
            "fRelativeDifference": fRel, "fTolerance": fRelTol,
            "bAgrees": bool(fRel <= fRelTol), "bTuned": bTuned,
            "sSource": sSource, "sNote": sNote}


def flistCoronagraphChecks(dictMission=None):
    """Core throughput and raw contrast against the DMVC6 curves of Stark et al. (2024) Fig. 25.

    These used to compare the parametric stand-in against a handful of points read off the figure
    by eye, at tolerances of 0.35 and 0.75. Those tolerances were loose enough to pass a curve of
    the wrong shape, and did: the read-off itself was wrong by a factor of three at 2 lambda/D
    against the figure's own vector data. The model now interpolates the digitized curve, so the
    comparison is against the same numbers it was built from and the tolerance can be tight --
    what it tests is that the table survived the round trip into the mission parameters intact,
    not that a fit is good.
    """
    dictTable = (dictMission or {}).get("dictCoronagraphTable")
    listOut = []
    for fSep, fPub in LIST_THROUGHPUT_READOFF:
        fModel = (float(cg.faCoreThroughputTable(np.array([fSep]), dictTable)[0]) if dictTable
                  else float(cg.faCoreThroughput(fSep)))
        listOut.append(fdictCheck(f"Upsilon_c at {fSep:g} lambda/D", fPub, fModel,
                                  0.02 if dictTable else 0.35,
                                  "Stark+2024 Fig. 25 (digitized from the PDF content stream)",
                                  bTuned=bool(dictTable),
                                  sNote="Round-trip of the digitized table into the mission "
                                        "parameters; the curve itself is tested against Stark's "
                                        "quoted 5% at 2 lambda/D, 0.45 maximum and 3.5 lambda/D "
                                        "IWA in tests/testPhysics.py." if dictTable else ""))
    for fSep, fPub in LIST_CONTRAST_READOFF:
        fModel = (float(cg.faRawContrastTable(np.array([fSep]), dictTable)[0]) if dictTable
                  else float(cg.faRawContrast(fSep)))
        listOut.append(fdictCheck(f"zeta at {fSep:g} lambda/D", fPub, fModel,
                                  0.05 if dictTable else 0.75,
                                  "Stark+2024 Fig. 25 (digitized from the PDF content stream)",
                                  bTuned=bool(dictTable),
                                  sNote="Round-trip of the digitized table." if dictTable else ""))
    return listOut


def flistPopulationChecks(dictBoxes):
    """The occurrence-rate anchor, which involves no telescope model at all."""
    return [fdictCheck("eta_Earth over the canonical box (SAG13)", 0.24,
                       oc.fnOccurrenceAnalytic(dictBoxes["canonical"], 0.38, -0.19, 0.26),
                       0.03, "Stark+2024 Table 1")]


def fdictFigure10Published(dictFig10):
    """Median, P(yield < 12) and fixed-eta mean read from the digitized Stark+2024 Fig. 10."""
    dictPurple, dictRed = dictFig10["including"], dictFig10["excluding"]
    faK, faP = np.array(dictPurple["iaYield"]), np.array(dictPurple["faProbability"])
    return {"fMedian": float(dictPurple["dictSummary"]["iMedian"]),
            "fBelow12": float(faP[faK < 12].sum()),
            "fFixedEtaMean": float(dictRed["dictSummary"]["fMeanTruncated"]),
            "fMode": fnSmoothedIntegerMode(faP)}


def fnSmoothedIntegerMode(faP):
    """Peak of a 3-bin running mean over P(yield = k); the same estimator for model and paper."""
    return float(np.argmax(np.convolve(faP, np.ones(3) / 3.0, mode="same")))


def faIntegerHistogram(faRealized, iMax=50):
    """P(yield = k), k = 0..iMax-1, from integer yield samples."""
    faCounts = np.bincount(np.clip(np.round(faRealized), 0, iMax).astype(int),
                           minlength=iMax + 1)[:iMax]
    return faCounts / float(len(faRealized))


def flistFigure10Checks(dictSurvey, faRealized, dictFig10):
    """Level and shape of the realized-yield distribution against digitized Fig. 10."""
    dictPub = fdictFigure10Published(dictFig10)
    sSource = "Stark+2024 Fig. 10 (digitized from the PDF content stream)"
    return [
        fdictCheck("Fixed-eta yield mean (Fig. 10 red)", dictPub["fFixedEtaMean"],
                   dictSurvey["fYieldBaseline"], 0.10, sSource,
                   sNote="Albedo and exozodi drawn, eta fixed: the level every other "
                         "distribution check rests on."),
        fdictCheck("Realized-yield distribution median", dictPub["fMedian"],
                   float(np.median(faRealized)), 0.15, sSource),
        fdictCheck("Realized-yield fraction below 12 EECs", dictPub["fBelow12"],
                   float(np.mean(faRealized < 12)), 0.35, sSource,
                   sNote="The low tail that the published mode near 10 reflects."),
        fdictCheck("Realized-yield distribution mean", 21.0,
                   float(np.mean(faRealized)), 0.30, "Stark+2024 Fig. 10 (dotted line)"),
        fdictCheck("Realized-yield distribution mode", dictPub["fMode"],
                   fnSmoothedIntegerMode(faIntegerHistogram(faRealized)), 0.25, sSource,
                   sNote="Peak of a 3-bin running mean, same estimator for both. Stark's curve "
                         "rests on 498 runs x 1000 planet draws, so its peak is well determined; "
                         "the tolerance reflects its flat top, not sampling noise."),
    ]


def flistYieldChecks(dictCal, dictSurvey, dictPred, dictAperture, rng, faRealized):
    """Yields, the sampling distribution, the albedo penalty and the aperture scaling."""
    fPlanning = dictSurvey.get("fYieldPlanningBaseline", dictSurvey["fYieldBaseline"])
    faPoisson = rng.poisson(dictCal["fCalibratedYield"], size=400000).astype(float)
    dictCanon = dictPred["dictByBox"]["canonical"]["dictDistribution"]
    faD = ("6", "7", "8", "9")
    fExponent = np.log(dictAperture["dictByDiameter"]["9"]["fExpectedYieldAtBaselineEta"] /
                       dictAperture["dictByDiameter"]["6"]["fExpectedYieldAtBaselineEta"]) / \
        np.log(1.5)
    bFitted = dictCal.get("bThroughputFitted", True)
    listOut = [
        fdictCheck("Uncalibrated 6 m yield", 22.5, dictCal["fUncalibratedYield"], 0.20,
                   "Stark+2024 Sec. 3.1", sNote="Independent: no fitted parameter."),
        fdictCheck("Throughput factor needed to reach 22.5", 1.0,
                   dictCal.get("fFittedThroughputFactor", dictCal["fCalibratedThroughputFactor"]),
                   1.0, "Plausibility band, not a published value",
                   sNote="Diagnostic. A factor beyond 2x would absorb physics rather than an "
                         "unknown." + ("" if bFitted else " Not applied downstream.")),
        fdictCheck("Sampling distribution mean (Fig. 4)", 22.5, float(np.mean(faPoisson)), 0.20,
                   "Stark+2024 Fig. 4", bTuned=bFitted,
                   sNote="Fig. 4 is sampling only, at A_G = 0.2 and fixed exozodi, i.e. around "
                         "the 6 m planning yield, which is tuned only when the throughput "
                         "factor is fitted. It was previously compared against the albedo-drawn "
                         "yield, the wrong quantity."),
        fdictCheck("Sampling distribution sigma (Fig. 4)", 5.0, float(np.std(faPoisson)), 0.25,
                   "Stark+2024 Fig. 4 (read off)"),
        fdictCheck("Albedo penalty on expected yield", 0.12,
                   1.0 - dictSurvey["fYieldBaseline"] / fPlanning if fPlanning else 0.0, 0.5,
                   "Stark+2024 Sec. 3.2 (22.5 -> 19.8)"),
        fdictCheck("P25 including sigma_eta, 6 m", 0.32,
                   dictCanon["fProbabilityAtLeastGoal"], 0.25, "Stark+2024 Sec. 3.5"),
        fdictCheck("Yield-aperture exponent", 1.90, float(fExponent), 0.15,
                   "Stark+2019 DMVC band and Stark+2024 Fig. 15"),
    ]
    if bFitted:
        listOut.insert(1, fdictCheck("Calibrated 6 m yield", 22.5, dictCal["fCalibratedYield"],
                                     0.02, "Stark+2024 Sec. 3.1", bTuned=True,
                                     sNote="Fitted to this value."))
    listOut += flistDustAndCharacterizationChecks(dictSurvey, dictPred, dictAperture)
    for s, fPub in zip(faD, (0.32, 0.53, 0.67, 0.78)):
        listOut.append(fdictCheck(
            f"P25 including sigma_eta at {s} m", fPub,
            dictAperture["dictByDiameter"][s]["fProbability25IncludingSigmaEta"], 0.25,
            "Stark+2024 Fig. 15" + ("" if s == "6" else " (read off)")))
    return listOut


def flistDustAndCharacterizationChecks(dictSurvey, dictPred, dictAperture):
    """Exozodi penalty, the eta-fixed P25, and Sec. 4.1's characterization times.

    These became checkable when exozodi became a random variable of the pipeline (2026-09-24):
    the penalty is the mean over draws, not one seed, and the eta-fixed P25 is Stark's red curve
    at 6 m. The characterization times are the means of Stark's Fig. 14 over the first 18 EECs,
    which the aperture step now reproduces draw by draw.
    """
    dictDraws = dictSurvey.get("dictDraws", {})
    dictByD = dictAperture["dictByDiameter"]
    fSix, fNine = (dictByD[s].get("fMeanCharDaysFirstN", float("nan")) for s in ("6", "9"))
    dictFixed = dictPred["dictByBox"]["canonical"].get("dictDistributionFixedEta", {})
    return [
        fdictCheck("Exozodi-sampling penalty on expected yield", 0.111,
                   dictDraws.get("fExozodiPenalty", 0.0), 0.5,
                   "Stark+2024 Sec. 3.3 (19.8 -> 17.6)",
                   sNote=f"Mean over {dictSurvey.get('iExozodiDraws', 0)} exozodi draws."),
        fdictCheck("P25 excluding sigma_eta, 6 m", 0.06,
                   dictFixed.get("fProbabilityAtLeastGoal", 0.0), 0.5, "Stark+2024 Sec. 3.5",
                   sNote="Fig. 10 red curve: eta fixed, albedo and exozodi drawn."),
        fdictCheck("Mean characterization time, first 18 EECs, 6 m (days)", 22.0, fSix, 0.3,
                   "Stark+2024 Sec. 4.1"),
        fdictCheck("Mean characterization time, first 18 EECs, 9 m (days)", 3.5, fNine, 0.3,
                   "Stark+2024 Sec. 4.1"),
        fdictCheck("Characterization-time ratio 6 m / 9 m", 6.2, fSix / fNine, 0.2,
                   "Stark+2024 Sec. 4.1"),
    ]


def flistTargetChecks(dfCatalog, dictNpz):
    """The selected-target envelope, against what Fig. 11 shows AYO choosing."""
    faComp = dictNpz["faComp_canonical"][:, -1, -1]
    bUsed = faComp > 0.01
    faLum = dictNpz["faLuminosityLsun"][bUsed]
    faDist = dictNpz["faDistancePc"][bUsed]
    return [
        fdictCheck("Maximum luminosity among reachable targets", 20.0,
                   float(np.max(faLum)) if faLum.size else 0.0, 0.6,
                   "Stark+2024 Fig. 11 (read off)"),
        fdictCheck("Maximum distance among reachable targets", 25.0,
                   float(np.max(faDist)) if faDist.size else 0.0, 0.6,
                   "Stark+2024 Fig. 11 (read off)"),
    ]


def fdictParseArgs():
    """Command-line configuration for the published-record verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--survey", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--aperture", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--yield-samples", required=True)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--fig10-digitised", default="starkFigure10Digitised.json")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-verification", default="publishedVerification.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    rng = np.random.default_rng(dictArgs["seed"])
    dictLoad = {k: json.load(open(dictArgs[k])) for k in
                ("mission_parameters", "calibration", "survey", "prediction", "aperture")}
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    faRealized = np.load(dictArgs["yield_samples"])["faObserved_canonical"].astype(float)
    with open(dictArgs["fig10_digitised"]) as oFile:
        dictFig10 = json.load(oFile)
    listChecks = (flistCoronagraphChecks(dictLoad["mission_parameters"]["dictMission"]) +
                  flistPopulationChecks(dictLoad["mission_parameters"]["dictBoxes"]) +
                  flistYieldChecks(dictLoad["calibration"], dictLoad["survey"],
                                   dictLoad["prediction"], dictLoad["aperture"], rng,
                                   faRealized) +
                  flistFigure10Checks(dictLoad["survey"], faRealized, dictFig10) +
                  flistTargetChecks(pd.read_csv(dictArgs["target_catalog"]), dictNpz))
    listIndependent = [d for d in listChecks if not d["bTuned"]]
    dictOut = {
        "listChecks": listChecks,
        "iChecks": len(listChecks),
        "iIndependent": len(listIndependent),
        "iIndependentAgreeing": sum(d["bAgrees"] for d in listIndependent),
        "listDisagreeing": [d["sName"] for d in listIndependent if not d["bAgrees"]],
    }
    with open(dictArgs["out_verification"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'check':<44}{'published':>11}{'model':>11}{'diff':>8}  verdict")
    print("-" * 88)
    for d in listChecks:
        sVerdict = "TUNED" if d["bTuned"] else ("ok" if d["bAgrees"] else "DISAGREES")
        print(f"{d['sName']:<44}{d['fPublished']:>11.4g}{d['fModel']:>11.4g}"
              f"{100*d['fRelativeDifference']:>7.0f}%  {sVerdict}")
    print(f"\nindependent checks agreeing: {dictOut['iIndependentAgreeing']}"
          f"/{dictOut['iIndependent']}")
    if dictOut["listDisagreeing"]:
        print("disagreeing:")
        for s in dictOut["listDisagreeing"]:
            print("  -", s)


if __name__ == "__main__":
    main()
