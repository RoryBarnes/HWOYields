#!/usr/bin/env python3
"""Collect, for every Stark et al. (2024) figure with a like-for-like model counterpart, the model
series in the published figure's own axes and units.

Only figures whose quantity this pipeline computes under the same assumptions are included, so
each pair can be read side by side. The published figures themselves are copied unaltered from the
arXiv e-print source of arXiv:2405.19418 (ms_v2.tex) into reference/, keeping their figure
numbers:

  Fig. 4   exoplanet sampling only, A_G = 0.2, exozodi fixed (tuned: its mean is the calibrated 22.5)
  Fig. 7   green: albedo drawn, exozodi fixed; orange: exozodi drawn as well
  Fig. 9   drawn exozodi levels (left, maximum-likelihood curve) and the eta-fixed yield (right)
  Fig. 10  realized yield including (purple) and excluding (red) eta_Earth uncertainty
  Fig. 11  per-target completeness, one representative exozodi draw, 6 m
  Fig. 12  realized yield distributions at 6, 7, 8 and 9 m
  Fig. 14  characterization times of the first 18 EECs at 6-9 m
  Fig. 15  P25 against inscribed diameter
  Fig. 25  DMVC6 contrast and core throughput (the digitized input, a round-trip check)

Figures without a counterpart (the optical layouts, the throughput and QE curves, the design-change
scenarios) are not reproduced.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import yielddistribution as yd  # noqa: E402

I_YIELD_MAX = 60


def fdictFigure04And07(dictSurvey):
    """Sampling-only and albedo/exozodi yield distributions at eta = 0.24 (6 m)."""
    faByDraw = np.asarray(dictSurvey["dictDraws"]["faYieldByDraw"])
    return {"fig04": {"faPmf": yd.faPoissonMixturePmf(
                          [dictSurvey["fYieldPlanningFixedExozodi"]], I_YIELD_MAX).tolist(),
                      "fMean": dictSurvey["fYieldPlanningFixedExozodi"]},
            "fig07": {"faPmfAlbedo": yd.faPoissonMixturePmf(
                          [dictSurvey["fYieldFixedExozodi"]], I_YIELD_MAX).tolist(),
                      "faPmfAlbedoExozodi": yd.faPoissonMixturePmf(faByDraw, I_YIELD_MAX).tolist(),
                      "fMeanAlbedo": dictSurvey["fYieldFixedExozodi"],
                      "fMeanAlbedoExozodi": float(np.mean(faByDraw))}}


def fdictFigure09(faZodiDraws, dictSurvey, iPublishedDraws=5_000_000):
    """Histogram of drawn levels in 2-zodi bins, scaled to Stark's 500 x 10k draws."""
    faEdges = np.arange(0.0, 1002.0, 2.0)
    faCounts = np.histogram(np.ravel(faZodiDraws), faEdges)[0].astype(float)
    faByDraw = np.asarray(dictSurvey["dictDraws"]["faYieldByDraw"])
    return {"faZodiEdges": faEdges.tolist(),
            "faNumberScaled": (faCounts * iPublishedDraws / faZodiDraws.size).tolist(),
            "fMedianZodi": float(np.median(faZodiDraws)),
            "faPmfEtaFixed": yd.faPoissonMixturePmf(faByDraw, I_YIELD_MAX).tolist()}


def fdictFigure10(dictSamples):
    """Realized yield including and excluding eta_Earth uncertainty (6 m, canonical box)."""
    faIncl = dictSamples["faExpected_canonical"]
    faExcl = dictSamples["faExpectedFixedEta_canonical"]
    return {"faPmfIncluding": yd.faPoissonMixturePmf(faIncl, I_YIELD_MAX).tolist(),
            "faPmfExcluding": yd.faPoissonMixturePmf(faExcl, I_YIELD_MAX).tolist(),
            "fMeanIncluding": float(np.mean(faIncl)), "fMeanExcluding": float(np.mean(faExcl))}


def fdictFigure11(dictTargets, dfCatalog):
    """Representative-draw completeness per target, over the whole input list in grey."""
    faC = np.asarray(dictTargets["faCompletenessRepresentative"])
    bUsed = faC > 0
    bList = (dfCatalog["fDistancePc"] <= 40) & (dfCatalog["fLuminosityLsun"] > 0)
    return {"faDistanceUsed": np.asarray(dictTargets["faDistancePc"])[bUsed].tolist(),
            "faLuminosityUsed": np.asarray(dictTargets["faLuminosityLsun"])[bUsed].tolist(),
            "faCompletenessUsed": faC[bUsed].tolist(),
            "faDistanceAll": dfCatalog.loc[bList, "fDistancePc"].tolist(),
            "faLuminosityAll": dfCatalog.loc[bList, "fLuminosityLsun"].tolist(),
            "iRepresentativeDraw": dictTargets["iRepresentativeDraw"]}


def fdictAperture(dictAperture):
    """Figs. 12, 14 and 15 from the aperture step, keyed by diameter."""
    dictBy = dictAperture["dictByDiameter"]
    return {"fig12": {s: {"faPmfIncluding": d["faPmfIncludingSigmaEta"][:I_YIELD_MAX],
                          "faPmfExcluding": d["faPmfExcludingSigmaEta"][:I_YIELD_MAX]}
                      for s, d in dictBy.items()},
            "fig14": {s: {"faEdgesDays": d["faCharDaysEdges"],
                          "faFrequency": d["faCharDaysHistogram"],
                          "fMeanDays": d["fMeanCharDaysFirstN"]} for s, d in dictBy.items()},
            "fig15": {"faDiameterM": [float(s) for s in dictBy],
                      "faP25Including": [d["fProbability25IncludingSigmaEta"]
                                         for d in dictBy.values()],
                      "faP25Excluding": [d["fProbability25ExcludingSigmaEta"]
                                         for d in dictBy.values()]}}


def fdictParseArgs():
    """Command-line configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coronagraph-curves", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--survey", required=True)
    p.add_argument("--yield-samples", required=True)
    p.add_argument("--aperture", required=True)
    p.add_argument("--target-comparison", required=True)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--out-json", default="publishedComparisonSeries.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    dictLoad = {k: json.load(open(dictArgs[k])) for k in ("survey", "aperture",
                                                          "target_comparison")}
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    dfCurves = pd.read_csv(dictArgs["coronagraph_curves"])
    dictOut = {"iYieldMax": I_YIELD_MAX, "sPublishedSource": "arXiv:2405.19418 e-print (ms_v2)",
               **fdictFigure04And07(dictLoad["survey"]),
               "fig09": fdictFigure09(dictNpz["faZodiDraws"], dictLoad["survey"]),
               "fig10": fdictFigure10(np.load(dictArgs["yield_samples"])),
               "fig11": fdictFigure11(dictLoad["target_comparison"],
                                      pd.read_csv(dictArgs["target_catalog"])),
               **fdictAperture(dictLoad["aperture"]),
               "fig25": {sCol: dfCurves[sCol].tolist() for sCol in dfCurves.columns}}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile)
    print(json.dumps({"fig04Mean": dictOut["fig04"]["fMean"],
                      "fig07Means": [dictOut["fig07"]["fMeanAlbedo"],
                                     dictOut["fig07"]["fMeanAlbedoExozodi"]],
                      "fig10Means": [dictOut["fig10"]["fMeanIncluding"],
                                     dictOut["fig10"]["fMeanExcluding"]],
                      "fig15": dictOut["fig15"]}, indent=2))


if __name__ == "__main__":
    main()
