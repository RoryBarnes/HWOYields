#!/usr/bin/env python3
"""Infer the eta_Earth distribution that turns Stark's Fig. 10 red curve into his purple curve.

The purple curve (with sigma_eta) is, to a good approximation, the red curve's Poisson sampling
smeared over eta_Earth. Holding this pipeline's optimized yield-vs-eta curve (A07's faYieldGrid)
and rescaling it so eta = 0.24 gives Stark's red mean, the predicted purple distribution for a
lognormal eta law (median, ln-sigma) is an exact Poisson mixture, computed by quadrature with no
Monte Carlo noise. A grid search over (median, ln-sigma) minimizes the total-variation distance to
the digitized purple curve. The implied mean and 86 percent interval are then compared with the
0.26 (+0.29/-0.14) Stark quotes: agreement means the lognormal matched to his quoted interval
should reproduce his figure; disagreement means his figure implies a different eta distribution
from the one his text describes, or something besides eta broadens it. The quoted interval is
also tested read as 68 percent (one sigma) rather than 86.
"""

import argparse
import json

import numpy as np
from scipy import stats

I_MAX_YIELD = 50


def faYieldAtEta(faEta, dictBox, fScale):
    """Rescaled log-interpolation of the pipeline's optimized expected yield."""
    faEtaGrid, faYieldGrid = np.array(dictBox["faEtaGrid"]), np.array(dictBox["faYieldGrid"])
    return fScale * np.interp(np.log(np.clip(faEta, faEtaGrid[0], faEtaGrid[-1])),
                              np.log(faEtaGrid), faYieldGrid)


def faPredictedPurple(fMedian, fSigma, dictBox, fScale, iNodes=400):
    """Exact P(yield = k), k = 0..49, for a lognormal eta law, by quadrature in ln(eta)."""
    faZ = np.linspace(-5.0, 5.0, iNodes)
    faW = stats.norm.pdf(faZ)
    faW /= faW.sum()
    faLambda = faYieldAtEta(fMedian * np.exp(fSigma * faZ), dictBox, fScale)
    faK = np.arange(I_MAX_YIELD)
    return np.sum(faW[:, None] * stats.poisson.pmf(faK[None, :], faLambda[:, None]), axis=0)


def ffTotalVariation(faModel, faPublished):
    """Total-variation distance on 0..49, each renormalized."""
    return 0.5 * float(np.abs(faModel / faModel.sum() - faPublished / faPublished.sum()).sum())


def fdictImplied(fMedian, fSigma):
    """Mean and 86 percent interval of a lognormal eta law."""
    return {"fMedian": fMedian, "fSigma": fSigma, "fMean": fMedian * np.exp(0.5 * fSigma ** 2),
            "fP07": fMedian * np.exp(-1.4757910281791712 * fSigma),
            "fP93": fMedian * np.exp(1.4757910281791712 * fSigma)}


def fdictShape(faP):
    """Mode, median and P(< 12) of a distribution on 0..49."""
    faCdf = np.cumsum(faP) / faP.sum()
    return {"iMode": int(np.argmax(faP)), "iMedian": int(np.searchsorted(faCdf, 0.5)),
            "fBelow12": float(faP[:12].sum() / faP.sum())}


def fdictGridSearch(dictBox, fScale, faPurple):
    """Best (median, sigma) over a grid, plus the TV of the quoted-interval lognormal."""
    listRows = [(ffTotalVariation(faPredictedPurple(fM, fS, dictBox, fScale), faPurple), fM, fS)
                for fM in np.linspace(0.10, 0.40, 61) for fS in np.linspace(0.20, 1.40, 61)]
    fTv, fM, fS = min(listRows)
    return {"fTotalVariation": fTv, **fdictImplied(fM, fS),
            "dictShape": fdictShape(faPredictedPurple(fM, fS, dictBox, fScale))}


def fdictQuotedAs68(dictBox, fScale, faPurple):
    """The quoted 0.26 (+0.29/-0.14) read as a 68 percent (one-sigma) interval instead of 86."""
    fSigma = 0.5 * (np.log(0.55) - np.log(0.12))
    faP = faPredictedPurple(np.sqrt(0.12 * 0.55), fSigma, dictBox, fScale)
    return {"fTotalVariation": ffTotalVariation(faP, faPurple),
            **fdictImplied(np.sqrt(0.12 * 0.55), fSigma), "dictShape": fdictShape(faP)}


def fdictLowEndByLevel(dictBox, fScaleStark, faPurple, fSigma, faLevels=(1.0, 0.95, 0.90)):
    """P(yield = k) for k < 15 with the quoted-as-68 law, at Stark's red level and scaled from it.

    fLevel = 1.0 is Stark's red level; the pipeline's own fixed-eta level is 19.34 / 17.35 = 1.115
    times that, and is included so the rising side can be compared at both levels.
    """
    dictOut = {"faPurple": faPurple[:15].tolist()}
    for fLevel in tuple(faLevels) + (19.34 / 17.347,):
        faP = faPredictedPurple(np.sqrt(0.12 * 0.55), fSigma, dictBox, fScaleStark * fLevel)
        dictOut[f"level{fLevel:.3f}"] = {"faLowEnd": faP[:15].tolist(),
                                         "fTotalVariation": ffTotalVariation(faP, faPurple),
                                         "dictShape": fdictShape(faP)}
    return dictOut


def fdictYieldLinearity(dictBox, fEtaNominal):
    """Expected yield relative to proportional scaling from the nominal eta, at low eta."""
    fY0 = float(faYieldAtEta(np.array([fEtaNominal]), dictBox, 1.0)[0])
    return {f"{fEta:g}": float(faYieldAtEta(np.array([fEta]), dictBox, 1.0)[0] /
                               (fY0 * fEta / fEtaNominal))
            for fEta in (0.03, 0.06, 0.10, 0.15, 0.24, 0.40, 0.60)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digitised", default="starkFigure10Digitised.json")
    p.add_argument("--prediction", default="../predictRedefinedYield/yieldPrediction.json")
    p.add_argument("--eta-nominal", type=float, default=0.24)
    p.add_argument("--out-json", default="output/etaLawFromStarkFigure10.json")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["digitised"]) as oFile:
        dictDig = json.load(oFile)
    with open(dictArgs["prediction"]) as oFile:
        dictBox = json.load(oFile)["dictByBox"]["canonical"]
    faPurple = np.zeros(I_MAX_YIELD)
    faPurple[dictDig["including"]["iaYield"]] = dictDig["including"]["faProbability"]
    fRedMean = dictDig["excluding"]["dictSummary"]["fMeanTruncated"]
    fScale = fRedMean / float(faYieldAtEta(np.array([dictArgs["eta_nominal"]]), dictBox, 1.0)[0])
    fSigQ = (np.log(0.55) - np.log(0.12)) / (2.0 * 1.4757910281791712)
    faQuoted = faPredictedPurple(np.sqrt(0.12 * 0.55), fSigQ, dictBox, fScale)
    dictOut = {"fYieldScale": fScale, "fRedMean": fRedMean,
               "dictBestFit": fdictGridSearch(dictBox, fScale, faPurple),
               "dictQuotedInterval": {"fTotalVariation": ffTotalVariation(faQuoted, faPurple),
                                      **fdictImplied(np.sqrt(0.12 * 0.55), fSigQ),
                                      "dictShape": fdictShape(faQuoted)},
               "dictQuotedAs68Percent": fdictQuotedAs68(dictBox, fScale, faPurple),
               "dictPublishedPurple": fdictShape(faPurple),
               "dictLowEndByLevel": fdictLowEndByLevel(dictBox, fScale, faPurple,
                                                       0.5 * (np.log(0.55) - np.log(0.12))),
               "dictYieldOverProportional": fdictYieldLinearity(dictBox, dictArgs["eta_nominal"]),
               "dictStarkQuoted": {"fMean": 0.26, "fP07": 0.12, "fP93": 0.55}}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    fnPrintLowEnd(dictOut)


def fnPrintLowEnd(dictOut):
    """Rising side of each level against Stark, plus yield linearity in eta."""
    dictLow = dictOut["dictLowEndByLevel"]
    listKeys = [k for k in dictLow if k.startswith("level")]
    print(f"{'k':>3}{'Stark':>8}" + "".join(f"{k.replace('level', 'x'):>9}" for k in listKeys))
    for k in range(15):
        print(f"{k:>3}{dictLow['faPurple'][k]:>8.4f}" +
              "".join(f"{dictLow[s]['faLowEnd'][k]:>9.4f}" for s in listKeys))
    print("TV:", {s: round(dictLow[s]["fTotalVariation"], 3) for s in listKeys})
    print("shape:", {s: dictLow[s]["dictShape"] for s in listKeys})
    print("yield / proportional-from-0.24:",
          {k: round(v, 3) for k, v in dictOut["dictYieldOverProportional"].items()})


if __name__ == "__main__":
    main()
