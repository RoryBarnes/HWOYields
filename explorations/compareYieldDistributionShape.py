#!/usr/bin/env python3
"""Compare the realized-yield distribution's shape with digitized Stark (2024) Fig. 10 under candidate eta laws.

Each candidate eta distribution is pushed through this pipeline's own yield-versus-eta curve
(A07's faYieldGrid, an expected yield already optimized at each eta) and Poisson-drawn, then
compared with the digitized purple curve on the same integer-yield grid. Candidates, all built
only from what Stark (2024) Sec. 3.5 states:
  pipeline      - A07's saved samples, as the report currently uses them
  lnMedian      - lognormal matched to [0.12, 0.55] at 86 percent (A06's current law)
  lnMean        - same width, but 0.26 read as the MEAN, which is what Sec. 3.5 says it is
  brysonMixture - equal mixture of two lognormals (Bryson 2021 completeness cases 1 and 2,
                  "uniformly randomly drawing from both cases"), constrained to mean 0.26 and
                  the 86 percent interval [0.12, 0.55]; the case separation is the one free
                  parameter and is scanned
  brysonDigitised - equal mixture of the two Bryson (2021) extrapolation cases digitized from
                  his Fig. 14 (digitiseBrysonEtaEarthFigure.py), rescaled by one factor so its
                  mean is Stark's 0.26; Stark's 86 percent interval [0.12, 0.55] is then an
                  independent check on whether that shape carries over to Stark's domain
The red curve (eta fixed) is compared with a Poisson draw at the nominal eta as a baseline check.
--yield-scale multiplies the yield curve, to ask what the shape would be if the fixed-eta level
matched Stark's red curve (17.3) instead of this pipeline's. (A bootstrap of the published mode from 498
resampled yields was removed: Stark's 498 runs each carry 1000 planet draws, so it overstated the
mode's uncertainty.)
"""

import argparse
import json

import numpy as np
from scipy import optimize, stats

F_Z_86 = 1.4757910281791712
I_MAX_YIELD = 50


def faYieldAtEta(faEta, dictBox):
    """Log-interpolate the pipeline's optimized expected yield at each eta."""
    faEtaGrid, faYieldGrid = np.array(dictBox["faEtaGrid"]), np.array(dictBox["faYieldGrid"])
    return dictBox.get("fYieldScale", 1.0) * np.interp(
        np.log(np.clip(faEta, faEtaGrid[0], faEtaGrid[-1])), np.log(faEtaGrid), faYieldGrid)


def faProbabilityPerYield(faCounts):
    """Probability of each integer yield 0..49, normalized over all draws (tail kept out)."""
    faHist = np.bincount(np.clip(faCounts, 0, I_MAX_YIELD).astype(int),
                         minlength=I_MAX_YIELD + 1)[:I_MAX_YIELD]
    return faHist / float(len(faCounts))


def fdictShape(faP, faCounts=None):
    """Mode (3-bin smoothed), median, 90 percent plateau, P25 and truncated mean."""
    faK = np.arange(len(faP))
    faSmooth = np.convolve(faP, np.ones(3) / 3.0, mode="same")
    faCdf = np.cumsum(faP) / faP.sum()
    faTop = faK[faSmooth >= 0.9 * faSmooth.max()]
    dictOut = {"iMode": int(faK[np.argmax(faP)]), "iModeSmoothed": int(faK[np.argmax(faSmooth)]),
               "iMedian": int(faK[np.searchsorted(faCdf, 0.5)]),
               "iaPlateau90": [int(faTop.min()), int(faTop.max())],
               "fPBelow12": float(faP[:12].sum() / faP.sum())}
    if faCounts is not None:
        dictOut.update({"fMean": float(np.mean(faCounts)), "fP25": float(np.mean(faCounts >= 25))})
    return dictOut


def ffDistance(faModel, faPublished):
    """Total-variation distance between two yield distributions on 0..49 (each renormalized)."""
    return 0.5 * float(np.abs(faModel / faModel.sum() - faPublished / faPublished.sum()).sum())


def fdictLogNormalFromInterval(fLo, fHi, fMean=None):
    """Lognormal (mu, sigma) whose 86 percent interval width matches; centred on median or mean."""
    fSigma = (np.log(fHi) - np.log(fLo)) / (2.0 * F_Z_86)
    fMu = 0.5 * (np.log(fLo) + np.log(fHi)) if fMean is None else np.log(fMean) - 0.5 * fSigma ** 2
    return {"fMu": float(fMu), "fSigma": float(fSigma)}


def ffMixtureCdf(fX, fMu, fSigma, fLnRatio):
    """CDF of an equal mixture of two lognormals separated by fLnRatio in ln(eta)."""
    return 0.5 * (stats.norm.cdf((np.log(fX) - fMu + 0.5 * fLnRatio) / fSigma) +
                  stats.norm.cdf((np.log(fX) - fMu - 0.5 * fLnRatio) / fSigma))


def fdictBrysonMixture(fLnRatio, fMean, fLo, fHi):
    """Fit (mu, sigma) of the two-case mixture to the 86 percent interval, then report its mean."""
    def faResid(faP):
        return [ffMixtureCdf(fLo, faP[0], abs(faP[1]), fLnRatio) - 0.07,
                ffMixtureCdf(fHi, faP[0], abs(faP[1]), fLnRatio) - 0.93]
    faSol = optimize.fsolve(faResid, [np.log(np.sqrt(fLo * fHi)), 0.4])
    fMu, fSigma = float(faSol[0]), float(abs(faSol[1]))
    fMixMean = float(np.exp(fMu + 0.5 * fSigma ** 2) * np.cosh(0.5 * fLnRatio))
    return {"fMu": fMu, "fSigma": fSigma, "fLnRatio": fLnRatio, "fImpliedMean": fMixMean,
            "fTargetMean": fMean}


def faDrawMixture(dictMix, iDraws, rng):
    """Draw eta from the two-case mixture."""
    faSign = np.where(rng.random(iDraws) < 0.5, -0.5, 0.5)
    return np.exp(rng.normal(dictMix["fMu"] + faSign * dictMix["fLnRatio"], dictMix["fSigma"]))


def faDrawBrysonMixture(dictBryson, iDraws, rng):
    """Equal mixture of the two digitized Bryson cases, uniform within each 0.1-wide bin."""
    faOut = np.empty(iDraws)
    faPick = rng.random(iDraws) < 0.5
    for sCase, bMask in (("extrapConst", faPick), ("extrapZero", ~faPick)):
        faEdges = np.array(dictBryson[sCase]["faEdges"])
        faP = np.array(dictBryson[sCase]["faProbability"])
        iaBin = rng.choice(len(faP), size=int(bMask.sum()), p=faP / faP.sum())
        faOut[bMask] = faEdges[iaBin] + rng.random(len(iaBin)) * np.diff(faEdges)[iaBin]
    return faOut


def fdictEtaInterval(faEta):
    """Mean, median and the two-sided 86 percent interval Stark quotes."""
    return {"fMean": float(np.mean(faEta)), "fMedian": float(np.median(faEta)),
            "fP07": float(np.percentile(faEta, 7)), "fP93": float(np.percentile(faEta, 93))}


def fdictCandidate(faEta, dictBox, faPublished, rng):
    """Push an eta sample through the yield curve, Poisson-draw, and score against Stark."""
    faCounts = rng.poisson(faYieldAtEta(faEta, dictBox))
    faP = faProbabilityPerYield(faCounts)
    return {"dictShape": fdictShape(faP, faCounts), "fTotalVariation": ffDistance(faP, faPublished),
            "fEtaMean": float(np.mean(faEta)), "fEtaMedian": float(np.median(faEta)),
            "faProbability": faP.tolist()}


def fdictParseArgs():
    """Command-line configuration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--digitised", default="starkFigure10Digitised.json")
    p.add_argument("--prediction", default="../predictRedefinedYield/yieldPrediction.json")
    p.add_argument("--samples", default="../predictRedefinedYield/yieldSamples.npz")
    p.add_argument("--eta-mean", type=float, default=0.26)
    p.add_argument("--eta-lo", type=float, default=0.12)
    p.add_argument("--eta-hi", type=float, default=0.55)
    p.add_argument("--eta-nominal", type=float, default=0.24)
    p.add_argument("--draws", type=int, default=400000)
    p.add_argument("--bryson-digitised", default="brysonFigure13EtaEarthDigitised.json")
    p.add_argument("--yield-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--out-json", default="yieldDistributionShape.json")
    return vars(p.parse_args())


def faPublishedCurve(dictDigitised, sKey):
    """Digitized probability on the integer grid 0..49."""
    faP = np.zeros(I_MAX_YIELD)
    faP[np.array(dictDigitised[sKey]["iaYield"])] = dictDigitised[sKey]["faProbability"]
    return faP


def fdictCandidates(dictArgs, dictBox, faPurple, rng):
    """Every candidate eta law, each scored against the purple curve."""
    iN, fLo, fHi = dictArgs["draws"], dictArgs["eta_lo"], dictArgs["eta_hi"]
    dictMed = fdictLogNormalFromInterval(fLo, fHi)
    dictMean = fdictLogNormalFromInterval(fLo, fHi, dictArgs["eta_mean"])
    dictOut = {
        "lnMedian": fdictCandidate(np.exp(rng.normal(dictMed["fMu"], dictMed["fSigma"], iN)),
                                   dictBox, faPurple, rng),
        "lnMean": fdictCandidate(np.exp(rng.normal(dictMean["fMu"], dictMean["fSigma"], iN)),
                                 dictBox, faPurple, rng)}
    for fLnRatio in (0.25, 0.5, 0.75, 1.0):
        dictMix = fdictBrysonMixture(fLnRatio, dictArgs["eta_mean"], fLo, fHi)
        dictCand = fdictCandidate(faDrawMixture(dictMix, iN, rng), dictBox, faPurple, rng)
        dictOut[f"brysonMixture_lnRatio{fLnRatio:g}"] = {**dictCand, "dictMixture": dictMix}
    if dictArgs.get("bryson_digitised"):
        with open(dictArgs["bryson_digitised"]) as oFile:
            faRaw = faDrawBrysonMixture(json.load(oFile), iN, rng)
        faEta = faRaw * dictArgs["eta_mean"] / np.mean(faRaw)
        dictOut["brysonDigitised"] = {**fdictCandidate(faEta, dictBox, faPurple, rng),
                                      "dictEta": fdictEtaInterval(faEta),
                                      "dictEtaStark": {"fMean": dictArgs["eta_mean"],
                                                       "fP07": fLo, "fP93": fHi}}
    return dictOut


def fnPrintLowEnd(dictOut, iMax=24):
    """P(yield = k) on the rising side, pipeline against Stark's purple curve."""
    faPipe, faPub = np.array(dictOut["pipeline"]["faProbability"]), np.array(dictOut["faPurple"])
    print(f"{'k':>3}{'Stark':>9}{'pipeline':>10}{'ratio':>8}")
    for k in range(iMax):
        print(f"{k:>3}{faPub[k]:>9.4f}{faPipe[k]:>10.4f}"
              f"{faPipe[k] / faPub[k] if faPub[k] > 0 else float('nan'):>8.2f}")


def fnPrintRow(sName, dictC):
    """One summary line."""
    d = dictC["dictShape"]
    print(f"{sName:<28}{dictC.get('fEtaMean', float('nan')):>7.3f}{d['iModeSmoothed']:>6}"
          f"{d['iMedian']:>6}{str(d['iaPlateau90']):>10}{d.get('fMean', float('nan')):>7.2f}"
          f"{d.get('fP25', float('nan')):>7.3f}{d['fPBelow12']:>7.3f}"
          f"{dictC['fTotalVariation']:>7.3f}")


def main():
    dictArgs = fdictParseArgs()
    rng = np.random.default_rng(dictArgs["seed"])
    with open(dictArgs["digitised"]) as oFile:
        dictDig = json.load(oFile)
    with open(dictArgs["prediction"]) as oFile:
        dictBox = json.load(oFile)["dictByBox"]["canonical"]
    dictBox["fYieldScale"] = dictArgs["yield_scale"]
    faPurple, faRed = faPublishedCurve(dictDig, "including"), faPublishedCurve(dictDig, "excluding")
    faPipe = np.load(dictArgs["samples"])["faObserved_canonical"].astype(int)
    faRedModel = rng.poisson(faYieldAtEta(np.full(dictArgs["draws"], dictArgs["eta_nominal"]), dictBox))
    dictOut = {
        "published": {"including": {"dictShape": fdictShape(faPurple), "fTotalVariation": 0.0},
                      "excluding": {"dictShape": fdictShape(faRed), "fTotalVariation": 0.0}},
        "redAtNominalEta": {"dictShape": fdictShape(faProbabilityPerYield(faRedModel), faRedModel),
                            "fTotalVariation": ffDistance(faProbabilityPerYield(faRedModel), faRed)},
        "pipeline": {"dictShape": fdictShape(faProbabilityPerYield(faPipe), faPipe),
                     "fTotalVariation": ffDistance(faProbabilityPerYield(faPipe), faPurple),
                     "faProbability": faProbabilityPerYield(faPipe).tolist()},
        "faPurple": faPurple.tolist(),
        "dictCandidates": fdictCandidates(dictArgs, dictBox, faPurple, rng),
        "fYieldScale": dictArgs["yield_scale"]}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'distribution':<28}{'<eta>':>7}{'mode':>6}{'med':>6}{'plateau':>10}{'mean':>7}"
          f"{'P25':>7}{'P<12':>7}{'TV':>7}")
    fnPrintRow("Stark purple (digitized)", dictOut["published"]["including"])
    fnPrintRow("Stark red (digitized)", dictOut["published"]["excluding"])
    fnPrintRow("model red, eta nominal", dictOut["redAtNominalEta"])
    fnPrintRow("pipeline (A07 samples)", dictOut["pipeline"])
    for sName, dictC in dictOut["dictCandidates"].items():
        fnPrintRow(sName, dictC)
    fnPrintLowEnd(dictOut)
    if "brysonDigitised" in dictOut["dictCandidates"]:
        print("Bryson-shape eta, rescaled to mean 0.26:",
              {k: round(v, 3) for k, v in dictOut["dictCandidates"]["brysonDigitised"]["dictEta"].items()},
              "| Stark: mean 0.26, 86% [0.12, 0.55]")


if __name__ == "__main__":
    main()
