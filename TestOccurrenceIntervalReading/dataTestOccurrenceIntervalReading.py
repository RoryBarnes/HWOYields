#!/usr/bin/env python3
"""Test whether Stark's quoted eta_Earth interval is an 86% interval of his own source distribution, or a 68% one.

Stark et al. (2024) Sec. 3.6 states "mean and 86% confidence interval eta_Earth = 0.26 +0.29/-0.14",
built by uniformly mixing the two Bryson et al. (2021) completeness-extrapolation cases. "86%" is a
digit transposition away from the standard "68%", appears exactly once in that paper, and the
adopted pipeline reproduces his Fig. 10 only when the interval is read at 68%.

This script settles it from his own source rather than from the fit: it takes the two Bryson cases
digitized from Bryson Fig. 13, mixes them uniformly as Stark describes, rescales the mixture by one
factor so its mean is Stark's stated 0.26, and reports what probability mass actually falls in his
quoted [0.12, 0.55] -- plus the intervals that mixture gives at 68% and at 86%, for comparison with
what he wrote. No free parameter is fitted; the single rescaling is fixed by his stated mean.
"""

import argparse
import json

import numpy as np

LIST_CASES = ["extrapZero", "extrapConst"]


def ftMixtureFromDigitised(dictBryson, iSamplesPerCase, oRng):
    """Uniform mixture of the two Bryson cases, sampled from the digitized histograms."""
    listDraws = []
    for sCase in LIST_CASES:
        dictCase = dictBryson[sCase]
        faEdges = np.asarray(dictCase["faEdges"], dtype=float)
        faProbability = np.asarray(dictCase["faProbability"], dtype=float)
        faWeights = faProbability / faProbability.sum()
        iaBin = oRng.choice(len(faWeights), size=iSamplesPerCase, p=faWeights)
        listDraws.append(oRng.uniform(faEdges[iaBin], faEdges[iaBin + 1]))
    return np.concatenate(listDraws)


def fdictIntervalTest(faEta, fLo, fHi, fTargetMean):
    """Mass inside the quoted interval, and the intervals at 68% and 86%, after rescaling."""
    faScaled = faEta * (fTargetMean / faEta.mean())
    fMass = float(((faScaled >= fLo) & (faScaled <= fHi)).mean())
    dictOut = {"fMean": float(faScaled.mean()), "fMedian": float(np.median(faScaled)),
               "fMassInQuotedInterval": fMass,
               "fImpliedConfidencePercent": 100.0 * fMass}
    for sName, fLevel in [("68", 0.68), ("86", 0.86)]:
        fTail = 0.5 * (1.0 - fLevel)
        faBounds = np.percentile(faScaled, [100 * fTail, 100 * (1 - fTail)])
        dictOut[f"faInterval{sName}"] = [float(faBounds[0]), float(faBounds[1])]
    return dictOut


def fnPrint(dictOut, fLo, fHi):
    """Report the comparison."""
    print(f"Bryson mixture rescaled to Stark's stated mean {dictOut['fMean']:.3f} "
          f"(median {dictOut['fMedian']:.3f})")
    print(f"  mass inside Stark's quoted [{fLo}, {fHi}]  = "
          f"{dictOut['fMassInQuotedInterval']:.3f}  ->  a "
          f"{dictOut['fImpliedConfidencePercent']:.0f}% interval")
    print(f"  the mixture's own 68% interval           = "
          f"[{dictOut['faInterval68'][0]:.3f}, {dictOut['faInterval68'][1]:.3f}]")
    print(f"  the mixture's own 86% interval           = "
          f"[{dictOut['faInterval86'][0]:.3f}, {dictOut['faInterval86'][1]:.3f}]")
    print(f"  Stark's text says [{fLo}, {fHi}] is the 86% interval")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bryson", default="reference/brysonFigure13EtaEarthDigitised.json")
    p.add_argument("--eta-mean", type=float, default=0.26)
    p.add_argument("--eta-lo", type=float, default=0.12)
    p.add_argument("--eta-hi", type=float, default=0.55)
    p.add_argument("--samples-per-case", type=int, default=400000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-json", default="etaIntervalConfidenceLevel.json")
    dictArgs = vars(p.parse_args())
    dictBryson = json.load(open(dictArgs["bryson"]))
    oRng = np.random.default_rng(dictArgs["seed"])
    faEta = ftMixtureFromDigitised(dictBryson, dictArgs["samples_per_case"], oRng)
    dictOut = fdictIntervalTest(faEta, dictArgs["eta_lo"], dictArgs["eta_hi"],
                                dictArgs["eta_mean"])
    dictOut["sSource"] = dictBryson["sSource"]
    dictOut["iSamplesPerCase"] = dictArgs["samples_per_case"]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    fnPrint(dictOut, dictArgs["eta_lo"], dictArgs["eta_hi"])


if __name__ == "__main__":
    main()
