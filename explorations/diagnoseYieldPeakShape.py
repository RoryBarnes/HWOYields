#!/usr/bin/env python3
"""Compare the shape of this pipeline's eta posterior with the published one it should reproduce.

Stark et al. (2024) Fig. 10 peaks near 10 EECs including eta_Earth uncertainty, with a long
right tail. A realized-yield distribution that instead peaks near the median and looks roughly
symmetric implies an eta distribution that is too narrow and insufficiently right-skewed, since
yield is close to proportional to eta. This measures the mode, spread and skew of both.
"""

import argparse
import json

import numpy as np

F_Z_86 = 1.4757910281791712


def faSampleSplitNormal(fCentre, fMinus, fPlus, iDraws, rng):
    """Draw from the split-normal implied by a published asymmetric interval."""
    fSigmaLo, fSigmaHi = fMinus / F_Z_86, fPlus / F_Z_86
    faPick = rng.random(iDraws) < fSigmaLo / (fSigmaLo + fSigmaHi)
    faDraw = np.where(faPick,
                      fCentre - np.abs(rng.normal(0.0, fSigmaLo, iDraws)),
                      fCentre + np.abs(rng.normal(0.0, fSigmaHi, iDraws)))
    return faDraw[faDraw > 0.0]


def fnMode(faValues, iBins=120):
    """Histogram mode of a sample."""
    faCounts, faEdges = np.histogram(faValues, bins=iBins)
    i = int(np.argmax(faCounts))
    return float(0.5 * (faEdges[i] + faEdges[i + 1]))


def fdictShape(faValues, sName):
    """Mode, median, mean and a skew summary for one sample."""
    fMode, fMedian, fMean = fnMode(faValues), float(np.median(faValues)), float(np.mean(faValues))
    return {"sName": sName, "fMode": fMode, "fMedian": fMedian, "fMean": fMean,
            "fP16": float(np.percentile(faValues, 16)),
            "fP84": float(np.percentile(faValues, 84)),
            "fMeanOverMode": fMean / fMode if fMode else float("nan"),
            "fUpperOverLower": (float(np.percentile(faValues, 84)) - fMedian) /
                               max(fMedian - float(np.percentile(faValues, 16)), 1e-12)}


def fdictParseArgs():
    """Command-line configuration for the yield-peak diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--posterior", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--eta-centre", type=float, default=0.26)
    p.add_argument("--eta-plus", type=float, default=0.29)
    p.add_argument("--eta-minus", type=float, default=0.14)
    p.add_argument("--draws", type=int, default=400000)
    p.add_argument("--bias-factor", type=float, default=17.3 / 22.5,
                   help="Stark's expected yield falls 22.5 -> 17.3 once albedo and exozodi "
                        "uncertainty are included; this pipeline models neither")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="yieldPeakShape.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    rng = np.random.default_rng(dictArgs["seed"])
    faMine = np.load(dictArgs["posterior"])["faEta_canonical"]
    faPublished = faSampleSplitNormal(dictArgs["eta_centre"], dictArgs["eta_minus"],
                                      dictArgs["eta_plus"], dictArgs["draws"], rng)
    with open(dictArgs["prediction"]) as oFile:
        dictPred = json.load(oFile)["dictByBox"]["canonical"]
    faEtaGrid = np.array(dictPred["faEtaGrid"])
    faYieldGrid = np.array(dictPred["faYieldGrid"])

    def faYield(faEta):
        return np.interp(np.log(np.clip(faEta, faEtaGrid[0], faEtaGrid[-1])),
                         np.log(faEtaGrid), faYieldGrid)

    dictOut = {
        "dictEtaShape": {"mine": fdictShape(faMine, "this pipeline"),
                         "published": fdictShape(faPublished, "Stark 0.26 +0.29/-0.14")},
        "dictYieldShape": {"mine": fdictShape(rng.poisson(faYield(faMine)).astype(float), "mine"),
                           "published": fdictShape(
                               rng.poisson(faYield(faPublished)).astype(float), "published eta")},
        "dictYieldShapeWithBias": {
            "mine": fdictShape(rng.poisson(faYield(faMine) * dictArgs["bias_factor"]
                                           ).astype(float), "mine x bias"),
        },
        "fBiasFactor": dictArgs["bias_factor"],
        "fStarkFig10PeakApprox": 10.0,
        "fStarkFig10MeanApprox": 21.0,
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sKey, sLabel in (("dictEtaShape", "eta"), ("dictYieldShape", "realized yield")):
        print(f"--- {sLabel} ---")
        print(f"{'':12}{'mode':>9}{'median':>9}{'mean':>9}{'mean/mode':>11}{'skew(u/l)':>11}")
        for sWhich, d in dictOut[sKey].items():
            print(f"{sWhich:12}{d['fMode']:>9.3f}{d['fMedian']:>9.3f}{d['fMean']:>9.3f}"
                  f"{d['fMeanOverMode']:>11.2f}{d['fUpperOverLower']:>11.2f}")
    d = dictOut["dictYieldShapeWithBias"]["mine"]
    print(f"\n--- realized yield, x{dictArgs['bias_factor']:.3f} for unmodelled albedo+exozodi ---")
    print(f"{'mine x bias':12}{d['fMode']:>9.3f}{d['fMedian']:>9.3f}{d['fMean']:>9.3f}"
          f"{d['fMeanOverMode']:>11.2f}{d['fUpperOverLower']:>11.2f}")
    print(f"{'Stark Fig.10':12}{dictOut['fStarkFig10PeakApprox']:>9.1f}"
          f"{'--':>9}{dictOut['fStarkFig10MeanApprox']:>9.1f}")


if __name__ == "__main__":
    main()
