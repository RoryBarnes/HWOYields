#!/usr/bin/env python3
"""Overlay the realized EEC yield distribution of each one-variable model on Stark's Fig. 10, and locate the peak.

The peak is the question, and a raw argmax over a histogram of ~14400 Poisson draws is not a safe
answer to it: near the top the distribution is flat, so the bin that happens to be highest moves
under resampling. The realized yield is a Poisson mixture over the sampled expected yields, so
this script builds that mixture EXACTLY with yieldlib.yielddistribution -- removing the Poisson
counting noise entirely and leaving only the finite eta posterior -- and reports the argmax, a
bootstrap over the eta samples, and the plateau of bins within 90% of the peak height.
The published purple curve (eta_Earth uncertainty included) is digitized from the e-print vector
PDF and carries a well-determined mode of 10, because each of Stark's 498 runs holds 1000 planet
draws.
"""

import argparse
import json
import os
import sys

import numpy as np

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))
from yieldlib import plotstyle as ps  # noqa: E402
from yieldlib.yielddistribution import faPoissonMixturePmf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DICT_LABELS = {"baseline": "Adopted model", "albedoRecompute": "Albedo: recomputed exposure",
               "etaInterval86": r"$\eta_\oplus$ interval as 86% (every documented choice)",
               "skyThroughputConstant": r"Constant $T_{\rm sky}$",
               "albedoRecomputeEta86": r"Recomputed albedo and 86% $\eta_\oplus$",
               "etaInterval86ConstantSky": r"86% $\eta_\oplus$ and constant $T_{\rm sky}$",
               "brysonMixtureEta": r"Stark's own $\eta_\oplus$ law (Bryson mixture)"}
S_VIOLET = "#7a5bd6"
S_TEAL = "#0e8f88"
DICT_COLOUR = {"baseline": ps.LIST_SERIES[0], "albedoRecompute": ps.LIST_SERIES[1],
               "etaInterval86": ps.LIST_SERIES[2], "skyThroughputConstant": ps.LIST_SERIES[3],
               "albedoRecomputeEta86": ps.LIST_SERIES[4], "etaInterval86ConstantSky": S_VIOLET,
               "brysonMixtureEta": S_TEAL}
LIST_EMPHASIS = ["baseline", "brysonMixtureEta", "etaInterval86", "etaInterval86ConstantSky"]


def faPmf(faExpected, iGrid):
    """Exact Poisson-mixture mass on 0..iGrid-1, renormalized over that range."""
    faP = faPoissonMixturePmf(faExpected, iMax=iGrid)
    return faP / faP.sum()


def fdictPeak(faExpected, iGrid, iBootstrap, oRng, fPlateauFraction=0.9):
    """Argmax, its bootstrap interval over the eta samples, and the plateau near the peak."""
    faP = faPmf(faExpected, iGrid)
    iMode = int(np.argmax(faP))
    iaPlateau = np.flatnonzero(faP >= fPlateauFraction * faP.max())
    listModes = []
    for _ in range(iBootstrap):
        faResample = oRng.choice(faExpected, size=faExpected.size, replace=True)
        listModes.append(int(np.argmax(faPmf(faResample, iGrid))))
    faModes = np.array(listModes)
    return {"iMode": iMode, "iModeP05": int(np.percentile(faModes, 5)),
            "iModeP95": int(np.percentile(faModes, 95)),
            "iaPlateau": [int(iaPlateau[0]), int(iaPlateau[-1])],
            "fPlateauFraction": fPlateauFraction, "faPmf": faP}


def fdictLevelScan(faExpected, iGrid, faScales):
    """Where the peak lands when the expected yields are scaled, holding the shape fixed.

    Separates a level offset from a shape difference: if scaling the adopted model to Stark's
    level also moves its peak onto Stark's, the peak offset carries no information the level
    does not already carry.
    """
    listRows = []
    for fScale in faScales:
        faP = faPmf(faExpected * fScale, iGrid)
        listRows.append({"fScale": float(fScale), "iMode": int(np.argmax(faP)),
                         "fMeanExpected": float((faExpected * fScale).mean())})
    return listRows


def fnDrawModel(oAxes, sName, dictPeak, bEmphasis):
    """One model's PMF; emphasized curves carry a peak marker, probes are drawn back."""
    faP, sColour = dictPeak["faPmf"], DICT_COLOUR.get(sName, ps.S_INK_SECONDARY)
    oAxes.step(np.arange(faP.size), faP, where="mid", color=sColour,
               linewidth=2.0 if bEmphasis else 1.1, alpha=1.0 if bEmphasis else 0.5,
               zorder=3 if bEmphasis else 2,
               label=f"{DICT_LABELS.get(sName, sName)} (peak {dictPeak['iMode']})")
    if bEmphasis:
        oAxes.plot([dictPeak["iMode"]], [faP[dictPeak["iMode"]]], marker="v", color=sColour,
                   markersize=7, linestyle="none", zorder=6)


def fnPlot(dictPeaks, faPublished, dictPublishedSummary, sOutPdf, iXMax):
    """The overlay figure."""
    ps.fnApplyStyle()
    oFigure, oAxes = plt.subplots(figsize=(7.2, 4.4))
    oAxes.fill_between(np.arange(faPublished.size), faPublished, step="mid",
                       color=ps.S_INK_SECONDARY, alpha=0.16, linewidth=0)
    oAxes.step(np.arange(faPublished.size), faPublished, where="mid", color=ps.S_INK_PRIMARY,
               linewidth=2.6, zorder=5,
               label=f"Stark+2024 Fig. 10 (peak {dictPublishedSummary['iMode']})")
    for sName, dictPeak in sorted(dictPeaks.items(), key=lambda t: t[0] in LIST_EMPHASIS):
        fnDrawModel(oAxes, sName, dictPeak, sName in LIST_EMPHASIS)
    oAxes.set_xlim(0, iXMax)
    ps.fnFinishAxes(oAxes, "Realized exoEarth candidates (canonical box)", "Probability",
                    "Where the EEC yield distribution peaks")
    oAxes.legend(loc="upper right")
    oFigure.tight_layout()
    oFigure.savefig(sOutPdf)
    plt.close(oFigure)


def fdictLoadSamples(sBaselineSamples, sOutRoot, listNames):
    """Sampled EXPECTED canonical-box yields (the Poisson mixture's components)."""
    dictOut = {}
    for sName in listNames:
        sPath = (sBaselineSamples if sName == "baseline"
                 else os.path.join(sOutRoot, sName, "yieldSamples.npz"))
        dictOut[sName] = np.load(sPath)["faExpected_canonical"]
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-samples", required=True)
    p.add_argument("--out-root", default="variants")
    p.add_argument("--digitised-figure-ten", required=True)
    p.add_argument("--models", default="baseline,albedoRecompute,etaInterval86,"
                                       "skyThroughputConstant,albedoRecomputeEta86,"
                                       "etaInterval86ConstantSky,brysonMixtureEta")
    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--x-max", type=int, default=45)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-pdf", default="../Plot/figYieldPeakComparison.pdf")
    p.add_argument("--level-scan-model", default="baseline")
    p.add_argument("--level-scan", default="0.80,0.85,0.90,0.95,1.00,1.05,1.10",
                   help="multiplicative scalings of the expected yield to scan the peak over")
    p.add_argument("--out-json", default="yieldPeakComparison.json")
    dictArgs = vars(p.parse_args())
    dictFigure = json.load(open(dictArgs["digitised_figure_ten"]))["including"]
    faPublished = np.array(dictFigure["faProbability"])
    listNames = [s.strip() for s in dictArgs["models"].split(",") if s.strip()]
    dictSamples = fdictLoadSamples(dictArgs["baseline_samples"], dictArgs["out_root"],
                                   listNames)
    oRng = np.random.default_rng(dictArgs["seed"])
    dictPeaks = {s: fdictPeak(fa, faPublished.size, dictArgs["bootstrap"], oRng)
                 for s, fa in dictSamples.items()}
    os.makedirs(os.path.dirname(dictArgs["out_pdf"]), exist_ok=True)
    fnPlot(dictPeaks, faPublished, dictFigure["dictSummary"], dictArgs["out_pdf"],
           dictArgs["x_max"])
    dictOut = {s: {k: v for k, v in d.items() if k != "faPmf"} for s, d in dictPeaks.items()}
    dictOut["published"] = dictFigure["dictSummary"]
    faScales = np.array([float(s) for s in dictArgs["level_scan"].split(",")])
    dictOut["listLevelScan"] = fdictLevelScan(dictSamples[dictArgs["level_scan_model"]],
                                              faPublished.size, faScales)
    dictOut["sLevelScanModel"] = dictArgs["level_scan_model"]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    for sName in listNames:
        dictPeak = dictOut[sName]
        print(f"{sName:<24} peak {dictPeak['iMode']:>3}  bootstrap 90% "
              f"[{dictPeak['iModeP05']}, {dictPeak['iModeP95']}]  plateau "
              f"{dictPeak['iaPlateau'][0]}-{dictPeak['iaPlateau'][1]}")
    print(f"{'Stark+2024 (published)':<24} peak {dictOut['published']['iMode']:>3}  "
          f"plateau (90% of peak) {dictOut['published']['iaPlateau90']}")
    print(f"\nlevel scan on {dictOut['sLevelScanModel']} (shape fixed, expected yields scaled):")
    for dictRow in dictOut["listLevelScan"]:
        print(f"  scale {dictRow['fScale']:.2f}  mean expected "
              f"{dictRow['fMeanExpected']:6.2f}  peak {dictRow['iMode']}")


if __name__ == "__main__":
    main()
