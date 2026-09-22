#!/usr/bin/env python3
"""Plot the yield-versus-eta curves and the posterior-marginalized EEC yield distributions."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fnPlotYieldCurves(oAxes, dictPrediction):
    """Left panel: summed completeness against eta, which falls as characterization eats time.

    Plotting yield/eta rather than yield isolates the effect: if yield were proportional to
    eta this curve would be flat, and its decline is the characterization burden.
    """
    for sBox, dictBox in dictPrediction["dictByBox"].items():
        faEta = np.array(dictBox["faEtaGrid"])
        faYield = np.array(dictBox["faYieldGrid"])
        oAxes.plot(faEta, faYield / faEta, color=ps.DICT_BOX_COLOUR[sBox],
                   label=ps.DICT_BOX_LABEL[sBox])
    oAxes.set_xscale("log")
    ps.fnFinishAxes(oAxes, "$\\eta$ over the box", "Summed completeness  (yield / $\\eta$)",
                    "Flat would mean yield $\\propto\\eta$")
    oAxes.legend(loc="lower left")


def fnPlotDistributions(oAxes, dictSamples, fGoal):
    """Right panel: the realized (Poisson) yield distribution, with expected yield behind it.

    The heavy curve is the REALIZED yield -- a Poisson draw per posterior sample -- because that
    is the quantity Stark et al. (2024) Fig. 10 plots and the quantity P25 is computed from.
    The thin curve is the expected yield E[N|eta], which is narrower because it carries the
    eta posterior only and not the counting statistics on top of it.
    """
    for sKey in sorted(k for k in dictSamples.files if k.startswith("faObserved_")):
        sBox = sKey.replace("faObserved_", "")
        faObserved = dictSamples[sKey].astype(float)
        faExpected = dictSamples[f"faExpected_{sBox}"]
        oAxes.hist(faObserved, bins=np.arange(0, 61) - 0.5, histtype="step", linewidth=2.0,
                   density=True, color=ps.DICT_BOX_COLOUR[sBox],
                   label=f"{sBox}: median {np.median(faObserved):.0f}, "
                         f"$P_{{25}}$ = {100*np.mean(faObserved >= fGoal):.2g}%")
        oAxes.hist(faExpected, bins=60, range=(0, 60), histtype="step", linewidth=0.9,
                   density=True, color=ps.DICT_BOX_COLOUR[sBox], alpha=0.55)
    oAxes.axvline(fGoal, color=ps.S_INK_SECONDARY, lw=1.2, ls="--")
    oAxes.annotate("25 EEC goal", xy=(fGoal, 0.0), xycoords=("data", "axes fraction"),
                   xytext=(4, 60), textcoords="offset points",
                   color=ps.S_INK_SECONDARY, fontsize=8)
    oAxes.set_xlim(0, 60)
    ps.fnFinishAxes(oAxes, "EEC yield", "Probability density",
                    "Realized yield (thin = expected)")
    oAxes.legend(loc="upper right")


def fdictParseArgs():
    """Command-line configuration for the yield-prediction figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["prediction"]) as oFile:
        dictPrediction = json.load(oFile)
    dictSamples = np.load(dictArgs["samples"])
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fnPlotYieldCurves(oAxesPair[0], dictPrediction)
    fnPlotDistributions(oAxesPair[1], dictSamples, dictPrediction["fYieldGoal"])
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
