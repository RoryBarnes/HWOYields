#!/usr/bin/env python3
"""Plot the parametric DMVC core-throughput and raw-contrast profiles against separation."""

import argparse
import sys

import pandas as pd

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the coronagraph figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curves", required=True)
    p.add_argument("--iwa-lamd", type=float, default=3.5)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    dfCurves = pd.read_csv(dictArgs["curves"])
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 2.9))
    oAxes = oAxesPair[0]
    oAxes.plot(dfCurves["fSeparationLamD"], dfCurves["fCoreThroughput"],
               color=ps.LIST_SERIES[0])
    oAxes.axvline(dictArgs["iwa_lamd"], color=ps.S_INK_SECONDARY, lw=1.0, ls="--")
    oAxes.annotate("IWA 3.5 $\\lambda/D$", xy=(dictArgs["iwa_lamd"], 0.05),
                   xytext=(4.6, 0.05), color=ps.S_INK_SECONDARY, fontsize=8)
    oAxes.set_xscale("log")
    ps.fnFinishAxes(oAxes, "Separation ($\\lambda/D$)", "Core throughput $\\Upsilon_c$",
                    "Core throughput")
    oAxes = oAxesPair[1]
    oAxes.plot(dfCurves["fSeparationLamD"], dfCurves["fRawContrast"], color=ps.LIST_SERIES[1])
    oAxes.set_xscale("log")
    oAxes.set_yscale("log")
    ps.fnFinishAxes(oAxes, "Separation ($\\lambda/D$)", "Raw contrast $\\zeta$",
                    "Raw contrast (floored at $10^{-10}$)")
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
