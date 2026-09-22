#!/usr/bin/env python3
"""Plot the occurrence-rate posterior integrated over each selection box, and their ratio."""

import argparse
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the posterior figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--posterior", required=True)
    p.add_argument("--boxes", default="canonical,redefined")
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    dictNpz = np.load(dictArgs["posterior"])
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for sBox in [b.strip() for b in dictArgs["boxes"].split(",")]:
        faEta = dictNpz[f"faEta_{sBox}"]
        oAxesPair[0].hist(faEta, bins=60, histtype="step", linewidth=2.0, density=True,
                          color=ps.DICT_BOX_COLOUR[sBox],
                          label=f"{sBox}: {np.median(faEta):.3f}"
                                f"$^{{+{np.percentile(faEta,84)-np.median(faEta):.3f}}}"
                                f"_{{-{np.median(faEta)-np.percentile(faEta,16):.3f}}}$")
    oAxesPair[0].set_xscale("log")
    ps.fnFinishAxes(oAxesPair[0], "$\\eta$ integrated over the box", "Posterior density",
                    "Occurrence rate per box")
    oAxesPair[0].legend(loc="upper right")
    faRatio = dictNpz["faRatio"]
    oAxesPair[1].hist(faRatio, bins=60, histtype="step", linewidth=2.0, density=True,
                      color=ps.LIST_SERIES[2])
    oAxesPair[1].axvline(float(np.median(faRatio)), color=ps.S_INK_SECONDARY, lw=1.0, ls="--")
    oAxesPair[1].annotate(f"median {np.median(faRatio):.3f}",
                          xy=(float(np.median(faRatio)), 0.0), xytext=(0.215, 5.0),
                          color=ps.S_INK_SECONDARY, fontsize=8)
    ps.fnFinishAxes(oAxesPair[1], "$\\eta_{\\rm redefined}/\\eta_{\\rm canonical}$",
                    "Posterior density", "Ratio: $\\Gamma$ cancels exactly")
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
