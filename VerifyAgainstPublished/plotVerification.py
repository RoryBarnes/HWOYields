#!/usr/bin/env python3
"""Plot every published-record check as a signed relative difference, tuned entries marked."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the verification figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verification", required=True)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["verification"]) as oFile:
        listChecks = json.load(oFile)["listChecks"]
    listChecks = list(reversed(listChecks))
    faY = np.arange(len(listChecks))
    faDiff = np.array([100.0 * (d["fModel"] / d["fPublished"] - 1.0)
                       if d["fPublished"] else 0.0 for d in listChecks])
    listColour = [ps.S_INK_SECONDARY if d["bTuned"] else
                  (ps.LIST_SERIES[2] if d["bAgrees"] else ps.LIST_SERIES[1])
                  for d in listChecks]
    oFig, oAxes = plt.subplots(figsize=(7.2, 0.26 * len(listChecks) + 1.2))
    oAxes.barh(faY, np.clip(faDiff, -140, 140), color=listColour, height=0.66)
    oAxes.axvline(0.0, color=ps.S_INK_PRIMARY, lw=1.0)
    oAxes.set_yticks(faY)
    oAxes.set_yticklabels([d["sName"] for d in listChecks], fontsize=7)
    oAxes.set_xlim(-150, 150)
    ps.fnFinishAxes(oAxes, "model relative to published (%)", "",
                    "Green agrees, orange disagrees, grey was tuned")
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
