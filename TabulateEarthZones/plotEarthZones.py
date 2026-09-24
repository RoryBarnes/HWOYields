#!/usr/bin/env python3
"""Compare the Earth zone with the classic habitable zone, star by star and by distance."""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fnPlotPerStar(oAxes, dfStars):
    """Each star's expected Earth-zone yield against its classic-HZ yield."""
    dfUsed = dfStars[(dfStars["fYieldClassicHz"] > 1e-4) | (dfStars["fYieldEarthZone"] > 1e-4)]
    oScatter = oAxes.scatter(np.maximum(dfUsed["fYieldClassicHz"], 1e-4),
                             np.maximum(dfUsed["fYieldEarthZone"], 1e-4),
                             c=np.log10(dfUsed["fLuminosityLsun"]), cmap="viridis", s=14, lw=0)
    faLine = np.array([1e-4, 1.0])
    fMedian = float(np.median(dfUsed["fYieldEarthZone"] / np.maximum(dfUsed["fYieldClassicHz"],
                                                                     1e-9)))
    oAxes.plot(faLine, faLine, color=ps.S_INK_SECONDARY, lw=1.0, ls="--", label="equal")
    oAxes.set_xscale("log")
    oAxes.set_yscale("log")
    oAxes.set_xlim(1e-4, 0.5)
    oAxes.set_ylim(1e-4, 0.5)
    plt.colorbar(oScatter, ax=oAxes, label="log$_{10}$ L / L$_\\odot$")
    oAxes.legend(loc="upper left")
    ps.fnFinishAxes(oAxes, "expected EECs, classic HZ", "expected EECs, Earth zone",
                    "Per star (each zone's own survey)")
    return fMedian


def fnPlotCumulative(oAxes, dfStars):
    """Cumulative expected yield with distance for both zones."""
    dfSorted = dfStars.sort_values("fDistancePc")
    for sZone, sBox, sLabel in (("ClassicHz", "canonical", "classic HZ 0.95-1.67 AU"),
                                ("EarthZone", "redefined", "Earth zone 0.96-1.20 AU")):
        oAxes.plot(dfSorted["fDistancePc"], np.cumsum(dfSorted[f"fYield{sZone}"]),
                   color=ps.DICT_BOX_COLOUR[sBox], label=sLabel)
    oAxes.set_xlim(0, 30)
    oAxes.legend(loc="lower right")
    ps.fnFinishAxes(oAxes, "distance (pc)", "cumulative expected EECs",
                    "Where the yield comes from")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True)
    p.add_argument("sPlotPath")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    dfStars = pd.read_csv(dictArgs["table"])
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.4, 3.1))
    fnPlotPerStar(oAxesPair[0], dfStars)
    fnPlotCumulative(oAxesPair[1], dfStars)
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
