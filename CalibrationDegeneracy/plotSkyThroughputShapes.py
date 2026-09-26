#!/usr/bin/env python3
"""Plot the model's T_sky(r) against the AYO-shaped alternative and AYO's own benchmark values.

Left: T_sky and core throughput for the DMVC6 against separation. Right: the ratio T_sky /
Upsilon_c, which is constant for the model's reconstruction and rises toward the inner working
angle for the AYO-shaped one; AYO's T_sky / T_core for the ETC benchmark's vortex coronagraph is
overplotted, since that is the only published evidence of AYO's shape.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

LIST_SEPARATION_TICKS = [2, 3, 5, 10, 20]


def fnSeparationAxis(oAxes):
    """Log separation axis labeled with plain numbers."""
    oAxes.set_xscale("log")
    oAxes.xaxis.set_minor_locator(NullLocator())
    oAxes.set_xticks(LIST_SEPARATION_TICKS)
    oAxes.set_xticklabels([str(i) for i in LIST_SEPARATION_TICKS])


def fnLeftPanel(oAxes, listRows):
    """Absolute throughputs for the DMVC6."""
    faSep = [r["fSepCirc"] for r in listRows]
    oAxes.plot(faSep, [r["fUpsilon"] for r in listRows], color=ps.S_INK_SECONDARY,
               label=r"core throughput $\Upsilon_c$")
    oAxes.plot(faSep, [r["fTskyModel"] for r in listRows], color=ps.LIST_SERIES[0],
               label=r"$T_{\rm sky}$, this model")
    oAxes.plot(faSep, [r["fTskyAyoShaped"] for r in listRows], color=ps.LIST_SERIES[1], ls="--",
               label=r"$T_{\rm sky}$, AYO-like shape")
    fnSeparationAxis(oAxes)
    ps.fnFinishAxes(oAxes, r"Separation (circumscribed $\lambda/D$)", "Throughput")
    oAxes.legend(loc="lower right")


def fnRightPanel(oAxes, listRows, listOvc):
    """Ratio of sky to core throughput for both shapes, with AYO's benchmark values."""
    listKeep = [r for r in listRows if r["fModelOverUpsilon"] is not None]
    faSep = [r["fSepCirc"] for r in listKeep]
    oAxes.plot(faSep, [r["fModelOverUpsilon"] for r in listKeep], color=ps.LIST_SERIES[0],
               label="this model (DMVC6)")
    oAxes.plot(faSep, [r["fAyoShapedOverUpsilon"] for r in listKeep], color=ps.LIST_SERIES[1],
               ls="--", label="AYO-like shape (DMVC6)")
    oAxes.scatter([r["fSepCirc"] for r in listOvc], [r["fAyoTskyOverCore"] for r in listOvc],
                  color=ps.LIST_SERIES[2], s=16, zorder=3,
                  label="AYO, ETC benchmark (vortex)")
    fnSeparationAxis(oAxes)
    ps.fnFinishAxes(oAxes, r"Separation (circumscribed $\lambda/D$)",
                    r"$T_{\rm sky}\,/\,\Upsilon_c$")
    oAxes.legend(loc="upper right")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", default="skyThroughputShapes.json")
    p.add_argument("--out-pdf", default="../Plot/figSkyThroughputShapes.pdf")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    dictShapes = json.load(open(dictArgs["shapes"]))
    oFig, (oLeft, oRight) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fnLeftPanel(oLeft, dictShapes["listDmvc6"])
    fnRightPanel(oRight, dictShapes["listDmvc6"], dictShapes["listOvc"])
    oFig.tight_layout()
    oFig.savefig(dictArgs["out_pdf"])
    plt.close(oFig)


if __name__ == "__main__":
    main()
