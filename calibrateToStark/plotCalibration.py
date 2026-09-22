#!/usr/bin/env python3
"""Plot the calibration bisection: modelled yield against the coronagraph throughput factor."""

import argparse
import json
import sys

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def fdictParseArgs():
    """Command-line configuration for the calibration figure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", required=True)
    p.add_argument("sPlotPath")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["calibration"]) as oFile:
        dictCal = json.load(oFile)
    listTrace = sorted(dictCal["listTrace"], key=lambda d: d["fCalibration"])
    oFig, oAxes = plt.subplots(figsize=(5.0, 3.2))
    oAxes.plot([d["fCalibration"] for d in listTrace], [d["fYield"] for d in listTrace],
               marker="o", ms=4, color=ps.LIST_SERIES[0], label="Rederived AYO model")
    oAxes.axhline(dictCal["fTargetYield"], color=ps.LIST_SERIES[1], lw=1.5, ls="--",
                  label="Stark et al. (2024): 22.5 EECs")
    oAxes.axvline(dictCal["fCalibratedThroughputFactor"], color=ps.S_INK_SECONDARY,
                  lw=1.0, ls=":")
    oAxes.annotate(f"calibrated factor = {dictCal['fCalibratedThroughputFactor']:.3f}",
                   xy=(dictCal["fCalibratedThroughputFactor"], 0.02),
                   xycoords=("data", "axes fraction"), xytext=(4, 6),
                   textcoords="offset points", color=ps.S_INK_SECONDARY, fontsize=8)
    oAxes.set_xscale("log")
    ps.fnFinishAxes(oAxes, "Coronagraph throughput calibration factor",
                    "Expected EEC yield", "Calibration gate")
    oAxes.legend(loc="upper left")
    oFig.tight_layout()
    oFig.savefig(dictArgs["sPlotPath"])


if __name__ == "__main__":
    main()
