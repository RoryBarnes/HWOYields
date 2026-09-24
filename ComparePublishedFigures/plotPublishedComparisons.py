#!/usr/bin/env python3
"""Draw this model's version of each comparable Stark et al. (2024) figure in the published axes.

Axis ranges, scales, units, line styles and colours follow the published figure so the two can be
set side by side; nothing is rescaled to improve agreement. Writes figMatchStarkFigNN.pdf for
each figure number into the output directory.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

S_PURPLE, S_RED = "#6a1b6f", "#e8202a"
S_GREEN, S_ORANGE = "#1aa84a", "#f5a300"
LIST_DIAMETER_STYLE = [("6", "-"), ("7", ":"), ("8", "--"), ("9", "-.")]
T_FIGSIZE = (3.6, 2.6)


def foNewAxes():
    """One published-size panel."""
    oFig, oAxes = plt.subplots(figsize=T_FIGSIZE)
    oAxes.grid(False)
    return oFig, oAxes


def fnSave(oFig, sDir, iFigure):
    """Save as figMatchStarkFigNN.pdf."""
    oFig.tight_layout()
    oFig.savefig(os.path.join(sDir, f"figMatchStarkFig{iFigure:02d}.pdf"))
    plt.close(oFig)


def fnYieldAxes(oAxes, fYMax=0.1):
    """The EEC-yield frequency axes Stark uses throughout Sec. 3."""
    oAxes.set_xlim(0, 50)
    oAxes.set_ylim(0, fYMax)
    oAxes.set_xlabel("EEC Yield")
    oAxes.set_ylabel("Frequency")


def fnFigure04(dictS, sDir):
    """Exoplanet sampling only."""
    oFig, oAxes = foNewAxes()
    oAxes.plot(np.arange(len(dictS["fig04"]["faPmf"])), dictS["fig04"]["faPmf"], color="k", lw=1.2)
    fnYieldAxes(oAxes)
    oAxes.set_title(f"model: mean {dictS['fig04']['fMean']:.1f}", fontsize=8, loc="right")
    fnSave(oFig, sDir, 4)


def fnFigure07(dictS, sDir):
    """Albedo drawn (green) and albedo plus exozodi drawn (orange)."""
    oFig, oAxes = foNewAxes()
    d = dictS["fig07"]
    oAxes.plot(np.arange(len(d["faPmfAlbedo"])), d["faPmfAlbedo"], color=S_GREEN, lw=1.2)
    oAxes.plot(np.arange(len(d["faPmfAlbedoExozodi"])), d["faPmfAlbedoExozodi"], color=S_ORANGE,
               lw=1.2)
    fnYieldAxes(oAxes)
    oAxes.set_title(f"model: means {d['fMeanAlbedo']:.1f} / {d['fMeanAlbedoExozodi']:.1f}",
                    fontsize=8, loc="right")
    fnSave(oFig, sDir, 7)


def fnFigure09(dictS, sDir):
    """Drawn exozodi levels (left) and the eta-fixed yield they produce (right)."""
    oFig, oAxesPair = plt.subplots(1, 2, figsize=(7.2, 2.6))
    d = dictS["fig09"]
    faEdges = np.asarray(d["faZodiEdges"])
    oAxesPair[0].step(faEdges[:-1], np.maximum(d["faNumberScaled"], 1e-3), where="post",
                      color=S_RED, lw=0.8, label="Max. Likelihood (model draws)")
    oAxesPair[0].set_yscale("log")
    oAxesPair[0].set_xlim(0, 1000)
    oAxesPair[0].set_ylim(1e2, 1e7)
    oAxesPair[0].set_xlabel("Zodis")
    oAxesPair[0].set_ylabel("Number")
    oAxesPair[0].legend(loc="upper right", fontsize=7)
    oAxesPair[1].plot(np.arange(len(d["faPmfEtaFixed"])), d["faPmfEtaFixed"], color=S_RED, lw=1.2)
    fnYieldAxes(oAxesPair[1], 0.12)
    for oAxes in oAxesPair:
        oAxes.grid(False)
    fnSave(oFig, sDir, 9)


def fnFigure10(dictS, sDir):
    """Realized yield including and excluding sigma_eta."""
    oFig, oAxes = foNewAxes()
    d = dictS["fig10"]
    faK = np.arange(len(d["faPmfIncluding"]))
    oAxes.plot(faK, d["faPmfExcluding"], color=S_RED, lw=1.2, label="Excluding $\\sigma_{\\eta}$")
    oAxes.plot(faK, d["faPmfIncluding"], color=S_PURPLE, lw=1.2,
               label="Including $\\sigma_{\\eta}$")
    oAxes.vlines(d["fMeanIncluding"], 0, np.interp(d["fMeanIncluding"], faK, d["faPmfIncluding"]),
                 color="k", ls=":", lw=1.0)
    fnYieldAxes(oAxes)
    oAxes.legend(loc="upper right", fontsize=7)
    fnSave(oFig, sDir, 10)


def fnFigure11(dictS, sDir, fDiameterM=6.0):
    """Targets in distance-luminosity, coloured by completeness, input list in grey."""
    oFig, oAxes = plt.subplots(figsize=(3.4, 2.9))
    d = dictS["fig11"]
    oAxes.scatter(d["faDistanceAll"], d["faLuminosityAll"], s=1.5, color="#cfcfcf", lw=0)
    faOrder = np.argsort(d["faCompletenessUsed"])
    oScatter = oAxes.scatter(np.asarray(d["faDistanceUsed"])[faOrder],
                             np.asarray(d["faLuminosityUsed"])[faOrder],
                             c=np.asarray(d["faCompletenessUsed"])[faOrder], cmap="turbo",
                             vmin=0, vmax=1, s=9, lw=0)
    faD = np.linspace(0.5, 40, 200)
    fLamDArcsec = 1e-6 / fDiameterM * 206264.806
    oAxes.plot(faD, (1.5 * fLamDArcsec * faD / 1.67) ** 2, color=S_RED, ls="--", lw=0.8)
    oAxes.axhline(fnNoiseFloorLuminosity(), color=S_RED, ls="--", lw=0.8)
    oAxes.set_yscale("log")
    oAxes.set_xlim(0, 40)
    oAxes.set_ylim(1e-3, 40)
    oAxes.set_xlabel("d (pc)")
    oAxes.set_ylabel("L$_{\\rm star}$ (L$_\\odot$)")
    oFig.colorbar(oScatter, ax=oAxes, label="Completeness")
    oAxes.grid(False)
    fnSave(oFig, sDir, 11)


def fnNoiseFloorLuminosity(fRadiusEarth=1.4, fAlbedo=0.2, fFloorDeltaMag=26.5):
    """Luminosity at which a 1.4 R_Earth planet at the EEID, at quadrature, hits the floor."""
    return fAlbedo / np.pi * (fRadiusEarth * 4.25875e-5) ** 2 / 10 ** (-0.4 * fFloorDeltaMag)


def fnFigure12(dictS, sDir):
    """Realized yield at 6-9 m, including and excluding sigma_eta."""
    oFig, oAxes = foNewAxes()
    for sD, sStyle in LIST_DIAMETER_STYLE:
        d = dictS["fig12"].get(sD)
        if d is None:
            continue
        faK = np.arange(len(d["faPmfIncluding"]))
        oAxes.plot(faK, d["faPmfExcluding"], color=S_RED, ls=sStyle, lw=1.0)
        oAxes.plot(faK, d["faPmfIncluding"], color=S_PURPLE, ls=sStyle, lw=1.0)
        oAxes.annotate(f"{sD} m ID", (float(np.argmax(d["faPmfExcluding"])),
                                     max(d["faPmfExcluding"]) + 0.003), fontsize=7,
                       ha="center")
    fnYieldAxes(oAxes)
    fnSave(oFig, sDir, 12)


def fnFigure14(dictS, sDir):
    """Characterization times of the first 18 EECs at 6-9 m."""
    oFig, oAxes = foNewAxes()
    for sD, sStyle in LIST_DIAMETER_STYLE:
        d = dictS["fig14"].get(sD)
        if d is None:
            continue
        faEdges = np.asarray(d["faEdgesDays"])
        oAxes.plot(0.5 * (faEdges[1:] + faEdges[:-1]), np.maximum(d["faFrequency"], 1e-4),
                   color=S_PURPLE, ls=sStyle, lw=1.0, label=f"{sD} m (mean {d['fMeanDays']:.1f} d)")
    oAxes.set_yscale("log")
    oAxes.set_xlim(0, 60)
    oAxes.set_ylim(2e-3, 0.2)
    oAxes.set_xlabel("Time to detect water vapor (days)")
    oAxes.set_ylabel("Frequency")
    oAxes.legend(loc="upper right", fontsize=6)
    fnSave(oFig, sDir, 14)


def fnFigure15(dictS, sDir):
    """P25 against inscribed diameter."""
    oFig, oAxes = foNewAxes()
    d = dictS["fig15"]
    oAxes.plot(d["faDiameterM"], 100 * np.asarray(d["faP25Excluding"]), "-o", color=S_RED,
               ms=3, lw=1.0, label="Excluding $\\sigma_{\\eta}$")
    oAxes.plot(d["faDiameterM"], 100 * np.asarray(d["faP25Including"]), "-o", color=S_PURPLE,
               ms=3, lw=1.0, label="Including $\\sigma_{\\eta}$")
    oAxes.set_xlim(5, 10)
    oAxes.set_ylim(0, 100)
    oAxes.set_xlabel("Inscribed diameter (m)")
    oAxes.set_ylabel("P$_{25}$ (%)")
    oAxes.legend(loc="lower right", fontsize=7)
    fnSave(oFig, sDir, 15)


def fnFigure25(dictS, sDir):
    """DMVC6 raw contrast (left axis) and core throughput (right axis)."""
    oFig, oAxes = foNewAxes()
    d = dictS["fig25"]
    oAxes.plot(d["fSeparationLamD"], d["fRawContrast"], color="k", ls=":", lw=1.0)
    oAxes.set_yscale("log")
    oAxes.set_xlim(0, 30)
    oAxes.set_ylim(1e-11, 1e-8)
    oAxes.set_xlabel("Separation ($\\lambda$/D)")
    oAxes.set_ylabel("Contrast, $\\zeta$")
    oTwin = oAxes.twinx()
    oTwin.plot(d["fSeparationLamD"], d["fCoreThroughput"], color="k", lw=1.2)
    oTwin.set_ylim(0, 1)
    oTwin.set_ylabel("Coronagraph core throughput, $\\Upsilon_c$")
    oTwin.grid(False)
    fnSave(oFig, sDir, 25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", required=True)
    p.add_argument("--out-dir", required=True)
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True})
    with open(dictArgs["series"]) as oFile:
        dictS = json.load(oFile)
    for fnFigure in (fnFigure04, fnFigure07, fnFigure09, fnFigure10, fnFigure11, fnFigure12,
                     fnFigure14, fnFigure15, fnFigure25):
        fnFigure(dictS, dictArgs["out_dir"])


if __name__ == "__main__":
    main()
