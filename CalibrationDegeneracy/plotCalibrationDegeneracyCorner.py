#!/usr/bin/env python3
"""Corner plot of the (kappa, kappa_c, f_leak, s_sky) posteriors from fitCalibrationDegeneracy.py."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LIST_LABELS = [r"$\ln\kappa$", r"$\ln\kappa_c$ (spectra only)", r"$f_{\rm leak}$",
               r"$s_{\rm sky}$"]
DICT_POSTERIORS = {"yieldOnly": ("Yield 22.5 only", ps.LIST_SERIES[0]),
                   "all": ("All published constraints", ps.LIST_SERIES[1])}
FA_RANGE = [(np.log(0.6), np.log(3.0)), (np.log(0.4), np.log(2.5)), (1.0, 1.782), (0.0, 1.0)]


def faHpdLevels(faHist, listMass):
    """Density thresholds enclosing the given probability masses."""
    faSorted = np.sort(faHist.ravel())[::-1]
    faCum = np.cumsum(faSorted) / faSorted.sum()
    return sorted(faSorted[np.searchsorted(faCum, m)] for m in listMass)


def fnPanel2d(oAx, faX, faY, tRangeX, tRangeY, sColour):
    """68% and 95% contours of one posterior in one panel."""
    faH, faEx, faEy = np.histogram2d(faX, faY, bins=30, range=[tRangeX, tRangeY])
    faCx, faCy = 0.5 * (faEx[1:] + faEx[:-1]), 0.5 * (faEy[1:] + faEy[:-1])
    listLev = faHpdLevels(faH, [0.95, 0.68])
    if len(set(listLev)) == 2 and listLev[0] > 0:
        oAx.contour(faCx, faCy, faH.T, levels=listLev, colors=sColour,
                    linewidths=[1.0, 2.0])


def fnPanel1d(oAx, faX, tRange, sColour):
    """Normalized marginal histogram."""
    oAx.hist(faX, bins=30, range=tRange, density=True, histtype="step", color=sColour,
             linewidth=2.0)
    oAx.set_yticks([])


def fnCorner(dictChains, faAdopted, sOut):
    """Lower-triangle corner plot with both posteriors and the adopted point."""
    iN = len(LIST_LABELS)
    oFig, aAx = plt.subplots(iN, iN, figsize=(7.5, 7.5))
    for i in range(iN):
        for j in range(iN):
            oAx = aAx[i, j]
            if j > i:
                oAx.set_visible(False)
                continue
            for sKey, (sLabel, sColour) in DICT_POSTERIORS.items():
                faC = dictChains[sKey]
                if i == j:
                    fnPanel1d(oAx, faC[:, i], FA_RANGE[i], sColour)
                else:
                    fnPanel2d(oAx, faC[:, j], faC[:, i], FA_RANGE[j], FA_RANGE[i], sColour)
            if i != j:
                oAx.plot(faAdopted[j], faAdopted[i], marker="x", color=ps.S_INK_PRIMARY, ms=8)
                oAx.set_ylim(FA_RANGE[i])
            oAx.set_xlim(FA_RANGE[j])
            oAx.set_xlabel(LIST_LABELS[j] if i == iN - 1 else "")
            oAx.set_ylabel(LIST_LABELS[i] if (j == 0 and i > 0) else "")
            if i < iN - 1:
                oAx.set_xticklabels([])
            if j > 0 and i != j:
                oAx.set_yticklabels([])
    fnLegend(oFig)
    oFig.subplots_adjust(hspace=0.08, wspace=0.08)
    oFig.savefig(sOut, bbox_inches="tight")


def fnLegend(oFig):
    """Legend in the empty upper-right triangle."""
    listHandles = [plt.Line2D([], [], color=c, lw=2, label=l) for l, c in DICT_POSTERIORS.values()]
    listHandles.append(plt.Line2D([], [], color=ps.S_INK_PRIMARY, marker="x", lw=0,
                                  label=r"Adopted ($\kappa$=1.20, others off)"))
    oFig.legend(handles=listHandles, loc="upper right", bbox_to_anchor=(0.9, 0.88))
    oFig.text(0.52, 0.72, "Contours: 68% (thick), 95% (thin)", color=ps.S_INK_SECONDARY)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chains", default="calibrationDegeneracyChains.npz")
    p.add_argument("--adopted-kappa", type=float, default=1.197)
    p.add_argument("--out-pdf", default="../Plot/figCalibrationDegeneracyCorner.pdf")
    dictArgs = vars(p.parse_args())
    ps.fnApplyStyle()
    dictNpz = np.load(dictArgs["chains"])
    fnCorner({k: dictNpz[k] for k in DICT_POSTERIORS},
             np.array([np.log(dictArgs["adopted_kappa"]), 0.0, 1.0, 0.0]), dictArgs["out_pdf"])


if __name__ == "__main__":
    main()
