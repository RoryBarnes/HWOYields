#!/usr/bin/env python3
"""Digitize Stark et al. (2024) Fig. 10, the EEC yield distributions with and without eta uncertainty.

The figure is vector: each distribution is one stroked polyline in its own CMYK colour (red
"Excluding sigma_eta", purple "Including sigma_eta") and the purple mean is a dashed vertical line.
Reading the content stream replaces the by-eye "mode near 10, mean near 21" with the plotted
values. The x axis is calibrated from the tick labels 0..50; points sit at bin centres k + 0.5,
so the value at centre k + 0.5 is the probability of a yield of exactly k.
"""

import argparse
import json
import re
import zlib

import numpy as np

DICT_COLOURS = {"excluding": "0 1 1 0 K", "including": "0.399902 1 0 0.430908 K"}


def fsInflateContent(sPdfPath):
    """Return the inflated content stream that draws the axes and curves."""
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for baStream in re.findall(rb"stream\r?\n(.*?)endstream", baRaw, re.S):
        try:
            sOut = zlib.decompress(baStream).decode("latin-1")
        except zlib.error:
            continue
        if "EEC Yield" in sOut:
            return sOut
    raise RuntimeError("no content stream with the EEC Yield axis label")


def fdictXCalibration(sContent):
    """Device x of the 0 and 50 major ticks, from the tick-label text positions."""
    listTicks = [(float(a), s) for a, s in
                 re.findall(r"1 0 0 1 ([-\d.]+) [-\d.]+ Tm\s*\((\d+)\)Tj", sContent)]
    dictTicks = {int(s): 10.0 * fX for fX, s in listTicks}
    faMajor = [float(a) for a in re.findall(r"([\d.]+) 399\.117 m\s*[\d.]+ 444\.758 l", sContent)]
    fX0, fX50 = min(faMajor), max(faMajor)
    return {"fX0": fX0, "fScale": (fX50 - fX0) / 50.0, "dictLabelX": dictTicks}


def faPolylineAfter(sContent, sColour):
    """Vertices of the first polyline stroked after a colour command."""
    iStart = sContent.index(sColour) + len(sColour)
    sBlock = sContent[iStart:sContent.index("\nS", iStart)]
    return np.array([(float(a), float(b)) for a, b in
                     re.findall(r"([-\d.]+) ([-\d.]+) [ml]", sBlock)])


def fdictCurve(faXY, dictCal, fBaseline):
    """Convert one polyline to yields and a probability per integer yield."""
    faYield = (faXY[:, 0] - dictCal["fX0"]) / dictCal["fScale"]
    faHeight = np.clip(faXY[:, 1] - fBaseline, 0.0, None)
    bCentre = np.abs(faYield - np.round(faYield - 0.5) - 0.5) < 0.05
    faK = np.round(faYield[bCentre] - 0.5).astype(int)
    faP = faHeight[bCentre] / faHeight[bCentre].sum()
    return {"iaYield": faK.tolist(), "faProbability": faP.tolist(),
            "fHeightAtAxisEnd": float(faHeight[-1] / faHeight.max())}


def fdictSummary(dictCurve):
    """Mode, median, truncated mean and P25 of a digitized curve (range 0..49 only)."""
    faK, faP = np.array(dictCurve["iaYield"]), np.array(dictCurve["faProbability"])
    faCdf = np.cumsum(faP)
    return {"iMode": int(faK[np.argmax(faP)]),
            "iMedian": int(faK[np.searchsorted(faCdf, 0.5)]),
            "fMeanTruncated": float(np.sum(faK * faP)),
            "fP25": float(faP[faK >= 25].sum()),
            "iaPlateau90": [int(faK[faP >= 0.9 * faP.max()].min()),
                            int(faK[faP >= 0.9 * faP.max()].max())]}


def ffMeanLine(sContent, dictCal):
    """Yield marked by the dashed vertical mean line."""
    oMatch = re.search(r"\] 0 d\s*([\d.]+) [\d.]+ m\s*([\d.]+) [\d.]+ l", sContent)
    return (float(oMatch.group(1)) - dictCal["fX0"]) / dictCal["fScale"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", default="reference/starkFigure10.pdf")
    p.add_argument("--out-json", default="starkFigure10Digitised.json")
    dictArgs = vars(p.parse_args())
    sContent = fsInflateContent(dictArgs["pdf"])
    dictCal = fdictXCalibration(sContent)
    dictOut = {"sSource": "Stark et al. (2024) arXiv:2405.19418v1 Fig. 10 (eta_earth_uncertainty)",
               "dictCalibration": dictCal, "fMeanLineIncluding": ffMeanLine(sContent, dictCal)}
    for sKey, sColour in DICT_COLOURS.items():
        dictCurve = fdictCurve(faPolylineAfter(sContent, sColour), dictCal, 399.117)
        dictOut[sKey] = {**dictCurve, "dictSummary": fdictSummary(dictCurve)}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"mean line (including sigma_eta): {dictOut['fMeanLineIncluding']:.2f}")
    for sKey in DICT_COLOURS:
        print(sKey, json.dumps(dictOut[sKey]["dictSummary"]),
              f"tail height at 50 / peak = {dictOut[sKey]['fHeightAtAxisEnd']:.3f}")


if __name__ == "__main__":
    main()
