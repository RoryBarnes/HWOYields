#!/usr/bin/env python3
"""Digitize Stark et al. (2024) Fig. 11 from the published PDF's vector content stream.

The figure plots every selected target in stellar luminosity against distance, coloured by
habitable-zone completeness, and is the only published per-target completeness data available.
Rather than rasterizing it, this reads the PDF's drawing commands directly: each target is a
filled polygon with its own CMYK colour, and the colour bar is drawn in the same stream, so the
colour-to-completeness mapping is recovered from the figure itself instead of being guessed from
a named colormap. Axes are calibrated from the tick-label positions.
"""

import argparse
import json
import re
import zlib

import numpy as np


def fsInflateContent(sPdfPath):
    """Return the inflated content stream of the figure PDF."""
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for baStream in re.findall(rb"stream\r?\n(.*?)endstream", baRaw, re.S):
        try:
            baOut = zlib.decompress(baStream)
        except zlib.error:
            continue
        if b" k\n" in baOut or b" k\r" in baOut:
            return baOut.decode("latin-1")
    raise RuntimeError("no inflated content stream with CMYK fills")


def flistFilledPaths(sContent):
    """Every filled path: its CMYK colour, vertex count and bounding box."""
    listOut = []
    for oMatch in re.finditer(
            r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) k\s*(.*?)\bf\b", sContent, re.S):
        faColour = tuple(float(oMatch.group(i)) for i in range(1, 5))
        listPts = [(float(a), float(b)) for a, b in
                   re.findall(r"([-\d.]+) ([-\d.]+) (?:m|l)", oMatch.group(5))]
        if len(listPts) < 3:
            continue
        faPts = np.array(listPts)
        listOut.append({"faColour": faColour, "iVertices": len(listPts),
                        "faCentre": faPts.mean(axis=0),
                        "fWidth": float(np.ptp(faPts[:, 0])),
                        "fHeight": float(np.ptp(faPts[:, 1]))})
    return listOut


def fdictAxisCalibration(sContent):
    """Device-to-data mapping from the numeric tick labels and their tick positions."""
    listText = [(float(a), float(b), c) for a, b, c in
                re.findall(r"1 0 0 1 ([-\d.]+) ([-\d.]+) Tm\s*\((.*?)\)Tj", sContent)]
    dictX, dictY = {}, {}
    for fX, fY, sLabel in listText:
        try:
            fValue = float(sLabel)
        except ValueError:
            continue
        if fY < 40.0:
            dictX.setdefault(fValue, []).append(fX * 10.0)
        elif fX < 60.0:
            dictY.setdefault(fValue, []).append(fY * 10.0)
    return dictX, dictY


def fnLinearFit(dictTicks):
    """Least-squares slope and intercept mapping device coordinate to data value."""
    faValue = np.array(sorted(dictTicks))
    faDevice = np.array([np.mean(dictTicks[f]) for f in faValue])
    fSlope, fIntercept = np.polyfit(faDevice, faValue, 1)
    return fSlope, fIntercept


def faDecodePngPredictor(baData, iColumns, iColours):
    """Undo PNG predictors on an inflated PDF image stream, returning rows of samples."""
    iStride = iColumns * iColours
    faOut = np.zeros((len(baData) // (iStride + 1), iStride), dtype=np.int64)
    faPrior = np.zeros(iStride, dtype=np.int64)
    for iRow in range(faOut.shape[0]):
        iOffset = iRow * (iStride + 1)
        iFilter = baData[iOffset]
        faLine = np.frombuffer(baData[iOffset + 1:iOffset + 1 + iStride],
                               dtype=np.uint8).astype(np.int64).copy()
        for i in range(iStride):
            iLeft = faLine[i - iColours] if i >= iColours else 0
            iUp = faPrior[i]
            iUpLeft = faPrior[i - iColours] if i >= iColours else 0
            if iFilter == 1:
                faLine[i] += iLeft
            elif iFilter == 2:
                faLine[i] += iUp
            elif iFilter == 3:
                faLine[i] += (iLeft + iUp) // 2
            elif iFilter == 4:
                iP = iLeft + iUp - iUpLeft
                listCand = [abs(iP - iLeft), abs(iP - iUp), abs(iP - iUpLeft)]
                faLine[i] += [iLeft, iUp, iUpLeft][int(np.argmin(listCand))]
            faLine[i] &= 0xFF
        faOut[iRow] = faLine
        faPrior = faLine
    return faOut


def faColourBarFromImage(sPdfPath, iMinWidth=200, iMaxHeight=60):
    """Recover the colour bar from the figure's embedded CMYK image.

    The bar is not drawn as filled paths; it is a wide, short DeviceCMYK image. Decoding it
    gives the exact colour-to-completeness mapping the figure itself uses, rather than an
    assumption about which named colormap was applied.
    """
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for oMatch in re.finditer(
            rb"/Subtype\s*/Image\s*/ColorSpace\s*/DeviceCMYK\s*/Width (\d+)\s*/Height (\d+)"
            rb".*?stream\r?\n", baRaw, re.S):
        iWidth, iHeight = int(oMatch.group(1)), int(oMatch.group(2))
        if iWidth < iMinWidth or iHeight > iMaxHeight:
            continue
        iStart = oMatch.end()
        baStream = baRaw[iStart:baRaw.find(b"endstream", iStart)]
        faRows = faDecodePngPredictor(zlib.decompress(baStream), iWidth, 4)
        faMiddle = faRows[faRows.shape[0] // 2].reshape(iWidth, 4) / 255.0
        return faMiddle
    return None


def flistColourBar(listPaths):
    """The colour-bar swatches: wide, short, low on the page, ordered left to right."""
    listWide = [d for d in listPaths
                if d["fWidth"] > 1.6 * max(d["fHeight"], 1e-9) and d["iVertices"] <= 8]
    if len(listWide) < 8:
        return []
    fMedianY = np.median([d["faCentre"][1] for d in listWide])
    listBar = [d for d in listWide if abs(d["faCentre"][1] - fMedianY) < 60.0]
    return sorted(listBar, key=lambda d: d["faCentre"][0])


def flistColourBarRgb(faBarCmyk, iSamples=256):
    """The colour bar resampled to iSamples RGB triples, from zero to full completeness.

    DeviceCMYK converts to RGB as R = (1 - C)(1 - K) and likewise for G and B, which is how the
    colours render in the published figure. Plots of this model reuse the list as their colormap,
    so the two figures share one colour scale.
    """
    faCmyk = np.asarray(faBarCmyk, dtype=float)
    faRgb = (1.0 - faCmyk[:, :3]) * (1.0 - faCmyk[:, 3:4])
    faPositions = np.linspace(0.0, 1.0, len(faRgb))
    faOut = np.linspace(0.0, 1.0, iSamples)
    return [[float(np.interp(f, faPositions, faRgb[:, j])) for j in range(3)] for f in faOut]


def faCompletenessFromColour(faBarColours, faColour):
    """Nearest colour along the bar, expressed as a fraction of its length."""
    faPositions = np.linspace(0.0, 1.0, len(faBarColours))
    faDistance = np.linalg.norm(faBarColours - np.array(faColour), axis=1)
    return float(faPositions[int(np.argmin(faDistance))]), float(np.min(faDistance))


def fdictParseArgs():
    """Command-line configuration for the figure digitization."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True)
    p.add_argument("--max-colour-distance", type=float, default=0.25)
    p.add_argument("--out-json", default="starkFigure11Digitised.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    sContent = fsInflateContent(dictArgs["pdf"])
    listPaths = flistFilledPaths(sContent)
    faBar = faColourBarFromImage(dictArgs["pdf"])
    if faBar is None:
        raise RuntimeError("colour bar image not found in the figure PDF")
    dictX, dictY = fdictAxisCalibration(sContent)
    fXs, fXi = fnLinearFit(dictX)
    faLogY = {np.log10(f): v for f, v in dictY.items() if f > 0}
    fYs, fYi = fnLinearFit({k: v for k, v in faLogY.items()})
    listPoints = []
    for dictPath in listPaths:
        if dictPath["iVertices"] < 12 or dictPath["fWidth"] > 200.0:
            continue
        fComp, fDist = faCompletenessFromColour(faBar, dictPath["faColour"])
        if fDist > dictArgs["max_colour_distance"]:
            continue
        listPoints.append({
            "fDistancePc": float(fXs * dictPath["faCentre"][0] + fXi),
            "fLuminosityLsun": float(10.0 ** (fYs * dictPath["faCentre"][1] + fYi)),
            "fCompleteness": fComp})
    dictOut = {
        "sSource": "Stark et al. (2024) Fig. 11, digitized from the PDF content stream",
        "iColourBarSamples": int(len(faBar)),
        "listColourBarRgb": flistColourBarRgb(faBar),
        "dictXTicks": {str(k): float(np.mean(v)) for k, v in dictX.items()},
        "dictYTicks": {str(k): float(np.mean(v)) for k, v in dictY.items()},
        "iPoints": len(listPoints), "listPoints": listPoints,
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    faD = np.array([d["fDistancePc"] for d in listPoints])
    faL = np.array([d["fLuminosityLsun"] for d in listPoints])
    faC = np.array([d["fCompleteness"] for d in listPoints])
    print(f"colour-bar samples  : {len(faBar)}")
    print(f"x ticks             : {sorted(dictX)}")
    print(f"y ticks             : {sorted(dictY)}")
    print(f"points recovered    : {len(listPoints)}")
    if len(listPoints):
        print(f"distance range      : {faD.min():.1f} to {faD.max():.1f} pc")
        print(f"luminosity range    : {faL.min():.3f} to {faL.max():.2f} Lsun")
        print(f"completeness        : median {np.median(faC):.2f}, "
              f"mean {faC.mean():.2f}, sum {faC.sum():.1f}")


if __name__ == "__main__":
    main()
