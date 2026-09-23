#!/usr/bin/env python3
"""Digitize the DMVC6 contrast and core-throughput curves from Stark et al. (2024) Fig. 12.

The parametric coronagraph in this pipeline was fitted to a handful of points read off that
figure by eye, under a 0.75 relative tolerance that is loose enough to hide a wrong SHAPE. The
shape is now known to matter: spectral characterization at 1000 nm operates between about 1.5 and
4 lambda/D for the marginal targets, and that is exactly the band where a few read-off points
cannot constrain the curve. The published numbers this model misses -- a mean characterization
time for the first 18 EECs of 9.9 days against 22, and an aperture ratio of 10.5 against 6.2 --
are both consistent with the modelled contrast being too shallow inside the knee.

Rather than read more points off the image, this parses the figure's vector content stream, as
was done for Fig. 11. The figure distinguishes its four curves unambiguously in the drawing
state: CMYK 1 1 1 1 is the DMVC6 and 0 1 1 0 the PIAA-FPM2.5, a dashed pattern marks raw contrast
and a solid one marks core throughput, and the axis tick marks carry their own dash state, so
they can be separated from the data and used to calibrate. The left axis is logarithmic in
contrast and the right is linear in throughput.
"""

import argparse
import json
import re
import zlib

import numpy as np

S_BLACK = "1 1 1 1"
S_RED = "0 1 1 0"
S_DASHED = "[ 11.3386 28.3465 ] 0"
S_SOLID = "[ ] 0"
S_TICKS = "[] 0"


def fsInflateDrawingStream(sPdfPath):
    """Return the inflated content stream that carries the drawing operators."""
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for baStream in re.findall(rb"stream\r?\n(.*?)endstream", baRaw, re.S):
        try:
            sOut = zlib.decompress(baStream).decode("latin-1")
        except zlib.error:
            continue
        if " m\n" in sOut:
            return sOut
    raise RuntimeError("no drawing stream found in " + sPdfPath)


def fdictParsePolylines(sStream):
    """Collect every stroked polyline, keyed by the (colour, dash) state in force."""
    dictOut = {}
    sColour, sDash, listCurrent = "?", S_SOLID, []
    for sLine in sStream.split("\n"):
        sLine = sLine.strip()
        oColour = re.match(r"^([\d.]+ [\d.]+ [\d.]+ [\d.]+) K$", sLine)
        oDash = re.match(r"^(\[[^]]*\] [\d.]+) d$", sLine)
        oMove = re.match(r"^([\d.eE+-]+) ([\d.eE+-]+) m$", sLine)
        oLine = re.match(r"^([\d.eE+-]+) ([\d.eE+-]+) l$", sLine)
        if oColour:
            sColour = oColour.group(1)
        elif oDash:
            sDash = oDash.group(1)
        elif oMove:
            if len(listCurrent) > 1:
                dictOut.setdefault((sColour, sDash), []).append(np.array(listCurrent))
            listCurrent = [(float(oMove.group(1)), float(oMove.group(2)))]
        elif oLine:
            listCurrent.append((float(oLine.group(1)), float(oLine.group(2))))
        elif sLine == "S" and len(listCurrent) > 1:
            dictOut.setdefault((sColour, sDash), []).append(np.array(listCurrent))
            listCurrent = []
    if len(listCurrent) > 1:
        dictOut.setdefault((sColour, sDash), []).append(np.array(listCurrent))
    return dictOut


def flistTickPositions(listSegments, bVertical, fAxisCoord, fTolerance):
    """Positions of the tick marks growing off one axis line."""
    listOut = []
    for faSeg in listSegments:
        if len(faSeg) != 2:
            continue
        (fX0, fY0), (fX1, fY1) = faSeg[0], faSeg[1]
        if bVertical and abs(fX1 - fX0) < 1e-6 and abs(fY0 - fAxisCoord) < fTolerance:
            listOut.append(fX0)
        elif not bVertical and abs(fY1 - fY0) < 1e-6 and abs(fX0 - fAxisCoord) < fTolerance:
            listOut.append(fY0)
    return sorted(set(listOut))


def fdictLinearCalibration(listPositions, listValues):
    """Least-squares mapping from device position to axis value."""
    faP, faV = np.array(listPositions, dtype=float), np.array(listValues, dtype=float)
    fSlope, fIntercept = np.polyfit(faP, faV, 1)
    faPredicted = fSlope * faP + fIntercept
    return {"fSlope": float(fSlope), "fIntercept": float(fIntercept),
            "fMaxResidual": float(np.max(np.abs(faPredicted - faV)))}


def faCurveToAxes(faCurve, dictX, dictY, bLogY):
    """Convert one polyline from device coordinates to axis values."""
    faXv = dictX["fSlope"] * faCurve[:, 0] + dictX["fIntercept"]
    faYv = dictY["fSlope"] * faCurve[:, 1] + dictY["fIntercept"]
    if bLogY:
        faYv = 10.0 ** faYv
    faOrder = np.argsort(faXv)
    return faXv[faOrder], faYv[faOrder]


def flistMajorTicks(listSegments, bVertical, fAxisCoord, fLongLength):
    """Positions of the MAJOR ticks on one axis, identified by their greater length.

    Major and minor ticks differ only in length in this figure -- 43.09 against 21.54 on the
    abscissa, 50.46 against 25.23 on both ordinates -- so the labelled positions are recoverable
    without matching text, which would be unreliable because the labels are left-anchored and
    therefore not centred on their ticks.
    """
    listOut = []
    for faSeg in listSegments:
        if len(faSeg) != 2:
            continue
        (fX0, fY0), (fX1, fY1) = faSeg[0], faSeg[1]
        if bVertical and abs(fX1 - fX0) < 1e-6 and abs(fY0 - fAxisCoord) < 1.0 \
                and abs(abs(fY1 - fY0) - fLongLength) < 1.0:
            listOut.append(fX0)
        elif not bVertical and abs(fY1 - fY0) < 1e-6 and abs(fX0 - fAxisCoord) < 1.0 \
                and abs(abs(fX1 - fX0) - fLongLength) < 1.0:
            listOut.append(fY0)
    return sorted(set(listOut))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure-pdf", required=True)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    sStream = fsInflateDrawingStream(dictArgs["figure_pdf"])
    dictPolys = fdictParsePolylines(sStream)
    listSolidBlack = dictPolys.get((S_BLACK, S_SOLID), [])
    listTicks = [a for a in listSolidBlack if len(a) == 2]

    listX = flistMajorTicks(listTicks, True, 432.0, 43.09)
    listContrast = flistMajorTicks(listTicks, False, 540.0, 50.46)
    listThroughput = flistMajorTicks(listTicks, False, 3060.0, 50.45)
    if len(listX) != 7 or len(listContrast) != 4 or len(listThroughput) != 6:
        raise RuntimeError(f"unexpected tick counts: x={len(listX)} "
                           f"contrast={len(listContrast)} throughput={len(listThroughput)}")

    dictX = fdictLinearCalibration(listX, [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    dictContrast = fdictLinearCalibration(listContrast, [-11.0, -10.0, -9.0, -8.0])
    dictThroughput = fdictLinearCalibration(listThroughput, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    dictOut = {
        "sSource": "Stark et al. (2024) Fig. 12, digitized from the PDF vector content stream",
        "sCurve": "DMVC6, the coronagraph adopted for the LUVOIR-B baseline",
        "sContrastNote": "raw contrast for an on-axis source of diameter 0.1 lambda/D",
        "dictXCalibration": dictX,
        "dictContrastCalibration": dictContrast,
        "dictThroughputCalibration": dictThroughput,
    }
    listCurve = [a for a in listSolidBlack if len(a) > 2]
    if listCurve:
        faX, faY = faCurveToAxes(max(listCurve, key=len), dictX, dictThroughput, False)
        dictOut["listCoreThroughput"] = [{"fSeparationLamD": round(float(a), 4),
                                          "fUpsilon": round(float(b), 5)}
                                         for a, b in zip(faX, faY)]
    listDashed = dictPolys.get((S_BLACK, S_DASHED), [])
    if listDashed:
        faX, faY = faCurveToAxes(max(listDashed, key=len), dictX, dictContrast, True)
        dictOut["listRawContrast"] = [{"fSeparationLamD": round(float(a), 4),
                                       "fContrast": float(f"{b:.4g}")}
                                      for a, b in zip(faX, faY)]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if not k.startswith("list")}, indent=2))
    for sKey, sVal in (("listCoreThroughput", "fUpsilon"), ("listRawContrast", "fContrast")):
        listRows = dictOut.get(sKey, [])
        print(f"\n{sKey}: {len(listRows)} points")
        for fTarget in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0):
            listNear = [d for d in listRows if abs(d["fSeparationLamD"] - fTarget) < 0.25]
            if listNear:
                dictNear = min(listNear, key=lambda d: abs(d["fSeparationLamD"] - fTarget))
                print(f"   {fTarget:5.1f} lambda/D -> {dictNear[sVal]}")


if __name__ == "__main__":
    main()
