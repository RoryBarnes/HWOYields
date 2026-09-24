#!/usr/bin/env python3
"""Digitize Bryson et al. (2021) Fig. 14 (left): eta_Earth for the two completeness-extrapolation cases.

Stark et al. (2024) Sec. 3.5 build their eta_Earth distribution by drawing uniformly from the
Bryson et al. (2021) posteriors for two bounding cases -- completeness zero beyond 500 days and
completeness constant beyond 500 days -- but those posterior chains are not published (the
paper's GitHub repository omits the *_out directories, and the author's site holds only the
completeness contours). This figure is the closest published form: the conservative-HZ eta_Earth
distribution (model 1, hab2 stars, with input uncertainties, 0.5-1.5 R_Earth, 4800-6300 K) for each
case, drawn by matplotlib as two step histograms. x is linear in eta from the tick labels 0..5;
the marked medians and 68 percent intervals are recovered too, as a calibration check against
the paper's Table 7 (0.37 +0.48 -0.21 and 0.60 +0.90 -0.36).
"""

import argparse
import json
import re
import zlib

import numpy as np

DICT_CASES = {"extrapConst": "0.2156862745 0.4941176471 0.7215686275 RG",
              "extrapZero": "1 0.4980392157 0 RG"}


def fsPlotStream(sPdfPath):
    """Return the inflated content stream that carries the axes text."""
    with open(sPdfPath, "rb") as oFile:
        baRaw = oFile.read()
    for baStream in re.findall(rb"stream\r?\n(.*?)endstream", baRaw, re.S):
        try:
            sOut = zlib.decompress(baStream).decode("latin-1")
        except zlib.error:
            continue
        if "Relative Frequency" in sOut:
            return sOut
    raise RuntimeError("no stream with the Relative Frequency label")


def fdictXAxis(sContent):
    """Device x of the major ticks labelled 0 and 5."""
    listTicks = [float(a) for a in
                 re.findall(r"\n([\d.]+) 58\.85625 m\n[\d.]+ 55\.35625 l", sContent)]
    return {"fX0": min(listTicks), "fXPerUnit": (max(listTicks) - min(listTicks)) / 5.0,
            "fYBase": 58.85625}


def faStepVertices(sContent, sColour):
    """Vertices of the first stroked path after a colour: the case's step histogram.

    matplotlib wraps long lines, so a colour command can be split across a newline; the match
    therefore allows any whitespace between its tokens.
    """
    sPattern = r"\s+".join(re.escape(s) for s in sColour.split())
    iStart = re.search(sPattern, sContent).start()
    sBlock = sContent[iStart:sContent.index("\nS", iStart)]
    return np.array([(float(a), float(b)) for a, b in
                     re.findall(r"\n([\d.]+) ([\d.]+) [ml]", sBlock)])


def fdictHistogram(faXY, dictAxis):
    """Bin edges in eta and normalized bin probabilities from a step outline."""
    faX = (faXY[:, 0] - dictAxis["fX0"]) / dictAxis["fXPerUnit"]
    faH = faXY[:, 1] - dictAxis["fYBase"]
    faEdges = np.unique(np.round(faX, 6))
    faHeights = np.array([faH[(np.abs(faX - fLo) < 1e-5)].max()
                          for fLo in faEdges[:-1]])
    faHeights = np.clip(faHeights, 0.0, None)
    return {"faEdges": faEdges.tolist(), "faProbability": (faHeights / faHeights.sum()).tolist()}


def fdictSummary(dictHist):
    """Median and 16th/84th percentiles of a digitized histogram, uniform within bins."""
    faEdges, faP = np.array(dictHist["faEdges"]), np.array(dictHist["faProbability"])
    faCdf = np.concatenate([[0.0], np.cumsum(faP)])
    return {s: float(np.interp(q, faCdf, faEdges)) for s, q in
            (("fP16", 0.16), ("fMedian", 0.5), ("fP84", 0.84))} | \
        {"fMean": float(np.sum(0.5 * (faEdges[:-1] + faEdges[1:]) * faP))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", default="reference/brysonFigure13ConservativeHz.pdf")
    p.add_argument("--out-json", default="brysonFigure13EtaEarthDigitised.json")
    dictArgs = vars(p.parse_args())
    sContent = fsPlotStream(dictArgs["pdf"])
    dictAxis = fdictXAxis(sContent)
    dictOut = {"sSource": "Bryson et al. (2021) arXiv:2010.14812 Fig. 14 left (conservative HZ, "
                          "model 1, hab2, with input uncertainty; 0.5-1.5 R_Earth, 4800-6300 K)",
               "dictAxis": dictAxis,
               "dictPublishedTable7": {"extrapConst": [0.37, 0.48, 0.21],
                                       "extrapZero": [0.60, 0.90, 0.36]}}
    for sCase, sColour in DICT_CASES.items():
        dictHist = fdictHistogram(faStepVertices(sContent, sColour), dictAxis)
        dictOut[sCase] = {**dictHist, "dictSummary": fdictSummary(dictHist)}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sCase in DICT_CASES:
        print(sCase, {k: round(v, 3) for k, v in dictOut[sCase]["dictSummary"].items()},
              "table:", dictOut["dictPublishedTable7"][sCase])


if __name__ == "__main__":
    main()
