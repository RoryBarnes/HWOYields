#!/usr/bin/env python3
"""Tabulate the model's extended-source throughput T_sky(r) against the AYO-shaped alternative.

Stark et al. (2019) Eqs. 5-6 multiply the zodiacal and exozodiacal count rates by T_sky(x,y),
the PSF convolved with a uniform background. The DMVC6 map is not published. The model
reconstructs it as T_sky(r) = Upsilon_c(r) / EE(X), the core throughput divided by the Airy
encircled energy inside the photometric aperture of radius X, which makes T_sky / Upsilon_c
constant. The only AYO values available are for a different coronagraph, the USORT vortex of the
Stark et al. (2025) ETC benchmark; there AYO's T_sky / T_core rises toward the inner working
angle. reference/skyThroughputConvolutionTest.json condenses that into a factor F(u) on the
model's T_sky, linear in the core fraction u = Upsilon_c / Upsilon_c,max, which step A13's
s_sky = 1 applies. This script evaluates both shapes on the DMVC6 the pipeline uses, and lists
AYO's own benchmark values beside them, so the difference can be stated in numbers.
"""

import argparse
import json
import os
import sys

import numpy as np

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))
from yieldlib import completeness as cp  # noqa: E402
from yieldlib import coronagraph as cg  # noqa: E402

FA_SEPARATIONS_LAMD = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]


def flistDmvc6Rows(dictMission, dictShape):
    """Upsilon_c, the model's T_sky and the AYO-shaped T_sky at each separation."""
    dictTable = dictMission["dictCoronagraphTable"]
    faSep = np.array(FA_SEPARATIONS_LAMD)
    faUpsilon = cg.faCoreThroughputTable(faSep, dictTable)
    faModel = cp.faSkyThroughputAt(faUpsilon, dictMission)
    dictShaped = dict(dictMission, dictSkyThroughputShape=dictShape)
    faAyoShaped = cp.faSkyThroughputAt(faUpsilon, dictShaped)
    fUpsilonMax = max(dictTable["faUpsilon"])
    return [{"fSepCirc": float(s), "fUpsilon": float(u), "fCoreFraction": float(u / fUpsilonMax),
             "fTskyModel": float(m), "fTskyAyoShaped": float(a),
             "fModelOverUpsilon": float(m / u) if u > 0 else None,
             "fAyoShapedOverUpsilon": float(a / u) if u > 0 else None,
             "fAyoShapedOverModel": float(a / m) if m > 0 else None}
            for s, u, m, a in zip(faSep, faUpsilon, faModel, faAyoShaped)]


def flistOvcRows(dictReference):
    """AYO's own T_sky and T_core for the benchmark vortex, with the model's method beside them."""
    return [{"fSepCirc": r["fSepCirc"], "fCoreAyo": r["fCoreAyo"], "fTskyAyo": r["fTskyAyo"],
             "fAyoTskyOverCore": r["fTskyAyo"] / r["fCoreAyo"],
             "fModelMethodOverAyo": r["fUnconvolvedOverAyo"]}
            for r in dictReference["dictOvc"]["listRows"]]


def fdictSummary(dictMission, dictShape, listOvc):
    """The constants that define each shape."""
    fEe = cg.fnAiryEncircledEnergy(dictMission["fApertureRadiusLamD"])
    faRatio = np.array([r["fAyoTskyOverCore"] for r in listOvc])
    faMethod = np.array([r["fModelMethodOverAyo"] for r in listOvc])
    return {"fApertureRadiusLamD": dictMission["fApertureRadiusLamD"], "fAiryEncircledEnergy": fEe,
            "fModelTskyOverUpsilon": 1.0 / fEe,
            "fUpsilonMax": float(max(dictMission["dictCoronagraphTable"]["faUpsilon"])),
            "fTskyMax": float(cp.fnSkyThroughputFor(dictMission)),
            "faShapeCoreFraction": dictShape["faCoreFraction"], "faShapeFactor": dictShape["faFactor"],
            "fShapeFitRmsFrac": dictShape["fRmsFrac"],
            "faOvcSepRange": [min(r["fSepCirc"] for r in listOvc), max(r["fSepCirc"] for r in listOvc)],
            "faOvcTskyOverCoreRange": [float(faRatio.min()), float(faRatio.max())],
            "faOvcModelMethodOverAyoRange": [float(faMethod.min()), float(faMethod.max())]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--sky-shape", default="reference/skyThroughputConvolutionTest.json")
    p.add_argument("--out-json", default="skyThroughputShapes.json")
    dictArgs = vars(p.parse_args())
    dictMission = json.load(open(dictArgs["mission_parameters"]))["dictMission"]
    dictReference = json.load(open(dictArgs["sky_shape"]))
    dictShape = dictReference["dictEmpiricalShape"]
    listOvc = flistOvcRows(dictReference)
    dictOut = {"dictSummary": fdictSummary(dictMission, dictShape, listOvc),
               "listDmvc6": flistDmvc6Rows(dictMission, dictShape), "listOvc": listOvc}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    print(json.dumps(dictOut["dictSummary"], indent=1))


if __name__ == "__main__":
    main()
