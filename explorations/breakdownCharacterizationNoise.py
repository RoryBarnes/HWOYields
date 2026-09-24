#!/usr/bin/env python3
"""Break characterization and detection times into their count-rate terms for easy and hard stars.

This pipeline's characterization times are too short for the easiest targets (mean of the first
18 EECs 9.9 d vs Stark's 22 d) and far too long for distant, luminous ones (tens to ~100 d),
which prices those stars out of the survey. Stark et al. (2019) Sec. 3 define the detector term as
n_pix (xi + RN^2/tau_read + 6.73 CR_sat CIC) with CR_sat "equal to 10 times the count rate
expected for a PSF core pixel of an Earth twin at quadrature, evaluated around each star
individually". This pipeline instead sets CR_sat to the actual per-pixel count rate of the scene
(leaked starlight + zodi + exozodi + the planet). For an Earth twin at quadrature at each star's
EEID, at fixed median exozodi, this lists every term per characterization spectral bin and per
detection band, and the resulting times with CR_sat defined both ways.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import completeness as cp  # noqa: E402
from yieldlib import physics as ph  # noqa: E402

DICT_STARS = {"8102": "tau Cet", "16537": "eps Eri", "64924": "61 Vir", "70497": "tet Boo",
              "84862": "72 Her", "92043": "110 Her"}


def fdictEarthTwin(dictStar):
    """A single Earth twin at quadrature at the star's EEID, in the library's planet format."""
    fEeid = float(np.sqrt(dictStar["fLuminosityLsun"]))
    return ({"faRadiusEarth": np.array([1.0]), "faAxisAu": np.array([fEeid])},
            {"faSepAu": np.array([[fEeid]]), "faPhase": ph.faLambertianPhase(np.array([[np.pi / 2]]))})


def fdictTerms(dictStar, dictBand, dictMission, fSnr):
    """Every count-rate term for the twin in one band, and tau with both CR_sat definitions."""
    dictPlanets, dictGeom = fdictEarthTwin(dictStar)
    dictR = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    fP, fLeak = float(dictR["faPlanet"].ravel()[0]), float(dictR["faLeak"].ravel()[0])
    fZodi = float(np.ravel(dictR["fZodi"])[0])
    fExozodi = float(np.ravel(dictR["fExozodi"])[0])
    fAstro = fLeak + fZodi + fExozodi
    iPix = dictBand["iNumPixels"]
    fCic, fDark = dictMission["fClockInducedCharge"], dictMission["fDarkCurrent"]
    dictSat = {"scene": (fAstro + fP) / iPix, "earthTwin": 10.0 * fP / iPix}
    dictOut = {"fPlanet": fP, "fLeak": fLeak, "fZodi": fZodi,
               "fExozodi": fExozodi, "fDark": iPix * fDark,
               "fSepLamD": float(dictR["faSepLamD"].ravel()[0])}
    for sMode, fSat in dictSat.items():
        fCicRate = iPix * ph.F_GEIGER_CIC_FACTOR * fSat * fCic
        fB = fAstro + iPix * fDark + fCicRate
        dictOut[f"fCic_{sMode}"] = fCicRate
        dictOut[f"fTauDays_{sMode}"] = fSnr ** 2 * (fP + 2.0 * fB) / fP ** 2 / 86400.0
    return dictOut


def fdictParams(dictArgs):
    """Mission parameters at the current calibration and fixed median exozodi."""
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["dictMission"]["bDrawExozodiLevels"] = False
    return dictParams


def fnPrint(sName, dictStar, dictRow):
    """One star's characterization-bin breakdown on two lines."""
    d = dictRow["dictCharacterization"]
    print(f"{sName:<8} d={dictStar['fDistancePc']:5.1f} L={dictStar['fLuminosityLsun']:5.2f} "
          f"sep={d['fSepLamD']:4.1f} l/D | P {d['fPlanet']:.2e} leak {d['fLeak']:.2e} "
          f"zodi {d['fZodi']:.2e} exo {d['fExozodi']:.2e} dark {d['fDark']:.2e}")
    print(f"{'':8} CIC scene {d['fCic_scene']:.2e} -> tau {d['fTauDays_scene']:7.2f} d | "
          f"CIC twin {d['fCic_earthTwin']:.2e} -> tau {d['fTauDays_earthTwin']:7.2f} d | "
          f"det tau {dictRow['dictDetection']['fTauDays_scene'] * 24:.2f} h / "
          f"{dictRow['dictDetection']['fTauDays_earthTwin'] * 24:.2f} h")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--out-json", default="output/characterizationNoiseBreakdown.json")
    dictArgs = vars(p.parse_args())
    dictParams = fdictParams(dictArgs)
    dictMission = dictParams["dictMission"]
    dfCat = pd.read_csv(dictArgs["target_catalog"])
    dfCat["sHip"] = [str(int(h)) if h == h else "" for h in dfCat["sHipName"]]
    dictBandChar = dictParams["dictBands"]["dictBandCharacterization"]
    dictBandDet = dictParams["dictBands"]["listBandsDetection"][0]
    dictOut = {}
    for sHip, sName in DICT_STARS.items():
        dictStar = dfCat[dfCat["sHip"] == sHip].iloc[0].to_dict()
        dictOut[sName] = {
            "dictCharacterization": fdictTerms(dictStar, dictBandChar, dictMission,
                                               dictBandChar["fSignalToNoise"]),
            "dictDetection": fdictTerms(dictStar, dictBandDet, dictMission,
                                        dictBandDet["fSignalToNoise"])}
        fnPrint(sName, dictStar, dictOut[sName])
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)


if __name__ == "__main__":
    main()
