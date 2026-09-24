#!/usr/bin/env python3
"""Characterization time and noise terms for every bandpass option, for named 10-20 pc targets.

testDistantTargetReachability.py found that Stark et al. (2024) Fig. 11's 10-20 pc targets are
unreachable in this model because their spectra exceed the 60-day cap, and that their habitable
zones sit inside the coronagraph's inner working angle at 1000 nm. AYO chooses the spectral band
star by star from the S/N-versus-wavelength options of Stark's Fig. 3 (this model's nine options
match his 20 percent curve), and a shorter band puts the planet farther out in lambda/D. This asks
whether ANY option brings these stars under the cap, and which noise term decides it.

For each star, an Earth twin (R = 1, A_G = 0.2) at quadrature -- maximum elongation, the easiest
geometry a characterization can be scheduled at -- at the EEID and at the middle of the canonical
zone (1.31 EEID), is put through every option at 3 zodis with the pipeline's calibration. Reported
per option: separation in circumscribed lambda/D (the curves' units), core throughput, raw
contrast, every count rate, and the exposure time in days.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))

from yieldlib import completeness as cp  # noqa: E402
from yieldlib import physics as ph  # noqa: E402

LIST_STARS = ["tau Cet", "HD 147513", "54 Psc", "HD 72673", "HD 190360", "b Her", "e Vir",
              "171 Pup"]


def fdictOptionTerms(dictStar, fAxisScaled, dictBand, dictMission):
    """Every term of one option's exposure time for an Earth twin at quadrature."""
    fAxis = fAxisScaled * np.sqrt(dictStar["fLuminosityLsun"])
    dictPlanets = {"faRadiusEarth": np.array([1.0]), "faAxisAu": np.array([fAxis])}
    dictGeom = {"faSepAu": np.array([[fAxis]]),
                "faPhase": ph.faLambertianPhase(np.array([[np.pi / 2]]))}
    dictR = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    fTau = float(cp.faRequiredExposureTime([dictR], [dictBand], dictBand["fSignalToNoise"],
                                           dictMission).ravel()[0])
    fScale = dictMission.get("fCircumscribedRatio", 1.0)
    fSep = float(dictR["faSepLamD"].ravel()[0])
    fZeta = float(dictR["faLeak"].ravel()[0] / (dictR["faPlanet"].ravel()[0] /
                                                dictR["faFluxRatio"].ravel()[0]))
    return {"fLambdaNm": dictBand["fLambdaM"] * 1e9, "fSnr": dictBand["fSignalToNoise"],
            "fSepLamDCirc": fSep * fScale, "fUpsilon": float(dictR["faUpsilon"].ravel()[0]),
            "fContrast": fZeta, "fPlanet": float(dictR["faPlanet"].ravel()[0]),
            "fLeak": float(dictR["faLeak"].ravel()[0]),
            "fZodi": float(np.ravel(dictR["fZodi"])[0]),
            "fExozodi": float(np.ravel(dictR["fExozodi"])[0]),
            "fTauDays": fTau / 86400.0}


def fdictFindStar(dfCatalog, sName):
    """Catalog row whose SIMBAD name matches, whitespace-insensitive."""
    sKey = sName.replace(" ", "").lower()
    faMatch = dfCatalog["sSimbadName"].fillna("").str.replace("*", "").str.replace(
        " ", "").str.lower() == sKey
    return dfCatalog[faMatch].iloc[0].to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--calibration-json", default="../calibrateToStark/calibration.json")
    p.add_argument("--stars", default=",".join(LIST_STARS))
    p.add_argument("--out-json", default="output/characterizationBands.json")
    dictArgs = vars(p.parse_args())
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictMission = dictParams["dictMission"]
    with open(dictArgs["calibration_json"]) as oFile:
        dictMission["fThroughputCalibration"] = json.load(oFile)["fCalibratedThroughputFactor"]
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    dictOut = {}
    for sName in dictArgs["stars"].split(","):
        dictStar = fdictFindStar(dfCatalog, sName)
        dictStar["fExozodiLevel"] = dictMission["fExozodiLevel"]
        dictOut[sName] = {"fDistancePc": dictStar["fDistancePc"],
                          "fLuminosityLsun": dictStar["fLuminosityLsun"]}
        for sWhere, fAxis in (("eeid", 1.0), ("midZone", 1.31)):
            listRows = [fdictOptionTerms(dictStar, fAxis, b, dictMission)
                        for b in dictParams["dictBands"]["listBandsCharacterization"]]
            dictOut[sName][sWhere] = listRows
            dictBest = min(listRows, key=lambda d: d["fTauDays"])
            print(f"{sName:<10} d={dictStar['fDistancePc']:5.1f} L={dictStar['fLuminosityLsun']:5.2f} "
                  f"{sWhere:<7}| best {dictBest['fLambdaNm']:.0f} nm {dictBest['fTauDays']:8.1f} d | "
                  + " ".join(f"{d['fLambdaNm']:.0f}:{d['fTauDays']:.0f}d@{d['fSepLamDCirc']:.1f}"
                             for d in listRows[::2]))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)


if __name__ == "__main__":
    main()
