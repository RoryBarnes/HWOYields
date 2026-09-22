#!/usr/bin/env python3
"""Emit the baseline HWO mission parameter set and the parametric DMVC coronagraph curves.

Every value here is transcribed from Stark et al. (2024, JATIS 10, 034006) Tables 1 and 2,
except the coronagraph core-throughput and contrast profiles. Those are outputs of detailed
coronagraph simulations that the papers do not distribute, so they are reconstructed from the
three published anchors for the DMVC: IWA 3.5 lambda/D, ~45 percent core throughput at wide
separation, and useful throughput down to ~1.5 lambda/D, with contrast floored at 1e-10.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import coronagraph as cg  # noqa: E402

F_YEAR_S = 365.25 * 86400.0
F_HOUR_S = 3600.0


def fdictMissionParameters(fDiameterM, fExozodiLevel):
    """Baseline coronagraph-mission parameters, Stark et al. (2024) Tables 1 and 2."""
    return {
        "fDiameterM": fDiameterM,
        "fIwaLamD": cg.F_DEFAULT_IWA_LAMD,
        "fOwaLamD": cg.F_DEFAULT_OWA_LAMD,
        "fCoreThroughputMax": cg.F_DEFAULT_CORE_THROUGHPUT_MAX,
        "fContrastFloor": cg.F_DEFAULT_CONTRAST_FLOOR,
        "fNoiseFloorDeltaMag": 26.5,
        "fContaminationThroughput": 0.95,
        "fApertureRadiusLamD": 0.7,
        "fGeometricAlbedo": 0.2,
        "fZodiMagArcsec2": 23.0,
        "fExozodiMagArcsec2": 22.0,
        "fExozodiLevel": fExozodiLevel,
        "fDarkCurrent": 3.0e-5,
        "fReadNoise": 0.0,
        "fClockInducedCharge": 1.3e-3,
        "fQuantumEfficiency": 0.9,
        "fDetectiveQuantumEfficiency": 0.75,
        "fTotalScienceTimeS": 2.0 * F_YEAR_S,
        "fSlewOverheadS": 1.0 * F_HOUR_S,
        "fWavefrontOverheadS": 2.7 * F_HOUR_S,
        "fWavefrontMultiplier": 1.1,
        "fExposureLimitS": 60.0 * 86400.0,
        "fThroughputCalibration": 1.0,
    }


def fdictBandParameters():
    """Detection channels (parallel SW and LW) and the characterization channel."""
    return {
        "listBandsDetection": [
            {"sName": "LW", "fLambdaM": 550e-9, "fOpticalThroughput": 0.34,
             "fBandwidthFraction": 0.20, "fSignalToNoise": 7.0, "iNumPixels": 4},
            {"sName": "SW", "fLambdaM": 450e-9, "fOpticalThroughput": 0.15,
             "fBandwidthFraction": 0.20, "fSignalToNoise": 7.0, "iNumPixels": 4},
        ],
        "dictBandCharacterization": {
            "sName": "IFS", "fLambdaM": 1000e-9, "fOpticalThroughput": 0.23,
            "fBandwidthFraction": 1.0 / 140.0, "fSignalToNoise": 5.0, "iNumPixels": 96},
    }


def fdictSelectionBoxes():
    """The canonical HabEx/LUVOIR EEC box and the redefined mass-selected box."""
    return {
        "canonical": {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
                      "fHzInnerAu": 0.95, "fHzOuterAu": 1.67},
        "redefined": {"sRadiusMode": "mass", "fMassLoEarth": 0.5, "fMassHiEarth": 2.0,
                      "fMassRadiusExponent": 0.27, "fHzInnerAu": 0.96, "fHzOuterAu": 1.20},
        "hzOnly": {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
                   "fHzInnerAu": 0.96, "fHzOuterAu": 1.20},
    }


def fdfCoronagraphCurves(dictMission, iNumPoints):
    """Tabulate core throughput and raw contrast against separation in lambda/D."""
    faSep = np.logspace(np.log10(0.5), np.log10(dictMission["fOwaLamD"] * 1.2), iNumPoints)
    return pd.DataFrame({
        "fSeparationLamD": faSep,
        "fCoreThroughput": cg.faCoreThroughput(faSep, fIwaLamD=dictMission["fIwaLamD"],
                                               fOwaLamD=dictMission["fOwaLamD"],
                                               fThroughputMax=dictMission["fCoreThroughputMax"]),
        "fRawContrast": cg.faRawContrast(faSep, fContrastFloor=dictMission["fContrastFloor"],
                                         fOwaLamD=dictMission["fOwaLamD"]),
    })


def fdictParseArgs():
    """Command-line configuration for the mission and coronagraph model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diameter-m", type=float, default=6.0)
    p.add_argument("--exozodi-level", type=float, default=3.0)
    p.add_argument("--num-points", type=int, default=200)
    p.add_argument("--out-parameters", default="missionParameters.json")
    p.add_argument("--out-curves", default="coronagraphCurves.csv")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    dictMission = fdictMissionParameters(dictArgs["diameter_m"], dictArgs["exozodi_level"])
    dictOut = {"dictMission": dictMission, "dictBands": fdictBandParameters(),
               "dictBoxes": fdictSelectionBoxes(),
               "sProvenance": "Stark et al. 2024 JATIS 10 034006, Tables 1 and 2; coronagraph "
                              "profiles parametrized from published DMVC anchors."}
    with open(dictArgs["out_parameters"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    fdfCoronagraphCurves(dictMission, dictArgs["num_points"]).to_csv(
        dictArgs["out_curves"], index=False)
    print(json.dumps({"sWrote": dictArgs["out_parameters"],
                      "fIwaThroughput": float(cg.faCoreThroughput(dictMission["fIwaLamD"])),
                      "fPlateauThroughput": float(cg.faCoreThroughput(25.0))}, indent=2))


if __name__ == "__main__":
    main()
