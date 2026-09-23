#!/usr/bin/env python3
"""Name the constraint that makes distant targets unreachable, planet by planet.

Forty-one of the 154 targets in Stark et al. (2024) Fig. 11 come out of this model with a maximum
achievable completeness of exactly zero -- not expensive, impossible -- and that single fact now
accounts for every published check this pipeline still fails. The albedo penalty is 5% against a
published 12% and the exozodi penalty is 0.2% against 11%, both because a survey confined to
nearby stars is dominated by leaked starlight rather than by the zodiacal terms, and is therefore
insensitive to exactly the perturbations those two checks measure. P_25 and the mode of the
realized-yield distribution follow from the same place.

A completeness of zero means every injected planet failed at least one gate. There are only five
candidates, and they call for entirely different fixes, so this script counts them rather than
arguing about them. For a grid of stars spanning Fig. 11's distance and luminosity range it
injects EECs and tallies, for each, which gate stopped it first:

  * the astrophysical noise floor, delta-mag > 26.5, which no exposure time can beat;
  * zero coronagraph core throughput, inside the IWA or beyond the OWA;
  * detection time over the two-month limit;
  * characterization time over the two-month limit, which Stark et al. (2019) Sec. 5.3 says
    removes the planet from the yield entirely;
  * none of these, i.e. detectable.

The characterization gate is the one to watch: it applies at 1000 nm, where lambda/D is 1.8 times
larger than at the 550 nm detection wavelength, so a habitable zone that is comfortably resolved
for detection can sit inside the inner working angle for the spectrum that has to follow it.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import completeness as cp  # noqa: E402

F_REARTH_AU = 4.25875e-5


def fdictBlockerTally(dictStar, dictBox, dictParams, dictMission, iNumPlanets, iSeed):
    """Fraction of injected planets stopped by each gate, for one star."""
    rng = np.random.default_rng(iSeed)
    dictPlanets = cp.fdictInjectPlanets(dictBox, dictStar["fEeidAu"], iNumPlanets,
                                        dictParams["fAlpha"], dictParams["fBeta"], rng,
                                        int(dictMission.get("iMaxVisits", 1)))
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    listCharOptions = dictParams["dictBands"]["listBandsCharacterization"]
    faTauDet, faTauChar = cp.faDetectionTimes(dictStar, dictPlanets, dictGeom, listBandsDet,
                                              listCharOptions, dictMission, iNumPlanets)
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, listBandsDet[0], dictMission)
    faDeltaMag = -2.5 * np.log10(np.maximum(dictRates["faFluxRatio"], 1e-30))
    fCap = dictMission["fExposureLimitS"]

    bFloor = faDeltaMag > dictMission["fNoiseFloorDeltaMag"]
    bNoThroughput = (~bFloor) & (dictRates["faUpsilon"] <= 0.0)
    bDetSlow = (~bFloor) & (~bNoThroughput) & (~np.isfinite(faTauDet) | (faTauDet > fCap))
    bCharSlow = (~bFloor) & (~bNoThroughput) & (~bDetSlow) & \
        (~np.isfinite(faTauChar) | (faTauChar > fCap))
    bOk = (~bFloor) & (~bNoThroughput) & (~bDetSlow) & (~bCharSlow)
    iTotal = faTauDet.size
    faCharFinite = faTauChar[np.isfinite(faTauChar)]
    return {
        "fNoiseFloor": round(float(bFloor.sum()) / iTotal, 4),
        "fZeroThroughput": round(float(bNoThroughput.sum()) / iTotal, 4),
        "fDetectionOverCap": round(float(bDetSlow.sum()) / iTotal, 4),
        "fCharacterizationOverCap": round(float(bCharSlow.sum()) / iTotal, 4),
        "fDetectable": round(float(bOk.sum()) / iTotal, 4),
        "fMedianCharDaysWhereFinite": round(float(np.median(faCharFinite) / 86400.0), 2)
        if faCharFinite.size else None,
        "fMedianDetDaysWhereFinite": round(
            float(np.median(faTauDet[np.isfinite(faTauDet)]) / 86400.0), 3)
        if np.isfinite(faTauDet).any() else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--box", default="canonical")
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--teff-k", type=float, default=5772.0)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = dictParams["dictBands"]["listBandsCharacterization"]
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]

    listRows = []
    for fDistancePc in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        for fLum in (0.5, 1.0, 2.5, 5.0, 10.0):
            fTeff = dictArgs["teff_k"] * fLum ** 0.11
            dictStar = {"fTeffK": fTeff, "fLuminosityLsun": fLum,
                        "fRadiusRsun": np.sqrt(fLum) / (fTeff / 5772.0) ** 2,
                        "fDistancePc": fDistancePc, "fEeidAu": np.sqrt(fLum)}
            dictTally = fdictBlockerTally(dictStar, dictBox, dictParams, dictMission,
                                          dictArgs["num_planets"], dictArgs["seed"])
            dictTally.update({"fDistancePc": fDistancePc, "fLuminosityLsun": fLum,
                              "fEeidArcsec": round(float(np.sqrt(fLum) / fDistancePc), 4)})
            listRows.append(dictTally)
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"listRows": listRows}, oFile, indent=2)
    print(f"{'d(pc)':>6}{'L':>7}{'EEID\"':>8}{'floor':>8}{'noThru':>8}{'detCap':>8}"
          f"{'charCap':>9}{'OK':>8}{'charDays':>10}")
    for d in listRows:
        print(f"{d['fDistancePc']:6.0f}{d['fLuminosityLsun']:7.1f}{d['fEeidArcsec']:8.4f}"
              f"{d['fNoiseFloor']:8.3f}{d['fZeroThroughput']:8.3f}{d['fDetectionOverCap']:8.3f}"
              f"{d['fCharacterizationOverCap']:9.3f}{d['fDetectable']:8.3f}"
              f"{str(d['fMedianCharDaysWhereFinite']):>10}")


if __name__ == "__main__":
    main()
