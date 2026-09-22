#!/usr/bin/env python3
"""Why distant targets vanish: the characterization requirement, not detection.

Completeness here collapses beyond about 10 pc while Stark et al. (2024) Fig. 11 sustains a
median of 0.25 out to 30 pc. This separates the two possible causes -- that distant planets
cannot be DETECTED, or that they cannot be CHARACTERIZED -- by computing completeness under both
counting rules, and reports the characterization cost breakdown that drives the difference.

The decisive number is the ratio of characterization to detection exposure, which grows with
distance rather than staying constant. Characterization runs at 1000 nm where lambda/D is 1.8
times larger than at the 550 nm detection wavelength, so the planet sits proportionally closer to
the inner working angle, losing core throughput, while the photometric aperture grows as
(lambda/D)^2 and admits more exozodiacal light.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import completeness as cp  # noqa: E402
from yieldlib import physics as ph  # noqa: E402


def fdictAtDistance(dictParams, fDistancePc, iNumPlanets, iSeed):
    """Detection and characterization economics for a solar twin at one distance."""
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = None
    listBands = dictParams["dictBands"]["listBandsDetection"]
    dictChar = dictParams["dictBands"]["dictBandCharacterization"]
    dictStar = {"fTeffK": 5772.0, "fRadiusRsun": 1.0, "fLuminosityLsun": 1.0,
                "fDistancePc": fDistancePc}
    rng = np.random.default_rng(iSeed)
    dictPlanets = cp.fdictInjectPlanets(dictParams["dictBoxes"]["canonical"], 1.0, iNumPlanets,
                                        -0.19, 0.26, rng, iVisits=1)
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    faTauDet, faTauChar = cp.faDetectionTimes(dictStar, dictPlanets, dictGeom, listBands,
                                              [dictChar], dictMission, iNumPlanets)
    fCap = dictMission["fExposureLimitS"]
    bDet = np.isfinite(faTauDet) & (faTauDet <= fCap)
    bBoth = bDet & np.isfinite(faTauChar) & (faTauChar <= fCap)
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictChar, dictMission)
    faAstro = dictRates["faLeak"] + dictRates["fZodi"] + dictRates["faExozodi"]
    faDetector = ph.faDetectorCountRate((faAstro + dictRates["faPlanet"]) / dictChar["iNumPixels"],
                                        dictChar["iNumPixels"], dictMission["fDarkCurrent"],
                                        dictMission["fReadNoise"], None,
                                        dictMission["fClockInducedCharge"])
    faTotal = faAstro + faDetector
    return {
        "fDistancePc": fDistancePc,
        "fDetectableFraction": float(bDet.mean()),
        "fAlsoCharacterizableFraction": float(bBoth.mean()),
        "fMedianTauDetDays": float(np.median(faTauDet[bDet]) / 86400.0) if bDet.any() else None,
        "fMedianTauCharDays": float(np.median(faTauChar[bDet]) / 86400.0) if bDet.any() else None,
        "fCharOverDet": float(np.median(faTauChar[bDet]) / np.median(faTauDet[bDet]))
        if bDet.any() else None,
        "fExozodiShareOfCharBackground": float(np.median(
            (dictRates["faExozodi"] / faTotal)[bDet])) if bDet.any() else None,
        "fDetectorShareOfCharBackground": float(np.median(
            (faDetector / faTotal)[bDet])) if bDet.any() else None,
    }


def fdictParseArgs():
    """Command-line configuration for the characterization-economics diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--distances", default="5,10,15,20,25")
    p.add_argument("--num-planets", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="characterisationEconomics.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    with open(dictArgs["calibration_json"]) as oFile:
        dictParams["dictMission"]["fThroughputCalibration"] = \
            json.load(oFile)["fCalibratedThroughputFactor"]
    listRows = [fdictAtDistance(dictParams, float(s), dictArgs["num_planets"], dictArgs["seed"])
                for s in dictArgs["distances"].split(",")]
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"listRows": listRows}, oFile, indent=2)
    print(f"{'d(pc)':>6}{'tau_det d':>11}{'tau_char d':>12}{'ratio':>8}"
          f"{'detectable':>12}{'+charz':>9}{'exozodi%':>10}{'detector%':>11}")
    for d in listRows:
        print(f"{d['fDistancePc']:>6.0f}{d['fMedianTauDetDays']:>11.3f}"
              f"{d['fMedianTauCharDays']:>12.1f}{d['fCharOverDet']:>8.0f}"
              f"{100*d['fDetectableFraction']:>11.1f}%{100*d['fAlsoCharacterizableFraction']:>8.1f}%"
              f"{100*d['fExozodiShareOfCharBackground']:>9.0f}%"
              f"{100*d['fDetectorShareOfCharBackground']:>10.0f}%")


if __name__ == "__main__":
    main()
