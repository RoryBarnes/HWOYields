#!/usr/bin/env python3
"""Calibrate the coronagraph throughput factor so the canonical EEC box reproduces Stark's yield.

The one input the published papers do not distribute is the pair of simulated coronagraph maps
Upsilon_c(x, y) and I(x, y). Everything downstream of them is fixed by published equations and
tables, so a single scalar throughput factor absorbs the residual mismatch. It is fitted by
bisection against the 22.5 EEC expected yield that Stark et al. (2024) report for the baseline
6 m LUVOIR-B-like design, and the fitted value is itself the headline diagnostic: a factor near
unity means the parametrized coronagraph stands in for the simulated one, and a factor far from
unity means it does not and nothing downstream should be trusted.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import survey as sv  # noqa: E402


def fdictBisectCalibration(dfTargets, dictParams, dictBox, faTauGridS, dictArgs):
    """Bisect the throughput factor in log space until the modelled yield matches the target."""
    fLo, fHi = dictArgs["calibration_min"], dictArgs["calibration_max"]
    listTrace = []
    for _ in range(dictArgs["bisection_steps"]):
        fMid = float(np.sqrt(fLo * fHi))
        fYield = sv.fnYieldForCalibration(dfTargets, dictParams, dictBox, faTauGridS,
                                          dictArgs["num_planets"], dictArgs["seed"],
                                          dictArgs["eta_earth"], fMid)
        listTrace.append({"fCalibration": fMid, "fYield": fYield})
        if fYield < dictArgs["target_yield"]:
            fLo = fMid
        else:
            fHi = fMid
    return listTrace


def fdictParseArgs():
    """Command-line configuration for the calibration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--target-yield", type=float, default=22.5)
    p.add_argument("--eta-earth", type=float, default=0.24)
    p.add_argument("--box", default="canonical")
    p.add_argument("--num-planets", type=int, default=1200)
    p.add_argument("--num-tau-points", type=int, default=120)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--min-eeid-lamd", type=float, default=0.5)
    p.add_argument("--teff-min", type=float, default=2500.0)
    p.add_argument("--teff-max", type=float, default=7500.0)
    p.add_argument("--calibration-min", type=float, default=0.05)
    p.add_argument("--calibration-max", type=float, default=20.0)
    p.add_argument("--bisection-steps", type=int, default=14)
    p.add_argument("--tolerance-fraction", type=float, default=0.10)
    p.add_argument("--max-plausible-factor", type=float, default=2.0,
                   help="a fitted factor beyond this band absorbs physics, not an unknown")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-calibration", default="calibration.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictParams["fAlpha"], dictParams["fBeta"] = -0.19, 0.26
    dictBox = dictParams["dictBoxes"][dictArgs["box"]]
    dfTargets = sv.fdfScreenTargets(pd.read_csv(dictArgs["target_catalog"]),
                                    dictParams["dictMission"],
                                    dictParams["dictBands"]["listBandsDetection"][0],
                                    dictArgs["min_eeid_lamd"], dictArgs["max_stars"],
                                    dictArgs["teff_min"], dictArgs["teff_max"], dictBox)
    faTauGridS = np.logspace(1.0, np.log10(dictParams["dictMission"]["fExposureLimitS"]),
                             dictArgs["num_tau_points"])
    listTrace = fdictBisectCalibration(dfTargets, dictParams, dictBox, faTauGridS, dictArgs)
    dictBest = min(listTrace, key=lambda d: abs(d["fYield"] - dictArgs["target_yield"]))
    fRelativeError = abs(dictBest["fYield"] - dictArgs["target_yield"]) / dictArgs["target_yield"]
    dictOut = {
        "fTargetYield": dictArgs["target_yield"],
        "iStarsScreened": int(len(dfTargets)),
        "fCalibratedThroughputFactor": dictBest["fCalibration"],
        "fCalibratedYield": dictBest["fYield"],
        "fRelativeError": fRelativeError,
        "bTargetReached": bool(fRelativeError <= dictArgs["tolerance_fraction"]),
        "fMaxPlausibleFactor": dictArgs["max_plausible_factor"],
        "bFactorPlausible": bool(
            1.0 / dictArgs["max_plausible_factor"] <= dictBest["fCalibration"]
            <= dictArgs["max_plausible_factor"]),
        "bCalibrationGatePassed": bool(
            fRelativeError <= dictArgs["tolerance_fraction"]
            and 1.0 / dictArgs["max_plausible_factor"] <= dictBest["fCalibration"]
            <= dictArgs["max_plausible_factor"]),
        "sGateNote": "The gate requires BOTH that the target was reached AND that the fitted "
                     "factor stays within a factor of two of unity. Hitting the target with an "
                     "implausible factor means the scalar is absorbing missing physics rather "
                     "than an instrumental unknown, which is not a pass.",
        "fUncalibratedYield": sv.fnYieldForCalibration(
            dfTargets, dictParams, dictBox, faTauGridS, dictArgs["num_planets"],
            dictArgs["seed"], dictArgs["eta_earth"], 1.0),
        "listTrace": listTrace,
    }
    with open(dictArgs["out_calibration"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "listTrace"}, indent=2))


if __name__ == "__main__":
    main()
