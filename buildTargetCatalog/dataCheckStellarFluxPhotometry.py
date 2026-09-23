#!/usr/bin/env python3
"""Check the blackbody stellar photon rate against HPIC's measured V photometry, star by star.

The stellar photon rate is the most load-bearing number in the whole pipeline: the planet count
rate, the leaked-starlight count rate and hence every exposure time are all proportional to it.
It is computed from a Planck function at the catalog's effective temperature, scaled by the solid
angle the catalog's stellar radius subtends at the catalog's distance -- three quantities that a
catalog reports independently and which therefore need not agree with each other, nor with the
star's measured brightness.

This script converts the modelled photon flux at 550 nm into a V magnitude and differences it
against HPIC's sy_vmag. Two failure modes are separated:

  * a BIAS in the median residual, which would scale every count rate and be partly absorbed by
    the throughput calibration factor;
  * SCATTER, which cannot be calibrated away and instead misranks targets -- a star whose flux is
    overstated looks cheap to observe and displaces a better one from the survey.

It also reports the catalog's own internal consistency, L against 4*pi*R^2*sigma*Teff^4, because
the pipeline takes the EEID from the luminosity and the flux from the radius and temperature. Any
disagreement there puts a planet at one separation while lighting it with a star of a different
brightness.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import physics as ph  # noqa: E402

F_SIGMA_SB = 5.670374419e-8
F_LSUN_W = 3.828e26
F_RSUN_M = 6.957e8


def faModelledVmag(faTeffK, faRadiusRsun, faDistancePc):
    """V magnitude implied by a blackbody of the given temperature, radius and distance.

    The zero point is the same one the count-rate code uses, so this tests the pipeline's own
    normalisation rather than an independently chosen one.
    """
    faFlux = ph.faStellarPhotonFlux(ph.F_VBAND_LAMBDA_M, faTeffK, faRadiusRsun, faDistancePc)
    fZeroPoint = ph.F_VBAND_ZERO_PHOTONS * 1e6
    return -2.5 * np.log10(faFlux / fZeroPoint)


def faLuminosityFromRadiusTeff(faRadiusRsun, faTeffK):
    """Bolometric luminosity in solar units from the Stefan-Boltzmann law."""
    faAreaM2 = 4.0 * np.pi * (np.asarray(faRadiusRsun) * F_RSUN_M) ** 2
    return faAreaM2 * F_SIGMA_SB * np.asarray(faTeffK) ** 4 / F_LSUN_W


def fdictResidualStats(faResidual, sLabel):
    """Median, spread and tails of a residual array."""
    return {
        "sLabel": sLabel,
        "iStars": int(faResidual.size),
        "fMedian": float(np.median(faResidual)),
        "fMean": float(np.mean(faResidual)),
        "fStdDev": float(np.std(faResidual)),
        "fP16": float(np.percentile(faResidual, 16)),
        "fP84": float(np.percentile(faResidual, 84)),
        "fMaxAbs": float(np.max(np.abs(faResidual))),
    }


def fdictByTeffBin(dfWork, sColumn, faEdges):
    """Residual statistics split by effective temperature, to expose a colour-dependent trend."""
    dictOut = {}
    for fLo, fHi in zip(faEdges[:-1], faEdges[1:]):
        dfBin = dfWork[(dfWork["fTeffK"] >= fLo) & (dfWork["fTeffK"] < fHi)]
        if len(dfBin) < 5:
            continue
        dictOut[f"{int(fLo)}-{int(fHi)}"] = {
            "iStars": int(len(dfBin)),
            "fMedian": float(np.median(dfBin[sColumn])),
            "fStdDev": float(np.std(dfBin[sColumn])),
        }
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--max-distance-pc", type=float, default=30.0)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    dfCat = pd.read_csv(dictArgs["target_catalog"])
    dfWork = dfCat[np.isfinite(dfCat["fVmag"]) & (dfCat["fDistancePc"] <=
                                                  dictArgs["max_distance_pc"])].copy()
    dfWork["fModelVmag"] = faModelledVmag(dfWork["fTeffK"].to_numpy(),
                                          dfWork["fRadiusRsun"].to_numpy(),
                                          dfWork["fDistancePc"].to_numpy())
    dfWork["fVmagResidual"] = dfWork["fModelVmag"] - dfWork["fVmag"]
    dfWork["fLumFromRadiusTeff"] = faLuminosityFromRadiusTeff(dfWork["fRadiusRsun"].to_numpy(),
                                                              dfWork["fTeffK"].to_numpy())
    dfWork["fLumRatioDex"] = np.log10(dfWork["fLumFromRadiusTeff"] / dfWork["fLuminosityLsun"])

    faEdges = np.array([3000, 3900, 4500, 5300, 6000, 7300, 10000])
    dfSunLike = dfWork[(dfWork["fTeffK"] >= 5300) & (dfWork["fTeffK"] <= 7300)]
    dictOut = {
        "iStarsChecked": int(len(dfWork)),
        "dictSunLikeResidual": fdictResidualStats(
            dfSunLike["fVmagResidual"].to_numpy(),
            "model V minus catalog V over 5300-7300 K, the population carrying the yield"),
        "fMaxDistancePc": dictArgs["max_distance_pc"],
        "dictVmagResidual": fdictResidualStats(dfWork["fVmagResidual"].to_numpy(),
                                               "model V minus catalog V, magnitudes"),
        "dictVmagResidualByTeff": fdictByTeffBin(dfWork, "fVmagResidual", faEdges),
        "dictLuminosityConsistencyDex": fdictResidualStats(
            dfWork["fLumRatioDex"].to_numpy(),
            "log10 of (L from 4 pi R^2 sigma Teff^4) over (catalog L)"),
        "dictLuminosityConsistencyByTeff": fdictByTeffBin(dfWork, "fLumRatioDex", faEdges),
        "fFluxBiasFactor": float(10.0 ** (-0.4 * np.median(dfWork["fVmagResidual"]))),
        "listWorstOffenders": [
            {"sName": str(r["sSourceId"]), "fTeffK": round(float(r["fTeffK"]), 0),
             "fDistancePc": round(float(r["fDistancePc"]), 2),
             "fCatalogVmag": round(float(r["fVmag"]), 2),
             "fModelVmag": round(float(r["fModelVmag"]), 2),
             "fResidual": round(float(r["fVmagResidual"]), 2)}
            for _, r in dfWork.reindex(
                dfWork["fVmagResidual"].abs().sort_values(ascending=False).index).head(10)
            .iterrows()],
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
