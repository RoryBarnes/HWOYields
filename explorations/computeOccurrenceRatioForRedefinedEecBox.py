#!/usr/bin/env python3
"""Compare SAG13 occurrence in the canonical HabEx/LUVOIR exoEarth-candidate box against a
redefined box (0.5-2 Mearth with R = M^0.27, habitable zone 0.96-1.2 AU scaled by sqrt(L)),
and propagate the resulting occurrence ratio onto the Stark et al. (2024) HWO baseline yields."""

import argparse
import json

import numpy as np
from scipy.integrate import quad


def fnRadiusIntegral(fRLo, fRHi, fAlpha):
    """Integral of R**alpha dlnR between two radii in Earth radii."""
    if fRLo >= fRHi:
        return 0.0
    if abs(fAlpha) < 1e-12:
        return float(np.log(fRHi / fRLo))
    return float((fRHi ** fAlpha - fRLo ** fAlpha) / fAlpha)


def fnOccurrence(fALo, fAHi, fnRadiusFloor, fRHi, fGamma, fAlpha, fBeta):
    """SAG13 occurrence integrated over a scaled semi-major-axis box, solar-twin reference.

    Uses d2N/(dlnR dlnP) = fGamma * R**fAlpha * P**fBeta with P in years and R in Earth radii.
    At the solar-twin reference (L=1, M=1) the scaled axis equals AU and P = a**1.5, so
    dlnP = 1.5 dlna. ExoVista's sqrt(L)-scaled coordinate is what makes this one reference
    integral stand in for every FGK spectral type.
    """

    def fnIntegrand(fLnA):
        fA = float(np.exp(fLnA))
        return (fA ** 1.5) ** fBeta * fnRadiusIntegral(fnRadiusFloor(fA), fRHi, fAlpha)

    fIntegral, _ = quad(fnIntegrand, float(np.log(fALo)), float(np.log(fAHi)), limit=200)
    return 1.5 * fGamma * fIntegral


def fdictBoxes(dictCfg):
    """Integration bounds for the canonical box, an HZ-only variant, and the redefined box."""
    fRLoMass = dictCfg["fMassLo"] ** dictCfg["fMassRadiusExponent"]
    fRHiMass = dictCfg["fMassHi"] ** dictCfg["fMassRadiusExponent"]
    fnCanonicalFloor = lambda fA: 0.8 * fA ** -0.5
    return {
        "canonical": dict(fALo=dictCfg["fHzLoCanon"], fAHi=dictCfg["fHzHiCanon"],
                          fnRadiusFloor=fnCanonicalFloor, fRHi=1.4),
        "hzOnly": dict(fALo=dictCfg["fHzLo"], fAHi=dictCfg["fHzHi"],
                       fnRadiusFloor=fnCanonicalFloor, fRHi=1.4),
        "redefined": dict(fALo=dictCfg["fHzLo"], fAHi=dictCfg["fHzHi"],
                          fnRadiusFloor=lambda fA: fRLoMass, fRHi=fRHiMass),
    }


def faRatioSamples(dictBoxes, dictCfg, iSamples, iSeed):
    """Monte Carlo samples of the redefined/canonical ratio over the SAG13 shape errors.

    The normalisation fGamma cancels in the ratio, so only the shape parameters matter and
    the result is independent of which eta_Earth anchor value is adopted.
    """
    rng = np.random.default_rng(iSeed)
    faAlpha = rng.normal(dictCfg["fAlpha"], dictCfg["fAlphaErr"], iSamples)
    faBeta = rng.normal(dictCfg["fBeta"], dictCfg["fBetaErr"], iSamples)
    faOut = np.empty(iSamples)
    for i in range(iSamples):
        fNum = fnOccurrence(**dictBoxes["redefined"], fGamma=1.0, fAlpha=faAlpha[i], fBeta=faBeta[i])
        fDen = fnOccurrence(**dictBoxes["canonical"], fGamma=1.0, fAlpha=faAlpha[i], fBeta=faBeta[i])
        faOut[i] = fNum / fDen
    return faOut


def fdictParseArgs():
    """Command-line configuration for the occurrence-ratio comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fGamma", type=float, default=0.38, help="SAG13 small-planet normalisation")
    p.add_argument("--fAlpha", type=float, default=-0.19, help="SAG13 radius power-law index")
    p.add_argument("--fBeta", type=float, default=0.26, help="SAG13 period power-law index")
    p.add_argument("--fAlphaErr", type=float, default=0.27, help="approx 1-sigma on fAlpha")
    p.add_argument("--fBetaErr", type=float, default=0.29, help="approx 1-sigma on fBeta")
    p.add_argument("--fHzLoCanon", type=float, default=0.95, help="canonical inner HZ, scaled AU")
    p.add_argument("--fHzHiCanon", type=float, default=1.67, help="canonical outer HZ, scaled AU")
    p.add_argument("--fHzLo", type=float, default=0.96, help="redefined inner HZ, scaled AU")
    p.add_argument("--fHzHi", type=float, default=1.20, help="redefined outer HZ, scaled AU")
    p.add_argument("--fMassLo", type=float, default=0.5, help="lower planet mass, Earth masses")
    p.add_argument("--fMassHi", type=float, default=2.0, help="upper planet mass, Earth masses")
    p.add_argument("--fMassRadiusExponent", type=float, default=0.27, help="R = M**exponent")
    p.add_argument("--fStarkYieldNoBias", type=float, default=22.5, help="Stark+24 6m EEC yield")
    p.add_argument("--fStarkYieldWithBias", type=float, default=17.3, help="Stark+24 bias-included")
    p.add_argument("--iSamples", type=int, default=8000, help="Monte Carlo samples")
    p.add_argument("--iSeed", type=int, default=20260921, help="random seed")
    p.add_argument("--sOutPath", type=str,
                   default="explorations/occurrenceRatioRedefinedEecBox.json")
    return vars(p.parse_args())


def main():
    dictCfg = fdictParseArgs()
    dictBoxes = fdictBoxes(dictCfg)
    dictEta = {
        sName: fnOccurrence(**dictBox, fGamma=dictCfg["fGamma"],
                            fAlpha=dictCfg["fAlpha"], fBeta=dictCfg["fBeta"])
        for sName, dictBox in dictBoxes.items()
    }
    faRatio = faRatioSamples(dictBoxes, dictCfg, dictCfg["iSamples"], dictCfg["iSeed"])
    fRatio = dictEta["redefined"] / dictEta["canonical"]
    dictOut = {
        "dictConfig": dictCfg,
        "dictRadiusBoxRedefined": {
            "fRLo": dictCfg["fMassLo"] ** dictCfg["fMassRadiusExponent"],
            "fRHi": dictCfg["fMassHi"] ** dictCfg["fMassRadiusExponent"],
        },
        "dictEtaByBox": dictEta,
        "fEtaCanonicalSag13Check": dictEta["canonical"],
        "dictRatioDecomposition": {
            "fHzNarrowingOnly": dictEta["hzOnly"] / dictEta["canonical"],
            "fMassBoxGivenHz": dictEta["redefined"] / dictEta["hzOnly"],
            "fCombined": fRatio,
        },
        "dictRatioUncertainty": {
            "fMedian": float(np.median(faRatio)),
            "fP16": float(np.percentile(faRatio, 16)),
            "fP84": float(np.percentile(faRatio, 84)),
        },
        "dictYieldUpperBound": {
            "sCaveat": "Occurrence-only rescaling. Completeness also falls under both changes, "
                       "so these are UPPER BOUNDS on the redefined yield, not predictions.",
            "fFromNoBias": dictCfg["fStarkYieldNoBias"] * fRatio,
            "fFromWithBias": dictCfg["fStarkYieldWithBias"] * fRatio,
        },
    }
    with open(dictCfg["sOutPath"], "w") as f:
        json.dump(dictOut, f, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
