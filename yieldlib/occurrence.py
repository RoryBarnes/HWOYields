"""SAG13-style occurrence-rate integrals over exoEarth-candidate selection boxes."""

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
    """SAG13 occurrence integrated over a scaled semi-major-axis box at the solar-twin reference.

    d2N/(dlnR dlnP) = fGamma R**fAlpha P**fBeta with P in years and R in Earth radii. At the
    reference (L=1, M=1) the sqrt(L)-scaled axis equals AU and P = a**1.5, so dlnP = 1.5 dlna.
    """

    def fnIntegrand(fLnA):
        fA = float(np.exp(fLnA))
        return (fA ** 1.5) ** fBeta * fnRadiusIntegral(fnRadiusFloor(fA), fRHi, fAlpha)

    fIntegral, _ = quad(fnIntegrand, float(np.log(fALo)), float(np.log(fAHi)), limit=200)
    return 1.5 * fGamma * fIntegral


def fnCanonicalRadiusFloor(fA):
    """HabEx/LUVOIR exoEarth-candidate minimum radius, 0.8 (a/EEID)^-0.5 Earth radii."""
    return 0.8 * fA ** -0.5


def fdictBoxFromConfig(dictBox):
    """Translate a box config dict into the bounds and radius-floor callable fnOccurrence wants."""
    if dictBox["sRadiusMode"] == "canonical":
        fnFloor, fRHi = fnCanonicalRadiusFloor, dictBox["fRadiusMaxEarth"]
    else:
        fRLo = dictBox["fMassLoEarth"] ** dictBox["fMassRadiusExponent"]
        fRHi = dictBox["fMassHiEarth"] ** dictBox["fMassRadiusExponent"]
        fnFloor = lambda fA: fRLo
    return dict(fALo=dictBox["fHzInnerAu"], fAHi=dictBox["fHzOuterAu"],
                fnRadiusFloor=fnFloor, fRHi=fRHi)


def fnPowerIntegral(fLo, fHi, fIndex):
    """Integral of x**fIndex dlnx between two bounds."""
    if abs(fIndex) < 1e-12:
        return float(np.log(fHi / fLo))
    return float((fHi ** fIndex - fLo ** fIndex) / fIndex)


def fnOccurrenceAnalytic(dictBox, fGamma, fAlpha, fBeta):
    """Closed-form SAG13 occurrence over a selection box, equivalent to fnOccurrence.

    For the canonical box the a-dependent radius floor 0.8 a^-0.5 keeps the double integral
    separable because (0.8 a^-0.5)**alpha = 0.8**alpha a**(-0.5 alpha), which simply shifts the
    semi-major-axis power. Closed form matters because the MCMC evaluates this ~10^5 times.
    """
    fALo, fAHi = dictBox["fHzInnerAu"], dictBox["fHzOuterAu"]
    if dictBox["sRadiusMode"] == "canonical":
        if abs(fAlpha) < 1e-12:
            return fnOccurrence(**fdictBoxFromConfig(dictBox), fGamma=fGamma,
                                fAlpha=fAlpha, fBeta=fBeta)
        fTermHi = dictBox["fRadiusMaxEarth"] ** fAlpha * fnPowerIntegral(fALo, fAHi, 1.5 * fBeta)
        fTermLo = 0.8 ** fAlpha * fnPowerIntegral(fALo, fAHi, 1.5 * fBeta - 0.5 * fAlpha)
        return 1.5 * fGamma / fAlpha * (fTermHi - fTermLo)
    fRLo = dictBox["fMassLoEarth"] ** dictBox["fMassRadiusExponent"]
    fRHi = dictBox["fMassHiEarth"] ** dictBox["fMassRadiusExponent"]
    return 1.5 * fGamma * fnRadiusIntegral(fRLo, fRHi, fAlpha) * \
        fnPowerIntegral(fALo, fAHi, 1.5 * fBeta)
