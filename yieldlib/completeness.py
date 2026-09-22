"""ExoEarth-candidate injection and single-visit completeness curves C(tau) per target star."""

import numpy as np

from . import coronagraph as cg
from . import physics as ph

F_REARTH_AU = 4.25875e-5


def faSamplePowerLawInLog(faLo, faHi, fIndex, faUniform):
    """Inverse-CDF draw from a density proportional to x**fIndex dlnx between two bounds."""
    faLo = np.asarray(faLo, dtype=float)
    faHi = np.asarray(faHi, dtype=float)
    if abs(fIndex) < 1e-12:
        return faLo * (faHi / faLo) ** faUniform
    faLoP, faHiP = faLo ** fIndex, faHi ** fIndex
    return (faLoP + faUniform * (faHiP - faLoP)) ** (1.0 / fIndex)


def fdictInjectPlanets(dictBox, fEeidAu, iNumPlanets, fAlpha, fBeta, rng):
    """Draw EECs over the selection box, weighted by the SAG13 density, with random geometry."""
    faU1, faU2, faU3, faU4 = (rng.random(iNumPlanets) for _ in range(4))
    faAxisScaled = faSamplePowerLawInLog(dictBox["fHzInnerAu"], dictBox["fHzOuterAu"],
                                         1.5 * fBeta, faU1)
    if dictBox["sRadiusMode"] == "canonical":
        faRadiusLo = 0.8 * faAxisScaled ** -0.5
        faRadiusHi = np.full(iNumPlanets, dictBox["fRadiusMaxEarth"])
    else:
        faRadiusLo = np.full(iNumPlanets, dictBox["fMassLoEarth"] ** dictBox["fMassRadiusExponent"])
        faRadiusHi = np.full(iNumPlanets, dictBox["fMassHiEarth"] ** dictBox["fMassRadiusExponent"])
    faRadius = faSamplePowerLawInLog(faRadiusLo, faRadiusHi, fAlpha, faU2)
    faCosInc = 2.0 * faU3 - 1.0
    faTheta = 2.0 * np.pi * faU4
    return dict(faAxisAu=faAxisScaled * fEeidAu, faAxisScaled=faAxisScaled,
                faRadiusEarth=faRadius, faCosInc=faCosInc, faTheta=faTheta)


def fdictProjectOrbits(dictPlanets):
    """Projected separation in AU and Lambertian phase factor for circular orbits."""
    faCosInc = dictPlanets["faCosInc"]
    faSinInc = np.sqrt(np.maximum(0.0, 1.0 - faCosInc ** 2))
    faTheta = dictPlanets["faTheta"]
    faSepAu = dictPlanets["faAxisAu"] * np.sqrt(np.cos(faTheta) ** 2 +
                                                (np.sin(faTheta) * faCosInc) ** 2)
    faCosPhase = np.clip(np.sin(faTheta) * faSinInc, -1.0, 1.0)
    return dict(faSepAu=faSepAu, faPhase=ph.faLambertianPhase(np.arccos(faCosPhase)))


def fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission):
    """All count rates (planet, leaked starlight, zodi, exozodi, detector) in counts s^-1."""
    fLamD = cg.fnLambdaOverDArcsec(dictBand["fLambdaM"], dictMission["fDiameterM"])
    faSepArcsec = dictGeom["faSepAu"] / dictStar["fDistancePc"]
    faSepLamD = faSepArcsec / fLamD
    faUpsilon = cg.faCoreThroughput(faSepLamD, fIwaLamD=dictMission["fIwaLamD"],
                                    fOwaLamD=dictMission["fOwaLamD"],
                                    fThroughputMax=dictMission["fCoreThroughputMax"])
    faZeta = cg.faRawContrast(faSepLamD, fContrastFloor=dictMission["fContrastFloor"],
                              fOwaLamD=dictMission["fOwaLamD"])
    fArea = np.pi * (dictMission["fDiameterM"] / 2.0) ** 2
    fThroughput = dictBand["fOpticalThroughput"] * dictMission["fContaminationThroughput"] * \
        dictMission["fDetectiveQuantumEfficiency"] * dictMission["fQuantumEfficiency"] * \
        dictMission.get("fThroughputCalibration", 1.0)
    fBandwidthM = dictBand["fLambdaM"] * dictBand["fBandwidthFraction"]
    fStarFlux = float(ph.faStellarPhotonFlux(dictBand["fLambdaM"], dictStar["fTeffK"],
                                             dictStar["fRadiusRsun"], dictStar["fDistancePc"]))
    fStarRate = fStarFlux * fBandwidthM * fArea * fThroughput
    faFluxRatio = dictMission["fGeometricAlbedo"] * dictGeom["faPhase"] * \
        (dictPlanets["faRadiusEarth"] * F_REARTH_AU / dictPlanets["faAxisAu"]) ** 2
    fOmega = ph.fnPhotometricApertureSolidAngle(dictBand["fLambdaM"], dictMission["fDiameterM"],
                                                dictMission["fApertureRadiusLamD"])
    fZeroMag = ph.fnZeroMagPhotonFlux(dictBand["fLambdaM"]) * (fBandwidthM * 1e6)
    fZodi = fZeroMag * 10 ** (-0.4 * dictMission["fZodiMagArcsec2"]) * fOmega * fArea * fThroughput
    faExozodiMag = dictMission["fExozodiMagArcsec2"] + 5.0 * np.log10(dictPlanets["faAxisScaled"])
    faExozodi = dictMission["fExozodiLevel"] * fZeroMag * 10 ** (-0.4 * faExozodiMag) * \
        fOmega * fArea * fThroughput
    return dict(faPlanet=fStarRate * faUpsilon * faFluxRatio,
                faLeak=fStarRate * faUpsilon * faZeta, fZodi=fZodi, faExozodi=faExozodi,
                faFluxRatio=faFluxRatio, faSepLamD=faSepLamD, faUpsilon=faUpsilon)


def faRequiredExposureTime(listBandRates, listBands, fSignalToNoise, dictMission):
    """Combined exposure time over parallel coronagraph channels, infinite where undetectable.

    Stark et al. (2024) Table 2 requires S/N = 7 "summed over both coronagraphs", so the
    channels add in quadrature: 1/tau is the sum of each channel's CR_p^2 / (CR_p + 2 CR_b).
    """
    faInverseTau = np.zeros_like(listBandRates[0]["faPlanet"])
    for dictRates, dictBand in zip(listBandRates, listBands):
        faAstro = dictRates["faLeak"] + dictRates["fZodi"] + dictRates["faExozodi"]
        faBrightest = (faAstro + dictRates["faPlanet"]) / dictBand["iNumPixels"]
        faDetector = ph.faDetectorCountRate(faBrightest, dictBand["iNumPixels"],
                                            dictMission["fDarkCurrent"],
                                            dictMission["fReadNoise"], None,
                                            dictMission["fClockInducedCharge"])
        faTotalBackground = faAstro + faDetector
        with np.errstate(divide="ignore", invalid="ignore"):
            faTerm = dictRates["faPlanet"] ** 2 / (dictRates["faPlanet"] +
                                                   2.0 * faTotalBackground)
        faInverseTau += np.where(dictRates["faPlanet"] > 0.0, faTerm, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        faTau = fSignalToNoise ** 2 / faInverseTau
    faPrimary = listBandRates[0]
    faDeltaMag = -2.5 * np.log10(np.maximum(faPrimary["faFluxRatio"], 1e-30))
    faBlocked = ((faDeltaMag > dictMission["fNoiseFloorDeltaMag"]) |
                 (faPrimary["faUpsilon"] <= 0.0) | (faInverseTau <= 0.0))
    return np.where(faBlocked, np.inf, faTau)


def faCumulativeMeanCharTime(faTauDet, faTauChar, faTauGridS):
    """Mean characterization time over the planets actually counted at each exposure time.

    The budget charges the EXPECTED TOTAL characterization time, which is the number of
    detections times the mean cost per detection, so the mean is the correct statistic and it
    must be taken over the counted set rather than over every detectable planet. Detection is
    brightness-ordered: a short allocation finds only the brightest planets, which are also the
    cheapest to characterize, so a statistic over all detectable planets overcharges short
    allocations. faTauDet must already be infinite for planets that do not count.
    """
    faOrder = np.argsort(faTauDet)
    faDetSorted = faTauDet[faOrder]
    faCharSorted = np.where(np.isfinite(faTauChar[faOrder]), faTauChar[faOrder], 0.0)
    faCumulative = np.cumsum(faCharSorted)
    faCount = np.searchsorted(faDetSorted, faTauGridS, side="right")
    return np.where(faCount > 0, faCumulative[np.maximum(faCount - 1, 0)] /
                    np.maximum(faCount, 1), 0.0)


def fdictStarCompleteness(dictStar, dictBox, listBandsDet, dictBandChar, dictMission,
                          faTauGridS, iNumPlanets, fAlpha, fBeta, iSeed):
    """Completeness C(tau) on a shared time grid for one star, with its characterization cost.

    C(tau) is the fraction of ALL injected EECs that COUNT within tau. A planet counts only if
    it can be both detected and spectrally characterized inside the two-month cap: Stark et al.
    (2019) require both times to be under two months and state that "any planets that did not
    meet this criteria did not count toward the yield". Charging an uncharacterizable planet the
    cap instead of excluding it inflates the characterization budget badly, which is what an
    earlier version of this function did.
    """
    rng = np.random.default_rng(iSeed)
    dictPlanets = fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                     iNumPlanets, fAlpha, fBeta, rng)
    dictGeom = fdictProjectOrbits(dictPlanets)
    listDetRates = [fdictCountRates(dictStar, dictPlanets, dictGeom, b, dictMission)
                    for b in listBandsDet]
    faTauDet = faRequiredExposureTime(listDetRates, listBandsDet,
                                      listBandsDet[0]["fSignalToNoise"], dictMission)
    listCharOptions = dictMission.get("listBandsCharacterization") or [dictBandChar]
    faTauChar = np.full(iNumPlanets, np.inf)
    for dictOption in listCharOptions:
        faOption = faRequiredExposureTime(
            [fdictCountRates(dictStar, dictPlanets, dictGeom, dictOption, dictMission)],
            [dictOption], dictOption["fSignalToNoise"], dictMission)
        faTauChar = np.minimum(faTauChar, faOption)
    fCap = dictMission["fExposureLimitS"]
    bCounts = (np.isfinite(faTauDet) & (faTauDet <= fCap) &
               np.isfinite(faTauChar) & (faTauChar <= fCap))
    faTauDet = np.where(bCounts, faTauDet, np.inf)
    faDetSorted = np.sort(faTauDet[np.isfinite(faTauDet)])
    faComp = np.searchsorted(faDetSorted, faTauGridS, side="right") / float(iNumPlanets)
    return dict(faComp=faComp,
                faTauCharMeanS=faCumulativeMeanCharTime(faTauDet, faTauChar, faTauGridS),
                fTauCharS=float(np.median(faTauChar[bCounts])) if bCounts.any() else np.inf,
                fMaxCompleteness=float(faComp[-1]))
