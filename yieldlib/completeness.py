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


def fdictInjectPlanets(dictBox, fEeidAu, iNumPlanets, fAlpha, fBeta, rng, iVisits=1):
    """Draw EECs over the selection box, weighted by the SAG13 density, with random geometry.

    Each planet keeps ONE orbit and is sampled at iVisits independent epochs along it. A revisit
    therefore sees the same planet at a different phase and separation, which is the whole point
    of revisiting: a planet hidden inside the inner working angle or at crescent phase on one
    visit can be caught on another.

    The epochs are drawn at random rather than spaced evenly in mean anomaly. For a circular
    orbit the projected separation has period pi in true anomaly, so even spacing with an even
    visit count makes alternate visits duplicates of each other and silently removes the benefit
    of revisiting. Real visit timing is set by scheduling against orbital periods that differ
    star to star, so random phase is both safer and closer to the truth.
    """
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
    faTheta = 2.0 * np.pi * rng.random((iNumPlanets, iVisits))
    return dict(faAxisAu=faAxisScaled * fEeidAu, faAxisScaled=faAxisScaled,
                faRadiusEarth=faRadius, faCosInc=faCosInc, faTheta=faTheta,
                faAlbedo=rng.uniform(0.0, 1.0, iNumPlanets), iVisits=iVisits)


def fdictProjectOrbits(dictPlanets):
    """Projected separation in AU and Lambertian phase factor, per planet per visit epoch."""
    faCosInc = np.atleast_1d(dictPlanets["faCosInc"])[:, None]
    faSinInc = np.sqrt(np.maximum(0.0, 1.0 - faCosInc ** 2))
    faTheta = np.atleast_2d(dictPlanets["faTheta"])
    faAxis = np.atleast_1d(dictPlanets["faAxisAu"])[:, None]
    faSepAu = faAxis * np.sqrt(np.cos(faTheta) ** 2 + (np.sin(faTheta) * faCosInc) ** 2)
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
    faAlbedo = dictPlanets.get("faAlbedoDrawn")
    faAlbedo = dictMission["fGeometricAlbedo"] if faAlbedo is None else faAlbedo
    faAlbedo = np.atleast_1d(faAlbedo)[:, None] if np.ndim(faAlbedo) == 1 else faAlbedo
    faFluxRatio = faAlbedo * dictGeom["faPhase"] * \
        (np.atleast_1d(dictPlanets["faRadiusEarth"])[:, None] * F_REARTH_AU /
         np.atleast_1d(dictPlanets["faAxisAu"])[:, None]) ** 2
    fOmega = ph.fnPhotometricApertureSolidAngle(dictBand["fLambdaM"], dictMission["fDiameterM"],
                                                dictMission["fApertureRadiusLamD"])
    fZeroMag = ph.fnZeroMagPhotonFlux(dictBand["fLambdaM"]) * (fBandwidthM * 1e6)
    fZodi = fZeroMag * 10 ** (-0.4 * dictMission["fZodiMagArcsec2"]) * fOmega * fArea * fThroughput
    faExozodiMag = dictMission["fExozodiMagArcsec2"] + \
        5.0 * np.log10(np.atleast_1d(dictPlanets["faAxisScaled"])[:, None])
    fExozodiLevel = dictStar.get("fExozodiLevel", dictMission["fExozodiLevel"])
    faExozodi = fExozodiLevel * fZeroMag * 10 ** (-0.4 * faExozodiMag) * \
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


def faCountedTimes(faTauDet, faTauChar, fCap):
    """Per-planet detection time using the first k visits, for every k, infinite where it fails.

    Cumulative minimum along the visit axis: with k visits in hand a planet is caught at
    whichever of those epochs was cheapest. A planet counts only if it can also be characterized
    within the cap at that same standard (Stark et al. 2019).
    """
    faBestDet = np.minimum.accumulate(faTauDet, axis=1)
    faBestChar = np.minimum.accumulate(faTauChar, axis=1)
    bCounts = (np.isfinite(faBestDet) & (faBestDet <= fCap) &
               np.isfinite(faBestChar) & (faBestChar <= fCap))
    return np.where(bCounts, faBestDet, np.inf), faBestChar, bCounts


def faCompletenessPerVisitCount(faBestDet, faTauGridS, iNumPlanets):
    """Completeness curve for each visit count, shape (visits, grid)."""
    iVisits = faBestDet.shape[1]
    faOut = np.zeros((iVisits, faTauGridS.size))
    for k in range(iVisits):
        faSorted = np.sort(faBestDet[:, k][np.isfinite(faBestDet[:, k])])
        faOut[k] = np.searchsorted(faSorted, faTauGridS, side="right") / float(iNumPlanets)
    return faOut


def faCharMeanPerVisitCount(faBestDet, faBestChar, faTauGridS):
    """Mean characterization time over the counted planets, for each visit count."""
    iVisits = faBestDet.shape[1]
    faOut = np.zeros((iVisits, faTauGridS.size))
    for k in range(iVisits):
        faOut[k] = faCumulativeMeanCharTime(faBestDet[:, k], faBestChar[:, k], faTauGridS)
    return faOut


def faDetectionTimes(dictStar, dictPlanets, dictGeom, listBandsDet, listCharOptions,
                     dictMission, iNumPlanets):
    """Detection and characterization exposure times per planet per visit epoch."""
    listDetRates = [fdictCountRates(dictStar, dictPlanets, dictGeom, b, dictMission)
                    for b in listBandsDet]
    faTauDet = faRequiredExposureTime(listDetRates, listBandsDet,
                                      listBandsDet[0]["fSignalToNoise"], dictMission)
    faTauChar = np.full(faTauDet.shape, np.inf)
    for dictOption in listCharOptions:
        faTauChar = np.minimum(faTauChar, faRequiredExposureTime(
            [fdictCountRates(dictStar, dictPlanets, dictGeom, dictOption, dictMission)],
            [dictOption], dictOption["fSignalToNoise"], dictMission))
    return faTauDet, faTauChar


def faStarkAlbedoTimes(faTauDet, faFlux, faSepLamD, faFluxDrawn):
    """Exposure time at which a drawn-albedo planet passes Stark's per-visit detectability test.

    Stark et al. (2024) Sec. 3.2 do not re-derive the exposure time for a planet whose albedo
    differs from the planning value. They ask whether the fixed observation plan would have
    caught it, by requiring its albedo-adjusted flux to exceed the FAINTEST flux actually
    detected during that visit, and its separation to exceed the SMALLEST separation detected.
    Both thresholds fall monotonically as the exposure lengthens, so each planet passes for every
    exposure beyond some threshold, which is what this returns.

    This is a blunter test than re-deriving the exposure time, because it cannot reward a dark
    planet that happens to sit where the background is low. It is implemented alongside the
    re-derivation so the two can be compared rather than assumed equivalent.
    """
    faOut = np.full(faTauDet.shape, np.inf)
    for k in range(faTauDet.shape[1]):
        faOrder = np.argsort(faTauDet[:, k])
        faTauSorted = faTauDet[faOrder, k]
        bFinite = np.isfinite(faTauSorted)
        if not bFinite.any():
            continue
        faTauSorted = faTauSorted[bFinite]
        faFloorFlux = np.minimum.accumulate(faFlux[faOrder, k][bFinite])
        faFloorSep = np.minimum.accumulate(faSepLamD[faOrder, k][bFinite])
        iFlux = np.searchsorted(-faFloorFlux, -faFluxDrawn[:, k], side="left")
        iSep = np.searchsorted(-faFloorSep, -faSepLamD[:, k], side="left")
        iNeed = np.maximum(iFlux, iSep)
        bReach = iNeed < faTauSorted.size
        faOut[bReach, k] = faTauSorted[iNeed[bReach]]
    return faOut


def fdictStarCompleteness(dictStar, dictBox, listBandsDet, dictBandChar, dictMission,
                          faTauGridS, iNumPlanets, fAlpha, fBeta, iSeed):
    """Completeness for one star against exposure time AND visit count.

    Returns curves of shape (visits, grid). The optimizer picks both the exposure per visit and
    the number of visits, as AYO does: Stark et al. (2024) describe it as "simultaneously
    optimizing the selected targets, the number of visits to each star, and the exposure and
    delay times for each visit". Modelling a single visit, as an earlier version did, understates
    completeness per star because a planet inside the inner working angle or at crescent phase at
    one epoch is simply lost rather than caught on a later visit.

    Two curves per visit count. faComp assumes the planning albedo A_G = 0.2 and is what the
    optimizer allocates against; faCompAlbedo re-evaluates the SAME planets and orbits with
    albedos drawn from the adopted distribution, and is what the yield is read from.
    """
    rng = np.random.default_rng(iSeed)
    iVisits = int(dictMission.get("iMaxVisits", 1))
    dictPlanets = fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                     iNumPlanets, fAlpha, fBeta, rng, iVisits=iVisits)
    dictGeom = fdictProjectOrbits(dictPlanets)
    listCharOptions = dictMission.get("listBandsCharacterization") or [dictBandChar]
    fCap = dictMission["fExposureLimitS"]
    faTauDet, faTauChar = faDetectionTimes(dictStar, dictPlanets, dictGeom, listBandsDet,
                                           listCharOptions, dictMission, iNumPlanets)
    faBestDet, faBestChar, bCounts = faCountedTimes(faTauDet, faTauChar, fCap)
    faComp = faCompletenessPerVisitCount(faBestDet, faTauGridS, iNumPlanets)
    faCompAlbedo = faComp
    dictAlbedoRange = dictMission.get("dictAlbedoDistribution")
    if dictAlbedoRange:
        dictPlanets["faAlbedoDrawn"] = (
            dictAlbedoRange["fMin"] + dictPlanets["faAlbedo"] *
            (dictAlbedoRange["fMax"] - dictAlbedoRange["fMin"]))
        faTauDetA, faTauCharA = faDetectionTimes(dictStar, dictPlanets, dictGeom, listBandsDet,
                                                 listCharOptions, dictMission, iNumPlanets)
        if dictMission.get("sAlbedoMethod", "recompute") == "perVisitThreshold":
            dictRatesPlan = fdictCountRates(dictStar, dictPlanets, dictGeom, listBandsDet[0],
                                            dict(dictMission, listBandsCharacterization=None))
            faFluxPlan = dictRatesPlan["faFluxRatio"] * (
                dictMission["fGeometricAlbedo"] /
                np.atleast_1d(dictPlanets["faAlbedoDrawn"])[:, None])
            faTauDetA = faStarkAlbedoTimes(faTauDet, faFluxPlan, dictRatesPlan["faSepLamD"],
                                           dictRatesPlan["faFluxRatio"])
        faBestDetA, _, _ = faCountedTimes(faTauDetA, faTauCharA, fCap)
        faCompAlbedo = faCompletenessPerVisitCount(faBestDetA, faTauGridS, iNumPlanets)
    return dict(faComp=faComp, faCompAlbedo=faCompAlbedo,
                faTauCharMeanS=faCharMeanPerVisitCount(faBestDet, faBestChar, faTauGridS),
                fTauCharS=float(np.median(faBestChar[:, -1][bCounts[:, -1]]))
                if bCounts[:, -1].any() else np.inf,
                fMaxCompleteness=float(faComp[-1, -1]))
