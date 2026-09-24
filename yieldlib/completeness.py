"""ExoEarth-candidate injection and single-visit completeness curves C(tau) per target star."""

import numpy as np

from . import coronagraph as cg
from . import physics as ph

F_REARTH_AU = 4.25875e-5
I_DEFAULT_CHARACTERIZATION_PHASES = 40
""" Orbital phases sampled when scheduling characterization at its most favourable point.

Measured convergence against a 200-phase reference, for a solar twin at 10 pc
(explorations/profileCompletenessCost.py): 40 phases reproduces the counted planet set exactly
(1011 of 1011) with a median error of 0.18% and a worst case of 2.8%, while 100 phases costs
2.4 times as much for a median error of 0.00%. Dropping to 24 starts losing counted planets
(1006), which is the quantity the yield is built from, so 40 is the last value that is free.
"""


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


def fnSkyThroughputFor(dictMission):
    """T_sky for this mission: the coronagraph's throughput for an extended source.

    Stark et al. (2019) Eqs. 5 and 6 carry a factor T_sky(x,y) on the zodiacal and exozodiacal
    count rates that the point-source terms do not. An explicit mission value overrides; otherwise
    it is reconstructed from the core throughput and the photometric aperture. See
    yieldlib.coronagraph.fnSkyThroughput for the derivation.
    """
    fExplicit = dictMission.get("fSkyThroughput")
    if fExplicit is not None:
        return float(fExplicit)
    dictTable = dictMission.get("dictCoronagraphTable")
    fMax = (max(dictTable["faUpsilon"]) if dictTable
            else dictMission["fCoreThroughputMax"])
    return cg.fnSkyThroughput(fMax, dictMission["fApertureRadiusLamD"])


def faSkyThroughputAt(faUpsilon, dictMission):
    """T_sky at each planet's position: constant, or following the core-throughput curve.

    Stark et al. (2019) Eqs. 5-6 write the zodiacal and exozodiacal count rates with
    T_sky(x,y), "the instrument's throughput for extended sources", obtained "by first convolving
    the spatially-dependent PSF at all locations with a normalized uniform background". It is a
    map, not a number: background light near the inner working angle is attenuated by the same
    focal-plane mask that attenuates a planet there. This pipeline used the large-separation value
    (0.678) everywhere, which overcharges background for every planet observed near the IWA --
    the distant and luminous targets, and characterization at 1 um above all -- by a factor of up
    to Upsilon_c,max / Upsilon_c(r).

    With bSkyThroughputFollowsCore the map is reconstructed as T_sky(r) = T_sky,max *
    Upsilon_c(r) / Upsilon_c,max, i.e. the total off-axis transmission at r under the assumption
    that the fraction of an off-axis PSF inside the photometric core does not change with
    separation. The PSF convolution in Stark's definition smooths this over about one lambda/D,
    which the reconstruction omits; the published 2-D maps that would supply it are not public.
    """
    fMax = fnSkyThroughputFor(dictMission)
    if not dictMission.get("bSkyThroughputFollowsCore", False):
        return fMax
    dictTable = dictMission.get("dictCoronagraphTable")
    fUpsilonMax = (max(dictTable["faUpsilon"]) if dictTable
                   else dictMission["fCoreThroughputMax"])
    return fMax * np.asarray(faUpsilon) / fUpsilonMax


def faExozodiRadialFactor(faSepAu, dictStar, dictMission):
    """Exozodi surface brightness at the planet's projected separation relative to the EEID.

    Stark et al. (2019) Eq. 6 writes the exozodi term with z'(x,y), and its Table 1 notes the
    surface brightness "varies with spectral type and planet-star separation". Stark (2014)
    App. C had evaluated every planet at the EEID instead. With fExozodiRadialIndex = q the
    brightness scales as (s / EEID)^-q at projected separation s: q = 2.34 is 1/r^2 illumination
    times the zodiacal cloud's face-on optical depth, which falls as r^-0.34 for the Kelsall
    density profile (proportional to r^-1.34) that ZODIPIC uses. Inclination and scattering-phase
    effects are not modelled. q = 0 (the default) is the EEID treatment.
    """
    fIndex = float(dictMission.get("fExozodiRadialIndex", 0.0))
    if fIndex == 0.0:
        return 1.0
    faScaled = np.asarray(faSepAu) / np.sqrt(dictStar["fLuminosityLsun"])
    return np.maximum(faScaled, 1e-3) ** (-fIndex)


def fnCollectingAreaM2(dictMission):
    """Effective collecting area A in m^2 (Stark et al. 2019 Eq. 3).

    A is "the effective collecting area of the telescope aperture accounting for segment gaps and
    secondary mirror and strut obscurations" -- the FULL obscured primary, not the inscribed
    circle. That pairing is not optional: Upsilon_c is normalised to "the light entering the
    coronagraph... from the full obscured primary mirror, including the region exterior to the
    inscribed diameter" (Ref. stark2019 Sec. 6.2), and the light the Lyot stop then discards is
    exactly why Upsilon_c,max is 0.46 rather than 0.69. Taking A as the inscribed circle while
    using an Upsilon_c normalised to the full primary charges that discard twice.

    fApertureAreaM2 supplies the real figure when it is known; otherwise the area falls back to a
    circle of the quoted diameter, which is what this pipeline assumed until the normalisation
    was traced.
    """
    fArea = dictMission.get("fApertureAreaM2")
    if fArea is not None:
        return float(fArea)
    fCircumscribed = dictMission["fDiameterM"] * dictMission.get("fCircumscribedRatio", 1.0)
    return dictMission.get("fApertureFillFactor", 1.0) * np.pi * (fCircumscribed / 2.0) ** 2


def fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission):
    """All count rates (planet, leaked starlight, zodi, exozodi, detector) in counts s^-1.

    fCoronagraphScale converts a separation expressed in lambda/D, with D the diameter this
    pipeline quotes (the INSCRIBED diameter, which is how Stark et al. 2024 label their
    scenarios), into the units the published coronagraph curves are plotted in. Ref. stark2024
    Sec. 6.1 states that the x-axis of its coronagraph figure is the CIRCUMSCRIBED diameter, and
    that the DMVC6's apparent 3.5 lambda/D inner working angle is inflated relative to a circular
    aperture for that reason -- Ref. stark2019 Sec. 6.2 adds that normalised to the inscribed
    pupil the DMVC and the monolithic vortex "would look nearly identical", and the monolithic
    vortex's IWA is ~3 lambda/D. The ratio between those two readings is D_circ/D_inscribed.
    """
    fLamD = cg.fnLambdaOverDArcsec(dictBand["fLambdaM"], dictMission["fDiameterM"])
    faSepArcsec = dictGeom["faSepAu"] / dictStar["fDistancePc"]
    faSepLamD = faSepArcsec / fLamD
    faSepCurve = faSepLamD * dictMission.get("fCoronagraphScale",
                                             dictMission.get("fCircumscribedRatio", 1.0))
    dictTable = dictMission.get("dictCoronagraphTable")
    if dictTable:
        faUpsilon = cg.faCoreThroughputTable(faSepCurve, dictTable)
        faZeta = cg.faRawContrastTable(faSepCurve, dictTable, dictMission["fContrastFloor"])
    else:
        faUpsilon = cg.faCoreThroughput(faSepCurve, fIwaLamD=dictMission["fIwaLamD"],
                                        fOwaLamD=dictMission["fOwaLamD"],
                                        fThroughputMax=dictMission["fCoreThroughputMax"])
        faZeta = cg.faRawContrast(faSepCurve, fContrastFloor=dictMission["fContrastFloor"],
                                  fOwaLamD=dictMission["fOwaLamD"])
    fArea = fnCollectingAreaM2(dictMission)
    fCalibration = (dictMission.get("fThroughputCalibration", 1.0)
                    if dictBand.get("bApplyThroughputCalibration", True) else 1.0)
    fThroughput = dictBand["fOpticalThroughput"] * dictMission["fContaminationThroughput"] * \
        dictMission["fDetectiveQuantumEfficiency"] * dictMission["fQuantumEfficiency"] * \
        fCalibration
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
    fBackground = fOmega * fArea * fThroughput * faSkyThroughputAt(faUpsilon, dictMission)
    fZodi = fZeroMag * 10 ** (-0.4 * dictMission["fZodiMagArcsec2"]) * fBackground
    fExozodiScale = ph.fnExozodiSurfaceBrightnessScale(
        dictBand["fLambdaM"], dictStar["fTeffK"], dictStar["fRadiusRsun"],
        dictStar["fLuminosityLsun"])
    fExozodiLevel = dictStar.get("fExozodiLevel", dictMission["fExozodiLevel"])
    fExozodi = fExozodiLevel * fZeroMag * 10 ** (-0.4 * dictMission["fExozodiMagArcsec2"]) * \
        fExozodiScale * fBackground * faExozodiRadialFactor(dictGeom["faSepAu"], dictStar,
                                                            dictMission)
    return dict(faPlanet=fStarRate * faUpsilon * faFluxRatio,
                faLeak=fStarRate * faUpsilon * faZeta, fZodi=fZodi, fExozodi=fExozodi,
                faFluxRatio=faFluxRatio, faSepLamD=faSepLamD, faUpsilon=faUpsilon)


def faRequiredExposureTime(listBandRates, listBands, fSignalToNoise, dictMission,
                           fExozodiScale=1.0):
    """Combined exposure time over parallel coronagraph channels, infinite where undetectable.

    Stark et al. (2024) Table 2 requires S/N = 7 "summed over both coronagraphs", so the
    channels add in quadrature: 1/tau is the sum of each channel's CR_p^2 / (CR_p + 2 CR_b).

    fExozodiScale multiplies the exozodi count rate. Rates computed once at one zodi can then be
    re-used at any exozodi level, which is how completeness is tabulated over a grid of levels.
    """
    faInverseTau = np.zeros_like(listBandRates[0]["faPlanet"])
    for dictRates, dictBand in zip(listBandRates, listBands):
        faAstro = (dictRates["faLeak"] + dictRates["fZodi"] +
                   fExozodiScale * dictRates["fExozodi"])
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


def faCumulativeMeanCharTime(faTauDet, faTauChar, faTauGridS, fCap=np.inf):
    """Mean characterization time over the planets actually counted at each exposure time.

    The budget charges the EXPECTED TOTAL characterization time, which is the number of
    detections times the mean cost per detection, so the mean is the correct statistic and it
    must be taken over the counted set rather than over every detectable planet. Detection is
    brightness-ordered: a short allocation finds only the brightest planets, which are also the
    cheapest to characterize, so a statistic over all detectable planets overcharges short
    allocations. faTauDet must already be infinite for planets that do not count.

    A characterization that would exceed the two-month cap is charged as zero rather than at its
    nominal value, because it would never be attempted. Charging it was a real error: beyond
    about 15 pc the nominal characterization time runs to thousands of days, and feeding that
    into the cost made distant stars look ruinously expensive even under a counting rule that
    does not require characterization at all.
    """
    faOrder = np.argsort(faTauDet)
    faDetSorted = faTauDet[faOrder]
    faCharSorted = faTauChar[faOrder]
    faCharSorted = np.where(np.isfinite(faCharSorted) & (faCharSorted <= fCap),
                            faCharSorted, 0.0)
    faCumulative = np.cumsum(faCharSorted)
    faCount = np.searchsorted(faDetSorted, faTauGridS, side="right")
    return np.where(faCount > 0, faCumulative[np.maximum(faCount - 1, 0)] /
                    np.maximum(faCount, 1), 0.0)


def faNthSmallestAccumulated(faValues, iRequired):
    """Running iRequired-th smallest along the visit axis, infinite until that many exist.

    With iRequired = 1 this is the running minimum: one detection is enough. With 2 it is the
    running second-smallest, which is the exposure at which a planet has been caught TWICE.
    """
    if iRequired <= 1:
        return np.minimum.accumulate(faValues, axis=1)
    iStars, iVisits = faValues.shape
    faOut = np.full_like(faValues, np.inf)
    for k in range(iRequired - 1, iVisits):
        faPrefix = np.sort(faValues[:, :k + 1], axis=1)
        faOut[:, k] = faPrefix[:, iRequired - 1]
    return faOut


def ffPlanetCharacterizationCap(dictMission):
    """The cap a single planet's spectrum must meet to count: the time limit, or none.

    sCharacterizationGate "planet" (default) applies the two-month limit to each planet's own
    characterization time, the reading of Stark et al. (2019) Sec. 3 ("Any planets that did not
    meet this criteria did not count toward the yield"). "star" drops the per-planet gate; the
    limit is instead applied by the optimizer to the star's probabilistic characterization time
    eta * C * <t_c> of Stark et al. (2015) Eq. 12 (see optimizer.fdictStarCostCurve). Stark et
    al. (2025)'s benchmark shows AYO's own spectra of Earth twins at 12-18 pc take 37-104 days,
    which under the per-planet gate would exclude most planets at stars Stark (2024) Fig. 11
    completes to 0.5-0.8; the "star" reading is the diagnostic alternative.
    """
    if dictMission.get("sCharacterizationGate", "planet") == "star":
        return np.inf
    return dictMission["fExposureLimitS"]


def faCountedTimes(faTauDet, faTauChar, fCap, iRequiredDetections=1, fCharCap=None):
    """Per-planet detection time using the first k visits, for every k, infinite where it fails.

    A planet counts once it has been detected iRequiredDetections times. One detection suffices
    to find a planet, but Stark et al. (2024) budget characterization only after orbit
    determination, and cite Bruna et al. (2023) for two reflected-light detections being enough
    to constrain an orbit. Requiring two makes the second-cheapest epoch the binding one rather
    than the cheapest, which steepens C(tau): a marginal single detection no longer counts.
    """
    fCharCap = fCap if fCharCap is None else fCharCap
    faBestDet = faNthSmallestAccumulated(faTauDet, iRequiredDetections)
    faBestChar = np.minimum.accumulate(faTauChar, axis=1)
    bCounts = (np.isfinite(faBestDet) & (faBestDet <= fCap) &
               np.isfinite(faBestChar) & (faBestChar <= fCharCap))
    return np.where(bCounts, faBestDet, np.inf), faBestChar, bCounts


def faCompletenessPerVisitCount(faBestDet, faTauGridS, iNumPlanets):
    """Completeness curve for each visit count, shape (visits, grid)."""
    iVisits = faBestDet.shape[1]
    faOut = np.zeros((iVisits, faTauGridS.size))
    for k in range(iVisits):
        faSorted = np.sort(faBestDet[:, k][np.isfinite(faBestDet[:, k])])
        faOut[k] = np.searchsorted(faSorted, faTauGridS, side="right") / float(iNumPlanets)
    return faOut


def faCharMeanPerVisitCount(faBestDet, faBestChar, faTauGridS, fCap=np.inf):
    """Mean characterization time over the counted planets, for each visit count."""
    iVisits = faBestDet.shape[1]
    faOut = np.zeros((iVisits, faTauGridS.size))
    for k in range(iVisits):
        faOut[k] = faCumulativeMeanCharTime(faBestDet[:, k], faBestChar[:, k], faTauGridS,
                                            fCap)
    return faOut


def flistCharacterizationRates(dictStar, dictPlanets, dictGeom, listCharOptions, dictMission,
                               bBestPhase, iPhaseSamples):
    """Count rates for every characterization option, over orbital phase or at the visit epochs.

    With bBestPhase the orbit is resolved into iPhaseSamples epochs (see
    faCharacterizationTimeAtBestPhase); otherwise the rates are at the epochs of dictGeom.
    """
    if not bBestPhase:
        return [fdictCountRates(dictStar, dictPlanets, dictGeom, o, dictMission)
                for o in listCharOptions]
    faTheta = np.linspace(0.0, 2.0 * np.pi, iPhaseSamples, endpoint=False)
    dictOrbit = dict(dictPlanets)
    dictOrbit["faTheta"] = np.tile(faTheta, (len(np.atleast_1d(dictPlanets["faAxisAu"])), 1))
    dictGeomAll = fdictProjectOrbits(dictOrbit)
    return [fdictCountRates(dictStar, dictOrbit, dictGeomAll, o, dictMission)
            for o in listCharOptions]


def faCharacterizationTimeFromRates(listCharRates, listCharOptions, dictMission, bBestPhase,
                                    iVisits, fExozodiScale=1.0):
    """Cheapest characterization time over the options, per planet per visit epoch."""
    faBest = None
    for dictRates, dictOption in zip(listCharRates, listCharOptions):
        faTau = faRequiredExposureTime([dictRates], [dictOption], dictOption["fSignalToNoise"],
                                       dictMission, fExozodiScale)
        faBest = faTau if faBest is None else np.minimum(faBest, faTau)
    if not bBestPhase:
        return faBest
    return np.repeat(np.min(faBest, axis=1)[:, None], iVisits, axis=1)


def fiPhaseSamples(dictMission):
    """Orbital phases resolved when characterization is scheduled at its best phase."""
    return int(dictMission.get("iCharacterizationPhaseSamples",
                               I_DEFAULT_CHARACTERIZATION_PHASES))


def faCharacterizationTimeAtBestPhase(dictStar, dictPlanets, listCharOptions, dictMission,
                                      iPhaseSamples):
    """Characterization time for each planet at the most favourable phase on its orbit.

    Stark et al. (2019) Sec. 7.2: "We assumed that the orbit was well-determined, such that the
    phase of the planet could be optimized." Spectral characterization follows orbit determination
    and is therefore scheduled, not taken wherever the planet happened to sit when it was
    detected. Charging the detection phase instead -- which this pipeline did until this was
    traced -- bills every crescent-phase detection at its worst possible moment, and because the
    two-month cap converts "expensive" into "does not count toward the yield", it turns marginal
    targets into hard zeros rather than merely costly ones.

    The orbit is resolved into iPhaseSamples epochs, mirroring the 100 evenly spaced mean
    anomalies AYO uses, and the minimum over them is returned. The result depends only on the
    planet's orbit, not on which visit detected it, so it is one number per planet.
    """
    listRates = flistCharacterizationRates(dictStar, dictPlanets, None, listCharOptions,
                                           dictMission, True, iPhaseSamples)
    return faCharacterizationTimeFromRates(listRates, listCharOptions, dictMission, True, 1)[:, 0]


def faDetectionTimes(dictStar, dictPlanets, dictGeom, listBandsDet, listCharOptions,
                     dictMission, iNumPlanets, bCharacterization=True):
    """Detection and characterization exposure times per planet per visit epoch.

    bCharacterization=False skips the characterization calculation entirely and returns infinities
    in its place. The albedo re-evaluation uses it, because Ref. stark2024 Sec. 3.2 fixes the
    characterization budget at the planning albedo and the recomputed value is therefore thrown
    away. Characterization is 91% of this function's cost once it is scheduled over orbital
    phases, so not computing a discarded result is the single largest saving available.
    """
    listDetRates = [fdictCountRates(dictStar, dictPlanets, dictGeom, b, dictMission)
                    for b in listBandsDet]
    faTauDet = faRequiredExposureTime(listDetRates, listBandsDet,
                                      listBandsDet[0]["fSignalToNoise"], dictMission)
    if not bCharacterization:
        return faTauDet, np.full(faTauDet.shape, np.inf)
    bBest = dictMission.get("bOptimizeCharacterizationPhase", True)
    listCharRates = flistCharacterizationRates(dictStar, dictPlanets, dictGeom, listCharOptions,
                                               dictMission, bBest, fiPhaseSamples(dictMission))
    return faTauDet, faCharacterizationTimeFromRates(listCharRates, listCharOptions, dictMission,
                                                     bBest, faTauDet.shape[1])


def faStarkAlbedoTimes(faTauDet, faFlux, faSepLamD, faFluxDrawn, bBrightBound=False):
    """Exposure time at which a drawn-albedo planet passes Stark's per-visit detectability test.

    Stark et al. (2024) Sec. 3.2 do not re-derive the exposure time for a planet whose albedo
    differs from the planning value. They ask whether the fixed observation plan would have caught
    it, by requiring its albedo-adjusted flux to fall inside the range of fluxes actually detected
    during that visit: "The faintest detected planet flux along the orbit segment (usually
    corresponding to crescent phase) is limited by the exposure time, while the brightest planet
    flux along the segment (usually corresponding to gibbous phase) is constrained by the IWA of
    the coronagraph."

    bBrightBound adds that upper limit. It is not an approximation artifact, which is how an
    earlier note here described it: for an edge-on orbit gibbous phase IS small projected
    separation, so a flux above what the segment reached does imply the planet was inside the
    inner working angle. Since cos(i) is uniform, high inclinations dominate the detections, and
    the identification holds where it matters; it degrades toward face-on, where the flux barely
    varies along the orbit and the bound does not bind anyway.

    The lower bounds relax monotonically as the exposure lengthens, so each planet passes for every
    exposure beyond some threshold, which is what this returns. The bright bound does not depend on
    exposure, so it is applied as a veto.
    """
    faOut = np.full(faTauDet.shape, np.inf)
    for k in range(faTauDet.shape[1]):
        faOrder = np.argsort(faTauDet[:, k])
        faTauSorted = faTauDet[faOrder, k]
        bFinite = np.isfinite(faTauSorted)
        if not bFinite.any():
            continue
        faTauSorted = faTauSorted[bFinite]
        faFluxOrdered = faFlux[faOrder, k][bFinite]
        faFloorFlux = np.minimum.accumulate(faFluxOrdered)
        faFloorSep = np.minimum.accumulate(faSepLamD[faOrder, k][bFinite])
        iFlux = np.searchsorted(-faFloorFlux, -faFluxDrawn[:, k], side="left")
        iSep = np.searchsorted(-faFloorSep, -faSepLamD[:, k], side="left")
        iNeed = np.maximum(iFlux, iSep)
        bReach = iNeed < faTauSorted.size
        if bBrightBound:
            bReach &= faFluxDrawn[:, k] <= float(np.max(faFluxOrdered))
        faOut[bReach, k] = faTauSorted[iNeed[bReach]]
    return faOut


def fdictStarRates(dictStar, dictBox, listBandsDet, dictBandChar, dictMission, iNumPlanets,
                   fAlpha, fBeta, iSeed):
    """Inject one star's planets and compute every count rate once, with exozodi at ONE zodi.

    Everything expensive about a star's completeness -- the orbits, the coronagraph lookups, and
    characterization over orbital phase -- is independent of its exozodi level, which enters
    the exposure time only as a multiplier on one background term. Separating the two lets the
    completeness be evaluated at many exozodi levels for little more than the cost of one.
    """
    rng = np.random.default_rng(iSeed)
    iVisits = int(dictMission.get("iMaxVisits", 1))
    dictPlanets = fdictInjectPlanets(dictBox, np.sqrt(dictStar["fLuminosityLsun"]),
                                     iNumPlanets, fAlpha, fBeta, rng, iVisits=iVisits)
    dictGeom = fdictProjectOrbits(dictPlanets)
    dictUnit = dict(dictStar, fExozodiLevel=1.0)
    listCharOptions = dictMission.get("listBandsCharacterization") or [dictBandChar]
    bBest = dictMission.get("bOptimizeCharacterizationPhase", True)
    dictOut = dict(iVisits=iVisits, listCharOptions=listCharOptions, bBestPhase=bBest,
                   listDetRates=[fdictCountRates(dictUnit, dictPlanets, dictGeom, b, dictMission)
                                 for b in listBandsDet],
                   listCharRates=flistCharacterizationRates(
                       dictUnit, dictPlanets, dictGeom, listCharOptions, dictMission, bBest,
                       fiPhaseSamples(dictMission)))
    dictAlbedoRange = dictMission.get("dictAlbedoDistribution")
    if dictAlbedoRange:
        dictOut.update(fdictAlbedoRates(dictUnit, dictPlanets, dictGeom, listBandsDet,
                                        dictAlbedoRange, dictOut, dictMission))
    return dictOut


def fdictAlbedoRates(dictUnit, dictPlanets, dictGeom, listBandsDet, dictAlbedoRange, dictRates,
                     dictMission):
    """Count rates for the same planets with drawn albedos, plus what Stark's method needs."""
    dictPlanets["faAlbedoDrawn"] = (dictAlbedoRange["fMin"] + dictPlanets["faAlbedo"] *
                                    (dictAlbedoRange["fMax"] - dictAlbedoRange["fMin"]))
    dictOut = dict(listDetRatesAlbedo=[fdictCountRates(dictUnit, dictPlanets, dictGeom, b,
                                                       dictMission) for b in listBandsDet])
    if dictMission.get("bAdjustCharacterizationForAlbedo", False):
        dictOut["listCharRatesAlbedo"] = flistCharacterizationRates(
            dictUnit, dictPlanets, dictGeom, dictRates["listCharOptions"], dictMission,
            dictRates["bBestPhase"], fiPhaseSamples(dictMission))
    if dictMission.get("sAlbedoMethod", "recompute") == "perVisitThreshold":
        dictPlan = dictOut["listDetRatesAlbedo"][0]
        dictOut["faFluxPlan"] = dictPlan["faFluxRatio"] * (
            dictMission["fGeometricAlbedo"] / np.atleast_1d(dictPlanets["faAlbedoDrawn"])[:, None])
        dictOut["faFluxDrawn"] = dictPlan["faFluxRatio"]
        dictOut["faSepLamD"] = dictPlan["faSepLamD"]
    return dictOut


def fdictCompletenessAtZodi(dictRates, fZodi, listBandsDet, dictMission, faTauGridS,
                            iNumPlanets):
    """Completeness curves for one star at one exozodi level, from rates at one zodi."""
    fCap = dictMission["fExposureLimitS"]
    iRequired = int(dictMission.get("iRequiredDetections", 1))
    faTauDet = faRequiredExposureTime(dictRates["listDetRates"], listBandsDet,
                                      listBandsDet[0]["fSignalToNoise"], dictMission, fZodi)
    faTauChar = faCharacterizationTimeFromRates(
        dictRates["listCharRates"], dictRates["listCharOptions"], dictMission,
        dictRates["bBestPhase"], dictRates["iVisits"], fZodi)
    fCharCap = ffPlanetCharacterizationCap(dictMission)
    faBestDet, faBestChar, bCounts = faCountedTimes(faTauDet, faTauChar, fCap, iRequired,
                                                    fCharCap)
    faComp = faCompletenessPerVisitCount(faBestDet, faTauGridS, iNumPlanets)
    faDetOnly = faNthSmallestAccumulated(
        np.where(np.isfinite(faTauDet) & (faTauDet <= fCap), faTauDet, np.inf), iRequired)
    if not dictMission.get("bYieldRequiresCharacterization", True):
        faBestDet = faDetOnly
        faComp = faCompletenessPerVisitCount(faBestDet, faTauGridS, iNumPlanets)
    faCompDetectionOnly = faCompletenessPerVisitCount(faDetOnly, faTauGridS, iNumPlanets)
    faCompAlbedo = faComp
    if "listDetRatesAlbedo" in dictRates:
        faCompAlbedo = faAlbedoCompleteness(dictRates, fZodi, faTauDet, faTauChar, listBandsDet,
                                            dictMission, faTauGridS, iNumPlanets)
    return dict(faComp=faComp, faCompAlbedo=faCompAlbedo,
                faCompDetectionOnly=faCompDetectionOnly,
                faTauCharMeanS=faCharMeanPerVisitCount(faBestDet, faBestChar, faTauGridS,
                                                       fCharCap),
                fTauCharS=float(np.median(faBestChar[:, -1][bCounts[:, -1]]))
                if bCounts[:, -1].any() else np.inf,
                fMaxCompleteness=float(faComp[-1, -1]))


def faAlbedoCompleteness(dictRates, fZodi, faTauDet, faTauChar, listBandsDet, dictMission,
                         faTauGridS, iNumPlanets):
    """Completeness of the same planets re-evaluated with drawn albedos (see below)."""
    fCap = dictMission["fExposureLimitS"]
    faTauDetA = faRequiredExposureTime(dictRates["listDetRatesAlbedo"], listBandsDet,
                                       listBandsDet[0]["fSignalToNoise"], dictMission, fZodi)
    if "faFluxPlan" in dictRates:
        faTauDetA = faStarkAlbedoTimes(
            faTauDet, dictRates["faFluxPlan"], dictRates["faSepLamD"], dictRates["faFluxDrawn"],
            bBrightBound=bool(dictMission.get("bAlbedoBrightBound", False)))
    faTauCharA = faTauChar
    if "listCharRatesAlbedo" in dictRates:
        faTauCharA = faCharacterizationTimeFromRates(
            dictRates["listCharRatesAlbedo"], dictRates["listCharOptions"], dictMission,
            dictRates["bBestPhase"], dictRates["iVisits"], fZodi)
    faBestDetA, _, _ = faCountedTimes(faTauDetA, faTauCharA, fCap,
                                      int(dictMission.get("iRequiredDetections", 1)),
                                      ffPlanetCharacterizationCap(dictMission))
    return faCompletenessPerVisitCount(faBestDetA, faTauGridS, iNumPlanets)


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

    The albedo re-evaluation adjusts the DETECTION time only. Ref. stark2024 Sec. 3.2 is explicit:
    "Detection times are used to determine whether a planet of differing albedo would have been
    detected, but characterization times are not considered. Characterization times are budgeted
    for by AYO under the assumption that the planets have a single A_G = 0.2." The observation
    plan is set in stone before the draw, so the spectrum's cost cannot be revised after it.

    Adjusting both, as this module did until the published detection-rate-versus-albedo shape was
    checked, is what held the albedo penalty at 5% against a published 12%. Characterization binds
    beyond about 10 pc, so letting a bright planet also buy a cheaper spectrum lets it keep gaining
    above A_G = 0.2, where Stark's detection rate is flat; that rising tail cancels the loss from
    dark planets. bAdjustCharacterizationForAlbedo restores the old behaviour for comparison.
    """
    dictRates = fdictStarRates(dictStar, dictBox, listBandsDet, dictBandChar, dictMission,
                               iNumPlanets, fAlpha, fBeta, iSeed)
    fZodi = dictStar.get("fExozodiLevel", dictMission["fExozodiLevel"])
    return fdictCompletenessAtZodi(dictRates, fZodi, listBandsDet, dictMission, faTauGridS,
                                   iNumPlanets)


def fdictStarCompletenessZodiGrid(dictStar, dictBox, listBandsDet, dictBandChar, dictMission,
                                  faTauGridS, iNumPlanets, fAlpha, fBeta, iSeed, faZodiLevels):
    """Completeness curves at every exozodi level in faZodiLevels, shape (levels, visits, grid).

    The planets, orbits and count rates are those fdictStarCompleteness would use with the same
    seed, so the curve at any level equals a single-level evaluation at that level exactly.
    """
    dictRates = fdictStarRates(dictStar, dictBox, listBandsDet, dictBandChar, dictMission,
                               iNumPlanets, fAlpha, fBeta, iSeed)
    listLevels = [fdictCompletenessAtZodi(dictRates, float(f), listBandsDet, dictMission,
                                          faTauGridS, iNumPlanets) for f in faZodiLevels]
    return {sKey: np.stack([d[sKey] for d in listLevels])
            for sKey in ("faComp", "faCompAlbedo", "faCompDetectionOnly", "faTauCharMeanS")}
