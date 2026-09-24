"""Target screening and the completeness table that the survey optimizer consumes."""

import numpy as np

from . import completeness as cp
from . import coronagraph as cg
from . import optimizer as opt

F_REARTH_AU = 4.25875e-5


def fnMaxPlanetRadius(dictBox):
    """Largest planet radius the selection box admits, in Earth radii."""
    if dictBox["sRadiusMode"] == "canonical":
        return dictBox["fRadiusMaxEarth"]
    return dictBox["fMassHiEarth"] ** dictBox["fMassRadiusExponent"]


def fdfScreenTargets(dfCatalog, dictMission, dictBandPrimary, fMinEeidLamD, iMaxStars,
                     fTeffMin, fTeffMax, dictBox):
    """Drop stars whose habitable zone can never yield a detection, then keep the best iMaxStars.

    Two necessary conditions. The Earth-equivalent insolation distance must subtend enough of the
    diffraction scale for the coronagraph to pass light, and the BRIGHTEST planet the selection
    box admits must sit above the post-processing noise floor.

    "Brightest" is load-bearing and was got wrong once: screening on an Earth-sized planet at
    quadrature rejected every star above ~4.6 Lsun, whereas Stark et al. (2024) Fig. 11 selects
    targets out to ~20 Lsun. Luminous stars have wide habitable zones and are exactly the distant
    targets a larger aperture unlocks, so discarding them suppressed the yield's growth with
    telescope diameter. The bound here takes the largest radius in the box, the inner habitable
    zone edge, and full phase, so a star is dropped only when no planet in the box could ever be
    detected around it.
    """
    dfOut = dfCatalog[(dfCatalog["fTeffK"] >= fTeffMin) &
                      (dfCatalog["fTeffK"] <= fTeffMax)].copy()
    fLamD = cg.fnLambdaOverDArcsec(dictBandPrimary["fLambdaM"], dictMission["fDiameterM"])
    dfOut["fEeidLamD"] = dfOut["fEeidArcsec"] / fLamD
    faSemiMajorMin = dictBox["fHzInnerAu"] * dfOut["fEeidAu"]
    faFluxRatio = dictMission["fGeometricAlbedo"] * \
        (fnMaxPlanetRadius(dictBox) * F_REARTH_AU / faSemiMajorMin) ** 2
    dfOut["fBrightestDeltaMag"] = -2.5 * np.log10(faFluxRatio)
    dfOut = dfOut[(dfOut["fEeidLamD"] >= fMinEeidLamD) &
                  (dfOut["fBrightestDeltaMag"] <= dictMission["fNoiseFloorDeltaMag"])]
    return dfOut.sort_values("fEeidLamD", ascending=False).head(iMaxStars).reset_index(drop=True)


def faDrawExozodiLevels(dictMission, iStars, iSeed, saHip=None):
    """Per-star exozodi levels drawn from the LBTI HOSTS maximum-likelihood distribution.

    Stark et al. (2024) Sec. 3.3 draw each star's exozodi from the HOSTS best fit -- median three
    zodis, multi-modal with peaks at higher levels -- rather than giving every star the median,
    and report the mean yield falling from 19.8 to 17.6.

    bDrawExozodiLevels selects between the two rungs of that ladder, and it matters more than it
    looks. The published 22.5 comes from a single AYO run at a FIXED exozodi level, before the
    sampling section; 17.6 is what sampling costs. Drawing unconditionally -- which this module
    did until the exozodi penalty measured as exactly zero and exposed it -- anchors the
    throughput calibration against an exozodi-sampled yield while the number it is fitted to is
    not, and makes the published 11% penalty structurally unmeasurable because both sides of the
    comparison already contain it. The bias is asymmetric in the same way
    as albedo: a star drawn below the median gains little because its exposure was already short,
    while a high draw on a high-priority target lengthens its exposure enough that the optimizer
    must substitute a less productive star from a limited pool.

    The distribution is the one Stark plots in Fig. 9 (red, "Max. Likelihood"), digitized from
    the figure's vector content stream: 2-zodi bins to 1000 zodis, median 2.98, 44 percent of
    draws below 2 zodis and 21 percent above 100, plus 5 percent beyond the axis. A lognormal
    stand-in (median 3, ln-sigma 1.2) was used until the exozodi penalty re-measured at 0.7
    percent against the published 11: it put 0.2 percent of stars above 100 zodis, so it almost
    never scrubbed a high-priority target, which is the mechanism Stark describes. The lognormal
    form is still accepted (fMedianZodi, fLogSigma) for comparison.

    Stark also pins four stars to their LBTI-measured levels (dictPinnedExozodi, keyed by HIP
    number), which he says accounts for about a third of the shift. Pinning needs saHip, the
    screened targets' HIP numbers, and applies only when levels are drawn, as in Stark's
    sampling runs; the fixed-level run that the calibration targets pins nothing.
    """
    dictDraw = dictMission.get("dictExozodiDistribution")
    if not dictDraw or not dictMission.get("bDrawExozodiLevels", True):
        return np.full(iStars, dictMission["fExozodiLevel"])
    rng = np.random.default_rng(iSeed)
    faLevels = (faDrawEmpiricalZodi(dictDraw, iStars, rng) if "faEdgesZodi" in dictDraw else
                dictDraw["fMedianZodi"] * np.exp(rng.normal(0.0, dictDraw["fLogSigma"], iStars)))
    return faApplyPinnedZodi(faLevels, saHip, dictMission.get("dictPinnedExozodi", {}))


def faDrawEmpiricalZodi(dictDraw, iStars, rng):
    """Draw from binned probabilities, uniform within a bin; off-axis mass goes to fOffAxisZodi."""
    faEdges = np.asarray(dictDraw["faEdgesZodi"], dtype=float)
    faProb = np.append(np.asarray(dictDraw["faProbability"], dtype=float),
                       max(0.0, 1.0 - float(np.sum(dictDraw["faProbability"]))))
    iBins = len(faEdges) - 1
    iaBin = rng.choice(iBins + 1, size=iStars, p=faProb / faProb.sum())
    faUniform = rng.random(iStars)
    faInBin = faEdges[np.minimum(iaBin, iBins - 1)] + faUniform * np.diff(faEdges)[
        np.minimum(iaBin, iBins - 1)]
    return np.where(iaBin == iBins, float(dictDraw["fOffAxisZodi"]), faInBin)


def faApplyPinnedZodi(faLevels, saHip, dictPinned):
    """Overwrite drawn levels for stars with an LBTI-measured exozodi, matched by HIP number."""
    if saHip is None or not dictPinned:
        return faLevels
    faOut = np.array(faLevels, dtype=float)
    for i, sHip in enumerate(saHip):
        if sHip in dictPinned:
            faOut[i] = float(dictPinned[sHip]["fZodi"])
    return faOut


def flistHipNumbers(dfTargets):
    """HIP numbers of the screened targets as strings, empty where the catalog has none."""
    if "sHipName" not in dfTargets:
        return None
    return [str(int(h)) if h == h else "" for h in dfTargets["sHipName"]]


def fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed):
    """Completeness curves and characterization times for every screened star."""
    dictMission = dictParams["dictMission"]
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    dictBandChar = dictParams["dictBands"]["dictBandCharacterization"]
    dictMission = dict(dictMission)
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    iVisits = int(dictMission.get("iMaxVisits", 1))
    faComp = np.zeros((len(dfTargets), iVisits, len(faTauGridS)))
    faTauCharMean = np.zeros((len(dfTargets), iVisits, len(faTauGridS)))
    faCompAlbedo = np.zeros((len(dfTargets), iVisits, len(faTauGridS)))
    faCompDetection = np.zeros((len(dfTargets), iVisits, len(faTauGridS)))
    faTauChar = np.zeros(len(dfTargets))
    faExozodi = faDrawExozodiLevels(dictMission, len(dfTargets),
                                    int(dictMission.get("iExozodiSeed", iSeed + 977)),
                                    flistHipNumbers(dfTargets))
    for i, dictRow in enumerate(dfTargets.to_dict("records")):
        dictRow["fExozodiLevel"] = float(faExozodi[i])
        dictResult = cp.fdictStarCompleteness(dictRow, dictBox, listBandsDet, dictBandChar,
                                              dictMission, faTauGridS, iNumPlanets,
                                              dictParams["fAlpha"], dictParams["fBeta"],
                                              iSeed + i)
        faComp[i] = dictResult["faComp"]
        faTauCharMean[i] = dictResult["faTauCharMeanS"]
        faCompAlbedo[i] = dictResult["faCompAlbedo"]
        faCompDetection[i] = dictResult["faCompDetectionOnly"]
        faTauChar[i] = dictResult["fTauCharS"]
    return dict(faComp=faComp, faTauChar=faTauChar, faTauCharMean=faTauCharMean,
                faCompAlbedo=faCompAlbedo, faCompDetectionOnly=faCompDetection)


def flistStarsFromTable(dictTable, faTauGridS):
    """Package a completeness table into the per-star dicts the optimizer expects."""
    return [dict(faTauGridS=faTauGridS, faComp=dictTable["faComp"][i],
                 faCompYield=dictTable["faCompAlbedo"][i],
                 faTauCharMeanS=dictTable["faTauCharMean"][i],
                 fTauCharS=float(dictTable["faTauChar"][i]))
            for i in range(dictTable["faComp"].shape[0])]


def fnYieldForCalibration(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed,
                          fEtaEarth, fCalibration):
    """Expected EEC yield at one throughput calibration factor, at the PLANNING albedo.

    Stark et al. (2024) obtain 22.5 from a single AYO run with every EEC at A_G = 0.2, so the
    calibration must be fitted against that number rather than against the albedo-drawn yield,
    which is lower by construction.
    """
    dictParams["dictMission"]["fThroughputCalibration"] = fCalibration
    dictTable = fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                       iNumPlanets, iSeed)
    listStars = flistStarsFromTable(dictTable, faTauGridS)
    return float(opt.fdictOptimizeSurvey(listStars, fEtaEarth,
                                         dictParams["dictMission"])["fYieldPlanning"])
