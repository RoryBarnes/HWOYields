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
    faTauChar = np.zeros(len(dfTargets))
    for i, dictRow in enumerate(dfTargets.to_dict("records")):
        dictResult = cp.fdictStarCompleteness(dictRow, dictBox, listBandsDet, dictBandChar,
                                              dictMission, faTauGridS, iNumPlanets,
                                              dictParams["fAlpha"], dictParams["fBeta"],
                                              iSeed + i)
        faComp[i] = dictResult["faComp"]
        faTauCharMean[i] = dictResult["faTauCharMeanS"]
        faCompAlbedo[i] = dictResult["faCompAlbedo"]
        faTauChar[i] = dictResult["fTauCharS"]
    return dict(faComp=faComp, faTauChar=faTauChar, faTauCharMean=faTauCharMean,
                faCompAlbedo=faCompAlbedo)


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
