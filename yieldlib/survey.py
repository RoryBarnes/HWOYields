"""Target screening and the completeness table that the survey optimizer consumes."""

import numpy as np

from . import completeness as cp
from . import coronagraph as cg
from . import optimizer as opt

F_REARTH_AU = 4.25875e-5


def fdfScreenTargets(dfCatalog, dictMission, dictBandPrimary, fMinEeidLamD, iMaxStars,
                     fTeffMin, fTeffMax):
    """Drop stars whose habitable zone can never yield a detection, then keep the best iMaxStars.

    Two physical screens: the Earth-equivalent insolation distance must subtend enough of the
    diffraction scale for the coronagraph to pass light, and an Earth analogue at quadrature must
    sit above the post-processing noise floor. Both are necessary conditions, so nothing that
    could have contributed is removed.
    """
    dfOut = dfCatalog[(dfCatalog["fTeffK"] >= fTeffMin) &
                      (dfCatalog["fTeffK"] <= fTeffMax)].copy()
    fLamD = cg.fnLambdaOverDArcsec(dictBandPrimary["fLambdaM"], dictMission["fDiameterM"])
    dfOut["fEeidLamD"] = dfOut["fEeidArcsec"] / fLamD
    faFluxRatio = dictMission["fGeometricAlbedo"] * (1.0 / np.pi) * \
        (F_REARTH_AU / dfOut["fEeidAu"]) ** 2
    dfOut["fEeidDeltaMag"] = -2.5 * np.log10(faFluxRatio)
    dfOut = dfOut[(dfOut["fEeidLamD"] >= fMinEeidLamD) &
                  (dfOut["fEeidDeltaMag"] <= dictMission["fNoiseFloorDeltaMag"])]
    return dfOut.sort_values("fEeidLamD", ascending=False).head(iMaxStars).reset_index(drop=True)


def fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed):
    """Completeness curves and characterization times for every screened star."""
    dictMission = dictParams["dictMission"]
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    dictBandChar = dictParams["dictBands"]["dictBandCharacterization"]
    faComp = np.zeros((len(dfTargets), len(faTauGridS)))
    faTauChar = np.zeros(len(dfTargets))
    for i, dictRow in enumerate(dfTargets.to_dict("records")):
        dictResult = cp.fdictStarCompleteness(dictRow, dictBox, listBandsDet, dictBandChar,
                                              dictMission, faTauGridS, iNumPlanets,
                                              dictParams["fAlpha"], dictParams["fBeta"],
                                              iSeed + i)
        faComp[i] = dictResult["faComp"]
        faTauChar[i] = dictResult["fTauCharS"]
    return dict(faComp=faComp, faTauChar=faTauChar)


def flistStarsFromTable(dictTable, faTauGridS):
    """Package a completeness table into the per-star dicts the optimizer expects."""
    return [dict(faTauGridS=faTauGridS, faComp=dictTable["faComp"][i],
                 fTauCharS=float(dictTable["faTauChar"][i]))
            for i in range(dictTable["faComp"].shape[0])]


def fnYieldForCalibration(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed,
                          fEtaEarth, fCalibration):
    """Expected EEC yield at one value of the throughput calibration factor."""
    dictParams["dictMission"]["fThroughputCalibration"] = fCalibration
    dictTable = fdictCompletenessTable(dfTargets, dictParams, dictBox, faTauGridS,
                                       iNumPlanets, iSeed)
    listStars = flistStarsFromTable(dictTable, faTauGridS)
    return float(opt.fdictOptimizeSurvey(listStars, fEtaEarth,
                                         dictParams["dictMission"])["fYield"])
