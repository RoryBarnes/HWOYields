"""Exozodi as a random variable: completeness tabulated over exozodi level, and survey draws.

Stark et al. (2024) Sec. 3.3 draw every star's exozodi level from the LBTI HOSTS distribution and
re-run AYO for each draw, 500 times, because the survey is re-optimized around whatever the dust
turns out to be ("Even if the yield code replaces these high-zodi stars with other low-zodi
stars, the limited pool of targets means the code must replace a previously productive target
with a lower-productivity target"). A single draw, which this pipeline used until 2026-09-24,
turns that distribution into one realization of it; over 8 draws the fixed-eta yield scattered
by +/-0.5 EEC and the pipeline's seed sat 0.5 above the mean.

Exozodi enters a star's exposure times only as a multiplier on one background term, so the
expensive part of completeness -- the injected planets and their count rates -- is computed
once per star and the curves are tabulated on a grid of levels. Each draw then interpolates every
star's curves at its drawn level (linear in the level between grid nodes) and re-optimizes. The
interpolation is checked against direct evaluation by explorations/validateExozodiGrid.py.
"""

import numpy as np

from . import completeness as cp
from . import optimizer as opt
from . import survey as sv

LIST_ZODI_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0, 23.0, 32.0, 45.0,
                  64.0, 90.0, 128.0, 181.0, 256.0, 362.0, 512.0, 724.0, 1024.0]
""" Exozodi levels (zodis) at which completeness is tabulated.

The HOSTS maximum-likelihood distribution spans 0-1000 zodis with 44 percent of its mass below 2,
and the pinned LBTI stars reach 588. Spacing is about sqrt(2) in level above 2 zodis, where the
exposure time's dependence on level is close to logarithmic, and 0.5 below it. The fixed planning
level (3 zodis) is a node, so the fixed-exozodi survey needs no interpolation.
"""


def fdictCompletenessZodiGrid(dfTargets, dictParams, dictBox, faTauGridS, iNumPlanets, iSeed,
                              faZodiGrid=None):
    """Completeness of every screened star at every grid level, stored compactly.

    Completeness is a count of injected planets divided by iNumPlanets, so it is stored as the
    count (uint16), which is exact. A star with no completeness at the lowest level has none at
    any level, since exposure times only grow with exozodi, so its other levels are not evaluated
    and it is not stored; iaStar indexes the stored stars into dfTargets.
    """
    faZodiGrid = np.asarray(LIST_ZODI_GRID if faZodiGrid is None else faZodiGrid, dtype=float)
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    listKeep, listComp, listCompAlb, listChar = [], [], [], []
    for i, dictRow in enumerate(dfTargets.to_dict("records")):
        dictRates = cp.fdictStarRates(dictRow, dictBox, listBandsDet,
                                      dictParams["dictBands"]["dictBandCharacterization"],
                                      dictMission, iNumPlanets, dictParams["fAlpha"],
                                      dictParams["fBeta"], iSeed + i)
        listLevels = flistLevelsForStar(dictRates, faZodiGrid, listBandsDet, dictMission,
                                        faTauGridS, iNumPlanets)
        if listLevels is None:
            continue
        listKeep.append(i)
        listComp.append(np.stack([d["faComp"] for d in listLevels]))
        listCompAlb.append(np.stack([d["faCompAlbedo"] for d in listLevels]))
        listChar.append(np.stack([d["faTauCharMeanS"] for d in listLevels]))
    return fdictPackGrid(listKeep, listComp, listCompAlb, listChar, faZodiGrid, iNumPlanets)


def flistLevelsForStar(dictRates, faZodiGrid, listBandsDet, dictMission, faTauGridS,
                       iNumPlanets):
    """Curves at every level, or None when the star has no completeness even at the lowest."""
    dictFirst = cp.fdictCompletenessAtZodi(dictRates, float(faZodiGrid[0]), listBandsDet,
                                           dictMission, faTauGridS, iNumPlanets)
    if not np.any(dictFirst["faComp"] > 0):
        return None
    return [dictFirst] + [cp.fdictCompletenessAtZodi(dictRates, float(f), listBandsDet,
                                                     dictMission, faTauGridS, iNumPlanets)
                          for f in faZodiGrid[1:]]


def fdictPackGrid(listKeep, listComp, listCompAlb, listChar, faZodiGrid, iNumPlanets):
    """Stack per-star grids into arrays of shape (stars, levels, visits, tau)."""
    def faCounts(listArrays):
        return np.rint(np.stack(listArrays) * iNumPlanets).astype(np.uint16)
    if not listKeep:
        raise ValueError("no screened star has any completeness at the lowest exozodi level")
    return dict(iaStar=np.asarray(listKeep, dtype=np.int32), faZodiGrid=faZodiGrid,
                iaCompCounts=faCounts(listComp), iaCompAlbedoCounts=faCounts(listCompAlb),
                faTauCharMean=np.stack(listChar).astype(np.float32),
                iNumPlanets=int(iNumPlanets))


def faDrawZodiLevels(dictMission, iStars, iDraws, iSeed, saHip=None):
    """Exozodi levels for every screened star in every draw, shape (draws, stars).

    Each draw is the single-draw rule of survey.faDrawExozodiLevels with its own seed, pinned
    LBTI stars included, so draw j of a run is reproducible on its own.
    """
    dictDraw = dict(dictMission, bDrawExozodiLevels=True)
    return np.stack([sv.faDrawExozodiLevels(dictDraw, iStars, iSeed + j, saHip)
                     for j in range(iDraws)])


def fdictInterpolateAtZodi(dictGrid, faZodi):
    """Every stored star's curves interpolated at its level, linear in level between nodes.

    faZodi holds one level per STORED star (already indexed by iaStar). Levels above the top node
    take the top node's curves; none of the adopted distribution's draws exceed 1000 zodis.
    """
    faGrid = dictGrid["faZodiGrid"]
    faZ = np.clip(np.asarray(faZodi, dtype=float), faGrid[0], faGrid[-1])
    iaLo = np.clip(np.searchsorted(faGrid, faZ, side="right") - 1, 0, faGrid.size - 2)
    faW = ((faZ - faGrid[iaLo]) / (faGrid[iaLo + 1] - faGrid[iaLo]))[:, None, None]
    iaRow = np.arange(faZ.size)
    fScale = 1.0 / dictGrid["iNumPlanets"]

    def faBlend(faArray, fFactor):
        return fFactor * ((1.0 - faW) * faArray[iaRow, iaLo] + faW * faArray[iaRow, iaLo + 1])
    return dict(faComp=faBlend(dictGrid["iaCompCounts"].astype(float), fScale),
                faCompAlbedo=faBlend(dictGrid["iaCompAlbedoCounts"].astype(float), fScale),
                faTauCharMean=faBlend(dictGrid["faTauCharMean"].astype(float), 1.0))


def flistStarsAtZodi(dictGrid, faZodiAllStars, faTauGridS):
    """Optimizer-ready star dicts for the stored stars at one draw's levels (all-star array)."""
    dictAt = fdictInterpolateAtZodi(dictGrid, np.asarray(faZodiAllStars)[dictGrid["iaStar"]])
    return [dict(faTauGridS=faTauGridS, faComp=dictAt["faComp"][i],
                 faCompYield=dictAt["faCompAlbedo"][i], faTauCharMeanS=dictAt["faTauCharMean"][i])
            for i in range(dictAt["faComp"].shape[0])]


def fdictSurveyOverEta(listStars, faEta, dictMission):
    """Planning (A_G = 0.2) and albedo-drawn expected yields, and stars used, at each eta."""
    listRuns = [opt.fdictOptimizeSurvey(listStars, float(f), dictMission) for f in faEta]
    return dict(faYieldPlanning=np.array([d["fYieldPlanning"] for d in listRuns]),
                faYield=np.array([d["fYield"] for d in listRuns]),
                iaStarsUsed=np.array([d["iStarsUsed"] for d in listRuns]))


_DICT_SHARED = {}


def fdictSurveyOverDraws(dictGrid, faZodiDraws, faTauGridS, faEta, dictMission, iProcesses=1):
    """fdictSurveyOverEta for every exozodi draw; arrays of shape (draws, eta).

    Draws are independent, so they run in a forked process pool when iProcesses > 1. The grid is
    handed to the workers through a module-level dict inherited at fork, not pickled per job.
    """
    _DICT_SHARED.update(dictGrid=dictGrid, faZodiDraws=faZodiDraws, faTauGridS=faTauGridS,
                        faEta=faEta, dictMission=dictMission)
    iDraws = faZodiDraws.shape[0]
    if iProcesses > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(iProcesses) as oPool:
            listOut = oPool.map(fdictOneDraw, range(iDraws))
    else:
        listOut = [fdictOneDraw(j) for j in range(iDraws)]
    _DICT_SHARED.clear()
    return {sKey: np.stack([d[sKey] for d in listOut]) for sKey in listOut[0]}


def fdictOneDraw(iDraw):
    """Draw iDraw of fdictSurveyOverDraws (module level so a process pool can call it)."""
    d = _DICT_SHARED
    return fdictSurveyOverEta(flistStarsAtZodi(d["dictGrid"], d["faZodiDraws"][iDraw],
                                               d["faTauGridS"]), d["faEta"], d["dictMission"])


def fdictPerStarOverDraws(dictGrid, faZodiDraws, faTauGridS, fEta, dictMission, iProcesses=1):
    """Each stored star's allocation in every draw at one eta, arrays of shape (draws, stars).

    Returns the completeness reached (planning and albedo-drawn), the time charged and the mean
    characterization time per counted planet, as fdictOptimizeSurvey(bPerStar=True) reports
    them, for the stars in dictGrid["iaStar"] order.
    """
    _DICT_SHARED.update(dictGrid=dictGrid, faZodiDraws=faZodiDraws, faTauGridS=faTauGridS,
                        fEta=fEta, dictMission=dictMission)
    iDraws = faZodiDraws.shape[0]
    if iProcesses > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(iProcesses) as oPool:
            listOut = oPool.map(fdictOnePerStarDraw, range(iDraws))
    else:
        listOut = [fdictOnePerStarDraw(j) for j in range(iDraws)]
    _DICT_SHARED.clear()
    return {sKey: np.stack([d[sKey] for d in listOut]) for sKey in listOut[0]}


def fdictOnePerStarDraw(iDraw):
    """Draw iDraw of fdictPerStarOverDraws."""
    d = _DICT_SHARED
    dictRun = opt.fdictOptimizeSurvey(flistStarsAtZodi(d["dictGrid"], d["faZodiDraws"][iDraw],
                                                       d["faTauGridS"]),
                                      d["fEta"], d["dictMission"], bPerStar=True)
    return {k: dictRun[k] for k in ("faStarComp", "faStarCompYield", "faStarTimeS",
                                    "faStarTauCharMeanS", "faStarVisits", "faStarTauS")}


def fdictLoadGrid(dictNpz, sBox):
    """The exozodi-grid arrays computeCompleteness (A04) stored for one box."""
    return dict(iaStar=dictNpz[f"iaGridStar_{sBox}"], faZodiGrid=dictNpz["faZodiGrid"],
                iaCompCounts=dictNpz[f"iaCompCounts_{sBox}"],
                iaCompAlbedoCounts=dictNpz[f"iaCompAlbedoCounts_{sBox}"],
                faTauCharMean=dictNpz[f"faTauCharMeanGrid_{sBox}"],
                iNumPlanets=int(dictNpz["iNumPlanets"]))


def faFirstNCharacterizationTimes(faComp, faChar, fEta, iFirstN, faPriority):
    """Characterization times of the first N expected EECs, with their expected-count weights.

    Stark et al. (2024) Sec. 4.1 take "the first 18 EECs of any simulation". Stars are taken in
    priority order (completeness per unit time charged) and each contributes eta * C expected
    EECs at its mean characterization time until N have accumulated. Returns (times, weights).
    """
    faOrder = np.argsort(-faPriority)
    faCount = fEta * faComp[faOrder]
    faTake = np.clip(iFirstN - np.concatenate(([0.0], np.cumsum(faCount)[:-1])), 0.0, faCount)
    bKeep = (faTake > 0) & (faChar[faOrder] > 0)
    return faChar[faOrder][bKeep], faTake[bKeep]


def fdictPlanetCharacterizationFirstN(dfTargets, dictParams, dictBox, dictGrid, faZodiDraws,
                                      dictPerStar, iaDraws, fEta, iFirstN, iSeed):
    """Individual characterization times of the first N expected EECs, pooled over draws.

    faFirstNCharacterizationTimes works from each star's MEAN characterization time, which gives
    the right mean but not the distribution Stark et al. (2024) Fig. 14 plots, whose spread comes
    from individual planets: a planet near the inner working angle or with a small orbit costs far
    more than the star's average. This re-derives, for each star allocated time in the chosen
    draws, the planets counted at its allocation (visits and exposure) at that draw's exozodi
    level, with the same seed and planets the grid used, and weights each by eta / iNumPlanets so
    a star contributes its expected EECs. Returns pooled times (days) and weights.
    """
    dictMission = dict(dictParams["dictMission"])
    dictMission["listBandsCharacterization"] = \
        dictParams["dictBands"].get("listBandsCharacterization")
    listRows = dfTargets.to_dict("records")
    faAllocated = dictPerStar["faStarComp"][iaDraws] > 0
    dictTimes = {}
    for p in np.flatnonzero(faAllocated.any(axis=0)):
        i = int(dictGrid["iaStar"][p])
        dictRates = cp.fdictStarRates(listRows[i], dictBox,
                                      dictParams["dictBands"]["listBandsDetection"],
                                      dictParams["dictBands"]["dictBandCharacterization"],
                                      dictMission, dictGrid["iNumPlanets"], dictParams["fAlpha"],
                                      dictParams["fBeta"], iSeed + i)
        for jPos, j in enumerate(iaDraws):
            if faAllocated[jPos, p]:
                dictTimes[(j, p)] = faCountedCharDays(
                    dictRates, faZodiDraws[j][i], int(dictPerStar["faStarVisits"][j, p]),
                    dictPerStar["faStarTauS"][j, p], dictParams, dictMission)
    return fdictPoolFirstN(dictTimes, dictPerStar, iaDraws, fEta, iFirstN,
                           dictGrid["iNumPlanets"])


def faCountedCharDays(dictRates, fZodi, iVisits, fTauS, dictParams, dictMission):
    """Characterization times (days) of the planets counted at one allocation."""
    listBandsDet = dictParams["dictBands"]["listBandsDetection"]
    faTauDet = cp.faRequiredExposureTime(dictRates["listDetRates"], listBandsDet,
                                         listBandsDet[0]["fSignalToNoise"], dictMission, fZodi)
    faTauChar = cp.faCharacterizationTimeFromRates(
        dictRates["listCharRates"], dictRates["listCharOptions"], dictMission,
        dictRates["bBestPhase"], dictRates["iVisits"], fZodi)
    faBestDet, faBestChar, _ = cp.faCountedTimes(faTauDet, faTauChar,
                                                 cp.ffScienceTimeCap(dictMission),
                                                 int(dictMission.get("iRequiredDetections", 1)),
                                                 cp.ffPlanetCharacterizationCap(dictMission))
    bCounted = faBestDet[:, iVisits - 1] <= fTauS * (1.0 + 1e-9)
    return faBestChar[bCounted, iVisits - 1] / 86400.0


def fdictPoolFirstN(dictTimes, dictPerStar, iaDraws, fEta, iFirstN, iNumPlanets):
    """Take each draw's stars in priority order until N expected EECs, pooling planet times."""
    listTimes, listWeights = [], []
    for jPos, j in enumerate(iaDraws):
        faComp, faTime = dictPerStar["faStarComp"][j], dictPerStar["faStarTimeS"][j]
        faPriority = np.where(faTime > 0, faComp / np.maximum(faTime, 1e-30), -1.0)
        fAccrued = 0.0
        for p in np.argsort(-faPriority):
            faT = dictTimes.get((j, p))
            if faT is None or fAccrued >= iFirstN:
                continue
            fStar = fEta * faT.size / iNumPlanets
            fScale = min(1.0, (iFirstN - fAccrued) / fStar) if fStar > 0 else 0.0
            listTimes.append(faT)
            listWeights.append(np.full(faT.size, fScale * fEta / iNumPlanets))
            fAccrued += fScale * fStar
    return np.concatenate(listTimes), np.concatenate(listWeights)
