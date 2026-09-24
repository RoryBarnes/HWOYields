"""Altruistic Yield Optimization: the equal-slope survey allocation of Stark et al. (2014, 2019).

The optimum of sum(C_i) at fixed total time has dC_i/dT_i equal across all observations. Because
completeness is a Monte Carlo step function, the reachable allocations are the vertices of its
upper concave envelope, so the sweep runs over envelope vertices rather than a continuous tau.
Characterization time enters each star's cost through C_i itself, which is what makes the yield
a non-linear function of eta_Earth (Stark et al. 2024 Sec. 3.5: eta_Earth is "actionable").
"""

import numpy as np


def faUpperConcaveHull(faTime, faComp):
    """Indices of the upper concave envelope of (time, completeness) points, time ascending."""
    listHull = []
    for i in range(len(faTime)):
        while len(listHull) >= 2:
            iA, iB = listHull[-2], listHull[-1]
            fCross = ((faTime[iB] - faTime[iA]) * (faComp[i] - faComp[iA]) -
                      (faComp[iB] - faComp[iA]) * (faTime[i] - faTime[iA]))
            if fCross >= 0:
                listHull.pop()
            else:
                break
        listHull.append(i)
    return np.array(listHull, dtype=int)


def faParetoFront(faCompByCost):
    """Indices of points that raise completeness over every cheaper point, first point kept.

    A point no better than something cheaper can only sit on the envelope's flat or falling tail,
    which no allocation at a positive slope ever reaches, so dropping it leaves every allocation
    unchanged. It shrinks the envelope's input from ~900 pooled (visit, exposure) points to the
    ~100 distinct completeness steps, which is what makes re-optimizing per exozodi draw cheap.
    """
    faComp = np.asarray(faCompByCost, dtype=float)
    faPrevMax = np.concatenate(([-np.inf], np.maximum.accumulate(faComp)[:-1]))
    return np.flatnonzero(faComp > faPrevMax)


def fdictStarCostCurve(dictStar, fEtaEarth, dictMission):
    """Reachable (time, completeness) options for one star, over exposure AND visit count.

    Each visit count k offers its own curve: k visits of exposure tau cost k(tau' tau + overhead)
    and deliver C_k(tau). Pooling every (k, tau) pair and taking the upper concave envelope lets
    the equal-slope sweep choose both, which is what AYO does. Restricting to k = 1, as an
    earlier version did, forces the survey to buy completeness only by staring longer at each
    star, when revisiting is often the cheaper way to catch a planet that was behind the inner
    working angle.

    Characterization is charged as eta * C * (tau' * tau_char + overhead) and is paid once per
    expected detection, not once per visit. Under sCharacterizationGate "star" an option whose
    expected characterization time exceeds the time limit is not allowed (see
    completeness.ffPlanetCharacterizationCap).
    """
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    faTau = np.asarray(dictStar["faTauGridS"])
    faComp2 = np.atleast_2d(dictStar["faComp"])
    if not np.any(faComp2 > 0):
        return dict(faCost=np.zeros(1), faComp=np.zeros(1), faCompYield=np.zeros(1),
                    faTauCharMeanS=np.zeros(1), iaVisits=np.zeros(1, dtype=int),
                    faTauS=np.zeros(1))
    faChar2 = (np.zeros_like(faComp2) if dictStar.get("faTauCharMeanS") is None
               else np.atleast_2d(dictStar["faTauCharMeanS"]))
    faYield2 = np.atleast_2d(dictStar.get("faCompYield", dictStar["faComp"]))
    listCost, listComp, listYield = [np.zeros(1)], [np.zeros(1)], [np.zeros(1)]
    listChar, listVisits, listTau = [np.zeros(1)], [np.zeros(1, dtype=int)], [np.zeros(1)]
    bStarCap = dictMission.get("sCharacterizationGate", "planet") == "star"
    for k in range(faComp2.shape[0]):
        faCost = (k + 1) * (fMult * faTau + fOverhead)
        faCharTerm = np.where(faChar2[k] > 0.0, fMult * faChar2[k] + fOverhead, 0.0)
        faCharCost = fEtaEarth * faComp2[k] * faCharTerm
        listCost.append(faCost + faCharCost)
        listComp.append(np.where(bStarCap & (faCharCost > dictMission.get("fExposureLimitS",
                                                                          np.inf)),
                                 0.0, faComp2[k]))
        listYield.append(faYield2[k])
        listChar.append(faChar2[k])
        listVisits.append(np.full(faTau.size, k + 1))
        listTau.append(faTau)
    faVisits, faTauAll = np.concatenate(listVisits), np.concatenate(listTau)
    faCost = np.concatenate(listCost)
    faComp = np.concatenate(listComp)
    faYield = np.concatenate(listYield)
    faChar = np.concatenate(listChar)
    faOrder = np.argsort(faCost, kind="stable")
    faCost, faComp = faCost[faOrder], faComp[faOrder]
    faYield, faChar = faYield[faOrder], faChar[faOrder]
    faVisits, faTauAll = faVisits[faOrder], faTauAll[faOrder]
    faKeep = faParetoFront(faComp)
    faHull = faKeep[faUpperConcaveHull(faCost[faKeep], faComp[faKeep])]
    return dict(faCost=faCost[faHull], faComp=faComp[faHull], faCompYield=faYield[faHull],
                faTauCharMeanS=faChar[faHull], iaVisits=faVisits[faHull], faTauS=faTauAll[faHull])


def fdictAllocateAtSlope(listCurves, fSlope):
    """Pick each star's envelope vertex where the marginal return first falls below fSlope."""
    fTotalTime, fTotalComp, fTotalYield, iStarsUsed = 0.0, 0.0, 0.0, 0
    for dictCurve in listCurves:
        faCost, faComp = dictCurve["faCost"], dictCurve["faComp"]
        faSlopes = np.diff(faComp) / np.diff(faCost)
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex > 0:
            fTotalTime += faCost[iVertex]
            fTotalComp += faComp[iVertex]
            fTotalYield += dictCurve["faCompYield"][iVertex]
            iStarsUsed += 1
    return dict(fTotalTimeS=fTotalTime, fSummedCompleteness=fTotalComp,
                fSummedCompletenessYield=fTotalYield, iStarsUsed=iStarsUsed)


def fdictSegments(listCurves):
    """Every envelope segment of every star as flat arrays, positive slopes only."""
    listSlope, listCost, listComp, listYield, listStar = [], [], [], [], []
    for iStar, dictCurve in enumerate(listCurves):
        faSlopes = np.diff(dictCurve["faComp"]) / np.diff(dictCurve["faCost"])
        iPositive = int(np.sum(faSlopes > 0))
        listSlope.append(faSlopes[:iPositive])
        listCost.append(np.diff(dictCurve["faCost"])[:iPositive])
        listComp.append(np.diff(dictCurve["faComp"])[:iPositive])
        listYield.append(np.diff(dictCurve["faCompYield"])[:iPositive])
        listStar.append(np.full(iPositive, iStar))
    return {k: np.concatenate(v) for k, v in (("faSlope", listSlope), ("faCost", listCost),
                                               ("faComp", listComp), ("faYield", listYield),
                                               ("iaStar", listStar))}


def fdictOptimizeSurvey(listStars, fEtaEarth, dictMission, bPerStar=False):
    """Equal-slope allocation that exactly fills the science time budget, solved in one sort.

    Each star's envelope is concave, so its segments come in decreasing slope order, and the
    allocation at any common slope lambda is every segment steeper than lambda -- a prefix of all
    segments sorted by slope. The budget-filling lambda is therefore found by sorting once and
    taking the longest prefix whose cumulative cost fits, rather than by bisecting on lambda,
    which re-walked every star 80 times. The prefix stops at a tie group boundary so that it is a
    set of the form {slope >= lambda}, which is what the bisection returned; both agree to the
    last segment (tests/testOptimizer.py).

    bPerStar adds, for every input star, the completeness (planning and albedo-drawn), the time
    charged (detection, overheads and budgeted characterization), the mean characterization
    time per counted planet, and the allocation itself (visit count and exposure per visit).
    """
    listAll = [fdictStarCostCurve(s, fEtaEarth, dictMission) for s in listStars]
    iaInput = [i for i, c in enumerate(listAll) if len(c["faCost"]) > 1]
    listCurves = [listAll[i] for i in iaInput]
    dictSeg = fdictSegments(listCurves) if listCurves else {"faSlope": np.zeros(0)}
    if dictSeg["faSlope"].size == 0:
        return dict(fYield=0.0, fYieldPlanning=0.0, fSummedCompleteness=0.0,
                    fSummedCompletenessYield=0.0, iStarsUsed=0, fTotalTimeS=0.0, fSlope=1.0)
    faOrder = np.argsort(-dictSeg["faSlope"], kind="stable")
    faSlope = dictSeg["faSlope"][faOrder]
    iTake = fiPrefixWithinBudget(faSlope, np.cumsum(dictSeg["faCost"][faOrder]),
                                 dictMission["fTotalScienceTimeS"])
    faTaken = faOrder[:iTake]
    fSlope = (float(np.sqrt(faSlope[iTake - 1] * faSlope[iTake])) if 0 < iTake < faSlope.size
              else (float(faSlope[0]) * 2.0 if iTake == 0 else 1e-30))
    fComp = float(np.sum(dictSeg["faComp"][faTaken]))
    fCompYield = float(np.sum(dictSeg["faYield"][faTaken]))
    dictOut = dict(fTotalTimeS=float(np.sum(dictSeg["faCost"][faTaken])),
                   fSummedCompleteness=fComp, fSummedCompletenessYield=fCompYield,
                   iStarsUsed=int(np.unique(dictSeg["iaStar"][faTaken]).size),
                   fYieldPlanning=fEtaEarth * fComp, fYield=fEtaEarth * fCompYield,
                   fSlope=fSlope)
    if bPerStar:
        dictOut.update(fdictPerStar(listCurves, iaInput, len(listStars), dictSeg["iaStar"][faTaken]))
    return dictOut


def fdictPerStar(listCurves, iaInput, iStars, iaTakenStar):
    """Each input star's envelope vertex at the allocation, from its count of taken segments."""
    iaVertex = np.bincount(iaTakenStar, minlength=len(listCurves))
    dictOut = {k: np.zeros(iStars) for k in ("faStarComp", "faStarCompYield", "faStarTimeS",
                                               "faStarTauCharMeanS", "faStarVisits",
                                               "faStarTauS")}
    for iCurve, iVertex in enumerate(iaVertex):
        if iVertex == 0:
            continue
        dictCurve, i = listCurves[iCurve], iaInput[iCurve]
        dictOut["faStarComp"][i] = dictCurve["faComp"][iVertex]
        dictOut["faStarCompYield"][i] = dictCurve["faCompYield"][iVertex]
        dictOut["faStarTimeS"][i] = dictCurve["faCost"][iVertex]
        dictOut["faStarTauCharMeanS"][i] = dictCurve["faTauCharMeanS"][iVertex]
        dictOut["faStarVisits"][i] = dictCurve["iaVisits"][iVertex]
        dictOut["faStarTauS"][i] = dictCurve["faTauS"][iVertex]
    return dictOut


def fiPrefixWithinBudget(faSlopeSorted, faCumCost, fBudget):
    """Length of the longest {slope >= lambda} prefix whose cumulative cost fits the budget."""
    iTake = int(np.searchsorted(faCumCost, fBudget, side="right"))
    while 0 < iTake < faSlopeSorted.size and faSlopeSorted[iTake] == faSlopeSorted[iTake - 1]:
        iTake -= 1
    return iTake


def fdictOptimizeSurveyBisection(listStars, fEtaEarth, dictMission, iBisectionSteps=80):
    """The original bisection on the common slope, kept as the reference the sort must match."""
    listCurves = [fdictStarCostCurve(s, fEtaEarth, dictMission) for s in listStars]
    listCurves = [c for c in listCurves if len(c["faCost"]) > 1]
    if not listCurves:
        return dict(fYield=0.0, fYieldPlanning=0.0, fSummedCompleteness=0.0,
                    fSummedCompletenessYield=0.0, iStarsUsed=0, fTotalTimeS=0.0)
    fLo, fHi = 1e-30, 1.0
    for _ in range(iBisectionSteps):
        fMid = np.sqrt(fLo * fHi)
        dictTrial = fdictAllocateAtSlope(listCurves, fMid)
        if dictTrial["fTotalTimeS"] > dictMission["fTotalScienceTimeS"]:
            fLo = fMid
        else:
            fHi = fMid
    dictFinal = fdictAllocateAtSlope(listCurves, fHi)
    dictFinal["fYieldPlanning"] = fEtaEarth * dictFinal["fSummedCompleteness"]
    dictFinal["fYield"] = fEtaEarth * dictFinal["fSummedCompletenessYield"]
    dictFinal["fSlope"] = fHi
    return dictFinal
