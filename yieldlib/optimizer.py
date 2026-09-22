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


def fdictStarCostCurve(dictStar, fEtaEarth, dictMission):
    """Reachable (time, completeness) options for one star, over exposure AND visit count.

    Each visit count k offers its own curve: k visits of exposure tau cost k(tau' tau + overhead)
    and deliver C_k(tau). Pooling every (k, tau) pair and taking the upper concave envelope lets
    the equal-slope sweep choose both, which is what AYO does. Restricting to k = 1, as an
    earlier version did, forces the survey to buy completeness only by staring longer at each
    star, when revisiting is often the cheaper way to catch a planet that was behind the inner
    working angle.

    Characterization is charged as eta * C * (tau' * tau_char + overhead) and is paid once per
    expected detection, not once per visit.
    """
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    faTau = np.asarray(dictStar["faTauGridS"])
    faComp2 = np.atleast_2d(dictStar["faComp"])
    faChar2 = (np.zeros_like(faComp2) if dictStar.get("faTauCharMeanS") is None
               else np.atleast_2d(dictStar["faTauCharMeanS"]))
    faYield2 = np.atleast_2d(dictStar.get("faCompYield", dictStar["faComp"]))
    listCost, listComp, listYield = [np.zeros(1)], [np.zeros(1)], [np.zeros(1)]
    for k in range(faComp2.shape[0]):
        faCost = (k + 1) * (fMult * faTau + fOverhead)
        faCharTerm = np.where(faChar2[k] > 0.0, fMult * faChar2[k] + fOverhead, 0.0)
        listCost.append(faCost + fEtaEarth * faComp2[k] * faCharTerm)
        listComp.append(faComp2[k])
        listYield.append(faYield2[k])
    faCost = np.concatenate(listCost)
    faComp = np.concatenate(listComp)
    faYield = np.concatenate(listYield)
    faOrder = np.argsort(faCost, kind="stable")
    faCost, faComp, faYield = faCost[faOrder], faComp[faOrder], faYield[faOrder]
    faHull = faUpperConcaveHull(faCost, faComp)
    return dict(faCost=faCost[faHull], faComp=faComp[faHull], faCompYield=faYield[faHull])


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


def fdictOptimizeSurvey(listStars, fEtaEarth, dictMission, iBisectionSteps=80):
    """Bisect on the common slope until the allocation exactly fills the science time budget."""
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
