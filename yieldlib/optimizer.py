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
    """Time cost and completeness at every reachable allocation for one star, including its hull.

    Cost per visit is tau' * tau + tau_slew + tau_WFC; the characterization burden
    eta * C * (tau' * tau_char + overhead) is added because detections must be followed up.
    """
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    faTau = np.concatenate(([0.0], dictStar["faTauGridS"]))
    faComp = np.concatenate(([0.0], dictStar["faComp"]))
    faCost = np.where(faTau > 0.0, fMult * faTau + fOverhead, 0.0)
    if np.isfinite(dictStar["fTauCharS"]):
        faCost = faCost + fEtaEarth * faComp * (fMult * dictStar["fTauCharS"] + fOverhead)
    faHull = faUpperConcaveHull(faCost, faComp)
    return dict(faCost=faCost[faHull], faComp=faComp[faHull])


def fdictAllocateAtSlope(listCurves, fSlope):
    """Pick each star's envelope vertex where the marginal return first falls below fSlope."""
    fTotalTime, fTotalComp, iStarsUsed = 0.0, 0.0, 0
    for dictCurve in listCurves:
        faCost, faComp = dictCurve["faCost"], dictCurve["faComp"]
        faSlopes = np.diff(faComp) / np.diff(faCost)
        iVertex = int(np.searchsorted(-faSlopes, -fSlope, side="right"))
        if iVertex > 0:
            fTotalTime += faCost[iVertex]
            fTotalComp += faComp[iVertex]
            iStarsUsed += 1
    return dict(fTotalTimeS=fTotalTime, fSummedCompleteness=fTotalComp, iStarsUsed=iStarsUsed)


def fdictOptimizeSurvey(listStars, fEtaEarth, dictMission, iBisectionSteps=80):
    """Bisect on the common slope until the allocation exactly fills the science time budget."""
    listCurves = [fdictStarCostCurve(s, fEtaEarth, dictMission) for s in listStars]
    listCurves = [c for c in listCurves if len(c["faCost"]) > 1]
    if not listCurves:
        return dict(fYield=0.0, fSummedCompleteness=0.0, iStarsUsed=0, fTotalTimeS=0.0)
    fLo, fHi = 1e-30, 1.0
    for _ in range(iBisectionSteps):
        fMid = np.sqrt(fLo * fHi)
        dictTrial = fdictAllocateAtSlope(listCurves, fMid)
        if dictTrial["fTotalTimeS"] > dictMission["fTotalScienceTimeS"]:
            fLo = fMid
        else:
            fHi = fMid
    dictFinal = fdictAllocateAtSlope(listCurves, fHi)
    dictFinal["fYield"] = fEtaEarth * dictFinal["fSummedCompleteness"]
    dictFinal["fSlope"] = fHi
    return dictFinal
