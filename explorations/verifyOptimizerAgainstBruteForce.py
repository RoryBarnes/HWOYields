#!/usr/bin/env python3
"""Check the equal-slope allocation against an exact optimum found by dynamic programming.

Every number in this project passes through the equal-slope optimizer, which has been tested for
properties -- budget respected, yield monotonic in time, envelope concave -- but never against a
known-correct answer. The survey problem is a multiple-choice knapsack: pick one (visits,
exposure) option per star to maximise summed completeness within the time budget. That has an
exact solution by dynamic programming on a discretized budget, so the two can be compared
directly on a problem small enough to solve both ways.

Two comparisons matter. Against the options the optimizer itself considers -- the vertices of
each star's upper concave envelope -- it should be near-exact. Against the FULL option set,
including the interior points the envelope discards, it may not be: equal-slope solves the
continuous relaxation, in which a star can time-share between two vertices, while a real survey
must pick one allocation per star. If the exact optimum routinely uses discarded interior points,
the envelope is throwing away yield.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import optimizer as opt  # noqa: E402


def flistToyStars(iStars, iVisits, iGrid, fTauCharS, iSeed):
    """A small synthetic target list with saturating, visit-improving completeness curves."""
    rng = np.random.default_rng(iSeed)
    faTauGridS = np.logspace(3.5, np.log10(40 * 86400.0), iGrid)
    listStars = []
    for _ in range(iStars):
        fScale = 10.0 ** rng.uniform(4.0, 6.2)
        fPeak = rng.uniform(0.25, 0.95)
        faComp = np.stack([fPeak * (1.0 - (1.0 - 0.5) ** (k + 1)) *
                           (1.0 - np.exp(-faTauGridS / fScale)) for k in range(iVisits)])
        listStars.append(dict(faTauGridS=faTauGridS, faComp=faComp,
                              faTauCharMeanS=np.full_like(faComp, fTauCharS),
                              fTauCharS=fTauCharS))
    return listStars


def fdictFullOptionSet(dictStar, fEtaEarth, dictMission):
    """Every (visits, exposure) option a star offers, before any envelope is taken."""
    fOverhead = dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"]
    fMult = dictMission["fWavefrontMultiplier"]
    faTau = np.asarray(dictStar["faTauGridS"])
    faComp2 = np.atleast_2d(dictStar["faComp"])
    faChar2 = np.atleast_2d(dictStar["faTauCharMeanS"])
    listCost, listComp = [np.zeros(1)], [np.zeros(1)]
    for k in range(faComp2.shape[0]):
        faCharTerm = np.where(faChar2[k] > 0.0, fMult * faChar2[k] + fOverhead, 0.0)
        listCost.append((k + 1) * (fMult * faTau + fOverhead) +
                        fEtaEarth * faComp2[k] * faCharTerm)
        listComp.append(faComp2[k])
    return np.concatenate(listCost), np.concatenate(listComp)


def fnExactOptimum(listOptions, fBudget, iBins):
    """Exact multiple-choice knapsack optimum by dynamic programming on a discretized budget."""
    fStep = fBudget / iBins
    faBest = np.zeros(iBins + 1)
    for faCost, faComp in listOptions:
        faIndex = np.minimum((faCost / fStep).astype(int), iBins)
        faNext = faBest.copy()
        for iCost, fComp in zip(faIndex, faComp):
            if iCost > iBins:
                continue
            faShifted = np.full(iBins + 1, -np.inf)
            faShifted[iCost:] = faBest[:iBins + 1 - iCost] + fComp
            faNext = np.maximum(faNext, faShifted)
        faBest = faNext
    return float(np.max(faBest))


def flistRealStars(sCompletenessPath, iStars, iSeed):
    """A random subset of the pipeline's actual completeness curves.

    The synthetic curves are smooth and saturating. The real ones are not: this project's
    completeness is bimodal across the target list, with near stars saturated and distant ones at
    zero, which is exactly the shape that could defeat an envelope-based method. Verifying on
    them as well as on the toy set is the point.
    """
    dictNpz = np.load(sCompletenessPath, allow_pickle=True)
    faTauGridS = dictNpz["faTauGridS"]
    faComp = dictNpz["faComp_canonical"]
    faChar = dictNpz["faTauCharMean_canonical"]
    rng = np.random.default_rng(iSeed)
    faUsable = np.where(faComp[:, -1, -1] > 0.01)[0]
    faPick = rng.choice(faUsable, size=min(iStars, faUsable.size), replace=False)
    return [dict(faTauGridS=faTauGridS, faComp=faComp[i], faTauCharMeanS=faChar[i],
                 fTauCharS=float(np.max(faChar[i]))) for i in faPick]


def fdictParseArgs():
    """Command-line configuration for the optimizer verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stars", type=int, default=14)
    p.add_argument("--visits", type=int, default=4)
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--bins", type=int, default=4000)
    p.add_argument("--budget-days", type=float, default=120.0)
    p.add_argument("--tau-char-days", type=float, default=8.0)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--completeness", default=None,
                   help="use real completeness curves from this npz instead of synthetic ones")
    p.add_argument("--out-json", default="optimizerVerification.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    listRows = []
    for iTrial in range(dictArgs["trials"]):
        dictMission = {"fSlewOverheadS": 3600.0, "fWavefrontOverheadS": 9720.0,
                       "fWavefrontMultiplier": 1.1,
                       "fTotalScienceTimeS": dictArgs["budget_days"] * 86400.0}
        listStars = (flistRealStars(dictArgs["completeness"], dictArgs["stars"], 100 + iTrial)
                     if dictArgs["completeness"]
                     else flistToyStars(dictArgs["stars"], dictArgs["visits"],
                                        dictArgs["grid"],
                                        dictArgs["tau_char_days"] * 86400.0, 100 + iTrial))
        dictOpt = opt.fdictOptimizeSurvey(listStars, 0.24, dictMission)
        listHull = [(c["faCost"], c["faComp"]) for c in
                    (opt.fdictStarCostCurve(s, 0.24, dictMission) for s in listStars)]
        listFull = [fdictFullOptionSet(s, 0.24, dictMission) for s in listStars]
        fExactHull = fnExactOptimum(listHull, dictMission["fTotalScienceTimeS"],
                                    dictArgs["bins"])
        fExactFull = fnExactOptimum(listFull, dictMission["fTotalScienceTimeS"],
                                    dictArgs["bins"])
        listRows.append({
            "iTrial": iTrial,
            "fEqualSlope": float(dictOpt["fSummedCompleteness"]),
            "fExactOnEnvelope": fExactHull, "fExactOnFullOptions": fExactFull,
            "fShortfallVsEnvelope": 1.0 - dictOpt["fSummedCompleteness"] / fExactHull
            if fExactHull else 0.0,
            "fShortfallVsFull": 1.0 - dictOpt["fSummedCompleteness"] / fExactFull
            if fExactFull else 0.0,
            "fTimeUsedFraction": float(dictOpt["fTotalTimeS"] /
                                       dictMission["fTotalScienceTimeS"])})
    faEnv = np.array([d["fShortfallVsEnvelope"] for d in listRows])
    faFull = np.array([d["fShortfallVsFull"] for d in listRows])
    dictOut = {"listRows": listRows,
               "fMedianShortfallVsEnvelope": float(np.median(faEnv)),
               "fMaxShortfallVsEnvelope": float(np.max(faEnv)),
               "fMedianShortfallVsFull": float(np.median(faFull)),
               "fMaxShortfallVsFull": float(np.max(faFull))}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'trial':>6}{'equal-slope':>13}{'exact(env)':>12}{'exact(full)':>13}"
          f"{'short env':>11}{'short full':>12}{'time used':>11}")
    for d in listRows:
        print(f"{d['iTrial']:>6}{d['fEqualSlope']:>13.4f}{d['fExactOnEnvelope']:>12.4f}"
              f"{d['fExactOnFullOptions']:>13.4f}{100*d['fShortfallVsEnvelope']:>10.2f}%"
              f"{100*d['fShortfallVsFull']:>11.2f}%{100*d['fTimeUsedFraction']:>10.1f}%")
    print(f"\nmedian shortfall vs envelope optimum : "
          f"{100*dictOut['fMedianShortfallVsEnvelope']:.2f}%")
    print(f"median shortfall vs full option set  : "
          f"{100*dictOut['fMedianShortfallVsFull']:.2f}%")


if __name__ == "__main__":
    main()
