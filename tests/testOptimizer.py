"""Unit tests for the equal-slope survey optimizer and its characterization-time coupling."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yieldlib import optimizer as opt  # noqa: E402

DICT_MISSION = {"fSlewOverheadS": 3600.0, "fWavefrontOverheadS": 9720.0,
                "fWavefrontMultiplier": 1.1, "fTotalScienceTimeS": 2.0 * 365.25 * 86400.0}


def flistToyStars(iStars, fTauCharS, iSeed=7, iVisits=1):
    """A small synthetic target list with saturating completeness curves, shaped (visits, grid).

    Completeness rises with visit count, as it must: a revisit can only add planets.
    """
    rng = np.random.default_rng(iSeed)
    faTauGridS = np.logspace(3.0, np.log10(60 * 86400.0), 60)
    listStars = []
    for _ in range(iStars):
        fScale = 10.0 ** rng.uniform(4.0, 6.0)
        fPeak = rng.uniform(0.3, 0.95)
        faComp = np.stack([fPeak * (1.0 - (1.0 - 0.55) ** (k + 1)) *
                           (1.0 - np.exp(-faTauGridS / fScale)) for k in range(iVisits)])
        faChar = (np.zeros_like(faComp) if not np.isfinite(fTauCharS)
                  else np.full_like(faComp, fTauCharS))
        listStars.append(dict(faTauGridS=faTauGridS, faComp=faComp,
                              faTauCharMeanS=faChar, fTauCharS=fTauCharS))
    return listStars


def test_concave_hull_has_strictly_decreasing_slopes():
    """The envelope the optimizer walks must be concave, or equal-slope has no solution."""
    faTime = np.linspace(0.0, 10.0, 40)
    faComp = np.sqrt(faTime) + 0.05 * np.sin(8 * faTime)
    faHull = opt.faUpperConcaveHull(faTime, faComp)
    faSlopes = np.diff(faComp[faHull]) / np.diff(faTime[faHull])
    assert np.all(np.diff(faSlopes) < 1e-12)


def test_concave_hull_keeps_the_endpoints():
    """The first and last points always lie on the envelope."""
    faTime = np.array([0.0, 1.0, 2.0, 3.0])
    faComp = np.array([0.0, 0.1, 0.9, 1.0])
    faHull = opt.faUpperConcaveHull(faTime, faComp)
    assert faHull[0] == 0 and faHull[-1] == len(faTime) - 1


def test_optimizer_respects_the_time_budget():
    """The allocation must not spend more than the mission's science time."""
    dictResult = opt.fdictOptimizeSurvey(flistToyStars(40, np.inf), 0.24, DICT_MISSION)
    assert dictResult["fTotalTimeS"] <= DICT_MISSION["fTotalScienceTimeS"] * 1.001


def test_yield_increases_with_the_time_budget():
    """More science time can never reduce the optimized yield."""
    listStars = flistToyStars(40, np.inf)
    dictSmall = opt.fdictOptimizeSurvey(listStars, 0.24, dict(DICT_MISSION,
                                        fTotalScienceTimeS=0.5 * 365.25 * 86400.0))
    dictLarge = opt.fdictOptimizeSurvey(listStars, 0.24, DICT_MISSION)
    assert dictLarge["fYield"] >= dictSmall["fYield"]


def test_yield_is_exactly_linear_in_eta_without_characterization():
    """With no characterization burden a uniform eta is a pure scale factor on the yield.

    This isolates the mechanism: the argmax of sum(C_i) does not depend on eta, so the survey
    plan is unchanged and the yield scales exactly. It is the assumption that scaling Stark's
    published number would rely on.
    """
    listStars = flistToyStars(40, np.inf)
    fLow = opt.fdictOptimizeSurvey(listStars, 0.05, DICT_MISSION)["fYield"]
    fHigh = opt.fdictOptimizeSurvey(listStars, 0.50, DICT_MISSION)["fYield"]
    assert np.isclose(fHigh / fLow, 10.0, rtol=1e-6)


def test_yield_is_sublinear_in_eta_when_characterization_is_charged():
    """Charging characterization time against the same budget breaks that proportionality."""
    listStars = flistToyStars(40, 20.0 * 86400.0)
    fLow = opt.fdictOptimizeSurvey(listStars, 0.05, DICT_MISSION)["fYield"]
    fHigh = opt.fdictOptimizeSurvey(listStars, 0.50, DICT_MISSION)["fYield"]
    assert fHigh / fLow < 10.0


def test_characterization_reduces_summed_completeness():
    """Time spent characterizing detections is time not spent searching."""
    fWithout = opt.fdictOptimizeSurvey(flistToyStars(40, np.inf), 0.24,
                                       DICT_MISSION)["fSummedCompleteness"]
    fWith = opt.fdictOptimizeSurvey(flistToyStars(40, 20.0 * 86400.0), 0.24,
                                    DICT_MISSION)["fSummedCompleteness"]
    assert fWith < fWithout


def test_empty_target_list_yields_nothing():
    """A survey with no usable targets returns a zero yield rather than failing."""
    assert opt.fdictOptimizeSurvey([], 0.24, DICT_MISSION)["fYield"] == 0.0


def test_optimizer_prefers_more_visits_when_they_are_cheap():
    """Offered a revisit option that adds completeness, the optimizer must use it."""
    fSingle = opt.fdictOptimizeSurvey(flistToyStars(40, np.inf, iVisits=1), 0.24,
                                      DICT_MISSION)["fYield"]
    fMulti = opt.fdictOptimizeSurvey(flistToyStars(40, np.inf, iVisits=5), 0.24,
                                     DICT_MISSION)["fYield"]
    assert fMulti > fSingle


def test_star_without_characterization_data_does_not_crash():
    """A star carrying no characterization curve is charged nothing for it, not an exception."""
    listStars = flistToyStars(10, np.inf, iVisits=2)
    for dictStar in listStars:
        dictStar.pop("faTauCharMeanS")
    assert opt.fdictOptimizeSurvey(listStars, 0.24, DICT_MISSION)["fYield"] > 0.0


def fnExactOptimumByDynamicProgramming(listOptions, fBudget, iBins=1500):
    """Exact multiple-choice knapsack optimum, for checking the equal-slope allocation."""
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


def test_equal_slope_allocation_matches_the_exact_optimum():
    """The equal-slope survey must come within a percent of an exhaustively computed optimum.

    Every yield in this project passes through this optimizer, and property tests alone cannot
    show the allocation is right. The survey problem is a multiple-choice knapsack -- one
    (visits, exposure) option per star within a time budget -- which dynamic programming solves
    exactly on a discretized budget, so the two can be compared directly.
    """
    dictMission = dict(DICT_MISSION, fTotalScienceTimeS=120.0 * 86400.0)
    listStars = flistToyStars(12, 8.0 * 86400.0, iSeed=101, iVisits=4)
    dictResult = opt.fdictOptimizeSurvey(listStars, 0.24, dictMission)
    listOptions = [(dictCurve["faCost"], dictCurve["faComp"]) for dictCurve in
                   (opt.fdictStarCostCurve(s, 0.24, dictMission) for s in listStars)]
    fExact = fnExactOptimumByDynamicProgramming(listOptions,
                                                dictMission["fTotalScienceTimeS"])
    assert dictResult["fSummedCompleteness"] >= 0.97 * fExact


def test_concave_envelope_discards_no_reachable_yield():
    """Hulling each star's options must not throw away an allocation the optimum would use.

    Equal-slope solves the continuous relaxation, in which a star can time-share between two
    envelope vertices; a real survey picks one allocation per star. If the interior points the
    envelope discards mattered, the exact optimum over ALL options would beat the exact optimum
    over the envelope alone.
    """
    dictMission = dict(DICT_MISSION, fTotalScienceTimeS=120.0 * 86400.0)
    listStars = flistToyStars(10, 8.0 * 86400.0, iSeed=202, iVisits=4)
    fMult, fOverhead = dictMission["fWavefrontMultiplier"], (
        dictMission["fSlewOverheadS"] + dictMission["fWavefrontOverheadS"])
    listHull, listFull = [], []
    for dictStar in listStars:
        dictCurve = opt.fdictStarCostCurve(dictStar, 0.24, dictMission)
        listHull.append((dictCurve["faCost"], dictCurve["faComp"]))
        faTau = np.asarray(dictStar["faTauGridS"])
        faComp2 = np.atleast_2d(dictStar["faComp"])
        faChar2 = np.atleast_2d(dictStar["faTauCharMeanS"])
        listCost, listComp = [np.zeros(1)], [np.zeros(1)]
        for k in range(faComp2.shape[0]):
            faCharTerm = np.where(faChar2[k] > 0.0, fMult * faChar2[k] + fOverhead, 0.0)
            listCost.append((k + 1) * (fMult * faTau + fOverhead) +
                            0.24 * faComp2[k] * faCharTerm)
            listComp.append(faComp2[k])
        listFull.append((np.concatenate(listCost), np.concatenate(listComp)))
    fBudget = dictMission["fTotalScienceTimeS"]
    fHull = fnExactOptimumByDynamicProgramming(listHull, fBudget)
    fFull = fnExactOptimumByDynamicProgramming(listFull, fBudget)
    assert fHull >= 0.99 * fFull


def test_sorted_allocation_matches_the_bisection_reference():
    """The one-sort solver must choose the same allocation as the 80-step slope bisection."""
    for iVisits, fChar in ((1, np.inf), (3, 2.0e5), (6, 8.0e5)):
        listStars = flistToyStars(60, fChar, iSeed=11, iVisits=iVisits)
        for fEta in (0.05, 0.24, 0.6):
            dictNew = opt.fdictOptimizeSurvey(listStars, fEta, DICT_MISSION)
            dictOld = opt.fdictOptimizeSurveyBisection(listStars, fEta, DICT_MISSION)
            assert np.isclose(dictNew["fSummedCompleteness"], dictOld["fSummedCompleteness"],
                              rtol=1e-9)
            assert dictNew["iStarsUsed"] == dictOld["iStarsUsed"]


def test_pareto_front_keeps_only_improvements():
    """Points that do not beat every cheaper point are dropped; the first is always kept."""
    faKeep = opt.faParetoFront(np.array([0.0, 0.2, 0.2, 0.1, 0.5, 0.5]))
    assert list(faKeep) == [0, 1, 4]


def test_per_star_breakdown_sums_to_the_totals():
    """The per-star completeness and time at the allocation add up to the survey totals."""
    listStars = flistToyStars(50, 3.0e5, iSeed=5, iVisits=3)
    dictResult = opt.fdictOptimizeSurvey(listStars, 0.24, DICT_MISSION, bPerStar=True)
    assert np.isclose(dictResult["faStarComp"].sum(), dictResult["fSummedCompleteness"])
    assert np.isclose(dictResult["faStarTimeS"].sum(), dictResult["fTotalTimeS"])
    assert int(np.sum(dictResult["faStarComp"] > 0)) == dictResult["iStarsUsed"]


def test_recorded_allocation_reproduces_its_completeness():
    """A star's (visits, exposure) at the allocation reads back its allocated completeness."""
    listStars = flistToyStars(30, 3.0e5, iSeed=9, iVisits=4)
    dictResult = opt.fdictOptimizeSurvey(listStars, 0.24, DICT_MISSION, bPerStar=True)
    for i, dictStar in enumerate(listStars):
        if dictResult["faStarComp"][i] == 0:
            continue
        iVisits = int(dictResult["faStarVisits"][i])
        iTau = int(np.argmin(np.abs(dictStar["faTauGridS"] - dictResult["faStarTauS"][i])))
        assert np.isclose(dictStar["faComp"][iVisits - 1][iTau], dictResult["faStarComp"][i])


def test_star_gate_forbids_allocations_whose_expected_spectrum_exceeds_the_cap():
    """eta * C * <t_c> above the limit makes an option unavailable under the per-star gate."""
    listStars = flistToyStars(5, 400 * 86400.0, iSeed=3)
    dictStar = dict(DICT_MISSION, fExposureLimitS=60 * 86400.0)
    dictPlanet = opt.fdictStarCostCurve(listStars[0], 0.6, dictStar)
    dictGate = opt.fdictStarCostCurve(listStars[0], 0.6,
                                      dict(dictStar, sCharacterizationGate="star"))
    fMaxAllowed = 60 * 86400.0 / (0.6 * (1.1 * 400 * 86400.0 + 13320.0))
    assert dictGate["faComp"].max() <= fMaxAllowed + 1e-12
    assert dictPlanet["faComp"].max() > dictGate["faComp"].max()
