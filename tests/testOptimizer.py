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
