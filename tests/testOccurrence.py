"""Unit tests for the occurrence-rate integrals, including the published eta_Earth anchor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yieldlib import occurrence as oc  # noqa: E402

DICT_CANONICAL = {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
                  "fHzInnerAu": 0.95, "fHzOuterAu": 1.67}
DICT_REDEFINED = {"sRadiusMode": "mass", "fMassLoEarth": 0.5, "fMassHiEarth": 2.0,
                  "fMassRadiusExponent": 0.27, "fHzInnerAu": 0.96, "fHzOuterAu": 1.20}


def test_sag13_reproduces_the_published_eta_earth():
    """SAG13's small-planet parameters over the HabEx/LUVOIR EEC box must give eta_Earth = 0.24.

    This is the anchor for the whole analysis: Stark et al. (2024) Table 1 adopts 0.24 for
    exactly this box, so a rederivation that misses it has the population model wrong.
    """
    fEta = oc.fnOccurrenceAnalytic(DICT_CANONICAL, 0.38, -0.19, 0.26)
    assert np.isclose(fEta, 0.24, rtol=0.02)


@pytest.mark.parametrize("dictBox", [DICT_CANONICAL, DICT_REDEFINED])
@pytest.mark.parametrize("faTheta", [(0.38, -0.19, 0.26), (0.5, 0.3, -0.1), (0.2, -0.6, 0.55)])
def test_analytic_integral_matches_numerical_quadrature(dictBox, faTheta):
    """The closed form used by the MCMC must equal the numerically integrated definition."""
    fQuad = oc.fnOccurrence(**oc.fdictBoxFromConfig(dictBox), fGamma=faTheta[0],
                            fAlpha=faTheta[1], fBeta=faTheta[2])
    fAnalytic = oc.fnOccurrenceAnalytic(dictBox, *faTheta)
    assert np.isclose(fQuad, fAnalytic, rtol=1e-10)


def test_occurrence_is_linear_in_the_normalisation():
    """Gamma is a pure scale factor, which is why it cancels from the box ratio."""
    fOne = oc.fnOccurrenceAnalytic(DICT_CANONICAL, 1.0, -0.19, 0.26)
    fThree = oc.fnOccurrenceAnalytic(DICT_CANONICAL, 3.0, -0.19, 0.26)
    assert np.isclose(fThree, 3.0 * fOne)


def test_box_ratio_is_independent_of_the_normalisation():
    """The redefined/canonical ratio must not move when Gamma changes."""
    listRatios = [oc.fnOccurrenceAnalytic(DICT_REDEFINED, fG, -0.19, 0.26) /
                  oc.fnOccurrenceAnalytic(DICT_CANONICAL, fG, -0.19, 0.26)
                  for fG in (0.1, 0.38, 2.0)]
    assert np.allclose(listRatios, listRatios[0], rtol=1e-12)


def test_redefined_box_is_about_one_fifth_of_the_canonical_box():
    """The headline population result, pinned so a change in the box definitions is caught."""
    fRatio = (oc.fnOccurrenceAnalytic(DICT_REDEFINED, 0.38, -0.19, 0.26) /
              oc.fnOccurrenceAnalytic(DICT_CANONICAL, 0.38, -0.19, 0.26))
    assert np.isclose(fRatio, 0.2037, rtol=0.01)


def test_narrowing_the_habitable_zone_dominates_the_reduction():
    """Shrinking the HZ removes more occurrence than tightening the radius range does."""
    dictHzOnly = dict(DICT_CANONICAL, fHzInnerAu=0.96, fHzOuterAu=1.20)
    fHzFactor = (oc.fnOccurrenceAnalytic(dictHzOnly, 0.38, -0.19, 0.26) /
                 oc.fnOccurrenceAnalytic(DICT_CANONICAL, 0.38, -0.19, 0.26))
    fRadiusFactor = (oc.fnOccurrenceAnalytic(DICT_REDEFINED, 0.38, -0.19, 0.26) /
                     oc.fnOccurrenceAnalytic(dictHzOnly, 0.38, -0.19, 0.26))
    assert fHzFactor < fRadiusFactor


def test_mass_radius_conversion_bounds():
    """R = M^0.27 maps the 0.5-2 Earth-mass box onto 0.83-1.21 Earth radii."""
    assert np.isclose(0.5 ** 0.27, 0.8293, rtol=1e-3)
    assert np.isclose(2.0 ** 0.27, 1.2058, rtol=1e-3)


def test_radius_integral_handles_the_flat_limit():
    """At alpha = 0 the radius integral degenerates to a log ratio."""
    assert np.isclose(oc.fnRadiusIntegral(1.0, np.e, 0.0), 1.0)


def test_empty_radius_range_contributes_nothing():
    """An inverted radius range yields zero rather than a negative occurrence."""
    assert oc.fnRadiusIntegral(2.0, 1.0, -0.19) == 0.0
