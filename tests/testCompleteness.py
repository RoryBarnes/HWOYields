"""Unit tests for EEC injection, orbital projection and the completeness curve."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yieldlib import completeness as cp  # noqa: E402

DICT_BOX = {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
            "fHzInnerAu": 0.95, "fHzOuterAu": 1.67}
DICT_MISSION = {
    "fDiameterM": 6.0, "fIwaLamD": 3.5, "fOwaLamD": 32.0, "fCoreThroughputMax": 0.45,
    "fContrastFloor": 1e-10, "fNoiseFloorDeltaMag": 26.5, "fContaminationThroughput": 0.95,
    "fApertureRadiusLamD": 0.7, "fGeometricAlbedo": 0.2, "fZodiMagArcsec2": 23.0,
    "fExozodiMagArcsec2": 22.0, "fExozodiLevel": 3.0, "fDarkCurrent": 3e-5,
    "fReadNoise": 0.0, "fClockInducedCharge": 1.3e-3, "fQuantumEfficiency": 0.9,
    "fDetectiveQuantumEfficiency": 0.75, "fExposureLimitS": 60.0 * 86400.0,
    "fThroughputCalibration": 1.0}
LIST_BANDS_DET = [{"fLambdaM": 550e-9, "fOpticalThroughput": 0.34, "fBandwidthFraction": 0.20,
                   "fSignalToNoise": 7.0, "iNumPixels": 4}]
DICT_BAND_CHAR = {"fLambdaM": 1000e-9, "fOpticalThroughput": 0.23,
                  "fBandwidthFraction": 1.0 / 140.0, "fSignalToNoise": 5.0, "iNumPixels": 96}


def fdictSolarTwin(fDistancePc):
    """A Sun-like star at the requested distance."""
    return {"fTeffK": 5772.0, "fRadiusRsun": 1.0, "fLuminosityLsun": 1.0,
            "fDistancePc": fDistancePc}


def test_power_law_sampler_matches_its_analytic_quantiles():
    """The inverse-CDF draw must reproduce the density it claims to sample."""
    rng = np.random.default_rng(3)
    faU = rng.random(200000)
    faDraw = cp.faSamplePowerLawInLog(1.0, 4.0, 0.5, faU)
    fMedian = cp.faSamplePowerLawInLog(1.0, 4.0, 0.5, np.array([0.5]))[0]
    assert np.isclose(np.median(faDraw), fMedian, rtol=0.01)
    assert faDraw.min() >= 1.0 and faDraw.max() <= 4.0


def test_power_law_sampler_handles_the_flat_limit():
    """At index zero the draw is log-uniform between the bounds."""
    faDraw = cp.faSamplePowerLawInLog(1.0, np.e, 0.0, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(faDraw, [1.0, np.sqrt(np.e), np.e])


def test_projected_separation_never_exceeds_the_orbital_radius():
    """Projection can only shorten a circular orbit's apparent separation."""
    rng = np.random.default_rng(11)
    dictPlanets = cp.fdictInjectPlanets(DICT_BOX, 1.0, 5000, -0.19, 0.26, rng)
    dictGeom = cp.fdictProjectOrbits(dictPlanets)
    assert np.all(dictGeom["faSepAu"] <= dictPlanets["faAxisAu"][:, None] + 1e-12)


def test_phase_factor_stays_within_lambertian_bounds():
    """The Lambertian phase function is bounded by zero and unity."""
    rng = np.random.default_rng(12)
    dictPlanets = cp.fdictInjectPlanets(DICT_BOX, 1.0, 5000, -0.19, 0.26, rng)
    faPhase = cp.fdictProjectOrbits(dictPlanets)["faPhase"]
    assert np.all(faPhase >= 0.0) and np.all(faPhase <= 1.0)


def test_earth_sun_flux_ratio_at_quadrature():
    """An Earth analogue at quadrature has a flux ratio near 1.1e-10, the canonical value."""
    fRatio = DICT_MISSION["fGeometricAlbedo"] * (1.0 / np.pi) * (cp.F_REARTH_AU / 1.0) ** 2
    assert np.isclose(fRatio, 1.1e-10, rtol=0.1)


def test_injected_planets_stay_inside_the_box():
    """Semi-major axes and radii must respect the selection box they were drawn from."""
    rng = np.random.default_rng(5)
    dictPlanets = cp.fdictInjectPlanets(DICT_BOX, 1.0, 20000, -0.19, 0.26, rng)
    assert dictPlanets["faAxisScaled"].min() >= DICT_BOX["fHzInnerAu"] - 1e-12
    assert dictPlanets["faAxisScaled"].max() <= DICT_BOX["fHzOuterAu"] + 1e-12
    assert dictPlanets["faRadiusEarth"].max() <= DICT_BOX["fRadiusMaxEarth"] + 1e-12


def test_completeness_is_monotonic_and_bounded():
    """C(tau) is a cumulative fraction, so it never decreases and never exceeds one."""
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 120)
    dictResult = cp.fdictStarCompleteness(fdictSolarTwin(5.0), DICT_BOX, LIST_BANDS_DET,
                                          DICT_BAND_CHAR, DICT_MISSION, faTauGridS,
                                          3000, -0.19, 0.26, 42)
    faComp = dictResult["faComp"]
    assert np.all(np.diff(faComp, axis=-1) >= -1e-12)
    assert 0.0 <= faComp.min() and faComp.max() <= 1.0


def test_nearby_star_beats_a_distant_one():
    """Habitable zones shrink below the inner working angle with distance."""
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 120)
    listMax = []
    for fDistancePc in (5.0, 60.0):
        dictResult = cp.fdictStarCompleteness(fdictSolarTwin(fDistancePc), DICT_BOX,
                                              LIST_BANDS_DET, DICT_BAND_CHAR, DICT_MISSION,
                                              faTauGridS, 2000, -0.19, 0.26, 42)
        listMax.append(dictResult["faComp"][-1, -1])
    assert listMax[0] > listMax[1]


def test_completeness_is_reproducible_for_a_fixed_seed():
    """The same seed must give bit-identical completeness, or nothing downstream is stable."""
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 60)
    listRuns = [cp.fdictStarCompleteness(fdictSolarTwin(8.0), DICT_BOX, LIST_BANDS_DET,
                                         DICT_BAND_CHAR, DICT_MISSION, faTauGridS,
                                         1500, -0.19, 0.26, 99)["faComp"] for _ in range(2)]
    assert np.array_equal(listRuns[0], listRuns[1])


def test_completeness_never_decreases_with_visit_count():
    """A revisit can only add planets, never lose one already caught."""
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 60)
    dictMission = dict(DICT_MISSION, iMaxVisits=5)
    dictResult = cp.fdictStarCompleteness(fdictSolarTwin(6.0), DICT_BOX, LIST_BANDS_DET,
                                          DICT_BAND_CHAR, dictMission, faTauGridS,
                                          2000, -0.19, 0.26, 21)
    faComp = dictResult["faComp"]
    assert faComp.shape[0] == 5
    assert np.all(np.diff(faComp, axis=0) >= -1e-12)


def test_revisiting_catches_planets_a_single_visit_misses():
    """The point of a revisit: a planet behind the IWA or at crescent phase gets another chance."""
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 60)
    dictResult = cp.fdictStarCompleteness(fdictSolarTwin(6.0), DICT_BOX, LIST_BANDS_DET,
                                          DICT_BAND_CHAR, dict(DICT_MISSION, iMaxVisits=6),
                                          faTauGridS, 2000, -0.19, 0.26, 21)
    assert dictResult["faComp"][-1, -1] > dictResult["faComp"][0, -1] * 1.2


def test_visit_epochs_are_not_degenerate():
    """Evenly spaced mean anomalies would duplicate alternate visits; random epochs must not."""
    rng = np.random.default_rng(4)
    dictPlanets = cp.fdictInjectPlanets(DICT_BOX, 1.0, 400, -0.19, 0.26, rng, iVisits=4)
    faSep = cp.fdictProjectOrbits(dictPlanets)["faSepAu"]
    assert not np.allclose(faSep[:, 0], faSep[:, 2])
