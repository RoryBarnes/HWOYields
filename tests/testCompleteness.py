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


def fdictMinimalCountRateInputs():
    """One solar twin, one Earth twin at quadrature: the smallest input fdictCountRates accepts."""
    dictStar = fdictSolarTwin(10.0)
    dictPlanets = {"faRadiusEarth": np.array([1.0]), "faAxisAu": np.array([1.0]),
                   "faAxisScaled": np.array([1.0])}
    dictGeom = {"faSepAu": np.array([[1.0]]), "faPhase": np.array([[0.5]])}
    return dictStar, dictPlanets, dictGeom, LIST_BANDS_DET[0], DICT_MISSION


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


def test_background_count_rates_carry_the_sky_throughput():
    """Zodi and exozodi are attenuated by T_sky; the planet and leaked-starlight terms are not.

    Stark et al. (2019) Eqs. 5 and 6 put T_sky(x,y) on the extended-source terms only. Omitting it
    -- which this pipeline did until the equations were re-read -- overstates both backgrounds by
    1/T_sky = 1.48, which is harmless where leaked starlight dominates and severe for the distant
    targets whose exposure times the zodiacal terms set.
    """
    dictStar, dictPlanets, dictGeom, dictBand, dictMission = fdictMinimalCountRateInputs()
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    dictNoAttenuation = dict(dictMission, fSkyThroughput=1.0)
    dictBare = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictNoAttenuation)
    fSkyThroughput = cp.fnSkyThroughputFor(dictMission)
    assert np.isclose(dictRates["fZodi"], dictBare["fZodi"] * fSkyThroughput, rtol=1e-12)
    assert np.isclose(dictRates["fExozodi"], dictBare["fExozodi"] * fSkyThroughput, rtol=1e-12)
    assert np.allclose(dictRates["faPlanet"], dictBare["faPlanet"], rtol=1e-12)
    assert np.allclose(dictRates["faLeak"], dictBare["faLeak"], rtol=1e-12)


def test_exozodi_is_one_brightness_per_star_not_a_radial_profile():
    """Stark et al. (2014) apply the EEID surface brightness to every planet around a star.

    They computed the self-consistent radial and geometric treatment with ZODIPIC, found it moved
    the yield by a few percent, and deliberately did not adopt it. An earlier version of this
    module carried a 5*log10(a/sqrt(L)) radial term that is not in the published model.
    """
    dictStar, dictPlanets, dictGeom, dictBand, dictMission = fdictMinimalCountRateInputs()
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    assert np.ndim(dictRates["fExozodi"]) == 0


def test_collecting_area_uses_the_full_primary_not_the_inscribed_circle():
    """Upsilon_c is normalised to the light entering from the whole obscured primary.

    Stark et al. (2019) Sec. 6.2 says the DMVC's apparently low core throughput is caused by the
    Lyot stop discarding the pupil outside the inscribed diameter, which is already inside
    Upsilon_c. Pairing that with a collecting area taken as the inscribed circle would charge the
    same discard twice, so A must be the larger, full-primary area.
    """
    dictMission = dict(DICT_MISSION, fCircumscribedRatio=8.0 / 6.7, fApertureFillFactor=0.785)
    fArea = cp.fnCollectingAreaM2(dictMission)
    fInscribedCircle = np.pi * (DICT_MISSION["fDiameterM"] / 2.0) ** 2
    assert fArea > fInscribedCircle
    assert np.isclose(fArea / fInscribedCircle, 0.785 * (8.0 / 6.7) ** 2, rtol=1e-12)


def test_collecting_area_scales_with_the_square_of_the_diameter():
    """The aperture study varies the inscribed diameter; the area must follow it as D^2."""
    dictMission = dict(DICT_MISSION, fCircumscribedRatio=8.0 / 6.7, fApertureFillFactor=0.785)
    fSmall = cp.fnCollectingAreaM2(dict(dictMission, fDiameterM=6.0))
    fLarge = cp.fnCollectingAreaM2(dict(dictMission, fDiameterM=9.0))
    assert np.isclose(fLarge / fSmall, (9.0 / 6.0) ** 2, rtol=1e-12)


def test_coronagraph_curves_are_evaluated_in_circumscribed_lambda_over_d():
    """A planet at a fixed angle sits at MORE lambda/D once the circumscribed scale is applied.

    The published core-throughput and contrast curves are plotted against the circumscribed
    diameter while the scenarios are labelled by inscribed diameter. Applying the curves at
    inscribed lambda/D placed this pipeline's inner working angle 19% too far out in angle, which
    suppressed exactly the distant targets whose completeness fell short of Stark's Fig. 11.
    """
    dictStar, dictPlanets, dictGeom, dictBand, dictMission = fdictMinimalCountRateInputs()
    dictScaled = dict(dictMission, fCircumscribedRatio=8.0 / 6.7)
    dictPlain = dict(dictMission, fCircumscribedRatio=1.0)
    fUpsilonScaled = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand,
                                        dictScaled)["faUpsilon"]
    fUpsilonPlain = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand,
                                       dictPlain)["faUpsilon"]
    assert np.all(fUpsilonScaled > fUpsilonPlain)


def test_albedo_draw_does_not_revise_the_characterization_budget():
    """Stark et al. (2024) Sec. 3.2 fix the spectrum's cost at the planning albedo.

    "Detection times are used to determine whether a planet of differing albedo would have been
    detected, but characterization times are not considered. Characterization times are budgeted
    for by AYO under the assumption that the planets have a single A_G = 0.2." The observation plan
    is set in stone before the draw, so a brighter planet cannot also buy itself a cheaper
    spectrum. Letting it do so -- which this module did until the published detection-rate-versus-
    albedo shape was checked -- roughly halves the albedo penalty, because characterization is the
    binding constraint beyond about 10 pc.
    """
    faTauGridS = np.logspace(1.0, np.log10(DICT_MISSION["fExposureLimitS"]), 60)
    dictMission = dict(DICT_MISSION, iMaxVisits=2,
                       dictAlbedoDistribution={"fMin": 0.08, "fMax": 0.32})
    dictFixed = cp.fdictStarCompleteness(fdictSolarTwin(12.0), DICT_BOX, LIST_BANDS_DET,
                                         DICT_BAND_CHAR, dictMission, faTauGridS,
                                         1500, -0.19, 0.26, 7)
    dictAdjusted = cp.fdictStarCompleteness(
        fdictSolarTwin(12.0), DICT_BOX, LIST_BANDS_DET, DICT_BAND_CHAR,
        dict(dictMission, bAdjustCharacterizationForAlbedo=True), faTauGridS,
        1500, -0.19, 0.26, 7)
    assert not np.allclose(dictFixed["faCompAlbedo"], dictAdjusted["faCompAlbedo"])
    for dictRun in (dictFixed, dictAdjusted):
        faAlbedo = dictRun["faCompAlbedo"]
        assert np.all((faAlbedo >= 0.0) & (faAlbedo <= 1.0))
        assert np.all(np.diff(faAlbedo, axis=-1) >= -1e-12)

    # The per-star direction is not universal: holding the budget fixed lets a DARK planet keep
    # the cheaper A_G = 0.2 spectrum while denying a bright one an even cheaper one, and which
    # dominates depends on the star. Over the whole survey it costs yield, taking the albedo
    # penalty from 0.050 to 0.088 against a published 0.12
    # (explorations/compareAlbedoVariants.py). Asserting the survey-level direction star by star
    # would be asserting something this test cannot see, so it checks only that the flag is live
    # and that each curve stays a completeness: bounded in [0, 1] and non-decreasing in exposure.
    # Note faCompAlbedo is NOT bounded by faCompDetectionOnly, which is evaluated at the planning
    # albedo -- a planet drawn brighter than A_G = 0.2 can be caught at an exposure where the
    # planning planet was not.


def test_bright_flux_bound_can_only_remove_detections():
    """The upper end of the detected flux range is a veto, never a licence.

    For an edge-on orbit a flux above what the segment reached implies gibbous phase and therefore
    a separation inside the inner working angle, so applying it can only reject planets. Measured
    on the full survey it moves the albedo penalty by 0.002, so it is real but not the reason this
    model's penalty fell short of the published 12%.
    """
    faTauDet = np.array([[1.0e4], [2.0e4], [4.0e4], [np.inf]])
    faFlux = np.array([[3.0e-10], [2.0e-10], [1.0e-10], [5.0e-11]])
    faSep = np.array([[6.0], [7.0], [8.0], [9.0]])
    faDrawn = np.array([[9.0e-10], [2.0e-10], [1.5e-10], [1.0e-10]])
    faWithout = cp.faStarkAlbedoTimes(faTauDet, faFlux, faSep, faDrawn, bBrightBound=False)
    faWith = cp.faStarkAlbedoTimes(faTauDet, faFlux, faSep, faDrawn, bBrightBound=True)
    assert np.all(faWith >= faWithout)
    assert np.isinf(faWith[0, 0]) and np.isfinite(faWithout[0, 0])
