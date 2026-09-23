"""Unit tests for the radiometric core, checked against independently known physical values."""

import os
import sys

import numpy as np
import pytest
from scipy.special import lambertw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yieldlib import coronagraph as cg  # noqa: E402
from yieldlib import physics as ph  # noqa: E402


def test_geiger_cic_factor_matches_lambert_w_solution():
    """The 6.73 in Stark et al. (2019) Eq. 9 is -[1 + W_-1(-q/e)]^-1 at q = 0.99."""
    fExpected = -1.0 / (1.0 + lambertw(-0.99 / np.e, k=-1).real)
    assert np.isclose(ph.F_GEIGER_CIC_FACTOR, fExpected, rtol=1e-3)


def test_solar_flux_matches_v_band_zero_point():
    """A 5772 K blackbody at 10 pc must reproduce the Sun's absolute V magnitude of 4.83."""
    fFlux = float(ph.faStellarPhotonFlux(ph.F_VBAND_LAMBDA_M, ph.F_TEFF_SUN_K, 1.0, 10.0))
    fExpected = ph.F_VBAND_ZERO_PHOTONS * 1e6 * 10 ** (-0.4 * 4.83)
    assert np.isclose(fFlux, fExpected, rtol=0.05)


def test_zero_mag_flux_is_normalised_at_v_band():
    """The zero-magnitude reference is pinned to the adopted V-band zero point."""
    assert np.isclose(ph.fnZeroMagPhotonFlux(ph.F_VBAND_LAMBDA_M), ph.F_VBAND_ZERO_PHOTONS)


def test_planck_energy_radiance_obeys_wien_displacement():
    """Energy radiance is photon radiance times hc/lambda, and must peak at Wien's wavelength."""
    faLambda = np.linspace(100e-9, 3000e-9, 20000)
    faRadianceEnergy = ph.faPlanckPhotonRadiance(faLambda, 5772.0) / faLambda
    fPeak = faLambda[int(np.argmax(faRadianceEnergy))]
    assert np.isclose(fPeak, 2.897771955e-3 / 5772.0, rtol=0.01)


def test_planck_photon_radiance_peaks_at_the_photon_wien_constant():
    """The photon-number Planck peak sits at 3.6697e-3 m K / T, redward of the energy peak."""
    faLambda = np.linspace(100e-9, 3000e-9, 20000)
    fPeak = faLambda[int(np.argmax(ph.faPlanckPhotonRadiance(faLambda, 5772.0)))]
    assert np.isclose(fPeak, 3.6697e-3 / 5772.0, rtol=0.01)


def test_lambertian_phase_limits():
    """Full phase returns unity, quadrature returns 1/pi, and new phase returns zero."""
    assert np.isclose(float(ph.faLambertianPhase(0.0)), 1.0)
    assert np.isclose(float(ph.faLambertianPhase(np.pi / 2)), 1.0 / np.pi)
    assert np.isclose(float(ph.faLambertianPhase(np.pi)), 0.0, atol=1e-12)


def test_exposure_time_scales_as_snr_squared():
    """Doubling the required signal to noise quadruples the exposure time."""
    fLow = float(ph.faExposureTime(1.0, 10.0, 5.0))
    fHigh = float(ph.faExposureTime(1.0, 10.0, 10.0))
    assert np.isclose(fHigh / fLow, 4.0)


def test_exposure_time_is_infinite_without_planet_counts():
    """A planet contributing no counts can never be detected."""
    assert np.isinf(float(ph.faExposureTime(0.0, 10.0, 7.0)))


def test_detector_count_rate_has_no_read_noise_term_for_photon_counting():
    """With RN = 0 the detector rate is dark current plus the clock-induced-charge term."""
    fRate = float(ph.faDetectorCountRate(1e-3, 4, 3e-5, 0.0, None, 1.3e-3))
    fExpected = 4 * (3e-5 + ph.F_GEIGER_CIC_FACTOR * 1e-3 * 1.3e-3)
    assert np.isclose(fRate, fExpected)


def test_core_throughput_is_half_maximum_at_the_inner_working_angle():
    """IWA is defined as the separation where off-axis throughput reaches half its maximum."""
    fAtIwa = float(cg.faCoreThroughput(cg.F_DEFAULT_IWA_LAMD))
    assert np.isclose(fAtIwa, 0.5 * cg.F_DEFAULT_CORE_THROUGHPUT_MAX, rtol=1e-6)


def test_core_throughput_is_nonzero_inside_the_iwa_and_zero_beyond_the_owa():
    """Stark et al. note planets are detectable inside the formal IWA at a throughput penalty."""
    assert 0.0 < float(cg.faCoreThroughput(1.5)) < cg.F_DEFAULT_CORE_THROUGHPUT_MAX
    assert float(cg.faCoreThroughput(cg.F_DEFAULT_OWA_LAMD * 1.1)) == 0.0


def test_raw_contrast_is_floored_inside_the_dark_zone():
    """Contrast is substituted with zeta_floor wherever the design would do better."""
    assert float(cg.faRawContrast(10.0)) == cg.F_DEFAULT_CONTRAST_FLOOR


@pytest.mark.parametrize("fDiameterM,fExpectedArcsec", [(6.0, 0.0189), (12.0, 0.00945)])
def test_lambda_over_d_scale(fDiameterM, fExpectedArcsec):
    """lambda/D at 550 nm for the baseline and a doubled aperture."""
    assert np.isclose(cg.fnLambdaOverDArcsec(550e-9, fDiameterM), fExpectedArcsec, rtol=0.01)


def test_airy_encircled_energy_reproduces_starks_legacy_upsilon():
    """Stark et al. (2019) say Upsilon was 0.69 before it became a simulated curve.

    That number is the encircled energy of an unobscured Airy pattern inside the photometric
    aperture, whose radius Table 3 gives as X = 0.7 lambda/D. Recovering 0.69 from the aperture
    radius alone is what licenses reading Upsilon_c,max / EE(X) as the coronagraph's own
    transmission loss, which is the definition T_sky is reconstructed from.
    """
    assert np.isclose(cg.fnAiryEncircledEnergy(0.7), 0.69, atol=0.02)


def test_sky_throughput_is_the_coronagraph_loss_and_exceeds_the_core_throughput():
    """T_sky attenuates an extended source without the core-fraction penalty a planet PSF pays."""
    fSkyThroughput = cg.fnSkyThroughput()
    assert np.isclose(fSkyThroughput,
                      cg.F_DEFAULT_CORE_THROUGHPUT_MAX / cg.fnAiryEncircledEnergy(0.7),
                      rtol=1e-12)
    assert cg.F_DEFAULT_CORE_THROUGHPUT_MAX < fSkyThroughput < 1.0


def test_exozodi_scale_is_unity_for_a_solar_twin():
    """One zodi is defined relative to the Sun's own zodiacal cloud, so a solar twin scores one."""
    fScale = ph.fnExozodiSurfaceBrightnessScale(ph.F_VBAND_LAMBDA_M, ph.F_TEFF_SUN_K, 1.0, 1.0)
    assert np.isclose(fScale, 1.0, rtol=1e-12)


def test_exozodi_is_dimmer_around_late_type_stars_by_the_published_factor():
    """Stark et al. (2014) App. C: a zodi around a late M star is ~2.5x dimmer than around the Sun.

    The constant-surface-brightness treatment this replaced scored every star at 1.0, which is
    what "unfairly penalizes late type stars" means in the yield: their exozodi background was
    charged at the solar rate while their habitable zones are far more compact.
    """
    fScale = ph.fnExozodiSurfaceBrightnessScale(ph.F_VBAND_LAMBDA_M, 3300.0, 0.30, 0.0095)
    assert 1.0 / 4.0 < fScale < 1.0 / 2.0


def test_exozodi_scale_tracks_band_flux_over_bolometric_luminosity():
    """The law is 10^-0.4(M_lambda - M_lambda_sun) / L, so doubling L at fixed flux halves it."""
    fOne = ph.fnExozodiSurfaceBrightnessScale(ph.F_VBAND_LAMBDA_M, ph.F_TEFF_SUN_K, 1.0, 1.0)
    fTwo = ph.fnExozodiSurfaceBrightnessScale(ph.F_VBAND_LAMBDA_M, ph.F_TEFF_SUN_K, 1.0, 2.0)
    assert np.isclose(fTwo, 0.5 * fOne, rtol=1e-12)
