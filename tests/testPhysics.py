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
