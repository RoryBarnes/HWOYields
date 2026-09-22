"""Radiometry and exposure-time physics for the AYO rederivation (Stark et al. 2019 Eqs. 1-9)."""

import numpy as np

F_PLANCK_H = 6.62607015e-34
F_LIGHT_C = 2.99792458e8
F_BOLTZMANN_K = 1.380649e-23
F_AU_M = 1.495978707e11
F_PC_M = 3.0856775814913673e16
F_RSUN_M = 6.957e8
F_LSUN_W = 3.828e26
F_ARCSEC_PER_RAD = 206264.80624709636
F_TEFF_SUN_K = 5772.0
F_VBAND_ZERO_PHOTONS = 1.0e11  # photons s^-1 m^-2 um^-1 for a V=0 star
F_VBAND_LAMBDA_M = 550e-9
F_GEIGER_CIC_FACTOR = 6.73  # -[1 + W_{-1}(-q/e)]^{-1} at q = 0.99 (Stark+2019 Eq. 9)


def faPlanckPhotonRadiance(faLambdaM, faTeffK):
    """Planck radiance in photons s^-1 m^-2 sr^-1 m^-1 at the given wavelengths and temperatures."""
    faLambdaM = np.asarray(faLambdaM, dtype=float)
    faTeffK = np.asarray(faTeffK, dtype=float)
    fExponent = F_PLANCK_H * F_LIGHT_C / (faLambdaM * F_BOLTZMANN_K * faTeffK)
    faRadianceW = (2.0 * F_PLANCK_H * F_LIGHT_C ** 2 / faLambdaM ** 5) / np.expm1(fExponent)
    return faRadianceW * faLambdaM / (F_PLANCK_H * F_LIGHT_C)


def faStellarPhotonFlux(faLambdaM, faTeffK, faRadiusRsun, faDistancePc):
    """Stellar photon flux at the telescope in photons s^-1 m^-2 m^-1, from a blackbody SED.

    Using Teff and stellar radius rather than apparent magnitude avoids any dependence on a
    bandpass zero point or a colour transformation, at the cost of a blackbody approximation.
    """
    faRadiance = faPlanckPhotonRadiance(faLambdaM, faTeffK)
    faSolidAngle = np.pi * (np.asarray(faRadiusRsun) * F_RSUN_M /
                            (np.asarray(faDistancePc) * F_PC_M)) ** 2
    return faRadiance * faSolidAngle


def fnZeroMagPhotonFlux(fLambdaM):
    """Photon flux of a 0-mag solar-coloured source, photons s^-1 m^-2 um^-1, normalised at V."""
    fShape = faPlanckPhotonRadiance(fLambdaM, F_TEFF_SUN_K)
    fShapeV = faPlanckPhotonRadiance(F_VBAND_LAMBDA_M, F_TEFF_SUN_K)
    return F_VBAND_ZERO_PHOTONS * float(fShape / fShapeV)


def fnPhotometricApertureSolidAngle(fLambdaM, fDiameterM, fApertureRadiusLamD):
    """Solid angle of the photometric aperture in arcsec^2."""
    fRadiusArcsec = fApertureRadiusLamD * (fLambdaM / fDiameterM) * F_ARCSEC_PER_RAD
    return np.pi * fRadiusArcsec ** 2


def faLambertianPhase(faPhaseAngleRad):
    """Lambertian phase function normalised to unity at full phase."""
    faBeta = np.asarray(faPhaseAngleRad, dtype=float)
    return (np.sin(faBeta) + (np.pi - faBeta) * np.cos(faBeta)) / np.pi


def faDetectorCountRate(faCountRateBrightestPixel, iNumPixels, fDarkCurrent, fReadNoise,
                        fReadTimeS, fClockInducedCharge):
    """Detector noise count rate, Stark et al. (2019) Eq. 9, in counts s^-1."""
    fReadTerm = 0.0 if (fReadNoise == 0.0 or fReadTimeS is None) else fReadNoise ** 2 / fReadTimeS
    faCicTerm = F_GEIGER_CIC_FACTOR * np.asarray(faCountRateBrightestPixel) * fClockInducedCharge
    return iNumPixels * (fDarkCurrent + fReadTerm + faCicTerm)


def faExposureTime(faCountRatePlanet, faCountRateBackground, fSignalToNoise):
    """Exposure time from Stark et al. (2019) Eq. 1; infinite where the planet yields no counts."""
    faPlanet = np.asarray(faCountRatePlanet, dtype=float)
    faBackground = np.asarray(faCountRateBackground, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        faTau = fSignalToNoise ** 2 * (faPlanet + 2.0 * faBackground) / faPlanet ** 2
    return np.where(faPlanet > 0.0, faTau, np.inf)
