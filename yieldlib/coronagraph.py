"""Parametric DMVC coronagraph performance standing in for Stark's simulated Upsilon_c and I maps."""

import numpy as np

F_DEFAULT_CORE_THROUGHPUT_MAX = 0.45
F_DEFAULT_IWA_LAMD = 3.5
F_DEFAULT_OWA_LAMD = 32.0
F_DEFAULT_RAMP_INDEX = 2.6
F_DEFAULT_CONTRAST_FLOOR = 1.0e-10


def faCoreThroughput(faSeparationLamD, fThroughputMax=F_DEFAULT_CORE_THROUGHPUT_MAX,
                     fIwaLamD=F_DEFAULT_IWA_LAMD, fOwaLamD=F_DEFAULT_OWA_LAMD,
                     fRampIndex=F_DEFAULT_RAMP_INDEX):
    """Azimuthally-averaged coronagraph core throughput Upsilon_c as a function of separation.

    A logistic ramp in log-separation reproducing the three numbers Stark et al. (2019, 2024)
    publish for the DMVC: half of the maximum at the formal IWA of 3.5 lambda/D, a plateau near
    45 percent at wide separations, and non-zero throughput inside the IWA (planets are
    detectable interior to it at a throughput penalty).
    """
    faSep = np.asarray(faSeparationLamD, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        faRamp = 1.0 / (1.0 + (fIwaLamD / np.maximum(faSep, 1e-6)) ** fRampIndex)
    return np.where(faSep <= fOwaLamD, fThroughputMax * faRamp, 0.0)


def faRawContrast(faSeparationLamD, fContrastFloor=F_DEFAULT_CONTRAST_FLOOR,
                  fOwaLamD=F_DEFAULT_OWA_LAMD):
    """Raw contrast zeta as a function of separation, floored at zeta_floor inside the dark zone.

    Stark et al. (2019) substitute zeta_floor wherever the simulated contrast is better, so
    inside the dark zone the floor is the operative value for every EEC-relevant separation.
    Outside the dark zone the contrast is unity (no suppression); planets there are excluded by
    the core throughput falling to zero, so a finite value here just keeps the arithmetic clean.
    """
    faSep = np.asarray(faSeparationLamD, dtype=float)
    return np.where(faSep <= fOwaLamD, fContrastFloor, 1.0)


def fnLambdaOverDArcsec(fLambdaM, fDiameterM):
    """Diffraction scale lambda/D in arcseconds."""
    return (fLambdaM / fDiameterM) * 206264.80624709636
