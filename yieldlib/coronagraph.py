"""Parametric DMVC coronagraph performance standing in for Stark's simulated Upsilon_c and I maps."""

import numpy as np

F_DEFAULT_CORE_THROUGHPUT_MAX = 0.45
F_DEFAULT_IWA_LAMD = 3.5
F_DEFAULT_OWA_LAMD = 32.0
F_DEFAULT_RAMP_INDEX = 2.6
F_DEFAULT_CONTRAST_FLOOR = 1.0e-10
F_DEFAULT_CONTRAST_KNEE_LAMD = 4.2
F_DEFAULT_CONTRAST_INNER_INDEX = 4.0


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
                  fOwaLamD=F_DEFAULT_OWA_LAMD, fKneeLamD=F_DEFAULT_CONTRAST_KNEE_LAMD,
                  fInnerIndex=F_DEFAULT_CONTRAST_INNER_INDEX):
    """Raw contrast zeta against separation: the floor outside a knee, rising steeply inside it.

    Stark et al. (2019) take zeta = max(zeta_simulated, zeta_floor). The floor is the operative
    value only where the simulated contrast is better than it, which for the DMVC means outside
    about 4 lambda/D. Inside that the simulated contrast climbs steeply -- their Fig. 5 reads
    roughly 2.5e-10 at 3 lambda/D, 3e-9 at 2, and above 1e-8 below 1.5 -- so treating the floor
    as universal, which an earlier version of this module did, makes close-in planets far easier
    than they are. That matters most for distant stars, whose whole habitable zone sits inside
    the knee, and therefore biases how the yield scales with aperture.

    The inner rise is a power law fitted to that figure and is a read-off, not a simulation.
    Outside the dark zone the contrast is unity; planets there are excluded by the core
    throughput falling to zero, so a finite value just keeps the arithmetic clean.
    """
    faSep = np.asarray(faSeparationLamD, dtype=float)
    faInner = fContrastFloor * np.maximum(1.0, (fKneeLamD / np.maximum(faSep, 1e-6)) ** fInnerIndex)
    return np.where(faSep <= fOwaLamD, faInner, 1.0)


def fnLambdaOverDArcsec(fLambdaM, fDiameterM):
    """Diffraction scale lambda/D in arcseconds."""
    return (fLambdaM / fDiameterM) * 206264.80624709636
