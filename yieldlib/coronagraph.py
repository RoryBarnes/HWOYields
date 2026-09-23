"""Parametric DMVC coronagraph performance standing in for Stark's simulated Upsilon_c and I maps."""

import numpy as np

F_DEFAULT_CORE_THROUGHPUT_MAX = 0.46
F_DEFAULT_IWA_LAMD = 3.89
F_DEFAULT_OWA_LAMD = 32.0
F_DEFAULT_RAMP_INDEX = 2.8
F_DEFAULT_CONTRAST_FLOOR = 1.0e-10
F_DEFAULT_CONTRAST_KNEE_LAMD = 4.2
F_DEFAULT_CONTRAST_INNER_INDEX = 4.0


def faCoreThroughput(faSeparationLamD, fThroughputMax=F_DEFAULT_CORE_THROUGHPUT_MAX,
                     fIwaLamD=F_DEFAULT_IWA_LAMD, fOwaLamD=F_DEFAULT_OWA_LAMD,
                     fRampIndex=F_DEFAULT_RAMP_INDEX):
    """Azimuthally-averaged coronagraph core throughput Upsilon_c as a function of separation.

    A logistic ramp in log-separation fitted to the DMVC curve of Stark et al. (2019) rather
    than to its three headline numbers. The earlier fit, anchored on a half-maximum at the
    formal 3.5 lambda/D IWA, was 70 percent too generous at 2 lambda/D; this one reproduces the
    published curve to within about 25 percent there and a few percent everywhere outside
    3 lambda/D. Throughput inside the IWA is non-zero but small: planets are detectable interior
    to it at a penalty.
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


def fnAiryEncircledEnergy(fApertureRadiusLamD):
    """Fraction of an unobscured Airy pattern inside a circular aperture of the given radius.

    EE(r) = 1 - J0(u)^2 - J1(u)^2 with u = pi * r in units of lambda/D. At Stark's photometric
    aperture radius of 0.7 lambda/D this returns 0.679, which is the 0.69 that Ref. stark2019
    quotes as the value Upsilon took before it became a simulated, separation-dependent curve.
    """
    from scipy.special import j0, j1
    fU = np.pi * float(fApertureRadiusLamD)
    return float(1.0 - j0(fU) ** 2 - j1(fU) ** 2)


def fnSkyThroughput(fThroughputMax=F_DEFAULT_CORE_THROUGHPUT_MAX,
                    fApertureRadiusLamD=0.7):
    """T_sky, the coronagraph's throughput for an extended source (Stark et al. 2019 Eqs. 5-6).

    Both the zodiacal and exozodiacal count rates carry a factor T_sky(x,y) that the point-source
    terms do not, because a uniform background is attenuated by the coronagraph masks without
    also paying the core-fraction penalty a planet PSF pays. Stark computes it by convolving the
    spatially-dependent PSF with a normalised uniform background; that map is not published, so
    it is reconstructed here from the two numbers that are.

    Upsilon_c differs from an ideal system's encircled energy by exactly the coronagraph's own
    transmission loss -- the focal-plane mask, the Lyot stop, and for the DMVC the light beyond
    the inscribed pupil diameter that the Lyot stop must discard (Ref. stark2019 Sec. 6.2). That
    loss applies to an extended source too, while the encircled-energy penalty does not, so

        T_sky = Upsilon_c,max / EE(X)  =  0.46 / 0.679  =  0.677.

    Omitting it, as this pipeline did until this was found, overstates the zodiacal and
    exozodiacal backgrounds by a factor of 1.48. The error is nearly harmless for the nearest
    stars, whose noise budget is dominated by leaked starlight, and severe for distant ones,
    whose planets are faint enough that the zodiacal terms set the exposure time.
    """
    return min(1.0, float(fThroughputMax) / fnAiryEncircledEnergy(fApertureRadiusLamD))
