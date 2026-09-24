"""Realized-yield distributions: Poisson counting about a distribution of expected yields."""

import numpy as np
from scipy.stats import poisson


def faPoissonMixturePmf(faExpected, iMax=100):
    """P(yield = k), k < iMax, averaged over Poisson counts about each expected yield.

    Exact for the model rather than a histogram of Poisson samples, so the curve is as smooth as
    Stark et al. (2024)'s, which average 1000 planet draws over each of 498 runs. Poisson counting
    about the expected yield is exact here because planets around a star are counted
    independently (see the report's graphical model).
    """
    faK = np.arange(iMax)
    faLambda = np.maximum(np.asarray(faExpected, dtype=float), 0.0)
    return poisson.pmf(faK[None, :], faLambda[:, None]).mean(axis=0)
