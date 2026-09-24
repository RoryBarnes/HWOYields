#!/usr/bin/env python3
"""Sample a Bayesian posterior for the SAG13 occurrence parameters and integrate it over boxes.

The model is deliberately modest about what it infers. The SHAPE parameters alpha and beta carry
Gaussian priors taken from the SAG13 fit; the single likelihood term is the eta_Earth that Stark
et al. (2024) obtain by integrating the Bryson et al. (2021) Kepler DR25 posterior over the
canonical EEC box, quoted as 0.26 (+0.29/-0.14) at 86 percent confidence (but see
F_Z_INTERVAL_DEFAULT: this step reads that interval as one sigma). The posterior is therefore
prior-dominated in alpha and beta and likelihood-dominated in the normalisation, which is the
honest structure of the available evidence: there is one well-characterised integral constraint
and no published per-parameter posterior to re-fit. Redefining the selection box changes which
integral of the same rate density is reported, which is exactly what this step computes.
"""

import argparse
import json
import sys

import emcee
import numpy as np

sys.path.insert(0, "..")
from yieldlib import occurrence as oc  # noqa: E402

F_Z_86_PERCENT = 1.4757910281791712  # two-sided 86 percent interval of a standard normal
F_Z_INTERVAL_DEFAULT = 1.0
# Stark et al. (2024) Sec. 3.5 label 0.26 (+0.29/-0.14) an 86 percent interval, but their own
# Fig. 10 does not support that width. Holding this pipeline's yield-vs-eta curve at Stark's
# fixed-eta level (his red curve), the lognormal eta law that turns red into his purple curve has
# median 0.26 and ln-sigma 0.80 (total-variation distance 0.019; mode 11, median 17 against his
# 10 and 18). Reading the quoted interval as one sigma (68 percent) gives ln-sigma 0.76 and
# reproduces the figure nearly as well (0.024); reading it as 86 percent gives 0.52 and does not
# (0.142, mode 15). Bryson et al. (2021) Fig. 13 shows per-case ln-widths of about 0.8 too. The
# posterior chains that would settle it are not published. See
# explorations/inferEtaLawFromStarkFigure10.py. Pass --eta-interval-z 1.4758 for the text's reading.


def fnLogSplitNormal(fValue, fCentre, fSigmaLow, fSigmaHigh):
    """Log density of a split-normal, the natural reading of an asymmetric quoted interval."""
    fSigma = fSigmaHigh if fValue >= fCentre else fSigmaLow
    return -0.5 * ((fValue - fCentre) / fSigma) ** 2 - np.log(fSigma)


def fnLogPosterior(faTheta, dictBoxes, dictPriors, dictLikelihood):
    """Log posterior over (ln Gamma, alpha, beta)."""
    fLnGamma, fAlpha, fBeta = faTheta
    if not (-8.0 < fLnGamma < 4.0 and -3.0 < fAlpha < 3.0 and -3.0 < fBeta < 3.0):
        return -np.inf
    fLogPrior = (-0.5 * ((fLnGamma - dictPriors["fLnGammaMean"]) /
                         dictPriors["fLnGammaSigma"]) ** 2
                 - 0.5 * ((fAlpha - dictPriors["fAlphaMean"]) / dictPriors["fAlphaSigma"]) ** 2
                 - 0.5 * ((fBeta - dictPriors["fBetaMean"]) / dictPriors["fBetaSigma"]) ** 2)
    fEta = oc.fnOccurrenceAnalytic(dictBoxes["canonical"], np.exp(fLnGamma), fAlpha, fBeta)
    if not np.isfinite(fEta) or fEta <= 0.0:
        return -np.inf
    return fLogPrior + fnLogSplitNormal(fEta, dictLikelihood["fEtaCentre"],
                                        dictLikelihood["fSigmaLow"], dictLikelihood["fSigmaHigh"])


def faRunSampler(dictBoxes, dictPriors, dictLikelihood, iWalkers, iSteps, iBurn, iSeed):
    """Run emcee and return the flattened, burned-in, thinned chain.

    emcee's EnsembleSampler draws its proposal moves from numpy's GLOBAL legacy RNG, not from
    any generator passed in, so seeding only the walker start positions leaves the chain
    irreproducible. Both are seeded here. The determinism scanner cannot detect this class of
    bug -- it was caught by the quantitative test comparing the chain against its standard.
    """
    np.random.seed(iSeed)
    rng = np.random.default_rng(iSeed)
    faStart = np.column_stack([
        rng.normal(dictPriors["fLnGammaMean"], 0.05, iWalkers),
        rng.normal(dictPriors["fAlphaMean"], 0.02, iWalkers),
        rng.normal(dictPriors["fBetaMean"], 0.02, iWalkers)])
    oSampler = emcee.EnsembleSampler(iWalkers, 3, fnLogPosterior,
                                     args=(dictBoxes, dictPriors, dictLikelihood))
    oSampler.run_mcmc(faStart, iSteps, progress=False)
    return oSampler, oSampler.get_chain(discard=iBurn, thin=10, flat=True)


def faSampleEtaFromPublished(fCentre, fMinus, fPlus, iDraws, rng, fZ=F_Z_INTERVAL_DEFAULT):
    """Draw eta for the canonical box directly from the published posterior summary.

    The value this pipeline must reproduce is already a POSTERIOR -- Stark et al. (2024) obtain
    eta_Earth = 0.26 (+0.29/-0.14) at 86 percent by integrating the Bryson et al. (2021) chain.
    An earlier version treated it as a likelihood and multiplied it by priors on (lnGamma, alpha,
    beta), which double-counts information: the result came out 23 percent NARROWER than the
    constraint it was built from, with its mode dragged toward the prior centre and its right
    skew destroyed. Occurrence rates are positive and heavy-tailed, so a lognormal matched to the
    published interval is used here rather than a split-normal, which has far too light a tail.
    """
    fLo, fHi = fCentre - fMinus, fCentre + fPlus
    fMu = 0.5 * (np.log(fLo) + np.log(fHi))
    fSigma = (np.log(fHi) - np.log(fLo)) / (2.0 * fZ)
    return np.exp(rng.normal(fMu, fSigma, iDraws)), fMu, fSigma


def fdictBoxPosteriors(faChain, dictBoxes):
    """Integrate every posterior draw over each selection box."""
    dictOut = {}
    for sName, dictBox in dictBoxes.items():
        dictOut[sName] = np.array([
            oc.fnOccurrenceAnalytic(dictBox, np.exp(t[0]), t[1], t[2]) for t in faChain])
    return dictOut


def fdictSummarise(faValues):
    """Median and 16th/84th percentiles of a posterior sample."""
    return {"fMedian": float(np.median(faValues)),
            "fP16": float(np.percentile(faValues, 16)),
            "fP84": float(np.percentile(faValues, 84)),
            "fMean": float(np.mean(faValues))}


def fdictParseArgs():
    """Command-line configuration for the occurrence-rate posterior."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--eta-centre", type=float, default=0.26)
    p.add_argument("--eta-plus", type=float, default=0.29)
    p.add_argument("--eta-minus", type=float, default=0.14)
    p.add_argument("--eta-interval-z", type=float, default=F_Z_INTERVAL_DEFAULT,
                   help="standard-normal z of the quoted eta interval: 1.0 reads it as one sigma "
                        "(what Stark's Fig. 10 implies), 1.4758 as the stated 86 percent")
    p.add_argument("--gamma-prior-mean", type=float, default=0.38)
    p.add_argument("--gamma-prior-lnsigma", type=float, default=1.0)
    p.add_argument("--alpha-prior-mean", type=float, default=-0.19)
    p.add_argument("--alpha-prior-sigma", type=float, default=0.27)
    p.add_argument("--beta-prior-mean", type=float, default=0.26)
    p.add_argument("--beta-prior-sigma", type=float, default=0.29)
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--burn", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-samples", default="occurrencePosterior.npz")
    p.add_argument("--out-summary", default="occurrencePosteriorSummary.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictBoxes = json.load(oFile)["dictBoxes"]
    dictPriors = {"fLnGammaMean": float(np.log(dictArgs["gamma_prior_mean"])),
                  "fLnGammaSigma": dictArgs["gamma_prior_lnsigma"],
                  "fAlphaMean": dictArgs["alpha_prior_mean"],
                  "fAlphaSigma": dictArgs["alpha_prior_sigma"],
                  "fBetaMean": dictArgs["beta_prior_mean"],
                  "fBetaSigma": dictArgs["beta_prior_sigma"]}
    dictLikelihood = {"fEtaCentre": dictArgs["eta_centre"],
                      "fSigmaHigh": dictArgs["eta_plus"] / dictArgs["eta_interval_z"],
                      "fSigmaLow": dictArgs["eta_minus"] / dictArgs["eta_interval_z"]}
    oSampler, faChain = faRunSampler(dictBoxes, dictPriors, dictLikelihood,
                                     dictArgs["walkers"], dictArgs["steps"],
                                     dictArgs["burn"], dictArgs["seed"])
    dictEtaShape = fdictBoxPosteriors(faChain, dictBoxes)
    faRatio = dictEtaShape["redefined"] / dictEtaShape["canonical"]
    rngEta = np.random.default_rng(dictArgs["seed"] + 1)
    faEtaCanonical, fMu, fSigma = faSampleEtaFromPublished(
        dictArgs["eta_centre"], dictArgs["eta_minus"], dictArgs["eta_plus"],
        faRatio.size, rngEta, dictArgs["eta_interval_z"])
    dictEta = {"canonical": faEtaCanonical, "redefined": faEtaCanonical * faRatio,
               "hzOnly": faEtaCanonical * (dictEtaShape["hzOnly"] /
                                           dictEtaShape["canonical"])}
    np.savez_compressed(dictArgs["out_samples"], faChain=faChain, faRatio=faRatio,
                        **{f"faEta_{k}": v for k, v in dictEta.items()})
    dictSummary = {
        "sEtaNormalisation": "Canonical-box eta sampled directly from the published posterior "
                             f"(lognormal matched to its quoted interval read at z = {dictArgs['eta_interval_z']:g}); other boxes follow "
                             "by the Gamma-independent ratio from the shape chain.",
        "fEtaIntervalZ": float(dictArgs["eta_interval_z"]),
        "fEtaLogNormalMu": float(fMu), "fEtaLogNormalSigma": float(fSigma),
        "fEtaModeImplied": float(np.exp(fMu - fSigma ** 2)),
        "fEtaMeanImplied": float(np.exp(fMu + 0.5 * fSigma ** 2)),
        "iSamples": int(faChain.shape[0]),
        "fAcceptanceFraction": float(np.mean(oSampler.acceptance_fraction)),
        "fMaxAutocorrSteps": float(np.max(oSampler.get_autocorr_time(quiet=True))),
        "dictParameters": {s: fdictSummarise(faChain[:, i]) for i, s in
                           enumerate(("lnGamma", "alpha", "beta"))},
        "dictEtaByBox": {k: fdictSummarise(v) for k, v in dictEta.items()},
        "dictRatioRedefinedOverCanonical": fdictSummarise(faRatio),
    }
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps(dictSummary, indent=2))


if __name__ == "__main__":
    main()
