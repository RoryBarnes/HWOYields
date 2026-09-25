#!/usr/bin/env python3
"""Emulate the survey over (kappa, kappa_c, f_leak, s_sky), sample posteriors, and draw a corner plot.

Reads the design evaluated by runCalibrationDegeneracyDesign.py. Each observable is emulated by a
full quadratic in the four parameters (scaled to the unit box), and its leave-one-out error is
added in quadrature to the observational tolerance, so emulator error cannot pose as a
constraint. Two posteriors are sampled with emcee under uniform box priors:

  yieldOnly  the 6 m planning yield 22.5 (10%) -- what A03's calibration uses;
  all        that, plus the priority-first-18 mean characterization time at 6 m (22 d) and 9 m
             (3.5 d), 10% each, and the Fig. 11 median completeness in the 6-10, 10-15 and
             15-20 pc bins (Stark's medians, +-0.10).

Reports each posterior's covariance and correlation matrix, the emulator's local sensitivity
d(observable)/d(parameter) at the adopted point, its singular values (how many parameter
combinations the observables constrain at all), and the posterior-predicted observables, which
show which published numbers no point in the box can reach.
"""

import argparse
import glob
import json
import os
import sys

import emcee
import numpy as np

S_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(S_HERE))
sys.path.insert(0, S_HERE)
from dataCalibrationDegeneracyDesign import FA_HI, FA_LO, LIST_PARAMS  # noqa: E402

LIST_BINS = ["6-10", "10-15", "15-20"]
LIST_OBS = ["ln_yield6", "ln_char18_6", "ln_char18_9"] + [f"fig11_{b}" for b in LIST_BINS]


def ftLoadDesign(sDir):
    """(theta, observables) arrays from the design point files."""
    listTheta, listObs = [], []
    for sPath in sorted(glob.glob(os.path.join(sDir, "point*.json"))):
        d = json.load(open(sPath))
        listTheta.append([d["dictTheta"][k] for k in LIST_PARAMS])
        dictFig = d["dictFigElevenMedianByDistance"]
        listObs.append([np.log(d["fYield6"]), np.log(d["fChar18Days6"]),
                        np.log(d["fChar18Days9"])] + [dictFig.get(b) or 0.0 for b in LIST_BINS])
    return np.array(listTheta), np.array(listObs)


def faFeatures(faTheta):
    """Full quadratic features of the unit-scaled parameters."""
    faZ = (np.atleast_2d(faTheta) - FA_LO) / (FA_HI - FA_LO)
    listCols = [np.ones(len(faZ))] + [faZ[:, i] for i in range(faZ.shape[1])]
    listCols += [faZ[:, i] * faZ[:, j] for i in range(faZ.shape[1])
                 for j in range(i, faZ.shape[1])]
    return np.column_stack(listCols)


def ftFitEmulator(faTheta, faObs):
    """Least-squares coefficients and leave-one-out RMS error for every observable."""
    faX = faFeatures(faTheta)
    faCoef = np.linalg.lstsq(faX, faObs, rcond=None)[0]
    faHat = faX @ np.linalg.pinv(faX.T @ faX) @ faX.T
    faResid = (faObs - faX @ faCoef) / (1.0 - np.diag(faHat))[:, None]
    return faCoef, np.sqrt(np.mean(faResid ** 2, axis=0))


def fdictTargets(sFigBinned):
    """Published values and tolerances for each observable."""
    dictPub = json.load(open(sFigBinned))["dictFigureEleven"]["dictByDistance"]
    faY = np.array([np.log(22.5), np.log(22.0), np.log(3.5)] +
                   [dictPub[b]["fMedianPublished"] for b in LIST_BINS])
    faSig = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    return {"faY": faY, "faSig": faSig}


def ffLogPosterior(faTheta, faCoef, faY, faSig, faUse):
    """Uniform box prior times Gaussian likelihood on the chosen observables."""
    if np.any(faTheta < FA_LO) or np.any(faTheta > FA_HI):
        return -np.inf
    faMu = (faFeatures(faTheta) @ faCoef)[0]
    return float(-0.5 * np.sum(((faMu - faY) / faSig)[faUse] ** 2))


def faSample(faCoef, faY, faSig, faUse, iSeed, iSteps=4000):
    """emcee chain, burn-in removed and flattened."""
    rng = np.random.default_rng(iSeed)
    iWalkers = 32
    faStart = FA_LO + rng.random((iWalkers, len(FA_LO))) * (FA_HI - FA_LO)
    oSampler = emcee.EnsembleSampler(iWalkers, len(FA_LO), ffLogPosterior,
                                     args=(faCoef, faY, faSig, faUse))
    oSampler.run_mcmc(faStart, iSteps, progress=False)
    return oSampler.get_chain(discard=iSteps // 4, thin=10, flat=True)


def fdictSummary(faChain, faCoef):
    """Mean, covariance, correlation and predicted observables for one posterior."""
    faCov = np.cov(faChain.T)
    faSd = np.sqrt(np.diag(faCov))
    faPred = faFeatures(faChain) @ faCoef
    return {"faMean": faChain.mean(axis=0).tolist(), "faStd": faSd.tolist(),
            "faCovariance": faCov.tolist(), "faCorrelation": (faCov / np.outer(faSd, faSd)).tolist(),
            "dictPredicted": {k: [float(np.percentile(faPred[:, i], q)) for q in (5, 50, 95)]
                              for i, k in enumerate(LIST_OBS)}}


def fdictSensitivity(faCoef, faTheta0, faSigTotal):
    """Jacobian d(obs)/d(theta) at faTheta0, and singular values of the whitened Jacobian."""
    faJ = np.zeros((len(LIST_OBS), len(LIST_PARAMS)))
    for k in range(len(LIST_PARAMS)):
        faStep = np.zeros(len(LIST_PARAMS))
        faStep[k] = 1e-4 * (FA_HI[k] - FA_LO[k])
        faJ[:, k] = ((faFeatures(faTheta0 + faStep) - faFeatures(faTheta0 - faStep)) @ faCoef
                     )[0] / (2 * faStep[k])
    faScaled = faJ / faSigTotal[:, None] * (FA_HI - FA_LO)[None, :]
    faU, faS, faVt = np.linalg.svd(faScaled, full_matrices=False)
    return {"faJacobian": faJ.tolist(), "faSingularValues": faS.tolist(),
            "faDirections": faVt.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design-dir", default="design")
    p.add_argument("--fig11-published",
                   default="reference/figureElevenPublishedBins.json")
    p.add_argument("--adopted-kappa", type=float, default=1.197)
    p.add_argument("--out-json", default="calibrationDegeneracy.json")
    p.add_argument("--out-chains", default="calibrationDegeneracyChains.npz")
    dictArgs = vars(p.parse_args())
    faTheta, faObs = ftLoadDesign(dictArgs["design_dir"])
    faCoef, faLoo = ftFitEmulator(faTheta, faObs)
    dictT = fdictTargets(dictArgs["fig11_published"])
    faSigTotal = np.sqrt(dictT["faSig"] ** 2 + faLoo ** 2)
    dictUse = {"yieldOnly": np.arange(len(LIST_OBS)) == 0,
               "all": np.ones(len(LIST_OBS), dtype=bool)}
    dictChains = {s: faSample(faCoef, dictT["faY"], faSigTotal, u, 11 + i)
                  for i, (s, u) in enumerate(dictUse.items())}
    faTheta0 = np.array([np.log(dictArgs["adopted_kappa"]), 0.0, 1.0, 0.0])
    dictOut = {"listParams": LIST_PARAMS, "listObservables": LIST_OBS, "iDesignPoints": len(faTheta),
               "faEmulatorLooRms": faLoo.tolist(), "faTargets": dictT["faY"].tolist(),
               "faTolerances": faSigTotal.tolist(),
               "dictSensitivityAtAdopted": fdictSensitivity(faCoef, faTheta0, faSigTotal),
               "dictPosteriors": {s: fdictSummary(c, faCoef) for s, c in dictChains.items()}}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    np.savez(dictArgs["out_chains"], **dictChains, faDesignTheta=faTheta, faDesignObs=faObs)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "dictPosteriors"}, indent=1))
    for s, d in dictOut["dictPosteriors"].items():
        print(s, "corr", np.round(d["faCorrelation"], 2).tolist(), "pred", d["dictPredicted"])


if __name__ == "__main__":
    main()
