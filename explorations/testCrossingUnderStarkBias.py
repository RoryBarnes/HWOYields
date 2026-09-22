#!/usr/bin/env python3
"""Check whether the P25 crossing displacement is explained by the uncertainty sources not modelled.

POST-HOC and NOT independent evidence: it applies Stark et al. (2024)'s own published bias
factor (their expected yield falls from 22.5 to 17.3 once albedo, exozodi-distribution and
exozodi-sampling uncertainty are included) to this pipeline's excluding-sigma curve, and asks
whether the crossing then lands near the published 7.1 m. A yes means the discrepancy is
quantitatively consistent with the stated cause; it does not independently confirm anything.
"""

import argparse
import json

import numpy as np


def faProbabilityExcluding(faEtaGrid, faYields, fEtaBaseline, fBias, fGoal, iDraws, rng):
    """P25 with eta fixed, Poisson counting, and an optional multiplicative yield bias."""
    fExpected = float(np.interp(np.log(fEtaBaseline), np.log(faEtaGrid), faYields)) * fBias
    return float(np.mean(rng.poisson(fExpected, size=iDraws) >= fGoal)), fExpected


def fnCrossing(faDiameters, faIncluding, faExcluding):
    """Diameter where the excluding curve overtakes the including curve, by linear interpolation."""
    faDifference = np.asarray(faExcluding) - np.asarray(faIncluding)
    faIndex = np.where(np.diff(np.sign(faDifference)) != 0)[0]
    if faIndex.size == 0:
        return None
    i = int(faIndex[0])
    fSlope = faDifference[i + 1] - faDifference[i]
    return float(faDiameters[i] - faDifference[i] *
                 (faDiameters[i + 1] - faDiameters[i]) / fSlope) if fSlope else faDiameters[i]


def fdictParseArgs():
    """Command-line configuration for the crossing diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", required=True)
    p.add_argument("--bias-factor", type=float, default=17.3 / 22.5)
    p.add_argument("--draws", type=int, default=200000)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-json", default="crossingUnderStarkBias.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["comparison"]) as oFile:
        dictData = json.load(oFile)
    rng = np.random.default_rng(dictArgs["seed"])
    faEtaGrid = np.array(dictData["faEtaGrid"])
    faDiameters = [float(s) for s in dictData["dictByDiameter"]]
    faIncluding = [dictData["dictByDiameter"][s]["fProbability25IncludingSigmaEta"]
                   for s in dictData["dictByDiameter"]]
    dictRows = {}
    for sKey, fBias in (("unbiased", 1.0), ("starkBias", dictArgs["bias_factor"])):
        listP, listE = [], []
        for s in dictData["dictByDiameter"]:
            fP, fE = faProbabilityExcluding(faEtaGrid,
                                            np.array(dictData["dictByDiameter"][s]["faYieldGrid"]),
                                            0.24, fBias, 25.0, dictArgs["draws"], rng)
            listP.append(fP)
            listE.append(fE)
        dictRows[sKey] = {"faProbability": listP, "faExpectedYield": listE,
                          "fCrossingM": fnCrossing(faDiameters, faIncluding, listP)}
    dictOut = {
        "sCaveat": __doc__.strip().splitlines()[2],
        "fBiasFactorApplied": dictArgs["bias_factor"],
        "faDiameters": faDiameters,
        "faPublishedExcluding": [dictData["dictPublished"]["dictExcludingSigmaEta"][s]
                                 for s in dictData["dictByDiameter"]],
        "fCrossingPublishedM": dictData["fCrossingDiameterPublishedM"],
        "dictRows": dictRows,
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"bias factor applied: {dictArgs['bias_factor']:.3f} (Stark's 22.5 -> 17.3)\n")
    print(f"{'D':>4}{'excl unbiased':>15}{'excl w/ bias':>14}{'published':>11}")
    for i, f in enumerate(faDiameters):
        print(f"{f:>4.0f}{100*dictRows['unbiased']['faProbability'][i]:>14.1f}%"
              f"{100*dictRows['starkBias']['faProbability'][i]:>13.1f}%"
              f"{100*dictOut['faPublishedExcluding'][i]:>10.1f}%")
    print(f"\ncrossing, unbiased : {dictRows['unbiased']['fCrossingM']:.2f} m")
    print("crossing, with bias:", "none in range" if dictRows["starkBias"]["fCrossingM"] is None else f"{dictRows['starkBias']['fCrossingM']:.2f} m")
    print(f"crossing, published: {dictOut['fCrossingPublishedM']:.2f} m")


if __name__ == "__main__":
    main()
