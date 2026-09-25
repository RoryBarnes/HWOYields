#!/usr/bin/env python3
"""Set the pipeline run with the throughput factor fitted beside the run with it fixed at 1.

The two runs differ ONLY in the throughput factor kappa (A03's --fit-throughput): both carry the
Stark (2014) latitude-dependent zodi, the noise floor in the exposure-time equation and the time
limit including overheads, and every other choice is unchanged. Run A was snapshotted to
output/runA_kappaFitted before run B overwrote the step outputs. Reports every A09 check side by
side with its pass/fail, the headline yields, the redefined/canonical ratio and the per-target
Fig. 11 comparison.
"""

import argparse
import json
import os


def fdictLoad(sRoot, sRel):
    """One JSON output under a run root."""
    with open(os.path.join(sRoot, sRel)) as oFile:
        return json.load(oFile)


def fdictHeadline(sRoot):
    """Headline numbers of one run."""
    dictCal = fdictLoad(sRoot, "calibrateToStark/calibration.json")
    dictPred = fdictLoad(sRoot, "predictRedefinedYield/yieldPrediction.json")
    dictCanon = dictPred["dictByBox"]["canonical"]
    dictRedef = dictPred["dictByBox"]["redefined"]
    dictFig = fdictLoad(sRoot, "CompareTargetCompleteness/targetCompletenessComparison.json")
    return {"fKappaApplied": dictCal["fCalibratedThroughputFactor"],
            "fPlanningYield6m": dictCal["fCalibratedYield"],
            "fCanonicalMedian": dictCanon["dictDistribution"]["fMedianExpected"],
            "fCanonicalFixedEtaMean": dictCanon["dictDistributionFixedEta"]["fMeanExpected"],
            "fRedefinedMedian": dictRedef["dictDistribution"]["fMedianExpected"],
            "fYieldRatioMedian": dictPred["dictYieldRatio"]["fMedian"],
            "fP25Canonical": dictCanon["dictDistribution"]["fProbabilityAtLeastGoal"],
            "fFig11MedianModelDrawMean": dictFig["dictDrawMean"]["fMedianModel"],
            "fFig11MedianPublished": dictFig["dictDrawMean"]["fMedianPublished"]}


def flistChecks(sRoot):
    """A09's check list of one run."""
    return fdictLoad(sRoot, "VerifyAgainstPublished/publishedVerification.json")["listChecks"]


def fnPrint(dictA, dictB, listA, listB):
    """Side-by-side table."""
    print(f"{'headline':<42}{'A: kappa fitted':>16}{'B: kappa = 1':>14}")
    for k in dictA:
        print(f"{k:<42}{dictA[k]:>16.3f}{dictB[k]:>14.3f}")
    dictB = {c["sName"]: c for c in listB}
    print(f"\n{'A09 check':<58}{'published':>10}{'A':>9}{'B':>9}  pass A/B")
    for c in listA:
        if c["bTuned"]:
            continue
        cB = dictB.get(c["sName"], {})
        print(f"{c['sName'][:57]:<58}{c['fPublished']:>10.4g}{c['fModel']:>9.4g}"
              f"{cB.get('fModel', float('nan')):>9.4g}  {int(c['bAgrees'])}/"
              f"{int(cB.get('bAgrees', 0))}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", default="output/runA_kappaFitted")
    p.add_argument("--run-b", default="..")
    p.add_argument("--out-json", default="output/kappaFittedVsUnity.json")
    dictArgs = vars(p.parse_args())
    dictA, dictB = fdictHeadline(dictArgs["run_a"]), fdictHeadline(dictArgs["run_b"])
    listA, listB = flistChecks(dictArgs["run_a"]), flistChecks(dictArgs["run_b"])
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"dictHeadlineA": dictA, "dictHeadlineB": dictB, "listChecksA": listA,
                   "listChecksB": listB}, oFile, indent=1)
    fnPrint(dictA, dictB, listA, listB)


if __name__ == "__main__":
    main()
