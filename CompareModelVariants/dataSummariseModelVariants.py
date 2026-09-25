#!/usr/bin/env python3
"""Tabulate the one-variable experiments against the adopted baseline and the published values.

Reads the baseline step outputs and each variant directory written by
runOneVariableExperiments.py, and reports the deliverable (the redefined/canonical yield ratio)
plus the published-comparison quantities each choice was adopted to improve: the 6 m planning
yield (Stark 22.5), the fixed-eta realized level that Stark's Fig. 10 red curve sets (17.35), and
the exozodi and albedo penalties (11.1% and 12%). A stage a variant did not re-run is reported
from the baseline and marked unchanged, since the variant cannot have moved it.

The question each row answers is not "does this match Stark better" but "how much of the reported
answer depends on this choice" -- a deliverable that barely moves is robust to the choice, whatever
the choice does to the published comparisons.
"""

import argparse
import json
import os

import numpy as np

DICT_PUBLISHED = {"fYieldPlanningFixedExozodi": 22.5, "fYieldFixedEta": 17.35,
                  "fExozodiPenalty": 0.111, "fAlbedoPenalty": 0.12,
                  "iYieldMode": 10, "iYieldMedian": 18, "fYieldMean": 21.0,
                  "fProbBelow12": 0.29, "fProbAtLeast25": 0.32}
LIST_ROWS = [("fYieldRatio", "yield ratio (DELIVERABLE)"),
             ("fOccurrenceRatio", "occurrence ratio"),
             ("fEtaCanonical", "eta canonical (median)"),
             ("fEtaRedefined", "eta redefined (median)"),
             ("fYieldPlanningFixedExozodi", "6 m planning yield, fixed exozodi"),
             ("fYieldFixedEta", "fixed-eta realized level (Fig. 10 red)"),
             ("fExozodiPenalty", "exozodi penalty"),
             ("fAlbedoPenalty", "albedo penalty"),
             ("iStarsUsed", "stars observed"),
             ("iYieldMode", "realized yield mode"),
             ("iYieldMedian", "realized yield median"),
             ("fYieldMean", "realized yield mean"),
             ("fProbBelow12", "P(yield < 12)"),
             ("fProbAtLeast25", "P(yield >= 25)"),
             ("fTailAboveFigureAxis", "model mass beyond the figure axis"),
             ("fShapeDistanceToFig10", "shape distance to Fig. 10 purple")]


def fdictYieldShape(sSamplesPath, faPublishedPmf):
    """Shape of the realized canonical-box yield distribution, and its distance from Fig. 10.

    The mode is taken on the full support: clipping the model's tail into the figure's last bin
    puts a spurious spike there. The total-variation distance is then computed on the bins the
    published digitization covers, with BOTH distributions renormalized over that range, because
    Stark's figure is truncated at its axis end and carries mass beyond it too.
    """
    iaObserved = np.load(sSamplesPath)["faObserved_canonical"]
    faFull = np.bincount(iaObserved, minlength=len(faPublishedPmf)) / iaObserved.size
    iGrid = len(faPublishedPmf)
    faModel = faFull[:iGrid] / faFull[:iGrid].sum()
    faPublished = faPublishedPmf / faPublishedPmf.sum()
    return {"iYieldMode": int(np.argmax(faFull)), "iYieldMedian": int(np.median(iaObserved)),
            "fYieldMean": float(iaObserved.mean()),
            "fProbBelow12": float((iaObserved < 12).mean()),
            "fProbAtLeast25": float((iaObserved >= 25).mean()),
            "fTailAboveFigureAxis": float(faFull[iGrid:].sum()),
            "fShapeDistanceToFig10": float(0.5 * np.abs(faModel - faPublished).sum())}


def fdictReadRun(sDir, sBaselineDir, listStagesRerun, faPublishedPmf):
    """Observables for one run, falling back to the baseline for stages it did not re-run."""
    def fsPick(sFile, sStage):
        sOwn = os.path.join(sDir, sFile)
        bOwn = sStage in listStagesRerun and os.path.exists(sOwn)
        return (sOwn if bOwn else os.path.join(sBaselineDir, sFile)), bOwn
    sSurvey, bSurvey = fsPick("surveyResult.json", "survey")
    sPredict, bPredict = fsPick("yieldPrediction.json", "prediction")
    sSamples, _ = fsPick("yieldSamples.npz", "prediction")
    dictSurvey, dictPredict = json.load(open(sSurvey)), json.load(open(sPredict))
    dictDraws = dictSurvey["dictDraws"]
    return {**fdictYieldShape(sSamples, faPublishedPmf),
            "fYieldRatio": dictPredict["dictYieldRatio"]["fMedian"],
            "fYieldRatioP16": dictPredict["dictYieldRatio"]["fP16"],
            "fYieldRatioP84": dictPredict["dictYieldRatio"]["fP84"],
            "fOccurrenceRatio": dictPredict["dictYieldRatio"]["fOccurrenceRatioMedian"],
            "fEtaCanonical": dictPredict["dictByBox"]["canonical"]["fEtaMedian"],
            "fEtaRedefined": dictPredict["dictByBox"]["redefined"]["fEtaMedian"],
            "fYieldPlanningFixedExozodi": dictSurvey["fYieldPlanningFixedExozodi"],
            "fYieldFixedEta": dictSurvey["fYieldBaseline"],
            "fExozodiPenalty": dictDraws["fExozodiPenalty"],
            "fAlbedoPenalty": dictDraws["fAlbedoPenalty"],
            "iStarsUsed": dictSurvey["iStarsUsedBaseline"],
            "bSurveyRerun": bSurvey, "bPredictionRerun": bPredict}


def fsFormatCell(sKey, dictRun, dictBase):
    """One cell: the value, and its shift from the baseline where the variant could move it."""
    fVal, fBase = dictRun[sKey], dictBase[sKey]
    sVal = f"{fVal:.4f}" if abs(fVal) < 10 else f"{fVal:.2f}"
    bMovable = dictRun["bSurveyRerun"] or dictRun["bPredictionRerun"]
    if not bMovable or fBase == 0:
        return f"{sVal:>10}"
    fRel = (fVal - fBase) / abs(fBase)
    return f"{sVal:>10} ({fRel:+6.1%})" if abs(fRel) >= 0.0005 else f"{sVal:>10} (  same)"


def fnPrintTable(dictBase, dictRuns, dictManifests):
    """The comparison table."""
    listNames = list(dictRuns)
    print(f"{'observable':<40}{'baseline':>12}  " +
          "  ".join(f"{s:>19}" for s in listNames) + "   published")
    for sKey, sLabel in LIST_ROWS:
        fBase = dictBase[sKey]
        sBase = f"{fBase:.4f}" if abs(fBase) < 10 else f"{fBase:.2f}"
        sPub = (f"{DICT_PUBLISHED[sKey]:.3f}" if sKey in DICT_PUBLISHED else "-")
        print(f"{sLabel:<40}{sBase:>12}  " +
              "  ".join(fsFormatCell(sKey, dictRuns[s], dictBase) for s in listNames) +
              f"   {sPub}")
    print("\n90% interval on the deliverable:")
    print(f"  {'baseline':<24}[{dictBase['fYieldRatioP16']:.3f}, {dictBase['fYieldRatioP84']:.3f}]")
    for sName, dictRun in dictRuns.items():
        print(f"  {sName:<24}[{dictRun['fYieldRatioP16']:.3f}, {dictRun['fYieldRatioP84']:.3f}]"
              f"   ({dictManifests[sName]['sWhy']})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="..")
    p.add_argument("--out-root", default="variants")
    p.add_argument("--digitised-figure-ten", required=True)
    p.add_argument("--survey", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--yield-samples", required=True)
    p.add_argument("--out-json", default="modelVariantsSummary.json")
    dictArgs = vars(p.parse_args())
    sRoot = dictArgs["out_root"]
    dictManifests = {d["sVariant"]: d for d in
                     json.load(open(os.path.join(sRoot, "manifests.json")))["listVariants"]}
    faPublishedPmf = np.array(json.load(
        open(dictArgs["digitised_figure_ten"]))["including"]["faProbability"])
    dictBaselineDirs = {"surveyResult.json": dictArgs["survey"],
                        "yieldPrediction.json": dictArgs["prediction"],
                        "yieldSamples.npz": dictArgs["yield_samples"]}
    sBaselineStage = os.path.join(sRoot, "_baselineLinks")
    os.makedirs(sBaselineStage, exist_ok=True)
    for sFile, sPath in dictBaselineDirs.items():
        sLink = os.path.join(sBaselineStage, sFile)
        if os.path.islink(sLink) or os.path.exists(sLink):
            os.remove(sLink)
        os.symlink(os.path.abspath(sPath), sLink)
    dictBase = fdictReadRun(sBaselineStage, sBaselineStage, ["survey", "prediction"],
                            faPublishedPmf)
    dictRuns = {s: fdictReadRun(os.path.join(sRoot, s), sBaselineStage, d["listStagesRerun"],
                                faPublishedPmf)
                for s, d in dictManifests.items()}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump({"dictBaseline": dictBase, "dictVariants": dictRuns,
                   "dictManifests": dictManifests, "dictPublished": DICT_PUBLISHED}, oFile,
                  indent=1)
    fnPrintTable(dictBase, dictRuns, dictManifests)


if __name__ == "__main__":
    main()
