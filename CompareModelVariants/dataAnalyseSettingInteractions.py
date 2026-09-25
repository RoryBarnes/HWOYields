#!/usr/bin/env python3
"""Decompose the disagreement with Stark's Fig. 10 into main effects of the albedo method and the eta interval reading.

The adopted model and the fully paper-faithful model differ in exactly two settings, so the four
runs baseline / albedoPerVisit / etaInterval86 / paperFaithful form a complete 2x2 factorial:

                        albedo "recompute"      albedo "perVisitThreshold" (Stark's)
  eta at 68% (adopted)  baseline                albedoPerVisit
  eta at 86% (his text) etaInterval86           paperFaithful

For each observable this reports the two main effects and the interaction, in the units of the
observable. The question it answers is how much of the "faithful reproduction is worse" result is
attributable to each setting -- which is what decides whether the adopted model rests on one
documented departure or on a pattern of choosing whatever matches.
"""

import argparse
import json

DICT_CELLS = {("recompute", "68"): "baseline", ("perVisit", "68"): "albedoPerVisit",
              ("recompute", "86"): "etaInterval86", ("perVisit", "86"): "paperFaithful"}
LIST_OBSERVABLES = [("fShapeDistanceToFig10", "shape distance to Fig. 10", 4),
                    ("iYieldMode", "realized yield mode", 2),
                    ("fYieldMean", "realized yield mean", 2),
                    ("fProbBelow12", "P(yield < 12)", 4),
                    ("fYieldRatio", "yield ratio (deliverable)", 4),
                    ("fAlbedoPenalty", "albedo penalty", 4)]


def fdictEffects(dictCellValues):
    """Main effects and interaction of a 2x2, in the observable's own units."""
    fLowLow = dictCellValues[("recompute", "68")]
    fAlbedo = dictCellValues[("perVisit", "68")]
    fEta = dictCellValues[("recompute", "86")]
    fBoth = dictCellValues[("perVisit", "86")]
    fEffectAlbedo = 0.5 * ((fAlbedo - fLowLow) + (fBoth - fEta))
    fEffectEta = 0.5 * ((fEta - fLowLow) + (fBoth - fAlbedo))
    return {"fBaseline": fLowLow, "fBothChanged": fBoth,
            "fEffectAlbedoMethod": fEffectAlbedo, "fEffectEtaInterval": fEffectEta,
            "fInteraction": (fBoth - fEta) - (fAlbedo - fLowLow),
            "fTotalChange": fBoth - fLowLow}


def fdictObservable(sKey, dictRuns, dictBaseline):
    """One observable's 2x2 effects, reading the baseline cell from the pipeline outputs."""
    dictCellValues = {}
    for tCell, sRun in DICT_CELLS.items():
        dictCellValues[tCell] = (dictBaseline if sRun == "baseline" else dictRuns[sRun])[sKey]
    return fdictEffects(dictCellValues)


def fnPrint(dictOut):
    """The decomposition table."""
    print(f"{'observable':<28}{'adopted':>10}{'both changed':>14}"
          f"{'albedo effect':>15}{'eta effect':>12}{'interaction':>13}")
    for sKey, sLabel, iRound in LIST_OBSERVABLES:
        d = dictOut[sKey]
        print(f"{sLabel:<28}{d['fBaseline']:>10.{iRound}f}{d['fBothChanged']:>14.{iRound}f}"
              f"{d['fEffectAlbedoMethod']:>15.{iRound}f}{d['fEffectEtaInterval']:>12.{iRound}f}"
              f"{d['fInteraction']:>13.{iRound}f}")
    d = dictOut["fShapeDistanceToFig10"]
    fShare = abs(d["fEffectEtaInterval"]) / (abs(d["fEffectEtaInterval"])
                                             + abs(d["fEffectAlbedoMethod"]))
    print(f"\nof the movement in shape distance, the eta interval reading accounts for "
          f"{fShare:.1%} and the albedo method {1 - fShare:.1%}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="modelVariantsSummary.json")
    p.add_argument("--out-json", default="settingInteractions.json")
    dictArgs = vars(p.parse_args())
    dictSummary = json.load(open(dictArgs["summary"]))
    dictOut = {s: fdictObservable(s, dictSummary["dictVariants"], dictSummary["dictBaseline"])
               for s, _, _ in LIST_OBSERVABLES}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1)
    fnPrint(dictOut)


if __name__ == "__main__":
    main()
