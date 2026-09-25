#!/usr/bin/env python3
"""Decompose the disagreement with Stark's Fig. 10 into main effects of the albedo method and the eta interval reading.

The adopted model follows Stark's per-visit albedo test and reads his eta_Earth interval as 68%.
With the three variants that change one or both of those settings, the four runs form a complete
2x2 factorial:

                        albedo "recompute"      albedo "perVisitThreshold" (Stark's)
  eta at 68% (adopted)  albedoRecompute         baseline
  eta at 86% (his text) albedoRecomputeEta86    etaInterval86

For each observable this reports the two main effects and the interaction, in the units of the
observable, with each effect signed as (Stark's documented setting) minus (the alternative). The
question it answers is how much of the disagreement with Fig. 10 each setting carries.
"""

import argparse
import json

DICT_CELLS = {("recompute", "68"): "albedoRecompute", ("perVisit", "68"): "baseline",
              ("recompute", "86"): "albedoRecomputeEta86", ("perVisit", "86"): "etaInterval86"}
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
    return {"fAdopted": fAlbedo, "fAllDocumented": fBoth,
            "fEffectAlbedoMethod": fEffectAlbedo, "fEffectEtaInterval": fEffectEta,
            "fInteraction": (fBoth - fEta) - (fAlbedo - fLowLow),
            "fAdoptedToAllDocumented": fBoth - fAlbedo}


def fdictObservable(sKey, dictRuns, dictBaseline):
    """One observable's 2x2 effects, reading the baseline cell from the pipeline outputs."""
    dictCellValues = {}
    for tCell, sRun in DICT_CELLS.items():
        dictCellValues[tCell] = (dictBaseline if sRun == "baseline" else dictRuns[sRun])[sKey]
    return fdictEffects(dictCellValues)


def fnPrint(dictOut):
    """The decomposition table."""
    print(f"{'observable':<28}{'adopted':>10}{'all documented':>16}"
          f"{'albedo effect':>15}{'eta effect':>12}{'interaction':>13}")
    for sKey, sLabel, iRound in LIST_OBSERVABLES:
        d = dictOut[sKey]
        print(f"{sLabel:<28}{d['fAdopted']:>10.{iRound}f}{d['fAllDocumented']:>16.{iRound}f}"
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
