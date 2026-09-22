#!/usr/bin/env python3
"""Migrate every step's declared paths from repo-relative to step-relative, via vaibify-do.

Vaibify resolves saOutputDataFiles, saPlotFiles and dictTests.sFilePath relative to the step's
own directory, not the repository root. This project was authored with repo-relative paths, so
every declared output resolved to a doubled path such as
modelCoronagraph/modelCoronagraph/missionParameters.json and was reported missing. Figures live
in the repo-root Plot/ directory, so they need an explicit ../ prefix.
"""

import argparse
import json
import os
import subprocess


def flistStripStepPrefix(saPaths, sDirectory):
    """Drop a leading '<stepDirectory>/' from each declared output path."""
    sPrefix = sDirectory + "/"
    return [s[len(sPrefix):] if s.startswith(sPrefix) else s for s in saPaths]


def flistPlotPaths(saPaths, sPlotDirectory):
    """Point figure paths at the repo-root plot directory from inside the step directory."""
    sPrefix = sPlotDirectory + "/"
    return ["../" + s if s.startswith(sPrefix) else s for s in saPaths]


def fdictStepPatch(dictStep, sPlotDirectory):
    """The partial step object needed to correct one step's declared paths."""
    dictPatch = {
        "saOutputDataFiles": flistStripStepPrefix(dictStep.get("saOutputDataFiles", []),
                                                  dictStep["sDirectory"]),
        "saPlotFiles": flistPlotPaths(dictStep.get("saPlotFiles", []), sPlotDirectory),
    }
    dictTests = dictStep.get("dictTests")
    if dictTests:
        dictPatch["dictTests"] = {
            sCat: {"sFilePath": os.path.basename(dictCat["sFilePath"]),
                   "saCommands": dictCat["saCommands"]}
            for sCat, dictCat in dictTests.items()}
    return dictPatch


def fnApplyPatch(sLabel, dictPatch):
    """Send one step patch through vaibify-do so the edit is schema-validated and atomic."""
    oResult = subprocess.run(["vaibify-do", "update-step", sLabel, json.dumps(dictPatch)],
                             capture_output=True, text=True, timeout=120)
    return oResult.returncode, oResult.stdout[:200] + oResult.stderr[:200]


def fdictParseArgs():
    """Command-line configuration for the path-scope migration."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--dry-run", action="store_true")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    with open(dictArgs["project"]) as oFile:
        dictProject = json.load(oFile)
    sPlotDirectory = dictProject.get("sPlotDirectory", "Plot")
    for i, dictStep in enumerate(dictProject["listSteps"], start=1):
        sLabel = f"A{i:02d}"
        dictPatch = fdictStepPatch(dictStep, sPlotDirectory)
        if dictArgs["dry_run"]:
            print(sLabel, json.dumps(dictPatch))
            continue
        iCode, sOut = fnApplyPatch(sLabel, dictPatch)
        print(f"{sLabel} {dictStep['sName']:<28} exit={iCode} "
              f"outputs={dictPatch['saOutputDataFiles']} plots={dictPatch['saPlotFiles']}")


if __name__ == "__main__":
    main()
