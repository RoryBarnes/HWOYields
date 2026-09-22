#!/usr/bin/env python3
"""Assemble the project report as LaTeX and compile it, injecting every number from step outputs.

No figure caption or result in the report is typed by hand: each is read from the JSON a
pipeline step wrote, so a re-run that changes a number changes the report too rather than
leaving a stale claim in print.
"""

import argparse
import json
import os
import re
import subprocess


def fdictLoadResults(sRepoRoot):
    """Read every step summary the report quotes."""
    dictPaths = {
        "catalog": "buildTargetCatalog/targetCatalogSummary.json",
        "calibration": "calibrateToStark/calibration.json",
        "completeness": "computeCompleteness/completenessSummary.json",
        "survey": "optimizeSurvey/surveyResult.json",
        "posterior": "sampleOccurrencePosterior/occurrencePosteriorSummary.json",
        "prediction": "predictRedefinedYield/yieldPrediction.json",
    }
    dictOut = {}
    for sKey, sRel in dictPaths.items():
        with open(os.path.join(sRepoRoot, sRel)) as oFile:
            dictOut[sKey] = json.load(oFile)
    return dictOut


def fdictSubstitutions(dictResults):
    """Map every @@token@@ in the template onto a formatted value from the results."""
    dictCat, dictCal = dictResults["catalog"], dictResults["calibration"]
    dictPost, dictPred = dictResults["posterior"], dictResults["prediction"]
    dictSurvey, dictComp = dictResults["survey"], dictResults["completeness"]
    dictCanon = dictPred["dictByBox"]["canonical"]["dictDistribution"]
    dictRedef = dictPred["dictByBox"]["redefined"]["dictDistribution"]
    dictEtaC, dictEtaR = dictPost["dictEtaByBox"]["canonical"], dictPost["dictEtaByBox"]["redefined"]
    dictRatio, dictYRatio = dictPost["dictRatioRedefinedOverCanonical"], dictPred["dictYieldRatio"]
    return {
        "CATROWS": f"{dictCat['iCatalogRows']:,}",
        "CATGAIA": f"{dictCat['iGaiaRows']:,}",
        "CATHIP": f"{dictCat['iHipparcosSupplementRows']:,}",
        "CATFGK": f"{dictCat['iFgkStars']:,}",
        "CALFACTOR": f"{dictCal['fCalibratedThroughputFactor']:.3f}",
        "CALYIELD": f"{dictCal['fCalibratedYield']:.2f}",
        "CALUNCAL": f"{dictCal['fUncalibratedYield']:.2f}",
        "CALUNCALPCT": f"{100*(dictCal['fUncalibratedYield']/22.5-1):.0f}",
        "CALSTARS": f"{dictCal['iStarsScreened']:,}",
        "SURVEYYIELD": f"{dictSurvey['fYieldBaseline']:.2f}",
        "SURVEYSTARS": f"{dictSurvey['iStarsUsedBaseline']:,}",
        "COMPCANON": f"{dictComp['dictByBox']['canonical']['iStarsWithAnyCompleteness']:,}",
        "COMPREDEF": f"{dictComp['dictByBox']['redefined']['iStarsWithAnyCompleteness']:,}",
        "COMPCANONMAX": f"{dictComp['dictByBox']['canonical']['fMaxCompletenessBest']:.3f}",
        "COMPREDEFMAX": f"{dictComp['dictByBox']['redefined']['fMaxCompletenessBest']:.3f}",
        "ETACMED": f"{dictEtaC['fMedian']:.3f}",
        "ETACUP": f"{dictEtaC['fP84']-dictEtaC['fMedian']:.3f}",
        "ETACDN": f"{dictEtaC['fMedian']-dictEtaC['fP16']:.3f}",
        "ETARMED": f"{dictEtaR['fMedian']:.4f}",
        "ETARUP": f"{dictEtaR['fP84']-dictEtaR['fMedian']:.4f}",
        "ETARDN": f"{dictEtaR['fMedian']-dictEtaR['fP16']:.4f}",
        "OCCRATIO": f"{dictRatio['fMedian']:.3f}",
        "OCCRATIOUP": f"{dictRatio['fP84']-dictRatio['fMedian']:.3f}",
        "OCCRATIODN": f"{dictRatio['fMedian']-dictRatio['fP16']:.3f}",
        "YRATIO": f"{dictYRatio['fMedian']:.3f}",
        "YRATIOUP": f"{dictYRatio['fP84']-dictYRatio['fMedian']:.3f}",
        "YRATIODN": f"{dictYRatio['fMedian']-dictYRatio['fP16']:.3f}",
        "CANONMEAN": f"{dictCanon['fMeanExpected']:.1f}",
        "CANONMED": f"{dictCanon['fMedianExpected']:.1f}",
        "CANONP5": f"{dictCanon['fP05Expected']:.1f}",
        "CANONP95": f"{dictCanon['fP95Expected']:.1f}",
        "CANONP25": f"{100*dictCanon['fProbabilityAtLeastGoal']:.1f}",
        "REDEFMEAN": f"{dictRedef['fMeanExpected']:.1f}",
        "REDEFMED": f"{dictRedef['fMedianExpected']:.1f}",
        "REDEFP5": f"{dictRedef['fP05Expected']:.1f}",
        "REDEFP95": f"{dictRedef['fP95Expected']:.1f}",
        "REDEFP25": f"{100*dictRedef['fProbabilityAtLeastGoal']:.2f}",
        "POSTSAMPLES": f"{dictPost['iSamples']:,}",
        "POSTACCEPT": f"{dictPost['fAcceptanceFraction']:.2f}",
    }


def fsRenderTemplate(sTemplatePath, dictSubs):
    """Replace every @@TOKEN@@ in the template, failing loudly on an unknown token."""
    with open(sTemplatePath) as oFile:
        sText = oFile.read()

    def fnReplace(oMatch):
        sKey = oMatch.group(1)
        if sKey not in dictSubs:
            raise KeyError(f"template references unknown token @@{sKey}@@")
        return dictSubs[sKey]

    sOut = re.sub(r"@@([A-Z0-9]+)@@", fnReplace, sText)
    saLeft = re.findall(r"@@([A-Z0-9]+)@@", sOut)
    assert not saLeft, f"unsubstituted tokens remain: {saLeft}"
    return sOut


def fnCompile(sTexPath, sOutDir):
    """Run pdflatex twice so references and the table of contents settle."""
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", sOutDir, sTexPath],
                       check=True, capture_output=True)


def fdictParseArgs():
    """Command-line configuration for the report build."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default="..")
    p.add_argument("--template", default="reportTemplate.tex")
    p.add_argument("--out-tex", default="hwoYieldRederivationReport.tex")
    p.add_argument("--out-dir", default=".")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    sRepoRoot = os.path.abspath(dictArgs["repo_root"])
    dictSubs = fdictSubstitutions(fdictLoadResults(sRepoRoot))
    sRendered = fsRenderTemplate(dictArgs["template"], dictSubs)
    with open(dictArgs["out_tex"], "w") as oFile:
        oFile.write(sRendered)
    fnCompile(dictArgs["out_tex"], dictArgs["out_dir"])
    print(json.dumps({"sTex": dictArgs["out_tex"],
                      "sPdf": os.path.splitext(dictArgs["out_tex"])[0] + ".pdf",
                      "iSubstitutions": len(dictSubs)}, indent=2))


if __name__ == "__main__":
    main()
