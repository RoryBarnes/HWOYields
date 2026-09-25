#!/usr/bin/env python3
"""Assemble the project report as LaTeX and compile it, injecting every number from step outputs.

No figure caption or result in the report is typed by hand: each is read from the JSON a
pipeline step wrote, so a re-run that changes a number changes the report too rather than
leaving a stale claim in print.
"""

import argparse
import json
import math
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
        "aperture": "CompareApertureScaling/apertureScaling.json",
        "budget": "explorations/timeBudgetSplit.json",
        "budgetNoChar": "explorations/timeBudgetSplitNoChar.json",
        "screen": "explorations/targetSelectionAudit.json",
        "photometry": "buildTargetCatalog/stellarFluxPhotometryCheck.json",
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
    dictAp = dictResults["aperture"]
    dictBud, dictNoChar = dictResults["budget"]["listRows"], dictResults["budgetNoChar"]["listRows"]
    dictScr = dictResults["screen"]
    dictPhot = dictResults["photometry"]
    fExpWith = math.log(dictBud[1]["fYield"] / dictBud[0]["fYield"]) / math.log(1.5)
    fExpNo = math.log(dictNoChar[1]["fYield"] / dictNoChar[0]["fYield"]) / math.log(1.5)
    dictCanon = dictPred["dictByBox"]["canonical"]["dictDistribution"]
    dictRedef = dictPred["dictByBox"]["redefined"]["dictDistribution"]
    dictEtaC, dictEtaR = dictPost["dictEtaByBox"]["canonical"], dictPost["dictEtaByBox"]["redefined"]
    dictRatio, dictYRatio = dictPost["dictRatioRedefinedOverCanonical"], dictPred["dictYieldRatio"]
    return {
        "CATROWS": f"{dictCat['iCatalogRows']:,}",
        "CATFGK": f"{dictCat['iFgkStars']:,}",
        "PHOTN": f"{dictPhot['dictSunLikeResidual']['iStars']:,}",
        "PHOTMED": f"{dictPhot['dictSunLikeResidual']['fMedian']:+.3f}",
        "PHOTSTD": f"{dictPhot['dictSunLikeResidual']['fStdDev']:.2f}",
        "CALUNCALFRAC": f"{100*dictCal['fUncalibratedYield']/22.5:.0f}",
        "ALBEDOPEN": f"{100*(1.0 - dictSurvey['fYieldBaseline']/dictSurvey['fYieldPlanningBaseline']):.1f}",
        "CALFACTOR": f"{dictCal['fCalibratedThroughputFactor']:.3f}",
        "CALYIELD": f"{dictCal['fCalibratedYield']:.2f}",
        "CALUNCAL": f"{dictCal['fUncalibratedYield']:.2f}",
        "CALUNCALPCT": f"{100*(dictCal['fUncalibratedYield']/22.5-1):+.1f}",
        "CALUNCALPCTABS": f"{abs(100*(dictCal['fUncalibratedYield']/22.5-1)):.1f}",
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
        "APINCL": " / ".join(f"{100*dictAp['dictByDiameter'][s]['fProbability25IncludingSigmaEta']:.0f}"
                             for s in ("6", "7", "8", "9")),
        "APINCLPUB": " / ".join(f"{100*dictAp['dictPublished']['dictIncludingSigmaEta'][s]:.0f}"
                                for s in ("6", "7", "8", "9")),
        "APEXCL": " / ".join(f"{100*dictAp['dictByDiameter'][s]['fProbability25ExcludingSigmaEta']:.0f}"
                             for s in ("6", "7", "8", "9")),
        "APEXCLPUB": " / ".join(f"{100*dictAp['dictPublished']['dictExcludingSigmaEta'][s]:.0f}"
                                for s in ("6", "7", "8", "9")),
        "APMAXRESID": f"{100*max(abs(dictAp['dictResiduals'][s]['fIncludingSigmaEta']) for s in ('6','7','8','9')):.1f}",
        "APCROSS": f"{dictAp['fCrossingDiameterM']:.1f}",
        "APCROSSPUB": f"{dictAp['fCrossingDiameterPublishedM']:.1f}",
        "APYIELDS": " / ".join(f"{dictAp['dictByDiameter'][s]['fExpectedYieldAtBaselineEta']:.1f}"
                               for s in ("6", "7", "8", "9")),
        "SCREENLOST": f"{dictScr['iLostWrongly']:,}",
        "SCREENLMAX": f"{dictScr['fMaxLuminosityAdmittedAsWritten']:.1f}",
        "SCREENLMAXOK": f"{dictScr['fMaxLuminosityAdmittedCorrect']:.0f}",
        "EXPWITHCHAR": f"{fExpWith:.2f}",
        "EXPNOCHAR": f"{fExpNo:.2f}",
        "CHARFRAC6": f"{100*dictBud[0]['fCharFraction']:.0f}",
        "CHARFRAC9": f"{100*dictBud[1]['fCharFraction']:.0f}",
        "STARSUSED6": f"{dictBud[0]['iStarsUsed']}",
        "STARSUSED9": f"{dictBud[1]['iStarsUsed']}",
        "APEXPONENT": f"{math.log(dictAp['dictByDiameter']['9']['fExpectedYieldAtBaselineEta'] / dictAp['dictByDiameter']['6']['fExpectedYieldAtBaselineEta']) / math.log(1.5):.2f}",
        "ETASAG13": "0.2404",
        "CALFACTORINV": f"{1.0/dictCal['fCalibratedThroughputFactor']:.2f}",
        "GATEPASS": "passes" if dictCal.get("bCalibrationGatePassed") else "FAILS",
        "ETAMODE": f"{dictPost.get('fEtaModeImplied', float('nan')):.3f}",
        "BIASMEAN": f"{dictCanon['fMeanExpected'] * 17.3 / 22.5:.1f}",
        "FACTORPLAUSIBLE": "within" if dictCal.get("bFactorPlausible") else "outside",
    }


DICT_EXTRA_PATHS = {
    "verification": "VerifyAgainstPublished/publishedVerification.json",
    "fig10": "VerifyAgainstPublished/starkFigure10Digitised.json",
    "hosts": "modelCoronagraph/reference/starkExozodiHostsMaxLikelihood.json",
    "gridCheck": "explorations/output/exozodiGridValidation.json",
    "skyWhatIf": "explorations/output/whatIfSkyThroughput.json",
    "earthZone": "TabulateEarthZones/earthZoneSummary.json",
    "pairs": "ComparePublishedFigures/publishedComparisonSeries.json",
    "targets": "CompareTargetCompleteness/targetCompletenessComparison.json",
    "etaLaw": "explorations/output/etaLawFromStarkFigure10.json",
    "lumBins": "explorations/output/targetCompletenessBinned.json",
    "twin": "explorations/output/characterizationNoiseBreakdown.json",
    "degeneracy": "CalibrationDegeneracy/calibrationDegeneracySummary.json",
    "variants": "CompareModelVariants/modelVariantsSummary.json",
    "interactions": "CompareModelVariants/settingInteractions.json",
    "peaks": "CompareModelVariants/yieldPeakComparison.json",
    "etaInterval": "TestOccurrenceIntervalReading/etaIntervalConfidenceLevel.json",
}


def fdictDegeneracyTokens(dictDeg):
    """Tokens from the throughput-degeneracy design (A13)."""
    dictGate = dictDeg["dictYieldGatedEnvelope"]
    faSv = dictDeg["faSensitivitySingularValues"]
    dictPost = dictDeg["dictPosteriors"]["yieldOnly"]
    dictChar = [d for d in dictPost["listObservables"] if "char time, 6" in d["sLabel"]][0]
    return {
        "DEGPOINTS": f"{dictDeg['iDesignPoints']:d}",
        "DEGSVRATIO": f"{faSv[0] / faSv[1]:.1f}",
        "DEGSVLIST": ", ".join(f"{f:.2f}" for f in faSv),
        "DEGKAPPA": f"{dictPost['dictParameters']['kappa']['fMedian']:.2f}",
        "DEGKAPPAC": f"{dictPost['dictParameters']['kappa_c']['fMedian']:.2f}",
        "DEGCORR": f"{dictPost['fKappaKappaCCorrelation']:.2f}",
        "DEGCHARSIX": f"{dictChar['fMedian']:.1f}",
        "DEGCHARSIXLO": f"{dictChar['fLow']:.1f}",
        "DEGCHARSIXHI": f"{dictChar['fHigh']:.1f}",
        "DEGENVCHARMAX": f"{dictDeg['dictDesignEnvelope']['dictRange']['ln_char18_6']['fMax']:.1f}",
        "DEGGATEN": f"{dictGate['iPoints']:d}",
        "DEGGATECHARLO": f"{dictGate['fChar6Min']:.1f}",
        "DEGGATECHARHI": f"{dictGate['fChar6Max']:.1f}",
        "DEGGATECHARNINELO": f"{dictGate['fChar9Min']:.2f}",
        "DEGGATECHARNINEHI": f"{dictGate['fChar9Max']:.2f}"}


def fdictVariantTokens(dictVar, dictInt, dictPeaks, dictEta):
    """Tokens from the one-variable experiments (A14) and the interval test (A15)."""
    dictOut = {}
    for sName, sTag in [("baseline", "BASE"), ("albedoPerVisit", "ALB"),
                        ("etaInterval86", "ETA86"), ("skyThroughputConstant", "SKY"),
                        ("paperFaithful", "FAITH"), ("allThreeCombined", "ALL3"),
                        ("brysonMixtureEta", "BRY")]:
        dictRun = dictVar["dictBaseline"] if sName == "baseline" else dictVar["dictVariants"][sName]
        dictOut[f"VAR{sTag}RATIO"] = f"{dictRun['fYieldRatio']:.4f}"
        dictOut[f"VAR{sTag}TV"] = f"{dictRun['fShapeDistanceToFig10']:.3f}"
        dictOut[f"VAR{sTag}MEAN"] = f"{dictRun['fYieldMean']:.1f}"
        dictOut[f"VAR{sTag}BELOW"] = f"{dictRun['fProbBelow12']:.2f}"
        dictOut[f"PEAK{sTag}"] = f"{dictPeaks[sName]['iMode']:d}"
    dictShape = dictInt["fShapeDistanceToFig10"]
    fShare = abs(dictShape["fEffectEtaInterval"]) / (abs(dictShape["fEffectEtaInterval"]) +
                                                     abs(dictShape["fEffectAlbedoMethod"]))
    dictOut.update({
        "INTETA": f"{dictShape['fEffectEtaInterval']:+.4f}",
        "INTALB": f"{dictShape['fEffectAlbedoMethod']:+.4f}",
        "INTCROSS": f"{dictShape['fInteraction']:+.4f}",
        "INTETASHARE": f"{100 * fShare:.1f}",
        "INTALBSHARE": f"{100 * (1 - fShare):.1f}",
        "PEAKPUB": f"{dictPeaks['published']['iMode']:d}",
        "ETAMASSPCT": f"{100 * dictEta['fMassInQuotedInterval']:.0f}",
        "ETAMIXMEDIAN": f"{dictEta['fMedian']:.3f}",
        "ETAMIX68LO": f"{dictEta['faInterval68'][0]:.3f}",
        "ETAMIX68HI": f"{dictEta['faInterval68'][1]:.3f}",
        "ETAMIX86LO": f"{dictEta['faInterval86'][0]:.3f}",
        "ETAMIX86HI": f"{dictEta['faInterval86'][1]:.3f}"})
    return dictOut


def fdictLoadExtra(sRepoRoot):
    """Read the verification table and the diagnostic outputs the model sections quote."""
    dictOut = {}
    for sKey, sRel in DICT_EXTRA_PATHS.items():
        with open(os.path.join(sRepoRoot, sRel)) as oFile:
            dictOut[sKey] = json.load(oFile)
    return dictOut


def fsLatexEscape(sText):
    """Escape the characters that appear in check names and sources."""
    for sOld, sNew in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"),
                       ("#", "\\#")):
        sText = sText.replace(sOld, sNew)
    return sText


def fsVerificationRows(dictVerif):
    """One LaTeX table row per independent check, in A09's order."""
    listRows = []
    for d in dictVerif["listChecks"]:
        if d["bTuned"]:
            continue
        sVerdict = "agrees" if d["bAgrees"] else "\\textbf{disagrees}"
        listRows.append(f"{fsLatexEscape(d['sName'])} & {d['fPublished']:.4g} & "
                        f"{d['fModel']:.4g} & {100*d['fRelativeDifference']:.0f}\\% & "
                        f"{100*d['fTolerance']:.0f}\\% & {sVerdict} \\\\")
    return "\n".join(listRows)


def ffCheck(dictVerif, sName):
    """Model value of one named A09 check."""
    return next(d["fModel"] for d in dictVerif["listChecks"] if d["sName"] == sName)


def fdictModelTokens(dictX, dictResults, sRepoRoot):
    """Tokens for the graphical-model, validation and open-problem sections."""
    dictV, dictF, dictSurvey = dictX["verification"], dictX["fig10"], dictResults["survey"]
    dictPost, dictDraw = dictResults["posterior"], dictSurvey["dictDraws"]
    dictEta, dictLow = dictX["etaLaw"], dictX["lumBins"]["dictByLuminosity"]
    dictAp = dictResults["aperture"]["dictByDiameter"]
    listIndep = [d for d in dictV["listChecks"] if not d["bTuned"]]
    fRedMean = dictF["excluding"]["dictSummary"]["fMeanTruncated"]
    fSix, fNine = dictAp["6"]["fMeanCharDaysFirstN"], dictAp["9"]["fMeanCharDaysFirstN"]
    return {
        "NINDEP": str(len(listIndep)), "NAGREE": str(sum(d["bAgrees"] for d in listIndep)),
        "VERIFTABLE": fsVerificationRows(dictV),
        "ETASIGMA": f"{dictPost['fEtaLogNormalSigma']:.2f}",
        "ETAZ": f"{dictPost.get('fEtaIntervalZ', 1.4758):g}",
        "ETAMEAN": f"{dictPost['fEtaMeanImplied']:.3f}",
        "FIGREDMEAN": f"{fRedMean:.2f}",
        "FIGMEDIAN": f"{dictF['including']['dictSummary']['iMedian']}",
        "FIGMODE": f"{dictF['including']['dictSummary']['iMode']}",
        "FIGMEANLINE": f"{dictF['fMeanLineIncluding']:.1f}",
        "MODELMEDIAN": f"{ffCheck(dictV, 'Realized-yield distribution median'):.0f}",
        "MODELBELOW": f"{ffCheck(dictV, 'Realized-yield fraction below 12 EECs'):.2f}",
        "MODELMODE": f"{ffCheck(dictV, 'Realized-yield distribution mode'):.0f}",
        "FIXEDETA": f"{dictSurvey['fYieldBaseline']:.2f}",
        "PLANFIXED": f"{dictSurvey['fYieldPlanningFixedExozodi']:.2f}",
        "PLANDRAWN": f"{dictSurvey['fYieldPlanningBaseline']:.2f}",
        "HOSTSMEDIAN": f"{dictX['hosts']['dictSummary']['fMedianZodi']:.2f}",
        "HOSTSABOVE": f"{100*dictX['hosts']['dictSummary']['fFractionAbove100']:.0f}",
        "HOSTSMASS": f"{dictX['hosts']['fMassOnAxis']:.3f}",
        "NDRAWS": str(dictSurvey["iExozodiDraws"]),
        "DRAWMEAN": f"{dictDraw['fYieldDrawnMean']:.2f}",
        "DRAWSTD": f"{dictDraw['fYieldDrawnStd']:.2f}",
        "DRAWEXOPEN": f"{100*dictDraw['fExozodiPenalty']:.1f}",
        "GRIDERRPCT": f"{100*dictX['gridCheck']['fMaxRelativeYieldError']:.2f}",
        "LEVELPCT": f"{100*(dictSurvey['fYieldBaseline']/fRedMean-1):.1f}",
        "BESTSIGMA": f"{dictEta['dictBestFit']['fSigma']:.2f}",
        "BESTMEDIAN": f"{dictEta['dictBestFit']['fMedian']:.2f}",
        "BESTTV": f"{dictEta['dictBestFit']['fTotalVariation']:.3f}",
        "QSIXEIGHTTV": f"{dictEta['dictQuotedAs68Percent']['fTotalVariation']:.3f}",
        "QEIGHTSIXTV": f"{dictEta['dictQuotedInterval']['fTotalVariation']:.3f}",
        "QEIGHTSIXMODE": f"{dictEta['dictQuotedInterval']['dictShape']['iMode']}",
        "QSIXEIGHTMODE": f"{dictEta['dictQuotedAs68Percent']['dictShape']['iMode']}",
        "QEIGHTSIXSIGMA": f"{dictEta['dictQuotedInterval']['fSigma']:.2f}",
        "QSIXEIGHTSIGMA": f"{dictEta['dictQuotedAs68Percent']['fSigma']:.2f}",
        "CHARSIX": f"{fSix:.1f}", "CHARNINE": f"{fNine:.2f}", "CHARRATIO": f"{fSix / fNine:.1f}",
        "RATIOEXCESS": f"{100*(dictResults['prediction']['dictYieldRatio']['fMedian'] / dictResults['posterior']['dictRatioRedefinedOverCanonical']['fMedian'] - 1):.0f}",
        **fdictLuminosityTokens(dictLow), **fdictTwinTokens(dictX["twin"]),
        **fdictDistanceTokens(dictX["lumBins"]["dictByDistance"]),
        **fdictSkyTokens(dictX["skyWhatIf"]),
        **fdictEarthZoneTokens(dictX["earthZone"], sRepoRoot),
        "PAIRFIGURES": fsPairFigures(dictX, dictResults),
    }


def fdictSkyTokens(dictSky):
    """Constant-versus-map T_sky: uncalibrated yield, fitted factor, first-18 char times."""
    dictOut = {}
    for sVariant, sTok in (("current", "CONST"), ("skyFollowsCore", "SKY")):
        d = dictSky[sVariant]
        dictOut[f"WIUNCAL{sTok}"] = f"{d['fUncalibratedPlanning']:.2f}"
        dictOut[f"WIKAPPA{sTok}"] = f"{d['fCalibration']:.3f}"
        dictOut[f"WICHAR{sTok}"] = (f"{d['fMeanCharDaysPriorityFirst18']:.1f} / "
                                    f"{d['fMeanCharDaysPriorityFirst18_9m']:.2f}")
    return dictOut


def fsBinRows(dictBins):
    """LaTeX rows of the Earth-zone versus classic-zone binned comparison."""
    listRows = []
    for sLabel, d in dictBins.items():
        sRatio = f"{d['fYieldRatio']:.2f}" if d["fYieldRatio"] is not None else "--"
        listRows.append(f"{sLabel.replace('<', '$<$').replace('>', '$>$')} & {d['iStars']} & "
                        f"{d['iObservedClassicHz']} & {d['iObservedEarthZone']} & "
                        f"{d['fYieldClassicHz']:.2f} & {d['fYieldEarthZone']:.2f} & {sRatio} \\\\")
    return "\n".join(listRows)


F_EZ_LIST_MIN = 0.01


def fsStarRows(sCsvPath):
    """One longtable row per star contributing at least F_EZ_LIST_MIN EEC in either zone."""
    import csv
    listRows = []
    with open(sCsvPath) as oFile:
        for d in csv.DictReader(oFile):
            fHz, fEz = float(d["fYieldClassicHz"]), float(d["fYieldEarthZone"])
            if max(fHz, fEz) < F_EZ_LIST_MIN:
                continue
            sRatio = f"{fEz / fHz:.2f}" if fHz > 1e-4 else "--"
            sName = fsLatexEscape(d["sName"] or d["sSourceId"])
            listRows.append(
                f"{sName} & {fsLatexEscape(d['sSpectralType'][:7])} & "
                f"{float(d['fDistancePc']):.2f} & {float(d['fLuminosityLsun']):.3g} & "
                f"{float(d['fEarthZoneInnerMas']):.0f}--{float(d['fEarthZoneOuterMas']):.0f} & "
                f"{float(d['fClassicHzInnerMas']):.0f}--{float(d['fClassicHzOuterMas']):.0f} & "
                f"{float(d['fCompClassicHz']):.2f} & {float(d['fCompEarthZone']):.2f} & "
                f"{fHz:.3f} & {fEz:.3f} & {sRatio} \\\\")
    return "\n".join(listRows)


def fdictEarthZoneTokens(dictEz, sRepoRoot):
    """Tokens for the Earth-zone section and the per-star appendix (A11)."""
    dictAng, dictRatio = dictEz["dictAngular"], dictEz["dictPerStarYieldRatio"]
    sRows = fsStarRows(os.path.join(sRepoRoot, "TabulateEarthZones/earthZoneTable.csv"))
    return {
        "EZNSTARS": f"{dictEz['iStarsTabulated']:,}", "EZNLISTED": str(sRows.count("\\\\")),
        "EZLISTMIN": f"{F_EZ_LIST_MIN:g}",
        "EZETAC": f"{dictEz['dictEta']['canonical']:.3f}",
        "EZETAR": f"{dictEz['dictEta']['redefined']:.4f}",
        "EZETARATIO": f"{dictEz['fEtaRatio']:.3f}",
        "EZYIELDHZ": f"{dictEz['fYieldClassicHz']:.2f}", "EZYIELDEZ": f"{dictEz['fYieldEarthZone']:.2f}",
        "EZYIELDRATIO": f"{dictEz['fYieldRatio']:.3f}",
        "EZSTARRATIO": f"{dictRatio['fMedian']:.3f}", "EZSTARRATIOLO": f"{dictRatio['fP16']:.3f}",
        "EZSTARRATIOHI": f"{dictRatio['fP84']:.3f}",
        "EZNOBSHZ": str(dictEz["iStarsObservedClassicHz"]),
        "EZNOBSEZ": str(dictEz["iStarsObservedEarthZone"]),
        "EZCOMPHZ": f"{dictEz['fMeanCompletenessObservedClassicHz']:.2f}",
        "EZCOMPEZ": f"{dictEz['fMeanCompletenessObservedEarthZone']:.2f}",
        "EZIWA": f"{dictAng['fIwaInscribedLamD']:.2f}",
        "EZOUTIWAEZ": f"{100*dictAng['fEarthZoneMeanFractionOutsideIwa1000']:.0f}",
        "EZOUTIWAHZ": f"{100*dictAng['fClassicHzMeanFractionOutsideIwa1000']:.0f}",
        "EZOUTIWAEZSIX": f"{100*dictAng['fEarthZoneMeanFractionOutsideIwa550']:.0f}",
        "EZOUTIWAHZSIX": f"{100*dictAng['fClassicHzMeanFractionOutsideIwa550']:.0f}",
        "EZLUMROWS": fsBinRows(dictEz["dictByLuminosity"]),
        "EZDISTROWS": fsBinRows(dictEz["dictByDistance"]),
        "EZSTARROWS": sRows,
    }


def ffTotalVariation(faP, faQ, iN=50):
    """Total-variation distance over k < iN, each PMF renormalized there.

    The digitized Fig. 10 curves are normalized over the plotted range 0-49, so the model's PMF
    is truncated and renormalized the same way before the two are compared.
    """
    fSumP, fSumQ = sum(faP[:iN]), sum(faQ[:iN])
    return 0.5 * sum(abs(faP[k] / fSumP - faQ[k] / fSumQ) for k in range(iN))


def faDigitisedPmf(dictCurve, iN=60):
    """A digitized Fig. 10 curve as P(k), k < iN."""
    faOut = [0.0] * iN
    for k, f in zip(dictCurve["iaYield"], dictCurve["faProbability"]):
        if 0 <= k < iN:
            faOut[k] = f
    return faOut


def fdictPairCaptions(dictX, dictResults):
    """Quantitative caption text for each published/model figure pair."""
    dictS, dictSurvey, dictF = dictX["pairs"], dictResults["survey"], dictX["fig10"]
    dictAp, dictT = dictResults["aperture"], dictX["targets"]["dictRepresentative"]
    fTvRed = ffTotalVariation(faDigitisedPmf(dictF["excluding"]), dictS["fig10"]["faPmfExcluding"])
    fTvPur = ffTotalVariation(faDigitisedPmf(dictF["including"]), dictS["fig10"]["faPmfIncluding"])
    fYield6 = dictAp["dictByDiameter"]["6"]["fExpectedYieldAtBaselineEta"]
    fYield9 = dictAp["dictByDiameter"]["9"]["fExpectedYieldAtBaselineEta"]
    sP25 = lambda sKey: " / ".join(f"{100*dictAp['dictByDiameter'][s][sKey]:.0f}"
                                   for s in ("6", "7", "8", "9"))
    sChar = " / ".join(f"{dictAp['dictByDiameter'][s]['fMeanCharDaysFirstN']:.1f}"
                       for s in ("6", "7", "8", "9"))
    return {
        4: ("Exoplanet sampling only: $A_G = 0.2$, exozodi fixed at 3 zodis, Poisson counting. "
            f"Published mean 22.5, standard deviation 21\\% of the mean. Model mean "
            f"{dictS['fig04']['fMean']:.1f}, which is \\emph{{tuned}}: $\\kappa$ is fitted to 22.5."),
        7: ("Albedo drawn with exozodi fixed (green), and exozodi drawn as well (orange), $\\eta_\\oplus$ "
            f"fixed. Published means 19.8 and 17.6. Model means {dictS['fig07']['fMeanAlbedo']:.1f} and "
            f"{dictS['fig07']['fMeanAlbedoExozodi']:.1f} over {dictSurvey['iExozodiDraws']} draws."),
        9: ("Left: exozodi levels drawn from the LBTI HOSTS maximum-likelihood distribution (red); the "
            "model panel is the histogram of the levels the pipeline actually drew, scaled to "
            f"Stark's $500\\times10^4$ draws (median {dictS['fig09']['fMedianZodi']:.2f} zodis; Stark's "
            "other three distributions are not modelled). Right: the $\\eta_\\oplus$-fixed yield "
            "those levels produce (red)."),
        10: ("Realized yield including (purple) and excluding (red) $\\eta_\\oplus$ uncertainty; the "
             "dotted line marks the purple mean. Total-variation distance, model to published: "
             f"{fTvPur:.3f} (purple), {fTvRed:.3f} (red). Means: published "
             f"{dictF['fMeanLineIncluding']:.1f} / {dictF['excluding']['dictSummary']['fMeanTruncated']:.1f}, "
             f"model {dictS['fig10']['fMeanIncluding']:.1f} / {dictS['fig10']['fMeanExcluding']:.1f}."),
        11: ("Targets selected in one representative exozodi draw, coloured by completeness, over the "
             f"input list in grey. Published: {dictT['iPublishedTargets']} targets, summed completeness "
             f"{dictT['fPublishedSummedCompleteness']:.1f}, median {dictT['fMedianPublished']:.2f}. "
             f"Model (draw {dictX['targets']['iRepresentativeDraw']}, the median-yield draw): "
             f"{dictT['iModelStarsUsed']} targets, summed {dictT['fModelSummedCompleteness']:.1f}, "
             f"median {dictT['fMedianModel']:.2f} over the {dictT['iMatched']} matched targets. Red "
             "dashed lines: the noise floor for a 1.4\\,$R_\\oplus$ planet at the EEID, and the HZ outer "
             "edge at 1.5\\,$\\lambda/D$ at 1\\,$\\mu$m; the spectral-type lines are not redrawn."),
        12: ("Realized yield at 6, 7, 8 and 9\\,m inscribed diameter (solid, dotted, dashed, "
             "dot-dashed), including (purple) and excluding (red) $\\sigma_{\\eta_\\oplus}$, with the "
             "calibration frozen at 6\\,m. Published: excluding $\\sigma_{\\eta_\\oplus}$, 9\\,m gives "
             f"2.2 times the 6\\,m yield; model {fYield9 / fYield6:.2f} times."),
        14: ("Characterization times of individual EECs among the first 18 at 6--9\\,m (model: 20 exozodi "
             "draws re-derived planet by planet, 1-day bins). Published means 22 and 3.5 days at 6 and "
             f"9\\,m; model means at 6/7/8/9\\,m: {sChar} days. Stark does not state his normalization, "
             "and with 1-day bins his 9\\,m plateau alone would exceed his stated mean, so compare shapes, "
             "not heights: his distributions keep a plateau of 15--50-day spectra at every aperture, which "
             "the model's lack."),
        15: (f"$P_{{25}}$ against inscribed diameter. Published including $\\sigma_{{\\eta_\\oplus}}$ "
             f"32/53/67/78\\%, model {sP25('fProbability25IncludingSigmaEta')}\\%; excluding, published "
             f"6/49/89/99.5\\%, model {sP25('fProbability25ExcludingSigmaEta')}\\%."),
        25: ("DMVC6 raw contrast (dotted) and core throughput (solid) against separation in "
             "circumscribed $\\lambda/D$. The model curves are the digitized published ones, so this "
             "pair checks the digitization, not the model. The model's contrast is shown with the "
             "$10^{-10}$ floor applied, as the exposure times use it; the published curve is the raw "
             "simulation. The published figure also shows the PIAA-FPM2.5 (red), which is not used."),
    }


def fsPairFigures(dictX, dictResults):
    """One figure environment per published/model pair, published on the left."""
    listOut = []
    for iFig, sCaption in fdictPairCaptions(dictX, dictResults).items():
        listOut.append(
            "\\begin{figure}[htbp]\n\\centering\n"
            f"\\begin{{minipage}}{{0.48\\textwidth}}\\centering\\textbf{{Stark et al.\\ (2024) Fig.~{iFig}}}\\\\[2pt]\n"
            f"\\includegraphics[width=\\linewidth]{{../ComparePublishedFigures/reference/starkFigure{iFig:02d}.pdf}}\\end{{minipage}}\\hfill\n"
            f"\\begin{{minipage}}{{0.48\\textwidth}}\\centering\\textbf{{This model}}\\\\[2pt]\n"
            f"\\includegraphics[width=\\linewidth]{{figMatchStarkFig{iFig:02d}.pdf}}\\end{{minipage}}\n"
            f"\\caption{{{sCaption}}}\n\\label{{fig:pair{iFig:02d}}}\n\\end{{figure}}\n")
    return "\n".join(listOut)


def fdictDistanceTokens(dictDist):
    """Median published and model completeness in three distance bins (A10, gated)."""
    dictOut = {}
    for sBin, sTok in (("0-6", "A"), ("10-15", "B"), ("15-20", "C")):
        dictOut[f"DIST{sTok}PUB"] = f"{dictDist[sBin]['fMedianPublished']:.2f}"
        dictOut[f"DIST{sTok}MOD"] = f"{dictDist[sBin]['fMedianModel']:.2f}"
    return dictOut


def fdictLuminosityTokens(dictLow):
    """Median published and model completeness in two luminosity bins, and the near-zero share."""
    dictA, dictB = dictLow["1-2"], dictLow["2-5"]
    return {"LUMAPUB": f"{dictA['fMedianPublished']:.2f}", "LUMAMOD": f"{dictA['fMedianModel']:.2f}",
            "LUMBPUB": f"{dictB['fMedianPublished']:.2f}", "LUMBMOD": f"{dictB['fMedianModel']:.2f}",
            "LUMBZERO": f"{100*dictB['fFractionModelNearZero']:.0f}"}


def fdictTwinTokens(dictTwin):
    """Earth-twin characterization times and exozodi-to-planet ratios for named stars."""
    dictOut = {}
    for sName, sTok in (("tau Cet", "TAUCET"), ("61 Vir", "VIR"), ("tet Boo", "TETBOO"),
                        ("110 Her", "HER")):
        d = dictTwin[sName]["dictCharacterization"]
        dictOut[f"TWIN{sTok}"] = f"{d['fTauDays_scene']:.2f}" if d['fTauDays_scene'] < 1 \
            else f"{d['fTauDays_scene']:.0f}"
        dictOut[f"EXO{sTok}"] = f"{d['fExozodi']/d['fPlanet']:.0f}" if d['fExozodi'] > 2 * d['fPlanet'] \
            else f"{d['fExozodi']/d['fPlanet']:.1f}"
    return dictOut


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
    dictResults = fdictLoadResults(sRepoRoot)
    dictExtra = fdictLoadExtra(sRepoRoot)
    dictSubs = {**fdictSubstitutions(dictResults),
                **fdictModelTokens(dictExtra, dictResults, sRepoRoot),
                **fdictDegeneracyTokens(dictExtra["degeneracy"]),
                **fdictVariantTokens(dictExtra["variants"], dictExtra["interactions"],
                                     dictExtra["peaks"], dictExtra["etaInterval"])}
    sRendered = fsRenderTemplate(dictArgs["template"], dictSubs)
    with open(dictArgs["out_tex"], "w") as oFile:
        oFile.write(sRendered)
    fnCompile(dictArgs["out_tex"], dictArgs["out_dir"])
    print(json.dumps({"sTex": dictArgs["out_tex"],
                      "sPdf": os.path.splitext(dictArgs["out_tex"])[0] + ".pdf",
                      "iSubstitutions": len(dictSubs)}, indent=2))


if __name__ == "__main__":
    main()
