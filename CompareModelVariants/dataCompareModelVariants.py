#!/usr/bin/env python3
"""Re-run the yield chain under variants of the choices adopted because they matched a published number.

Three settings entered the pipeline because they improved agreement with Stark et al. (2024), not
because a source required them. This script measures what each is worth on its own, and what they
do together, with every other setting, seed and input held fixed:

  albedoPerVisit         sAlbedoMethod "perVisitThreshold" instead of "recompute" -- Stark's own
                         per-visit flux test rather than re-deriving the exposure at the drawn
                         albedo (affects A04, A05, A07).
  etaInterval86          A06 --eta-interval-z 1.4758: read Stark's eta_Earth interval as the 86%
                         his text states, instead of the 68% adopted on 2026-09-23 to match his
                         Fig. 10 width (affects A06, A07).
  skyThroughputConstant  bSkyThroughputFollowsCore false: the constant large-separation T_sky the
                         model used before 2026-09-24, instead of the reconstructed T_sky(r)
                         (affects A04, A05, A07).

  paperFaithful          both changes that move TOWARD the papers at once -- his per-visit albedo
                         test and his stated 86% interval -- while KEEPING the reconstructed
                         T_sky(r), which is sourced physics (Stark 2019 Eqs. 5-6) rather than a
                         tuning. This is the most faithful reading of the published method.
  allThreeCombined       the above plus T_sky(r) removed. Included because the combination was
                         asked for, but it is NOT more faithful: dropping T_sky(r) discards a
                         term the papers state, so any agreement it buys is a cancellation.

Stages a variant does not change reuse the baseline step outputs, so a difference is attributable
to the changed settings. The combined variants are not the sum of the individual ones: the albedo
method raises the level while the eta width narrows the distribution, and constant T_sky lowers
the level again, so they partly cancel -- which is the reason to run them together rather than
add up the single-variable shifts. kappa stays at the adopted 1 in every variant, including skyThroughputConstant
-- refitting it would let the calibration absorb the change, which is what removing kappa was for.

Variants run concurrently. Outputs go under explorations/output/oneVariableExperiments/<variant>/;
no pipeline step output is touched. summariseOneVariableExperiments.py tabulates the result.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys

S_HERE = os.path.dirname(os.path.abspath(__file__))
S_REPO = os.path.dirname(S_HERE)

DICT_VARIANTS = {
    "albedoPerVisit": {"dictMissionOverrides": {"sAlbedoMethod": "perVisitThreshold"},
                       "listStages": ["completeness", "survey", "prediction"],
                       "sWhy": "Stark 2024 Sec. 3.2 per-visit flux test vs recomputing at the "
                               "drawn albedo"},
    "etaInterval86": {"dictMissionOverrides": {}, "fEtaIntervalZ": 1.4758,
                      "listStages": ["occurrence", "prediction"],
                      "sWhy": "eta_Earth interval read as the stated 86% vs the adopted 68%"},
    "skyThroughputConstant": {"dictMissionOverrides": {"bSkyThroughputFollowsCore": False},
                              "listStages": ["completeness", "survey", "prediction"],
                              "sWhy": "constant large-separation T_sky vs the reconstructed "
                                      "T_sky(r)"},
    "paperFaithful": {"dictMissionOverrides": {"sAlbedoMethod": "perVisitThreshold"},
                      "fEtaIntervalZ": 1.4758,
                      "listStages": ["completeness", "survey", "occurrence", "prediction"],
                      "sWhy": "every documented Stark choice followed at once: his per-visit "
                              "albedo test AND his stated 86% eta interval, with the sourced "
                              "T_sky(r) of Stark 2019 Eqs. 5-6 KEPT"},
    "allThreeCombined": {"dictMissionOverrides": {"sAlbedoMethod": "perVisitThreshold",
                                                  "bSkyThroughputFollowsCore": False},
                         "fEtaIntervalZ": 1.4758,
                         "listStages": ["completeness", "survey", "occurrence", "prediction"],
                         "sWhy": "all three one-variable changes at once, T_sky(r) removed too "
                                 "-- NOT more faithful, since that removes sourced physics"},
    "brysonMixtureEta": {"dictMissionOverrides": {}, "sEtaLaw": "brysonMixture",
                         "listStages": ["occurrence", "prediction"],
                         "sWhy": "Stark's OWN eta construction -- the uniform mixture of Bryson's "
                                 "two cases rescaled to his stated mean 0.26 -- instead of a "
                                 "lognormal fitted to his quoted interval, removing both the "
                                 "86-vs-68 percent and mean-vs-median readings"}}


DICT_BASELINE = {}


def fsBaseline(sStem):
    """Absolute path of a baseline step artifact, as supplied on the command line."""
    return DICT_BASELINE[sStem]


def fsWriteVariantMission(dictOverrides, sOutDir):
    """Variant mission-parameters file, or the baseline path when nothing is overridden."""
    if not dictOverrides:
        return fsBaseline("mission")
    sOut = os.path.join(sOutDir, "missionParameters.json")
    with open(fsBaseline("mission")) as oFile:
        dictParams = json.load(oFile)
    dictParams["dictMission"].update(dictOverrides)
    with open(sOut, "w") as oFile:
        json.dump(dictParams, oFile, indent=1)
    return sOut


def flistStageCommands(sName, dictVariant, sMission, sOutDir, iProcesses):
    """(step directory, argv) for each stage this variant has to re-run, in order."""
    listStages, listOut = dictVariant["listStages"], []
    sCompleteness = (os.path.join(sOutDir, "completeness.npz") if "completeness" in listStages
                     else fsBaseline("completeness"))
    sOccurrence = (os.path.join(sOutDir, "occurrencePosterior.npz") if "occurrence" in listStages
                   else fsBaseline("occurrence"))
    if "completeness" in listStages:
        listOut.append(("computeCompleteness", [
            "dataComputeCompleteness.py", "--target-catalog", fsBaseline("catalog"),
            "--mission-parameters", sMission, "--calibration-json", fsBaseline("calibration"),
            "--out-completeness", sCompleteness,
            "--out-summary", os.path.join(sOutDir, "completenessSummary.json")]))
    if "survey" in listStages:
        listOut.append(("optimizeSurvey", [
            "dataOptimizeSurvey.py", "--completeness", sCompleteness,
            "--mission-parameters", sMission, "--calibration-json", fsBaseline("calibration"),
            "--box", "canonical", "--processes", str(iProcesses),
            "--out-survey", os.path.join(sOutDir, "surveyResult.json")]))
    if "occurrence" in listStages:
        listArgv = ["dataSampleOccurrencePosterior.py", "--mission-parameters", sMission]
        if dictVariant.get("fEtaIntervalZ") is not None:
            listArgv += ["--eta-interval-z", str(dictVariant["fEtaIntervalZ"])]
        if dictVariant.get("sEtaLaw"):
            listArgv += ["--eta-law", dictVariant["sEtaLaw"],
                         "--bryson-digitised", fsBaseline("bryson")]
        listOut.append(("sampleOccurrencePosterior", listArgv + [
            "--out-samples", sOccurrence,
            "--out-summary", os.path.join(sOutDir, "occurrencePosteriorSummary.json")]))
    listOut.append(("predictRedefinedYield", [
        "dataPredictRedefinedYield.py", "--completeness", sCompleteness,
        "--occurrence-posterior", sOccurrence, "--mission-parameters", sMission,
        "--calibration-json", fsBaseline("calibration"), "--processes", str(iProcesses),
        "--out-prediction", os.path.join(sOutDir, "yieldPrediction.json"),
        "--out-samples", os.path.join(sOutDir, "yieldSamples.npz")]))
    return listOut


def fdictRunVariant(tJob):
    """Run one variant's stages in order; returns its manifest."""
    sName, dictVariant, sOutRoot, iProcesses = tJob
    sOutDir = os.path.join(sOutRoot, sName)
    os.makedirs(sOutDir, exist_ok=True)
    sMission = fsWriteVariantMission(dictVariant["dictMissionOverrides"], sOutDir)
    listStages = flistStageCommands(sName, dictVariant, sMission, sOutDir, iProcesses)
    with open(os.path.join(sOutDir, "run.log"), "w") as oLog:
        for sStepDir, listArgv in listStages:
            oLog.write(f"### {sStepDir}: {' '.join(listArgv)}\n")
            oLog.flush()
            subprocess.run([sys.executable] + listArgv, cwd=os.path.join(S_REPO, sStepDir),
                           stdout=oLog, stderr=subprocess.STDOUT, check=True)
    dictManifest = {"sVariant": sName, "sWhy": dictVariant["sWhy"],
                    "dictMissionOverrides": dictVariant["dictMissionOverrides"],
                    "fEtaIntervalZ": dictVariant.get("fEtaIntervalZ"),
                    "sEtaLaw": dictVariant.get("sEtaLaw", "lognormal"),
                    "listStagesRerun": dictVariant["listStages"],
                    "sMissionParameters": sMission}
    with open(os.path.join(sOutDir, "manifest.json"), "w") as oFile:
        json.dump(dictManifest, oFile, indent=1)
    print(f"{sName} done", flush=True)
    return dictManifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--occurrence-posterior", required=True)
    p.add_argument("--bryson-digitised", required=True)
    p.add_argument("--out-root", default="variants")
    p.add_argument("--variants", default=",".join(DICT_VARIANTS),
                   help="comma-separated subset of the variant names")
    p.add_argument("--processes", type=int, default=2,
                   help="per-variant worker processes for the parallel stages")
    dictArgs = vars(p.parse_args())
    DICT_BASELINE.update({
        "catalog": os.path.abspath(dictArgs["target_catalog"]),
        "mission": os.path.abspath(dictArgs["mission_parameters"]),
        "calibration": os.path.abspath(dictArgs["calibration_json"]),
        "completeness": os.path.abspath(dictArgs["completeness"]),
        "occurrence": os.path.abspath(dictArgs["occurrence_posterior"]),
        "bryson": os.path.abspath(dictArgs["bryson_digitised"])})
    sOutRoot = os.path.abspath(dictArgs["out_root"])
    os.makedirs(sOutRoot, exist_ok=True)
    listNames = [s.strip() for s in dictArgs["variants"].split(",") if s.strip()]
    listJobs = [(s, DICT_VARIANTS[s], sOutRoot, dictArgs["processes"]) for s in listNames]
    with cf.ThreadPoolExecutor(max_workers=len(listJobs)) as oPool:
        listManifests = list(oPool.map(fdictRunVariant, listJobs))
    sManifests = os.path.join(sOutRoot, "manifests.json")
    dictKept = {}
    if os.path.exists(sManifests):
        dictKept = {d["sVariant"]: d for d in json.load(open(sManifests))["listVariants"]}
    dictKept.update({d["sVariant"]: d for d in listManifests})
    with open(sManifests, "w") as oFile:
        json.dump({"listVariants": list(dictKept.values())}, oFile, indent=1)


if __name__ == "__main__":
    main()
