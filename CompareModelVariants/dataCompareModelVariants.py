#!/usr/bin/env python3
"""Re-run the yield chain under variants of the settings that the published record leaves open.

The adopted model follows Stark et al. (2024) wherever the method is documented: his per-visit
albedo test, and the separation-dependent T_sky(r) of Stark (2019) Eqs. 5-6. It departs from his
text in one place, reading his eta_Earth interval as 68% rather than the stated 86% (A15). This
script measures what each setting is worth, with every other setting, seed and input held fixed:

  albedoRecompute          sAlbedoMethod "recompute": re-derive each drawn-albedo planet's exposure
                           time instead of Stark's per-visit flux test (affects A04, A05, A07).
  etaInterval86            A06 --eta-interval-z 1.4758: Stark's interval read as the 86% his text
                           states. With the albedo test already his, this is the model that follows
                           every documented choice (affects A06, A07).
  skyThroughputConstant    bSkyThroughputFollowsCore false: a constant large-separation T_sky
                           instead of the reconstructed T_sky(r) (affects A04, A05, A07).
  albedoRecomputeEta86     both of the first two at once; with the adopted model, albedoRecompute
                           and etaInterval86 it completes the 2x2 in albedo method x eta reading
                           that dataAnalyseSettingInteractions.py decomposes.
  etaInterval86ConstantSky the 86% reading with T_sky(r) removed too. NOT more faithful: dropping
                           T_sky(r) discards a term the papers state, so any agreement it buys is
                           a cancellation.
  brysonMixtureEta         Stark's own eta construction, the uniform mixture of Bryson's two cases
                           rescaled to his stated mean, instead of a lognormal.

Stages a variant does not change reuse the baseline step outputs, so a difference is attributable
to the changed settings. kappa stays at the adopted 1 in every variant: refitting it would let the
calibration absorb the change.

Variants run concurrently, under --out-root; no pipeline step output is touched.
dataSummariseModelVariants.py tabulates the result.
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
    "albedoRecompute": {"dictMissionOverrides": {"sAlbedoMethod": "recompute"},
                        "listStages": ["completeness", "survey", "prediction"],
                        "sWhy": "exposure recomputed at the drawn albedo vs Stark 2024 Sec. 3.2's "
                                "per-visit flux test"},
    "etaInterval86": {"dictMissionOverrides": {}, "fEtaIntervalZ": 1.4758,
                      "listStages": ["occurrence", "prediction"],
                      "sWhy": "eta_Earth interval read as the stated 86% vs the adopted 68%; "
                              "every documented Stark choice followed"},
    "skyThroughputConstant": {"dictMissionOverrides": {"bSkyThroughputFollowsCore": False},
                              "listStages": ["completeness", "survey", "prediction"],
                              "sWhy": "constant large-separation T_sky vs the reconstructed "
                                      "T_sky(r)"},
    "albedoRecomputeEta86": {"dictMissionOverrides": {"sAlbedoMethod": "recompute"},
                             "fEtaIntervalZ": 1.4758,
                             "listStages": ["completeness", "survey", "occurrence", "prediction"],
                             "sWhy": "recomputed albedo AND the 86% interval: the fourth corner of "
                                     "the albedo x eta 2x2"},
    "etaInterval86ConstantSky": {"dictMissionOverrides": {"bSkyThroughputFollowsCore": False},
                                 "fEtaIntervalZ": 1.4758,
                                 "listStages": ["completeness", "survey", "occurrence",
                                                "prediction"],
                                 "sWhy": "the 86% interval with T_sky(r) removed too -- NOT more "
                                         "faithful, since that removes sourced physics"},
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
        dictKept = {d["sVariant"]: d for d in json.load(open(sManifests))["listVariants"]
                    if d["sVariant"] in DICT_VARIANTS}
    dictKept.update({d["sVariant"]: d for d in listManifests})
    with open(sManifests, "w") as oFile:
        json.dump({"listVariants": list(dictKept.values())}, oFile, indent=1)


if __name__ == "__main__":
    main()
