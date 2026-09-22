#!/usr/bin/env python3
"""Reproduce Stark et al. (2024) Fig. 11 from this pipeline and audit the target screen.

Fig. 11 plots selected targets in stellar luminosity against distance, coloured by habitable-zone
completeness, over the full input catalog in grey. Stark's selected targets reach L ~ 20 Lsun,
which tests whether this pipeline's screening admits the same population. The screen is also
audited directly: it rejects a star when an EEC around it could never clear the post-processing
noise floor, so it must use the BRIGHTEST planet the selection box allows, not a typical one.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

F_REARTH_AU = 4.25875e-5


def faDeltaMagTypical(faLuminosity, fAlbedo):
    """Delta mag of an Earth-sized planet at the EEID at quadrature -- the screen as written."""
    faFlux = fAlbedo * (1.0 / np.pi) * (F_REARTH_AU / np.sqrt(faLuminosity)) ** 2
    return -2.5 * np.log10(faFlux)


def faDeltaMagBrightest(faLuminosity, fAlbedo, fRadiusMax, fHzInner):
    """Delta mag of the BRIGHTEST planet the box allows: largest radius, inner edge, full phase."""
    faSemiMajor = fHzInner * np.sqrt(faLuminosity)
    faFlux = fAlbedo * 1.0 * (fRadiusMax * F_REARTH_AU / faSemiMajor) ** 2
    return -2.5 * np.log10(faFlux)


def fdictAuditScreen(dfCatalog, dictMission, dictBox):
    """Count stars the screen rejects under the typical-planet and brightest-planet criteria."""
    faL = dfCatalog["fLuminosityLsun"].to_numpy()
    faTypical = faDeltaMagTypical(faL, dictMission["fGeometricAlbedo"])
    faBrightest = faDeltaMagBrightest(faL, dictMission["fGeometricAlbedo"],
                                      dictBox["fRadiusMaxEarth"], dictBox["fHzInnerAu"])
    fFloor = dictMission["fNoiseFloorDeltaMag"]
    bRejectedTypical = faTypical > fFloor
    bRejectedBrightest = faBrightest > fFloor
    bLostWrongly = bRejectedTypical & ~bRejectedBrightest
    return {
        "iCatalog": int(len(dfCatalog)),
        "iRejectedByScreenAsWritten": int(np.sum(bRejectedTypical)),
        "iRejectedByCorrectBound": int(np.sum(bRejectedBrightest)),
        "iLostWrongly": int(np.sum(bLostWrongly)),
        "fMaxLuminosityAdmittedAsWritten": float(np.max(faL[~bRejectedTypical])),
        "fMaxLuminosityAdmittedCorrect": float(np.max(faL[~bRejectedBrightest])),
        "fLuminosityCutAsWritten": float(np.min(faL[bRejectedTypical])) if bRejectedTypical.any()
        else None,
        "iLostWronglyWithinTwentyPc": int(np.sum(bLostWrongly &
                                                 (dfCatalog["fDistancePc"].to_numpy() < 20))),
    }


def fnPlotTargets(oAxes, dfCatalog, dictNpz, dictAudit):
    """Stark Fig. 11 layout: luminosity against distance, coloured by completeness."""
    oAxes.scatter(dfCatalog["fDistancePc"], dfCatalog["fLuminosityLsun"], s=3, c="#d9d8d4",
                  lw=0, label="input catalog")
    faComp = dictNpz["faComp_canonical"][:, -1]
    bSelected = faComp > 0.0
    oScatter = oAxes.scatter(dictNpz["faDistancePc"][bSelected],
                             dictNpz["faLuminosityLsun"][bSelected], c=faComp[bSelected],
                             s=16, cmap="viridis", vmin=0.0, vmax=1.0, lw=0)
    oAxes.axhline(dictAudit["fMaxLuminosityAdmittedAsWritten"], color=ps.LIST_SERIES[1],
                  lw=1.4, ls="--")
    oAxes.annotate("screen's luminosity ceiling", xy=(2, dictAudit["fMaxLuminosityAdmittedAsWritten"]),
                   xytext=(3, 6), textcoords="offset points", fontsize=7,
                   color=ps.LIST_SERIES[1])
    oAxes.set_yscale("log")
    oAxes.set_xlim(0, 40)
    oAxes.set_ylim(3e-3, 40)
    ps.fnFinishAxes(oAxes, "$d$ (pc)", "$L_\\star$ ($L_\\odot$)",
                    "Selected targets (cf. Stark+2024 Fig. 11)")
    return oScatter


def fdictParseArgs():
    """Command-line configuration for the target-selection diagnostic."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--completeness", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--out-json", default="targetSelectionAudit.json")
    p.add_argument("--out-plot", default="../Plot/figTargetSelection.pdf")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    ps.fnApplyStyle()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dfCatalog = pd.read_csv(dictArgs["target_catalog"])
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    dictAudit = fdictAuditScreen(dfCatalog, dictParams["dictMission"],
                                 dictParams["dictBoxes"]["canonical"])
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictAudit, oFile, indent=2)
    oFig, oAxes = plt.subplots(figsize=(5.4, 4.0))
    oScatter = fnPlotTargets(oAxes, dfCatalog, dictNpz, dictAudit)
    oFig.colorbar(oScatter, ax=oAxes, label="Completeness")
    oFig.tight_layout()
    oFig.savefig(dictArgs["out_plot"])
    print(json.dumps(dictAudit, indent=2))


if __name__ == "__main__":
    main()
