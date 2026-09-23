#!/usr/bin/env python3
"""Name the nearby targets the HPIC build drops relative to the Gaia reconstruction, and say why.

The screened HPIC list holds 169 stars inside 15 pc against the Gaia reconstruction's 195, and
the shortfall sits in the 5-10 pc shell that contributes most of the yield. HPIC is the more
complete catalog, so either this step's selection cuts are discarding real targets or the Gaia
reconstruction carried entries that are not real targets. Both are defects, and they call for
opposite fixes, so this script decides between them per star rather than in aggregate.

For every star the Gaia screen accepted in the shell, it locates the nearest HPIC row -- before
any cut -- and reports which specific condition that row fails: the distance limit, a missing
temperature, luminosity, radius or V magnitude, HPIC's own dwarf flag, the radius bound, the
surface-gravity bound, or the downstream target screen. A star with no HPIC row within the match
radius at all is reported separately, since that is the only outcome that would mean HPIC is
genuinely incomplete.
"""

import argparse
import io
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import survey as sv  # noqa: E402

LIST_COLS = ["star_name", "ra", "dec", "ra_J2016", "dec_J2016", "sy_dist", "st_teff", "st_lum",
             "st_rad", "st_logg", "sy_vmag", "dwarf_fl", "st_spectype", "wds_sep"]


def fdfRawHpic(sArchivePath):
    """Every HPIC row, with no cuts applied, so a rejection can be attributed to one condition."""
    with tarfile.open(sArchivePath, "r:gz") as oTar:
        baBody = oTar.extractfile("HPIC/full_HPIC.txt").read()
    dfRaw = pd.read_csv(io.BytesIO(baBody), delimiter="|", low_memory=False, usecols=LIST_COLS)
    for sCol in ("ra", "dec", "ra_J2016", "dec_J2016", "sy_dist", "st_teff", "st_lum", "st_rad",
                 "st_logg", "sy_vmag"):
        dfRaw[sCol] = pd.to_numeric(dfRaw[sCol], errors="coerce")
    return dfRaw


def faSeparationArcsec(faRa, faDec, fRa, fDec):
    """Angular separation in arcseconds from one position to an array of positions."""
    faDd = np.asarray(faDec) - fDec
    faDr = (np.asarray(faRa) - fRa) * np.cos(np.radians(fDec))
    return np.hypot(faDr, faDd) * 3600.0


def faBestEpochSeparation(dfHpic, fRa, fDec):
    """Separation to each HPIC row, taking the closer of its J2000 and J2016 positions.

    The Gaia half of the legacy catalog carries J2016 positions while its Hipparcos supplement
    and HPIC's default columns are J2000. Sixteen years of proper motion moves the nearest stars
    by more than ten arcseconds -- Barnard's Star by nearly three arcminutes -- so a single-epoch
    match reports the best targets in the sky as missing from the catalog that contains them.
    """
    faJ2000 = faSeparationArcsec(dfHpic["ra"].to_numpy(), dfHpic["dec"].to_numpy(), fRa, fDec)
    faJ2016 = faSeparationArcsec(dfHpic["ra_J2016"].to_numpy(), dfHpic["dec_J2016"].to_numpy(),
                                 fRa, fDec)
    return np.fmin(faJ2000, np.where(np.isfinite(faJ2016), faJ2016, np.inf))


def fsRejectionReason(dictRow, fMaxDistancePc, fMaxRadiusRsun, fMinLogg):
    """The first selection condition an HPIC row fails, or 'passes-build'."""
    if not np.isfinite(dictRow["sy_dist"]) or dictRow["sy_dist"] <= 0:
        return "no-distance"
    if dictRow["sy_dist"] > fMaxDistancePc:
        return "beyond-distance-limit"
    for sCol, sLabel in (("st_teff", "no-teff"), ("st_lum", "no-luminosity"),
                         ("st_rad", "no-radius"), ("sy_vmag", "no-vmag")):
        if not np.isfinite(dictRow[sCol]):
            return sLabel
    if str(dictRow["dwarf_fl"]) not in ("1", "1.0", "True"):
        return "dwarf-flag-zero"
    if not (0 < dictRow["st_rad"] <= fMaxRadiusRsun):
        return "radius-above-bound"
    if np.isfinite(dictRow["st_logg"]) and dictRow["st_logg"] < fMinLogg:
        return "logg-below-bound"
    return "passes-build"


def fdictAuditOneStar(dictLegacy, dfHpicRaw, dfHpicScreened, dictArgs):
    """Match one accepted Gaia target into HPIC and classify its fate."""
    faSep = faBestEpochSeparation(dfHpicRaw, dictLegacy["ra"], dictLegacy["dec"])
    iBest = int(np.nanargmin(faSep))
    if faSep[iBest] > dictArgs["match_arcsec"]:
        return {"sFate": "absent-from-hpic", "fNearestArcsec": float(faSep[iBest])}
    dictRow = dfHpicRaw.iloc[iBest].to_dict()
    sReason = fsRejectionReason(dictRow, dictArgs["max_distance_pc"],
                                dictArgs["max_radius_rsun"], dictArgs["min_logg"])
    if sReason != "passes-build":
        return {"sFate": sReason, "sHpicName": str(dictRow["star_name"]),
                "sSpectralType": str(dictRow["st_spectype"]),
                "fHpicDistancePc": float(dictRow["sy_dist"]),
                "fHpicRadiusRsun": float(dictRow["st_rad"]),
                "fHpicLogg": float(dictRow["st_logg"])}
    bInScreen = str(dictRow["star_name"]) in dictArgs["setScreenedNames"]
    return {"sFate": "in-screened-list" if bInScreen else "cut-by-target-screen",
            "sHpicName": str(dictRow["star_name"]),
            "sSpectralType": str(dictRow["st_spectype"]),
            "fHpicDistancePc": float(dictRow["sy_dist"]),
            "fHpicTeffK": float(dictRow["st_teff"]),
            "fHpicLumLsun": float(10.0 ** dictRow["st_lum"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-catalog", required=True)
    p.add_argument("--hpic-catalog", required=True)
    p.add_argument("--hpic-archive", required=True)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--shell-min-pc", type=float, default=0.0)
    p.add_argument("--shell-max-pc", type=float, default=15.0)
    p.add_argument("--max-distance-pc", type=float, default=50.0)
    p.add_argument("--max-radius-rsun", type=float, default=3.0)
    p.add_argument("--min-logg", type=float, default=3.5)
    p.add_argument("--max-stars", type=int, default=1500)
    p.add_argument("--match-arcsec", type=float, default=15.0)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictBox = dictParams["dictBoxes"]["canonical"]
    dfLegacy = sv.fdfScreenTargets(pd.read_csv(dictArgs["legacy_catalog"]),
                                   dictParams["dictMission"],
                                   dictParams["dictBands"]["listBandsDetection"][0],
                                   0.5, dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    dfHpicScreened = sv.fdfScreenTargets(pd.read_csv(dictArgs["hpic_catalog"]),
                                         dictParams["dictMission"],
                                         dictParams["dictBands"]["listBandsDetection"][0],
                                         0.5, dictArgs["max_stars"], 2500.0, 7500.0, dictBox)
    dfHpicRaw = fdfRawHpic(dictArgs["hpic_archive"])
    dictArgs["setScreenedNames"] = set(dfHpicScreened["sSourceId"].astype(str))
    dfShell = dfLegacy[(dfLegacy["fDistancePc"] >= dictArgs["shell_min_pc"]) &
                       (dfLegacy["fDistancePc"] < dictArgs["shell_max_pc"])]
    listAudit = []
    for dictLegacy in dfShell.to_dict("records"):
        dictFate = fdictAuditOneStar(dictLegacy, dfHpicRaw, dfHpicScreened, dictArgs)
        dictFate.update({"sLegacyName": str(dictLegacy["sSourceId"]),
                         "fLegacyDistancePc": round(float(dictLegacy["fDistancePc"]), 2),
                         "fLegacyTeffK": round(float(dictLegacy["fTeffK"]), 0),
                         "fLegacyLumLsun": round(float(dictLegacy["fLuminosityLsun"]), 3)})
        listAudit.append(dictFate)
    dictTally = {}
    for dictFate in listAudit:
        dictTally[dictFate["sFate"]] = dictTally.get(dictFate["sFate"], 0) + 1
    dictOut = {"iLegacyStarsInShell": len(listAudit), "dictFateTally": dictTally,
               "listNotInScreen": [d for d in listAudit if d["sFate"] != "in-screened-list"]}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({"iLegacyStarsInShell": dictOut["iLegacyStarsInShell"],
                      "dictFateTally": dictTally}, indent=2))


if __name__ == "__main__":
    main()
