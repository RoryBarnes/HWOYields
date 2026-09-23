#!/usr/bin/env python3
"""Compare the real HPIC target list against the Gaia DR3 + Hipparcos reconstruction used by A01.

Stark et al. (2024) sec. 2 states the input target list IS the HWO Preliminary Input Catalog
(Tuchow et al. 2024).  Step A01 currently rebuilds an approximation of it from Gaia DR3 plus a
Hipparcos bright-star supplement, because an earlier attempt to download HPIC failed (a 301
redirect returned a 169-byte stub).  The real catalog is now on disk, so this script measures
what the substitution cost: how many stars are lost, where they sit in distance and luminosity,
and whether the missing population lands in the 15-30 pc band where our per-target completeness
falls to zero against Stark's published Fig. 11.
"""

import argparse
import json

import numpy as np
import pandas as pd


def fdfLoadHpic(sPath, fMaxDistancePc):
    """Load HPIC and keep dwarfs inside the distance limit with the parameters the model needs."""
    dfRaw = pd.read_csv(sPath, delimiter="|", low_memory=False,
                        usecols=["star_name", "ra", "dec", "sy_vmag", "sy_gaiamag", "sy_dist",
                                 "st_teff", "st_lum", "st_rad", "st_mass", "st_spectype",
                                 "dwarf_fl", "wds_sep", "known_binary_fl", "st_logg"])
    dfOut = dfRaw[pd.to_numeric(dfRaw["sy_dist"], errors="coerce").notna()].copy()
    for sCol in ("sy_dist", "st_teff", "st_lum", "st_rad", "sy_vmag", "st_logg"):
        dfOut[sCol] = pd.to_numeric(dfOut[sCol], errors="coerce")
    dfOut = dfOut[(dfOut["sy_dist"] > 0) & (dfOut["sy_dist"] <= fMaxDistancePc)]
    dfOut["fLuminosityLsun"] = 10.0 ** dfOut["st_lum"]
    return dfOut.reset_index(drop=True)


def fdictCompleteness(dfHpic):
    """Count how many HPIC rows carry each parameter the yield model requires."""
    return {
        "iRows": int(len(dfHpic)),
        "iWithTeff": int(dfHpic["st_teff"].notna().sum()),
        "iWithLum": int(dfHpic["st_lum"].notna().sum()),
        "iWithRadius": int(dfHpic["st_rad"].notna().sum()),
        "iWithVmag": int(dfHpic["sy_vmag"].notna().sum()),
        "iWithAllFour": int((dfHpic["st_teff"].notna() & dfHpic["st_lum"].notna() &
                             dfHpic["st_rad"].notna() & dfHpic["sy_vmag"].notna()).sum()),
    }


def faDistanceHistogram(faDistancePc, faEdges):
    """Counts per distance bin."""
    return np.histogram(np.asarray(faDistancePc, dtype=float), bins=faEdges)[0]


def fdictBandComparison(dfHpic, dfOurs, faEdges, fTeffLo, fTeffHi):
    """Per-distance-bin FGK dwarf counts in both catalogs, plus the shortfall of the reconstruction."""
    bHpic = ((dfHpic["st_teff"] >= fTeffLo) & (dfHpic["st_teff"] <= fTeffHi) &
             dfHpic["st_lum"].notna() & dfHpic["st_rad"].notna() & dfHpic["sy_vmag"].notna())
    bOurs = (dfOurs["fTeffK"] >= fTeffLo) & (dfOurs["fTeffK"] <= fTeffHi)
    faHpic = faDistanceHistogram(dfHpic.loc[bHpic, "sy_dist"], faEdges)
    faOurs = faDistanceHistogram(dfOurs.loc[bOurs, "fDistancePc"], faEdges)
    return {
        "faEdgesPc": [float(f) for f in faEdges],
        "faHpicFgk": [int(i) for i in faHpic],
        "faOursFgk": [int(i) for i in faOurs],
        "faMissingFgk": [int(i) for i in (faHpic - faOurs)],
    }


def fdictBrightestLost(dfHpic, dfOurs, fMatchArcsec, iTop):
    """The largest-EEID HPIC stars with no counterpart in the reconstruction.

    EEID angular size sqrt(L)/d is the single strongest predictor of a target's yield value, so
    a star lost from the top of this ordering costs far more than one lost from the tail.
    """
    dfOk = dfHpic[dfHpic["st_lum"].notna() & dfHpic["st_teff"].notna()].copy()
    dfOk["fEeidArcsec"] = np.sqrt(dfOk["fLuminosityLsun"]) / dfOk["sy_dist"]
    dfOk = dfOk.sort_values("fEeidArcsec", ascending=False).head(400)
    faRaO, faDecO = dfOurs["ra"].to_numpy(), dfOurs["dec"].to_numpy()
    listLost = []
    for dictRow in dfOk.to_dict("records"):
        faDd = faDecO - dictRow["dec"]
        faDr = (faRaO - dictRow["ra"]) * np.cos(np.radians(dictRow["dec"]))
        if np.min(np.hypot(faDr, faDd)) * 3600.0 > fMatchArcsec:
            listLost.append({"sName": str(dictRow["star_name"]),
                             "sSimbad": str(dictRow.get("st_spectype", "")),
                             "fDistancePc": round(float(dictRow["sy_dist"]), 2),
                             "fTeffK": round(float(dictRow["st_teff"]), 0),
                             "fEeidArcsec": round(float(dictRow["fEeidArcsec"]), 4)})
    return {"iCheckedTop": int(len(dfOk)), "iLost": len(listLost), "listLost": listLost[:iTop]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hpic", required=True)
    p.add_argument("--our-catalog", required=True)
    p.add_argument("--max-distance-pc", type=float, default=50.0)
    p.add_argument("--teff-lo", type=float, default=3900.0)
    p.add_argument("--teff-hi", type=float, default=7300.0)
    p.add_argument("--match-arcsec", type=float, default=10.0)
    p.add_argument("--out-json", required=True)
    dictArgs = vars(p.parse_args())

    dfHpic = fdfLoadHpic(dictArgs["hpic"], dictArgs["max_distance_pc"])
    dfOurs = pd.read_csv(dictArgs["our_catalog"])
    faEdges = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    dictOut = {
        "dictHpicParameterCompleteness": fdictCompleteness(dfHpic),
        "iOurCatalogRows": int(len(dfOurs)),
        "dictFgkByDistance": fdictBandComparison(dfHpic, dfOurs, faEdges,
                                                 dictArgs["teff_lo"], dictArgs["teff_hi"]),
        "dictBrightestLost": fdictBrightestLost(dfHpic, dfOurs, dictArgs["match_arcsec"], 25),
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
