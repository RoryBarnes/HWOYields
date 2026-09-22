#!/usr/bin/env python3
"""Build the HWO target catalog of nearby dwarf stars from Gaia DR3 via the ESA archive TAP.

Stark et al. (2024) use the HWO Preliminary Input Catalog (HPIC), the union of the TESS Input
Catalog and Gaia DR3 within 50 pc complete to T = 12. HPIC itself is not distributed in a form
reachable from this container, so this step reconstructs an equivalent target list from Gaia
DR3 -- one of HPIC's two parent catalogs -- applying the same 50 pc distance limit and a
comparable magnitude limit, and keeping the homogeneous GSP-Phot/FLAME astrophysical
parameters the yield model needs (Teff, radius, luminosity).
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

S_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
S_VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
F_OBLIQUITY_DEG = 23.4392911


def faTeffFromColour(faColourBv):
    """Effective temperature from B-V using the Ballesteros (2012) blackbody calibration."""
    faX = 0.92 * np.asarray(faColourBv, dtype=float)
    return 4600.0 * (1.0 / (faX + 1.70) + 1.0 / (faX + 0.62))


def faEclipticLatitude(faRaDeg, faDecDeg):
    """Ecliptic latitude in degrees from equatorial coordinates."""
    fEps = np.radians(F_OBLIQUITY_DEG)
    faRa, faDec = np.radians(faRaDeg), np.radians(faDecDeg)
    faSinBeta = np.sin(faDec) * np.cos(fEps) - np.cos(faDec) * np.sin(fEps) * np.sin(faRa)
    return np.degrees(np.arcsin(np.clip(faSinBeta, -1.0, 1.0)))


def faBolometricCorrection(faTeffK):
    """V-band bolometric correction from the Torres (2010) recalibration of Flower (1996)."""
    faLogT = np.log10(np.asarray(faTeffK, dtype=float))
    listCold = [-0.190537291496456e5, 0.155144866764412e5, -0.421278819301717e4,
                0.381476328422343e3]
    listMid = [-0.370510203809015e5, 0.385672629965804e5, -0.150651486316025e5,
               0.261724637119416e4, -0.170623810323864e3]
    listHot = [-0.118115450538963e6, 0.137145973583929e6, -0.636233812100225e5,
               0.147412923562646e5, -0.170587278406872e4, 0.788731721804990e2]
    faOut = np.zeros_like(faLogT)
    for bMask, listCoef in ((faLogT < 3.70, listCold),
                            ((faLogT >= 3.70) & (faLogT < 3.90), listMid),
                            (faLogT >= 3.90, listHot)):
        faOut[bMask] = np.polyval(listCoef[::-1], faLogT[bMask])
    return faOut


def fdfBrightStarSupplement(fMaxDistancePc, fVmagLimit, sCachePath, iTimeoutS):
    """Bright nearby dwarfs from Hipparcos, covering the stars Gaia DR3 saturates on (G < 3).

    Gaia DR3 publishes no GSP-Phot/FLAME parameters brighter than G ~ 3, which removes several
    of the best HWO targets -- alpha Cen A and B, Procyon, Sirius -- whose habitable zones
    subtend the largest angles of any star in the sky. Hipparcos supplies V, B-V and parallax;
    Teff follows from B-V (Ballesteros 2012), the luminosity from the absolute magnitude plus a
    Torres (2010) bolometric correction, and the radius from L and Teff.
    """
    fMinParallaxMas = 1000.0 / fMaxDistancePc
    sQuery = (f"SELECT TOP 20000 hip, ra, de, plx, vmag, b_v FROM public.hipparcos "
              f"WHERE plx > {fMinParallaxMas} AND vmag < {fVmagLimit} AND b_v IS NOT NULL")
    dfRaw = pd.read_csv(fsFetchCatalog(sQuery, sCachePath, iTimeoutS))
    dfOut = dfRaw.dropna(subset=["plx", "vmag", "b_v", "ra", "de"]).copy()
    dfOut = dfOut[(dfOut["plx"] > fMinParallaxMas) & (dfOut["b_v"] > -0.4) &
                  (dfOut["b_v"] < 2.2)]
    dfOut["fDistancePc"] = 1000.0 / dfOut["plx"]
    dfOut["fTeffK"] = faTeffFromColour(dfOut["b_v"].to_numpy())
    faAbsV = dfOut["vmag"] + 5.0 - 5.0 * np.log10(dfOut["fDistancePc"])
    faBolometric = faAbsV + faBolometricCorrection(dfOut["fTeffK"].to_numpy())
    dfOut["fLuminosityLsun"] = 10.0 ** (-0.4 * (faBolometric - 4.74))
    dfOut["fRadiusRsun"] = np.sqrt(dfOut["fLuminosityLsun"]) / (dfOut["fTeffK"] / 5772.0) ** 2
    dfOut["fEclipticLatDeg"] = faEclipticLatitude(dfOut["ra"].to_numpy(),
                                                  dfOut["de"].to_numpy())
    dfOut["sSourceId"] = "HIP" + dfOut["hip"].astype(str)
    dfOut["fGmag"] = dfOut["vmag"]
    return dfOut.rename(columns={"de": "dec"})


def fdfMergeSupplement(dfGaia, dfSupplement, fMatchArcsec):
    """Append only those supplement stars with no Gaia counterpart within fMatchArcsec."""
    if dfSupplement.empty:
        return dfGaia.assign(sCatalogSource="gaiadr3")
    faRaG, faDecG = dfGaia["ra"].to_numpy(), dfGaia["dec"].to_numpy()
    listKeep = []
    for dictRow in dfSupplement.to_dict("records"):
        faDeltaDec = faDecG - dictRow["dec"]
        faDeltaRa = (faRaG - dictRow["ra"]) * np.cos(np.radians(dictRow["dec"]))
        if np.min(np.hypot(faDeltaRa, faDeltaDec)) * 3600.0 > fMatchArcsec:
            listKeep.append(dictRow)
    dfNew = pd.DataFrame(listKeep)
    dfGaia = dfGaia.assign(sCatalogSource="gaiadr3")
    if dfNew.empty or dfSupplement.empty:
        return dfGaia
    dfNew = dfNew.assign(sCatalogSource="hipparcos")
    listShared = [c for c in dfNew.columns if c in dfGaia.columns]
    return pd.concat([dfGaia, dfNew[listShared]], ignore_index=True)



def fsBuildQuery(fMaxDistancePc, fGmagLimit, iRowLimit):
    """ADQL query joining Gaia DR3 source astrometry to its astrophysical parameters."""
    fMinParallaxMas = 1000.0 / fMaxDistancePc
    return (
        f"SELECT TOP {iRowLimit} s.source_id, s.ra, s.dec, s.ecl_lat, s.parallax, "
        "s.phot_g_mean_mag, ap.teff_gspphot, ap.radius_gspphot, ap.lum_flame, "
        "ap.mass_flame, ap.logg_gspphot "
        "FROM gaiadr3.gaia_source AS s "
        "JOIN gaiadr3.astrophysical_parameters AS ap ON s.source_id = ap.source_id "
        f"WHERE s.parallax > {fMinParallaxMas} AND s.phot_g_mean_mag < {fGmagLimit} "
        "AND ap.teff_gspphot IS NOT NULL AND ap.radius_gspphot IS NOT NULL "
        "AND ap.lum_flame IS NOT NULL")


def fsFetchCatalog(sQuery, sCachePath, iTimeoutS):
    """Download the TAP result as CSV, reusing the on-disk cache when present."""
    if os.path.exists(sCachePath) and os.path.getsize(sCachePath) > 0:
        return sCachePath
    dictParams = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": sQuery}
    oRequest = urllib.request.Request(S_TAP_URL,
                                      data=urllib.parse.urlencode(dictParams).encode("utf-8"))
    with urllib.request.urlopen(oRequest, timeout=iTimeoutS) as oResponse:
        sBody = oResponse.read().decode("utf-8")
    with open(sCachePath, "w") as oFile:
        oFile.write(sBody)
    return sCachePath


def fdfDeriveColumns(dfRaw, fMaxDistancePc, fMinLogg, fMaxRadiusRsun):
    """Filter to main-sequence dwarfs and derive the quantities the yield model consumes."""
    dfOut = dfRaw.dropna(subset=["parallax", "teff_gspphot", "radius_gspphot",
                                 "lum_flame"]).copy()
    dfOut["fDistancePc"] = 1000.0 / dfOut["parallax"]
    dfOut = dfOut[(dfOut["fDistancePc"] > 0) & (dfOut["fDistancePc"] <= fMaxDistancePc)]
    dfOut = dfOut[(dfOut["radius_gspphot"] > 0) & (dfOut["radius_gspphot"] <= fMaxRadiusRsun)]
    dfOut = dfOut[dfOut["lum_flame"] > 0]
    dfOut = dfOut[dfOut["logg_gspphot"].fillna(fMinLogg + 1.0) >= fMinLogg]
    dfOut = dfOut.rename(columns={"teff_gspphot": "fTeffK", "radius_gspphot": "fRadiusRsun",
                                  "lum_flame": "fLuminosityLsun", "mass_flame": "fMassMsun",
                                  "ecl_lat": "fEclipticLatDeg", "source_id": "sSourceId",
                                  "phot_g_mean_mag": "fGmag"})
    return dfOut.reset_index(drop=True)


def fdfFinalise(dfMerged):
    """Derive the Earth-equivalent insolation distance and order the catalog by its angular size."""
    dfOut = dfMerged.copy()
    dfOut["fEeidAu"] = np.sqrt(dfOut["fLuminosityLsun"])
    dfOut["fEeidArcsec"] = dfOut["fEeidAu"] / dfOut["fDistancePc"]
    return dfOut.sort_values("fEeidArcsec", ascending=False).reset_index(drop=True)


def fdictParseArgs():
    """Command-line configuration for the target-catalog build."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-distance-pc", type=float, default=50.0)
    p.add_argument("--gmag-limit", type=float, default=13.0)
    p.add_argument("--min-logg", type=float, default=3.5)
    p.add_argument("--max-radius-rsun", type=float, default=3.0)
    p.add_argument("--row-limit", type=int, default=300000)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--cache-csv", default="gaiaRawQuery.csv")
    p.add_argument("--supplement-cache-csv", default="hipparcosRawQuery.csv")
    p.add_argument("--supplement-vmag-limit", type=float, default=6.0)
    p.add_argument("--match-arcsec", type=float, default=10.0)
    p.add_argument("--out-catalog", default="targetCatalog.csv")
    p.add_argument("--out-summary", default="targetCatalogSummary.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    sQuery = fsBuildQuery(dictArgs["max_distance_pc"], dictArgs["gmag_limit"],
                          dictArgs["row_limit"])
    sCsv = fsFetchCatalog(sQuery, dictArgs["cache_csv"], dictArgs["timeout_s"])
    dfRaw = pd.read_csv(sCsv)
    dfGaia = fdfDeriveColumns(dfRaw, dictArgs["max_distance_pc"], dictArgs["min_logg"],
                              dictArgs["max_radius_rsun"])
    try:
        dfSupp = fdfBrightStarSupplement(dictArgs["max_distance_pc"],
                                         dictArgs["supplement_vmag_limit"],
                                         dictArgs["supplement_cache_csv"], dictArgs["timeout_s"])
        dfSupp = dfSupp[dfSupp["fRadiusRsun"] <= dictArgs["max_radius_rsun"]]
        bSupplementFetched = True
    except Exception as oError:
        print(f"WARNING: bright-star supplement unavailable ({oError}); Gaia DR3 alone omits "
              "targets brighter than G ~ 3.")
        dfSupp, bSupplementFetched = pd.DataFrame(), False
    dfCat = fdfFinalise(fdfMergeSupplement(dfGaia, dfSupp, dictArgs["match_arcsec"]))
    dfCat.to_csv(dictArgs["out_catalog"], index=False)
    faTeff = dfCat["fTeffK"].to_numpy()
    dictSummary = {
        "sQuery": sQuery,
        "sSource": "Gaia DR3 (gaia_source x astrophysical_parameters), ESA archive TAP",
        "iRawRows": int(len(dfRaw)),
        "iCatalogRows": int(len(dfCat)),
        "iGaiaRows": int((dfCat["sCatalogSource"] == "gaiadr3").sum()),
        "iHipparcosSupplementRows": int((dfCat["sCatalogSource"] == "hipparcos").sum()),
        "bSupplementFetched": bSupplementFetched,
        "iFgkStars": int(np.sum((faTeff >= 3900) & (faTeff <= 7300))),
        "fMedianDistancePc": float(dfCat["fDistancePc"].median()),
        "fMaxEeidArcsec": float(dfCat["fEeidArcsec"].max()),
        "dictTeffPercentiles": {s: float(np.percentile(faTeff, int(s)))
                                for s in ("5", "50", "95")},
    }
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps(dictSummary, indent=2))


if __name__ == "__main__":
    main()
