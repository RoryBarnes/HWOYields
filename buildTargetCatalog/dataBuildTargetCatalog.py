#!/usr/bin/env python3
"""Build the HWO target catalog from the HWO Preliminary Input Catalog (HPIC), Tuchow et al. 2024.

Stark et al. (2024), Sec. 2: "We use the HWO Preliminary Input Catalog (HPIC) as our input
target list."  This step therefore reads HPIC itself rather than approximating it.

An earlier version of this step reconstructed a stand-in from Gaia DR3 plus a Hipparcos
bright-star supplement, on the mistaken belief that HPIC was not reachable from this container;
the download had in fact returned a 301 redirect stub, which was read as a missing file.  That
substitution cost real fidelity, and the numbers are worth recording because they bound how much
any surviving catalog difference can matter:

  * The reconstruction required a Gaia FLAME luminosity, which is published for a shrinking
    fraction of stars with distance.  Against HPIC it was short by 26% of FGK dwarfs at 15-20 pc,
    29% at 20-25 pc and 34% at 25-40 pc -- a deficit concentrated exactly where the survey
    optimizer looks once the nearest stars saturate.
  * Teff and luminosity for the bright supplement came from a B-V colour calibration plus a
    bolometric correction rather than from measurement.  For alpha Cen A that gave 5568 K against
    HPIC's 5790 K, a 4% error entering both the stellar flux and the EEID.

HPIC supplies distance, effective temperature, luminosity, radius, mass and V photometry
directly, so none of that derivation survives here.  The one HPIC product this step carries but
does not yet use is binarity (WDS and GCNS separations): Stark uses it to compute a per-star
stray-light background, which this pipeline does not model.  The columns are retained so that
omission stays visible and addressable.
"""

import argparse
import hashlib
import io
import json
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd

S_HPIC_URL = "https://emac.gsfc.nasa.gov/data/hpic/hpic_1.1.tar.gz"
S_HPIC_MEMBER = "HPIC/full_HPIC.txt"
F_OBLIQUITY_DEG = 23.4392911
LIST_NUMERIC_COLS = ["sy_dist", "st_teff", "st_lum", "st_rad", "st_mass", "st_logg", "sy_vmag",
                     "sy_bmag", "ra", "dec", "wds_sep"]
LIST_USED_COLS = ["star_name", "hip_name", "simbad_name", "ra", "dec", "sy_dist", "st_teff",
                  "st_lum", "st_rad", "st_mass", "st_logg", "sy_vmag", "sy_bmag",
                  "st_spectype", "dwarf_fl", "wds_sep", "known_binary_fl", "sy_planets_flag"]


def fsEnsureArchive(sCachePath, iTimeoutS):
    """Download the HPIC tarball to the cache path if it is not already there, and return it.

    urllib follows the 301 that emac.gsfc.nasa.gov issues to its ckan-files host; a plain curl
    without --location does not, and silently writes a 169-byte redirect page instead.
    """
    if os.path.exists(sCachePath) and os.path.getsize(sCachePath) > 1_000_000:
        return sCachePath
    with urllib.request.urlopen(S_HPIC_URL, timeout=iTimeoutS) as oResponse:
        baBody = oResponse.read()
    with open(sCachePath, "wb") as oFile:
        oFile.write(baBody)
    return sCachePath


def fsFileDigest(sPath):
    """SHA-256 of a file, recorded so a rerun can prove it read the same catalog."""
    oHash = hashlib.sha256()
    with open(sPath, "rb") as oFile:
        for baChunk in iter(lambda: oFile.read(1 << 20), b""):
            oHash.update(baChunk)
    return oHash.hexdigest()


def fdfReadHpic(sArchivePath):
    """Read the pipe-delimited HPIC table straight out of the tarball."""
    with tarfile.open(sArchivePath, "r:gz") as oTar:
        oMember = oTar.extractfile(S_HPIC_MEMBER)
        baBody = oMember.read()
    dfRaw = pd.read_csv(io.BytesIO(baBody), delimiter="|", low_memory=False,
                        usecols=LIST_USED_COLS)
    for sCol in LIST_NUMERIC_COLS:
        dfRaw[sCol] = pd.to_numeric(dfRaw[sCol], errors="coerce")
    return dfRaw


def faEclipticLatitude(faRaDeg, faDecDeg):
    """Ecliptic latitude in degrees from equatorial coordinates."""
    fEps = np.radians(F_OBLIQUITY_DEG)
    faRa, faDec = np.radians(faRaDeg), np.radians(faDecDeg)
    faSinBeta = np.sin(faDec) * np.cos(fEps) - np.cos(faDec) * np.sin(fEps) * np.sin(faRa)
    return np.degrees(np.arcsin(np.clip(faSinBeta, -1.0, 1.0)))


def fdfSelectDwarfs(dfRaw, fMaxDistancePc, fMaxRadiusRsun, fMinLogg):
    """Keep main-sequence stars inside the distance limit that carry every parameter we need.

    HPIC's own dwarf_fl is the primary cut; the radius and log(g) bounds catch the handful of
    rows whose spectral type and bulk parameters disagree (a "K1V" carrying a 9.2 Rsun radius).
    """
    dfOut = dfRaw.dropna(subset=["sy_dist", "st_teff", "st_lum", "st_rad", "sy_vmag"]).copy()
    dfOut = dfOut[(dfOut["sy_dist"] > 0) & (dfOut["sy_dist"] <= fMaxDistancePc)]
    dfOut = dfOut[dfOut["dwarf_fl"].astype(str).isin(["1", "1.0", "True"])]
    dfOut = dfOut[(dfOut["st_rad"] > 0) & (dfOut["st_rad"] <= fMaxRadiusRsun)]
    dfOut = dfOut[dfOut["st_logg"].fillna(fMinLogg + 1.0) >= fMinLogg]
    return dfOut.reset_index(drop=True)


def fdfDeriveColumns(dfDwarfs):
    """Rename HPIC columns to the pipeline's schema and derive the EEID."""
    dfOut = pd.DataFrame({
        "sSourceId": dfDwarfs["star_name"].astype(str),
        "sHipName": dfDwarfs["hip_name"].astype(str),
        "sSimbadName": dfDwarfs["simbad_name"].astype(str),
        "sSpectralType": dfDwarfs["st_spectype"].astype(str),
        "ra": dfDwarfs["ra"].to_numpy(),
        "dec": dfDwarfs["dec"].to_numpy(),
        "fVmag": dfDwarfs["sy_vmag"].to_numpy(),
        "fTeffK": dfDwarfs["st_teff"].to_numpy(),
        "fRadiusRsun": dfDwarfs["st_rad"].to_numpy(),
        "fLuminosityLsun": 10.0 ** dfDwarfs["st_lum"].to_numpy(),
        "fMassMsun": dfDwarfs["st_mass"].to_numpy(),
        "fLogg": dfDwarfs["st_logg"].to_numpy(),
        "fDistancePc": dfDwarfs["sy_dist"].to_numpy(),
        "fWdsSepArcsec": dfDwarfs["wds_sep"].to_numpy(),
        "bKnownBinary": dfDwarfs["known_binary_fl"].astype(str).isin(["1", "1.0", "True"]),
        "sCatalogSource": "hpic1.1",
    })
    dfOut["fEclipticLatDeg"] = faEclipticLatitude(dfOut["ra"].to_numpy(),
                                                  dfOut["dec"].to_numpy())
    dfOut["fEeidAu"] = np.sqrt(dfOut["fLuminosityLsun"])
    dfOut["fEeidArcsec"] = dfOut["fEeidAu"] / dfOut["fDistancePc"]
    return dfOut.sort_values("fEeidArcsec", ascending=False).reset_index(drop=True)


def fdictSummary(dfRaw, dfCat, sArchivePath):
    """Provenance and shape of the built catalog."""
    faTeff = dfCat["fTeffK"].to_numpy()
    faDist = dfCat["fDistancePc"].to_numpy()
    bFgk = (faTeff >= 3900) & (faTeff <= 7300)
    return {
        "sSource": "HWO Preliminary Input Catalog v1.1 (Tuchow et al. 2024), NASA EMAC",
        "sUrl": S_HPIC_URL,
        "sArchiveSha256": fsFileDigest(sArchivePath),
        "iRawRows": int(len(dfRaw)),
        "iCatalogRows": int(len(dfCat)),
        "iFgkStars": int(np.sum(bFgk)),
        "iKnownBinaries": int(dfCat["bKnownBinary"].sum()),
        "fMedianDistancePc": float(np.median(faDist)),
        "fMaxEeidArcsec": float(dfCat["fEeidArcsec"].max()),
        "dictFgkCountsByDistance": {s: int(np.sum(bFgk & (faDist <= float(s))))
                                    for s in ("10", "15", "20", "30", "50")},
        "dictTeffPercentiles": {s: float(np.percentile(faTeff, int(s)))
                                for s in ("5", "50", "95")},
    }


def fdictParseArgs():
    """Command-line configuration for the target-catalog build."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-distance-pc", type=float, default=50.0)
    p.add_argument("--max-radius-rsun", type=float, default=3.0)
    p.add_argument("--min-logg", type=float, default=3.5)
    p.add_argument("--timeout-s", type=int, default=1200)
    p.add_argument("--archive", default="reference/hpic_1.1.tar.gz")
    p.add_argument("--out-catalog", default="targetCatalog.csv")
    p.add_argument("--out-summary", default="targetCatalogSummary.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    os.makedirs(os.path.dirname(dictArgs["archive"]) or ".", exist_ok=True)
    sArchive = fsEnsureArchive(dictArgs["archive"], dictArgs["timeout_s"])
    dfRaw = fdfReadHpic(sArchive)
    dfDwarfs = fdfSelectDwarfs(dfRaw, dictArgs["max_distance_pc"], dictArgs["max_radius_rsun"],
                               dictArgs["min_logg"])
    dfCat = fdfDeriveColumns(dfDwarfs)
    dfCat.to_csv(dictArgs["out_catalog"], index=False)
    dictSummary = fdictSummary(dfRaw, dfCat, sArchive)
    with open(dictArgs["out_summary"], "w") as oFile:
        json.dump(dictSummary, oFile, indent=2)
    print(json.dumps(dictSummary, indent=2))


if __name__ == "__main__":
    main()
