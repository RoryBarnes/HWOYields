#!/usr/bin/env python3
"""Bound the yield carried by the four stars Stark (2024) Sec. 3.3 pins to LBTI-measured exozodi.

Stark assigns eps Eri 297, tet Boo 148, 72 Her 588 and 110 Her 235 zodis and attributes about a
third of the 19.8 -> 17.6 exozodi shift (about one EEC) to them. This pipeline gives every star a
lognormal draw instead. This reads each pinned star's best completeness from A04's table (the
most any allocation could buy, over visits and exposure) and converts it to EECs at the nominal
eta, which is an upper bound on what pinning could remove.
"""

import argparse
import json

import numpy as np

DICT_PINNED = {"16537": ("eps Eri", 297.0), "70497": ("tet Boo", 148.0),
               "84862": ("72 Her", 588.0), "92043": ("110 Her", 235.0)}


def fdictHipBySourceId(sCatalogPath):
    """Map catalog sSourceId to its HIP number string."""
    import pandas as pd
    dfCat = pd.read_csv(sCatalogPath, usecols=["sSourceId", "sHipName"])
    return {s: str(int(h)) for s, h in zip(dfCat["sSourceId"], dfCat["sHipName"]) if h == h}


def fdictStarBound(dictNpz, iIndex, fEta):
    """Maximum albedo-drawn completeness of one star, and its EEC equivalent."""
    fComp = float(np.max(dictNpz["faCompAlbedo_canonical"][iIndex]))
    return {"fMaxCompleteness": fComp, "fMaxEec": fComp * fEta,
            "fDistancePc": float(dictNpz["faDistancePc"][iIndex]),
            "fLuminosityLsun": float(dictNpz["faLuminosityLsun"][iIndex])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completeness", default="../computeCompleteness/completeness.npz")
    p.add_argument("--catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--eta", type=float, default=0.24)
    p.add_argument("--out-json", default="pinnedExozodiStarContribution.json")
    dictArgs = vars(p.parse_args())
    dictNpz = np.load(dictArgs["completeness"], allow_pickle=True)
    dictHip = fdictHipBySourceId(dictArgs["catalog"])
    dictOut = {}
    for i, sId in enumerate(dictNpz["saSourceId"]):
        sHip = dictHip.get(str(sId))
        if sHip in DICT_PINNED:
            sName, fZodi = DICT_PINNED[sHip]
            dictOut[sName] = {"fPinnedZodi": fZodi, **fdictStarBound(dictNpz, i, dictArgs["eta"])}
    dictOut["fTotalMaxEec"] = sum(d["fMaxEec"] for d in dictOut.values() if isinstance(d, dict))
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps(dictOut, indent=2))


if __name__ == "__main__":
    main()
