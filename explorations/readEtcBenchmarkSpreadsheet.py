#!/usr/bin/env python3
"""Extract AYO's and EXOSIMS's characterization count rates and times from the ETC benchmark.

Stark et al. (2025, arXiv:2502.18556) cross-validated AYO, EXOSIMS and EBS exposure-time
calculators on five fiducial F-K stars, and published their spreadsheets
(https://www.starkspace.com/permanent/ETC_cal_char.xlsx, copied to reference/). This reads the
per-star rows for H2O at 1000 nm and O2 at 800 nm with the standard library (no spreadsheet
package is in the image), so AYO's own characterization times can be set beside this model's.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile

S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
S_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DICT_COLUMNS = {"AYO_800": 4, "EXOSIMS_800": 6, "AYO_1000": 10, "EXOSIMS_1000": 12}


def flistSharedStrings(oZip):
    """The workbook's shared-string table."""
    if "xl/sharedStrings.xml" not in oZip.namelist():
        return []
    oRoot = ET.fromstring(oZip.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{{{S_NS}}}t"))
            for si in oRoot.findall(f"{{{S_NS}}}si")]


def fdictSheetRows(oZip, sTarget, listShared):
    """{row number: {column number: text}} for one worksheet."""
    dictRows = {}
    for oCell in ET.fromstring(oZip.read("xl/" + sTarget)).iter(f"{{{S_NS}}}c"):
        oValue = oCell.find(f"{{{S_NS}}}v")
        if oValue is None:
            continue
        sText = listShared[int(oValue.text)] if oCell.get("t") == "s" else oValue.text
        oMatch = re.match(r"([A-Z]+)(\d+)", oCell.get("r"))
        iCol = 0
        for sChar in oMatch.group(1):
            iCol = iCol * 26 + ord(sChar) - 64
        dictRows.setdefault(int(oMatch.group(2)), {})[iCol] = sText
    return dictRows


def fdictStar(dictRows):
    """Named parameters for each code/band column of one star's sheet."""
    dictOut = {s: {} for s in DICT_COLUMNS}
    for dictRow in dictRows.values():
        sName = dictRow.get(1, "")
        if not sName or sName.startswith("Check"):
            continue
        for sKey, iCol in DICT_COLUMNS.items():
            try:
                dictOut[sKey][sName] = float(dictRow[iCol])
            except (KeyError, ValueError):
                pass
    return dictOut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xlsx", default="reference/ETC_cal_char.xlsx")
    p.add_argument("--out-json", default="output/etcBenchmarkCharacterization.json")
    dictArgs = vars(p.parse_args())
    oZip = zipfile.ZipFile(dictArgs["xlsx"])
    listShared = flistSharedStrings(oZip)
    oWorkbook = ET.fromstring(oZip.read("xl/workbook.xml"))
    dictTargets = {r.get("Id"): r.get("Target").replace("/xl/", "")
                   for r in ET.fromstring(oZip.read("xl/_rels/workbook.xml.rels"))}
    dictOut = {}
    for oSheet in oWorkbook.find(f"{{{S_NS}}}sheets"):
        if "HIP" not in oSheet.get("name"):
            continue
        dictRows = fdictSheetRows(oZip, dictTargets[oSheet.get(f"{{{S_REL}}}id")], listShared)
        dictOut[oSheet.get("name").replace(" (Char.)", "")] = fdictStar(dictRows)
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    for sStar, d in dictOut.items():
        dictAyo = d["AYO_1000"]
        print(f"{sStar:<11} d={dictAyo.get('dist', 0):5.1f} pc  L={dictAyo.get('L_star', 0):5.2f}  "
              f"AYO H2O t_science={dictAyo.get('t_science', 0) / 86400:6.1f} d  "
              f"CR_p={dictAyo.get('CR_p', 0):.2e}  CR_bez={dictAyo.get('CR_bez', 0):.2e}  "
              f"CR_bd={dictAyo.get('CR_bd', 0):.2e}")


if __name__ == "__main__":
    main()
