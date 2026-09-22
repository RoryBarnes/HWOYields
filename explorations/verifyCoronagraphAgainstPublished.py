#!/usr/bin/env python3
"""Verify the parametrized DMVC curves against the published ones read off Stark et al. (2019).

The coronagraph maps are the one input the papers do not distribute, so the parametrization
standing in for them is this pipeline's largest approximation and the first place to look when
the calibration demands an implausible factor. Both quantities are checked here: the core
throughput Upsilon_c, which turns out to agree, and the raw contrast zeta, which did not until
the inner rise was added.
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "..")
from yieldlib import coronagraph as cg  # noqa: E402

LIST_THROUGHPUT_READOFF = [(1.0, 0.00), (2.0, 0.05), (3.0, 0.15), (3.5, 0.20), (4.0, 0.24),
                           (5.0, 0.30), (6.0, 0.35), (8.0, 0.40), (10.0, 0.42), (15.0, 0.44),
                           (20.0, 0.45)]
LIST_CONTRAST_READOFF = [(1.5, 1.0e-8), (2.0, 3.0e-9), (2.5, 6.0e-10), (3.0, 2.5e-10),
                         (4.0, 1.2e-10), (6.0, 1.0e-10), (10.0, 1.0e-10)]
S_SOURCE = "Stark et al. (2019), DMVC contrast and core-throughput figure; values read off."


def fdictCompare(listReadoff, fnModel, bLogarithmic):
    """Model against read-off values, with the residual expressed appropriately."""
    listRows = []
    for fSep, fPublished in listReadoff:
        fModel = float(fnModel(fSep))
        fResidual = (np.log10(fModel / fPublished) if bLogarithmic and fPublished > 0
                     else fModel - fPublished)
        listRows.append({"fSeparationLamD": fSep, "fPublished": fPublished,
                         "fModel": fModel, "fResidual": fResidual})
    return listRows


def fdictParseArgs():
    """Command-line configuration for the coronagraph verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-throughput-error", type=float, default=0.05)
    p.add_argument("--max-contrast-dex", type=float, default=0.35)
    p.add_argument("--out-json", default="coronagraphVerification.json")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    listThroughput = fdictCompare(LIST_THROUGHPUT_READOFF, cg.faCoreThroughput, False)
    listContrast = fdictCompare(LIST_CONTRAST_READOFF, cg.faRawContrast, True)
    fWorstThroughput = max(abs(d["fResidual"]) for d in listThroughput)
    fWorstContrast = max(abs(d["fResidual"]) for d in listContrast)
    dictOut = {
        "sSource": S_SOURCE,
        "listThroughput": listThroughput, "listContrast": listContrast,
        "fWorstThroughputError": fWorstThroughput,
        "fWorstContrastDex": fWorstContrast,
        "bThroughputAgrees": bool(fWorstThroughput <= dictArgs["max_throughput_error"]),
        "bContrastAgrees": bool(fWorstContrast <= dictArgs["max_contrast_dex"]),
    }
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(f"{'sep':>6}{'Upsilon pub':>13}{'model':>9}{'diff':>9}")
    for d in listThroughput:
        print(f"{d['fSeparationLamD']:>6.1f}{d['fPublished']:>13.2f}{d['fModel']:>9.3f}"
              f"{d['fResidual']:>+9.3f}")
    print(f"\n{'sep':>6}{'zeta pub':>12}{'model':>12}{'dex':>8}")
    for d in listContrast:
        print(f"{d['fSeparationLamD']:>6.1f}{d['fPublished']:>12.1e}{d['fModel']:>12.1e}"
              f"{d['fResidual']:>+8.2f}")
    print(f"\nthroughput agrees: {dictOut['bThroughputAgrees']} "
          f"(worst {fWorstThroughput:.3f})")
    print(f"contrast agrees  : {dictOut['bContrastAgrees']} (worst {fWorstContrast:.2f} dex)")


if __name__ == "__main__":
    main()
