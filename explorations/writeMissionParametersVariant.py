#!/usr/bin/env python3
"""Write a copy of A02's missionParameters.json with named dictMission overrides, for what-if runs.

Step scripts take the mission parameters as a file, so a counterfactual (exozodi fixed, a
different noise floor, ...) is run by pointing a step's own script at a variant written here,
with its output kept under explorations/. Values are parsed as JSON (true, 26.5, "text").
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSONVALUE")
    p.add_argument("--out", required=True)
    dictArgs = vars(p.parse_args())
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    for sPair in dictArgs["set"]:
        sKey, sValue = sPair.split("=", 1)
        dictParams["dictMission"][sKey] = json.loads(sValue)
    dictParams["dictVariantOverrides"] = dictArgs["set"]
    with open(dictArgs["out"], "w") as oFile:
        json.dump(dictParams, oFile, indent=2)
    print(f"wrote {dictArgs['out']} with {dictArgs['set']}")


if __name__ == "__main__":
    main()
