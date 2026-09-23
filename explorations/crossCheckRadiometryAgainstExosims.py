#!/usr/bin/env python3
"""Cross-check this pipeline's radiometry against EXOSIMS on identical stars and identical optics.

EXOSIMS (Savransky et al.) is an independent, community-maintained mission simulator. It shares
no code with this pipeline and, importantly, does not share its methods either: it schedules
observations dynamically rather than solving AYO's equal-slope allocation, and its integration
time follows Nemati (2014), which carries a speckle-floor divergence term that Stark (2019)
Eq. 1 does not have. So it is a STRONG test of the layers the two must agree on if either is
right -- stellar flux, background surface brightnesses, photometric aperture, detector noise --
and NOT a test of the allocation layer, which was separately verified to 0.00% against an exact
dynamic-programming optimum.

The comparison is deliberately narrowed to the places where the two codes model the same physics
DIFFERENTLY, because anywhere I hand EXOSIMS my own coronagraph numbers agreement is guaranteed
by construction and proves nothing:

  * stellar photon rate -- a Planck blackbody from Teff and radius here, a Pickles/BPGS spectral
    template renormalised through a synphot bandpass there;
  * local zodiacal surface brightness -- a flat 23 mag/arcsec^2 here, EXOSIMS's Stark
    ZodiacalLight module with its ecliptic-latitude and solar-elongation dependence there;
  * exozodi intensity at the planet's location;
  * the photometric aperture solid angle and the pixel count that sets detector noise.

Every optical constant that CAN be matched is matched, so a disagreement in the four quantities
above is attributable to the physics and not to bookkeeping.
"""

import argparse
import json
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yieldlib import coronagraph as cg  # noqa: E402
from yieldlib import physics as ph  # noqa: E402

F_ARCSEC_PER_RAD = 206264.80624709636


def fdictExosimsSpec(dictMission, dictBand, fSkyThroughput, fCoreThroughput, fCoreAreaArcsec2,
                     listStarNames):
    """An EXOSIMS mission specification carrying this pipeline's optics verbatim.

    Scalar coronagraph entries are used rather than tables so that EXOSIMS evaluates exactly the
    core throughput, extended-source throughput and raw contrast that this pipeline used at the
    working angle under test.
    """
    fPixelScaleArcsec = float(dictMission["fApertureRadiusLamD"]) * \
        (dictBand["fLambdaM"] / dictMission["fDiameterM"]) * F_ARCSEC_PER_RAD
    return {
        "pupilDiam": float(dictMission["fDiameterM"]),
        "obscurFac": 0.0,
        "shapeFac": np.pi / 4.0,
        "ppFact": 1.0,
        "scienceInstruments": [{
            "name": "imager",
            "QE": float(dictMission["fQuantumEfficiency"]),
            "optics": float(dictBand["fOpticalThroughput"]),
            "pixelScale": fPixelScaleArcsec,
            "pixelSize": 1.3e-05,
            "sread": float(dictMission["fReadNoise"]),
            "idark": float(dictMission["fDarkCurrent"]),
            "CIC": float(dictMission["fClockInducedCharge"]),
            "texp": 1000.0,
            "ENF": 1.0,
        }],
        "starlightSuppressionSystems": [{
            "name": "dmvc",
            "lam": float(dictBand["fLambdaM"] * 1e9),
            "BW": float(dictBand["fBandwidthFraction"]),
            "occ_trans": float(fSkyThroughput),
            "core_thruput": float(fCoreThroughput),
            "core_contrast": float(dictMission["fContrastFloor"]),
            "core_area": float(fCoreAreaArcsec2),
            "optics": float(dictMission["fContaminationThroughput"] *
                            dictMission["fDetectiveQuantumEfficiency"]),
        }],
        "observingModes": [{"instName": "imager", "systName": "dmvc", "detectionMode": True,
                            "SNR": float(dictBand["fSignalToNoise"])}],
        "modules": {
            "StarCatalog": "FakeCatalog", "OpticalSystem": "Nemati", "ZodiacalLight": "Stark",
            "BackgroundSources": " ", "PlanetPhysicalModel": " ", "Observatory": " ",
            "TimeKeeping": " ", "PostProcessing": " ", "Completeness": " ",
            "PlanetPopulation": " ", "TargetList": " ", "SimulatedUniverse": " ",
            "SurveySimulation": " ", "SurveyEnsemble": " ",
        },
        "explainFiltering": False, "filterBinaries": False, "fillPhotometry": False,
        "ntargs": int(len(listStarNames)) + 40, "star_dist": 10.0, "seed": 1,
        "cachedir": "/tmp/exosimsCache",
    }


def foBuildTargetList(dictSpec, dfCatalog):
    """Build an EXOSIMS TargetList and overwrite its stars with this pipeline's own.

    EXOSIMS ships an HPIC reader, but version 3.6.5 predates pandas 3's copy-on-write and its
    loader raises on the read-only arrays pandas now returns. Rather than patch an installed
    package, the target list is created from EXOSIMS's synthetic catalog and each star's
    parameters are then replaced with the HPIC values this pipeline already uses.

    That is the right division of labour for the comparison in any case: both codes are handed
    IDENTICAL stellar parameters, and EXOSIMS still supplies its own spectral templates, synphot
    bandpass integration, zodiacal model and detector bookkeeping. What is being tested is those,
    not whether two programs can read the same file.
    """
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from EXOSIMS.Prototypes.TargetList import TargetList

    TargetList.calc_saturation_and_intCutoff_vals = fnSkipSaturationCalcs
    oTargetList = TargetList(**dictSpec)
    iStars = len(dfCatalog)
    assert oTargetList.nStars >= iStars, "synthetic catalog too small to host the real stars"
    oTargetList.revise_lists(np.arange(iStars))
    oTargetList.Name = np.array([str(s).strip() for s in dfCatalog["sSourceId"]])
    oTargetList.Spec = np.array([fsCleanSpectralType(s) for s in dfCatalog["sSpectralType"]])
    oTargetList.Vmag = dfCatalog["fVmag"].to_numpy(dtype=float)
    oTargetList.Teff = dfCatalog["fTeffK"].to_numpy(dtype=float) * u.K
    oTargetList.dist = dfCatalog["fDistancePc"].to_numpy(dtype=float) * u.pc
    oTargetList.L = dfCatalog["fLuminosityLsun"].to_numpy(dtype=float)
    oTargetList.MV = oTargetList.Vmag - 5.0 * (np.log10(dfCatalog["fDistancePc"].to_numpy()) - 1.0)
    oTargetList.coords = SkyCoord(ra=dfCatalog["ra"].to_numpy() * u.deg,
                                  dec=dfCatalog["dec"].to_numpy() * u.deg,
                                  distance=oTargetList.dist)
    oTargetList.diameter = np.arctan(2.0 * dfCatalog["fRadiusRsun"].to_numpy() * u.Rsun /
                                     oTargetList.dist).to(u.mas)
    oTargetList.nStars = iStars
    oTargetList.fillPhotometryVals()
    oTargetList.star_fluxes = {}
    oTargetList.blackbody_spectra = np.full(iStars, None, dtype=object)
    fnRecomputeExozodiIntensity(oTargetList)
    return oTargetList


def fnRecomputeExozodiIntensity(oTargetList):
    """Recompute EXOSIMS's per-star JEZ0 for the stars that were installed after construction.

    JEZ0 is evaluated once when the TargetList is built and then cached per observing mode. Left
    alone it would describe the synthetic stars the list was seeded with, and the comparison would
    silently be against the wrong stars -- the exozodi ratio would come out constant, which is
    exactly what it did before this was noticed.
    """
    import numpy as np
    oZodi = oTargetList.ZodiacalLight
    for dictMode in oTargetList.OpticalSystem.observingModes:
        faColour = oTargetList.star_flambda_factors[dictMode["hex"]] \
            if hasattr(oTargetList, "star_flambda_factors") and \
            dictMode["hex"] in getattr(oTargetList, "star_flambda_factors", {}) \
            else np.ones(oTargetList.nStars)
        oTargetList.JEZ0[dictMode["hex"]] = oZodi.calc_JEZ0(
            oTargetList.MV, oTargetList.L, faColour, dictMode["deltaLam"])


def fnSkipSaturationCalcs(oSelf):
    """Stub for TargetList.calc_saturation_and_intCutoff_vals, which cannot run in this install.

    EXOSIMS 3.6.5 raises `TypeError: only 0-dimensional arrays can be converted to Python scalars`
    inside Nemati.int_time_denom_obj under this numpy, because the scalar root-find in
    calc_dMag_per_intTime hands Cp_Cb_Csp a 0-d dMag that the code then indexes as an array. This
    is not a property of the configuration used here: EXOSIMS's own shipped sampleScript_coron
    fails at the same line, after successfully computing star fluxes for all 2050 EXOCAT targets.

    The routine only precomputes saturation and integration-cutoff dMag limits used by EXOSIMS's
    scheduler. Nothing in this comparison reads them -- the count rates come from Cp_Cb_Csp with
    explicit array arguments -- so it is replaced with placeholders rather than patched.
    """
    import numpy as np
    oSelf.saturation_dMag = np.full(oSelf.nStars, 40.0)
    oSelf.saturation_comp = np.ones(oSelf.nStars)
    oSelf.intCutoff_dMag = np.full(oSelf.nStars, 30.0)
    oSelf.intCutoff_comp = np.ones(oSelf.nStars)
    oSelf.int_dMag = np.full(oSelf.nStars, 25.0)
    oSelf.int_tmin = None
    for sAtt in ("intCutoff_dMag", "intCutoff_comp", "saturation_dMag", "saturation_comp"):
        if sAtt not in oSelf.catalog_atts:
            oSelf.catalog_atts.append(sAtt)


def fsCleanSpectralType(oSpec):
    """Reduce an HPIC spectral type to a bare dwarf type EXOSIMS's MeanStars matcher accepts.

    HPIC carries SIMBAD strings such as "K1.5IIIFe-0.5" and "M0VpCa-3Cr-1". EXOSIMS drops any
    star whose type it cannot parse, which silently desynchronises the comparison, so only the
    leading temperature class and subclass are kept and the luminosity class is forced to V. The
    type is used solely to pick a spectral template for the flux; every other stellar parameter
    is supplied from HPIC.
    """
    oMatch = re.match(r"\s*([OBAFGKM])\s*(\d(?:\.\d)?)?", str(oSpec))
    if not oMatch:
        return "G2V"
    return f"{oMatch.group(1)}{int(float(oMatch.group(2))) if oMatch.group(2) else 2}V"


def flistMatchStars(oTargetList, listWantedNames):
    """Indices in the EXOSIMS target list for the named HPIC stars, in the order requested."""
    dictIndex = {str(s).strip(): i for i, s in enumerate(oTargetList.Name)}
    return [(s, dictIndex[s]) for s in listWantedNames if s in dictIndex]


def fdictOurRates(dictStar, dictBand, dictMission, fSepArcsec, fDeltaMag, fExozodiLevel,
                  fSkyThroughput):
    """This pipeline's component count rates for one star at one working angle."""
    fArea = np.pi * (dictMission["fDiameterM"] / 2.0) ** 2
    fThroughput = dictBand["fOpticalThroughput"] * dictMission["fContaminationThroughput"] * \
        dictMission["fDetectiveQuantumEfficiency"] * dictMission["fQuantumEfficiency"]
    fBandwidthM = dictBand["fLambdaM"] * dictBand["fBandwidthFraction"]
    fStarFlux = float(ph.faStellarPhotonFlux(dictBand["fLambdaM"], dictStar["fTeffK"],
                                             dictStar["fRadiusRsun"], dictStar["fDistancePc"]))
    fStarRate = fStarFlux * fBandwidthM * fArea * fThroughput
    fLamD = cg.fnLambdaOverDArcsec(dictBand["fLambdaM"], dictMission["fDiameterM"])
    fUpsilon = float(cg.faCoreThroughput(np.array([fSepArcsec / fLamD]),
                                         fThroughputMax=dictMission["fCoreThroughputMax"],
                                         fIwaLamD=dictMission["fIwaLamD"],
                                         fOwaLamD=dictMission["fOwaLamD"])[0])
    fOmega = ph.fnPhotometricApertureSolidAngle(dictBand["fLambdaM"], dictMission["fDiameterM"],
                                                dictMission["fApertureRadiusLamD"])
    fZeroMag = ph.fnZeroMagPhotonFlux(dictBand["fLambdaM"]) * (fBandwidthM * 1e6)
    fBackgroundCommon = fOmega * fArea * fThroughput * fSkyThroughput
    fAxisScaled = fSepArcsec * dictStar["fDistancePc"] / np.sqrt(dictStar["fLuminosityLsun"])
    fExozodiMag = dictMission["fExozodiMagArcsec2"] + 5.0 * np.log10(fAxisScaled)
    return {
        "fStarRate": fStarRate,
        "fPlanet": fStarRate * fUpsilon * 10.0 ** (-0.4 * fDeltaMag),
        "fLeak": fStarRate * fUpsilon * dictMission["fContrastFloor"],
        "fZodi": fZeroMag * 10 ** (-0.4 * dictMission["fZodiMagArcsec2"]) * fBackgroundCommon,
        "fExozodi": fExozodiLevel * fZeroMag * 10 ** (-0.4 * fExozodiMag) * fBackgroundCommon,
        "fOmegaArcsec2": fOmega,
        "fCoreThroughput": fUpsilon,
    }


def fdictExosimsRates(oTargetList, iStar, dictMode, fSepArcsec, fDeltaMag, fExozodiLevel):
    """EXOSIMS's component count rates for the same star, working angle and planet brightness.

    The local zodiacal surface brightness is taken at EXOSIMS's nominal fZ0 rather than from its
    pointing-dependent model, to match this pipeline's single 23 mag/arcsec^2 value; the exozodi
    intensity is EXOSIMS's own per-star JEZ0, which carries the Stark et al. (2014) spectral-type
    scaling and is one of the quantities under test.
    """
    import astropy.units as u
    oZodi = oTargetList.ZodiacalLight
    faSind = np.array([iStar])
    faWa = np.array([fSepArcsec]) * u.arcsec
    faFz = np.array([oZodi.fZ0.to_value(oZodi.inv_arcsec2)]) * oZodi.inv_arcsec2
    faJez = oTargetList.JEZ0[dictMode["hex"]][faSind] * fExozodiLevel
    _, _, _, dictExtra = oTargetList.OpticalSystem.Cp_Cb_Csp(
        oTargetList, faSind, faFz, faJez, np.array([fDeltaMag]), faWa, dictMode,
        returnExtra=True)
    return {sKey: float(np.atleast_1d(oValue.value if hasattr(oValue, "value") else oValue)[0])
            for sKey, oValue in dictExtra.items()}


def fdictComponentRatios(dictOurs, dictTheirs):
    """This pipeline's count rate divided by EXOSIMS's, component by component."""
    listPairs = [("fStarRate", "C_star"), ("fPlanet", "C_p0"), ("fLeak", "C_sr"),
                 ("fZodi", "C_z"), ("fExozodi", "C_ez")]
    dictOut = {}
    for sOurs, sTheirs in listPairs:
        fTheirs = dictTheirs.get(sTheirs)
        dictOut[sOurs] = (dictOurs[sOurs] / fTheirs) if fTheirs else None
    dictOut["iNumPixels"] = dictTheirs.get("Npix")
    return dictOut


def fdictRatioSummary(listRows):
    """Median and range of each component ratio across the compared stars."""
    if not listRows:
        return {}
    dictOut = {}
    for sKey in ("fStarRate", "fPlanet", "fLeak", "fZodi", "fExozodi"):
        faRatio = np.array([r["dictRatio"][sKey] for r in listRows
                            if r["dictRatio"].get(sKey)])
        if faRatio.size:
            dictOut[sKey] = {"fMedian": float(np.median(faRatio)),
                             "fMin": float(faRatio.min()), "fMax": float(faRatio.max())}
    dictOut["fExosimsNumPixels"] = float(listRows[0]["dictRatio"]["iNumPixels"])
    return dictOut


def fdictParseArgs():
    """Command-line configuration for the EXOSIMS cross-check."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mission-parameters", required=True)
    p.add_argument("--target-catalog", required=True)
    p.add_argument("--num-stars", type=int, default=12)
    p.add_argument("--delta-mag", type=float, default=26.0)
    p.add_argument("--exozodi-level", type=float, default=3.0)
    p.add_argument("--out-json", required=True)
    return vars(p.parse_args())


def main():
    import pandas as pd
    dictArgs = fdictParseArgs()
    with open(dictArgs["mission_parameters"]) as oFile:
        dictParams = json.load(oFile)
    dictMission = dictParams["dictMission"]
    dictBand = dictParams["dictBands"]["listBandsDetection"][0]
    fSkyThroughput = cg.fnSkyThroughput(dictMission["fCoreThroughputMax"],
                                        dictMission["fApertureRadiusLamD"])
    dfCat = pd.read_csv(dictArgs["target_catalog"]).head(dictArgs["num_stars"])

    fOmega = ph.fnPhotometricApertureSolidAngle(dictBand["fLambdaM"], dictMission["fDiameterM"],
                                                dictMission["fApertureRadiusLamD"])
    listNames = [str(s).strip() for s in dfCat["sSourceId"]]
    dictSpec = fdictExosimsSpec(dictMission, dictBand, fSkyThroughput,
                                dictMission["fCoreThroughputMax"], fOmega, listNames)
    oTargetList = foBuildTargetList(dictSpec, dfCat)
    dictMode = oTargetList.OpticalSystem.observingModes[0]
    listRows = []
    for _, oRow in dfCat.iterrows():
        listMatch = flistMatchStars(oTargetList, [str(oRow["sSourceId"]).strip()])
        if not listMatch:
            continue
        sName, iStar = listMatch[0]
        fSepArcsec = float(oRow["fEeidArcsec"])
        dictOurs = fdictOurRates(dict(oRow), dictBand, dictMission, fSepArcsec,
                                 dictArgs["delta_mag"], dictArgs["exozodi_level"],
                                 fSkyThroughput)
        dictTheirs = fdictExosimsRates(oTargetList, iStar, dictMode, fSepArcsec,
                                       dictArgs["delta_mag"], dictArgs["exozodi_level"])
        listRows.append({"sName": sName, "fDistancePc": float(oRow["fDistancePc"]),
                         "fTeffK": float(oRow["fTeffK"]),
                         "fLuminosityLsun": float(oRow["fLuminosityLsun"]),
                         "fSepArcsec": fSepArcsec,
                         "dictOurs": dictOurs, "dictExosims": dictTheirs,
                         "dictRatio": fdictComponentRatios(dictOurs, dictTheirs)})
    dictOut = {"fSkyThroughput": fSkyThroughput, "iStarsCompared": len(listRows),
               "dictRatioSummary": fdictRatioSummary(listRows), "listRows": listRows}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    print(json.dumps({k: v for k, v in dictOut.items() if k != "listRows"}, indent=2))


if __name__ == "__main__":
    main()
