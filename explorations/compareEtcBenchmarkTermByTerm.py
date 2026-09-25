#!/usr/bin/env python3
"""Compare this model's exposure-time constants and functional forms with AYO's, term by term.

Stark et al. (2025, arXiv:2502.18556) published AYO's full exposure-time chain for five fiducial
Earth twins (reference/ETC_cal_char.xlsx, reference/ETC_cal_detect.xlsx): inputs, astrophysical
fluxes, coronagraph terms, every count rate and the final times. That benchmark is for the USORT
telescope (7.87 m circumscribed, 6.5 m inscribed, optical vortex coronagraph), not Stark et al.
(2024)'s 6 m EAC, so constants that belong to the mission are expected to differ. This script
separates the two questions:

1. FORMS. yieldlib's own count-rate and exposure-time code is run on the benchmark's mission
   constants and its coronagraph values at the planet's separation, so any residual comes from
   the model's astrophysical flux models or its equations, not from the mission.
2. CONSTANTS. The same Earth twins are run through the model's Stark (2024) configuration
   (modelCoronagraph/missionParameters.json) and each term is set beside AYO's.

It also checks AYO's own internal forms (T_sky, CIC, noise floor, overheads) by reconstructing its
count rates from its inputs, which is how the units of "skytrans" were pinned down.
"""

import argparse
import copy
import csv
import json
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yieldlib import completeness as cp  # noqa: E402
from yieldlib import coronagraph as cg  # noqa: E402
from yieldlib import physics as ph  # noqa: E402
from readEtcBenchmarkSpreadsheet import S_NS, S_REL, flistSharedStrings, fdictSheetRows  # noqa: E402

DICT_SCENARIOS = {
    "char800": ("char", 4, 800e-9), "char1000": ("char", 10, 1000e-9),
    "det500": ("detect", 4, 500e-9), "det1000": ("detect", 10, 1000e-9)}
F_USORT_CIRC_M = 7.87
F_USORT_INSCRIBED_M = 6.5
F_TSKY_SUBSAMPLES = 16.0  # skytrans / 16 reproduces CR_bz and CR_bez; see fdictAyoInternal
F_TO_CGS_FLUX = 1e-4 * 1e-9  # photons s^-1 m^-2 m^-1 -> photons s^-1 cm^-2 nm^-1


def fdictReadWorkbook(sPath, iColumn):
    """{HIP name: {parameter: value}} for one code column of one ETC workbook."""
    oZip = zipfile.ZipFile(sPath)
    listShared = flistSharedStrings(oZip)
    dictTargets = {r.get("Id"): r.get("Target").replace("/xl/", "")
                   for r in ET.fromstring(oZip.read("xl/_rels/workbook.xml.rels"))}
    dictOut = {}
    for oSheet in ET.fromstring(oZip.read("xl/workbook.xml")).find(f"{{{S_NS}}}sheets"):
        if "HIP" not in oSheet.get("name"):
            continue
        dictRows = fdictSheetRows(oZip, dictTargets[oSheet.get(f"{{{S_REL}}}id")], listShared)
        dictOut[oSheet.get("name").split(" (")[0]] = fdictColumn(dictRows, iColumn)
    return dictOut


def fdictColumn(dictRows, iColumn):
    """Named numeric parameters in one column of a sheet."""
    dictOut = {}
    for dictRow in dictRows.values():
        sName = dictRow.get(1, "")
        try:
            dictOut[sName] = float(dictRow[iColumn])
        except (KeyError, ValueError):
            pass
    return dictOut


def fdictReadCatalog(sPath, listHip):
    """Catalog rows for the benchmark stars, keyed 'HIP n'."""
    dictOut = {}
    with open(sPath) as oFile:
        for dictRow in csv.DictReader(oFile):
            sHip = dictRow["sHipName"].split(".")[0]
            if f"HIP {sHip}" in listHip:
                dictOut[f"HIP {sHip}"] = {k: float(dictRow[k]) for k in
                                          ("fTeffK", "fRadiusRsun", "fLuminosityLsun",
                                           "fDistancePc", "fEeidAu", "fEclipticLatDeg")}
    return dictOut


def ffLamDArcsec(fLambdaM, fDiameterM):
    """lambda/D in arcsec."""
    return cg.fnLambdaOverDArcsec(fLambdaM, fDiameterM)


def ffThroughput(dictA):
    """AYO's T: optical x QE x dQE (contamination is inside T_optical in the sheet)."""
    return dictA["T_optical"] * dictA["QE"] * dictA["dQE"]


def ffPerSkyArea(dictA, sFlux, fCountRate):
    """Omega * T_sky (arcsec^2) implied by a sky count rate and its surface brightness."""
    return fCountRate / (dictA[sFlux] * dictA["A"] * ffThroughput(dictA) * dictA["Δλ"])


def fdictAyoInternal(dictA, fLambdaM):
    """AYO's own forms, reconstructed from its inputs and compared with its count rates."""
    fLamD2 = ffLamDArcsec(fLambdaM, F_USORT_CIRC_M) ** 2
    fOmegaTsky = ffPerSkyArea(dictA, "F_zodi", dictA["CR_bz"]) / fLamD2
    fCrSat = 1.0 / (ph.F_GEIGER_CIC_FACTOR * dictA["t_photon_count"])
    fSum = sum(dictA[k] for k in ("CR_bs", "CR_bz", "CR_bez", "CR_bstray", "CR_bd"))
    fDenom = dictA["CR_p"] ** 2 - dictA["SNR"] ** 2 * dictA["CR_NF"] ** 2
    return dict(
        fTskyImplied=fOmegaTsky / dictA["Ω_core"],
        fTskyFromSkytrans=dictA["skytrans"] / F_TSKY_SUBSAMPLES,
        fExozodiOmegaTskyRatio=ffPerSkyArea(dictA, "F_exozodi", dictA["CR_bez"]) /
        fLamD2 / fOmegaTsky,
        fTskyOverTcore=fOmegaTsky / dictA["Ω_core"] / dictA["T_core"],
        fCrSatOverMeanPixel=fCrSat / ((fSum - dictA["CR_bd"] + dictA["CR_p"]) /
                                      dictA["det_npix"]),
        fCrBdCheck=dictA["det_npix"] * (dictA["det_DC"] + dictA["det_CIC"] /
                                        dictA["t_photon_count"]) / dictA["CR_bd"],
        fNoiseFloorOverLeak=dictA["CR_NF"] / dictA["CR_bs"],
        fNoiseFloorTimePenalty=dictA["CR_p"] ** 2 / fDenom,
        fTscienceCheck=dictA["SNR"] ** 2 * (dictA["CR_p"] + 2 * fSum) / fDenom /
        dictA["t_science"],
        fOverheadStaticS=dictA["t_exp"] - dictA["t_overhead,dynamic"] * dictA["t_science"])


def fdictEarthTwin(dictStar):
    """One Earth twin at quadrature at the EEID, in the model's planet/geometry dicts."""
    fEeid = np.sqrt(dictStar["fLuminosityLsun"])
    dictPlanets = {"faRadiusEarth": np.array([1.0]), "faAxisAu": np.array([fEeid])}
    dictGeom = {"faSepAu": np.array([[fEeid]]), "faPhase": np.array([[1.0 / np.pi]])}
    return dictPlanets, dictGeom


def fdictFlatCoronagraph(fUpsilon, fContrast):
    """A coronagraph table flat in separation, holding one (Upsilon, zeta) pair."""
    return {"faSeparationUpsilon": [1e-3, 1e3], "faUpsilon": [fUpsilon, fUpsilon],
            "faSeparationContrast": [1e-3, 1e3], "faContrast": [fContrast, fContrast],
            "fInnerEdgeLamD": 1e-3, "fOuterEdgeLamD": 1e3, "fInnerLogSlope": 0.0}


def fdictUsortMission(dictBase, dictA, fLambdaM):
    """The model's mission dict re-parameterized to the benchmark telescope and coronagraph."""
    dictM = copy.deepcopy(dictBase)
    fLeak = dictA["I_star"] * dictA["Ω_core"] / dictA["T_core"]
    dictM.update(fDiameterM=F_USORT_INSCRIBED_M, fApertureAreaM2=dictA["A"] * 1e-4,
                 fCircumscribedRatio=F_USORT_CIRC_M / F_USORT_INSCRIBED_M,
                 dictCoronagraphTable=fdictFlatCoronagraph(dictA["T_core"], fLeak),
                 fContrastFloor=0.0, fThroughputCalibration=1.0, fContaminationThroughput=1.0,
                 fQuantumEfficiency=dictA["QE"], fDetectiveQuantumEfficiency=dictA["dQE"],
                 fSkyThroughput=fdictAyoInternal(dictA, fLambdaM)["fTskyImplied"],
                 bSkyThroughputFollowsCore=False, fNoiseFloorDeltaMag=99.0,
                 fExozodiLevel=dictA["nzodis"], fExozodiRadialIndex=0.0)
    return dictM


def fdictBandFor(dictA, fLambdaM):
    """The model's band dict carrying the benchmark's throughput, bandwidth, S/N and pixels."""
    return {"sName": "ETC", "fLambdaM": fLambdaM, "fOpticalThroughput": dictA["T_optical"],
            "fBandwidthFraction": dictA["Δλ"] * 1e-9 / fLambdaM,
            "fSignalToNoise": dictA["SNR"], "iNumPixels": dictA["det_npix"],
            "bApplyThroughputCalibration": False}


def fdictModelRates(dictStar, dictBand, dictMission):
    """The model's count rates and science time for one Earth twin, as scalars."""
    dictPlanets, dictGeom = fdictEarthTwin(dictStar)
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    fTau = cp.faRequiredExposureTime([dictRates], [dictBand], dictBand["fSignalToNoise"],
                                     dictMission)
    faBrightest = (dictRates["faLeak"] + dictRates["fZodi"] + dictRates["fExozodi"] +
                   dictRates["faPlanet"]) / dictBand["iNumPixels"]
    fDet = ph.faDetectorCountRate(faBrightest, dictBand["iNumPixels"], dictMission["fDarkCurrent"],
                                  dictMission["fReadNoise"], None,
                                  dictMission["fClockInducedCharge"])
    return dict(CR_p=float(np.ravel(dictRates["faPlanet"])[0]),
                CR_bs=float(np.ravel(dictRates["faLeak"])[0]),
                CR_bz=float(np.ravel(dictRates["fZodi"])[0]),
                CR_bez=float(np.ravel(dictRates["fExozodi"])[0]),
                CR_bd=float(np.ravel(fDet)[0]), t_science=float(np.ravel(fTau)[0]),
                fSepLamDCirc=float(np.ravel(dictRates["faSepLamD"])[0] *
                                   dictMission.get("fCircumscribedRatio", 1.0)),
                fUpsilon=float(np.ravel(dictRates["faUpsilon"])[0]))


def fdictAstroTerms(dictStar, dictA, fLambdaM, dictMission):
    """The model's astrophysical fluxes in AYO's units, as ratios model/AYO."""
    fStar = float(ph.faStellarPhotonFlux(fLambdaM, dictStar["fTeffK"], dictStar["fRadiusRsun"],
                                         dictStar["fDistancePc"])) * F_TO_CGS_FLUX
    fZero = ph.fnZeroMagPhotonFlux(fLambdaM) * 1e-7  # per um per m^2 -> per nm per cm^2
    fZodi = fZero * 10 ** (-0.4 * dictMission["fZodiMagArcsec2"])
    fExo = dictA["nzodis"] * fZero * 10 ** (-0.4 * dictMission["fExozodiMagArcsec2"]) * \
        ph.fnExozodiSurfaceBrightnessScale(fLambdaM, dictStar["fTeffK"],
                                           dictStar["fRadiusRsun"], dictStar["fLuminosityLsun"])
    fContrast = dictMission["fGeometricAlbedo"] / np.pi * \
        (cp.F_REARTH_AU / np.sqrt(dictStar["fLuminosityLsun"])) ** 2
    fZodiLat = ph.fnZodiPhotonSurfaceBrightness(fLambdaM, dictStar["fEclipticLatDeg"]) * 1e-7
    return dict(F_star=fStar / dictA["F_star"], F_zodi=fZodi / dictA["F_zodi"],
                F_zodi_stark2014=fZodiLat / dictA["F_zodi"],
                F_exozodi=fExo / dictA["F_exozodi"],
                F_exozodi_over_F_star=(fExo / fStar) / (dictA["F_exozodi"] / dictA["F_star"]),
                planet_contrast=fContrast / (dictA["F_p"] / dictA["F_star"]))


def fdictRatios(dictModel, dictA, listKeys):
    """model / AYO for each named count rate or time."""
    return {k: dictModel[k] / dictA[k] if dictA.get(k) else None for k in listKeys}


def fdictModelOwnSetup(dictStar, sScenario, dictParams):
    """The model's Stark (2024) configuration for the same Earth twin and wavelength."""
    dictMission = dictParams["dictMission"]
    if sScenario == "char1000":
        dictBand = dictParams["dictBands"]["dictBandCharacterization"]
        return fdictModelRates(dictStar, dictBand, dictMission), dictBand
    if sScenario == "char800":
        dictBand = [b for b in dictParams["dictBands"]["listBandsCharacterization"]
                    if b["sName"] == "IFS824"][0]
        return fdictModelRates(dictStar, dictBand, dictMission), dictBand
    return None, None


def fdictModelDetection(dictStar, dictParams):
    """The model's two-channel detection time for the Earth twin (Stark 2024: 450 + 550 nm)."""
    dictMission = dictParams["dictMission"]
    listBands = dictParams["dictBands"]["listBandsDetection"]
    dictPlanets, dictGeom = fdictEarthTwin(dictStar)
    listRates = [cp.fdictCountRates(dictStar, dictPlanets, dictGeom, b, dictMission)
                 for b in listBands]
    fTau = cp.faRequiredExposureTime(listRates, listBands, listBands[0]["fSignalToNoise"],
                                     dictMission)
    return float(np.ravel(fTau)[0])


def ffAiryPeakOmegaOverCore(fApertureRadiusLamD):
    """PSF_peak * Omega / Upsilon_c for an Airy core: AYO's leak factor over the model's.

    Stark et al. (2019) Eq. 4 writes the leak as zeta * PSF_peak * Omega (their I/theta^2 replaces
    Stark 2014's "zeta PSF_peak"); the model writes zeta * Upsilon_c. For an unobscured Airy
    pattern the peak is pi/4 of the total per (lambda/D)^2, so the ratio is
    (pi/4)(pi X^2) / EE(X) = 1.78 at X = 0.7. A real off-axis PSF near the IWA differs.
    """
    fPeakOmega = (np.pi / 4.0) * np.pi * fApertureRadiusLamD ** 2
    return fPeakOmega / cg.fnAiryEncircledEnergy(fApertureRadiusLamD)


def fdictLeverTimes(dictStar, dictBand, dictMission, fTskyFactor):
    """Char time of the model's own setup with the peak-normalized leak and/or AYO-shaped T_sky."""
    dictPlanets, dictGeom = fdictEarthTwin(dictStar)
    dictRates = cp.fdictCountRates(dictStar, dictPlanets, dictGeom, dictBand, dictMission)
    fPeak = ffAiryPeakOmegaOverCore(dictMission["fApertureRadiusLamD"])
    dictOut = {}
    for sName, fLeak, fSky in (("base", 1.0, 1.0), ("peakLeak", fPeak, 1.0),
                               ("tskyAyoShape", 1.0, fTskyFactor), ("both", fPeak, fTskyFactor)):
        dictR = dict(dictRates, faLeak=dictRates["faLeak"] * fLeak,
                     fZodi=dictRates["fZodi"] * fSky, fExozodi=dictRates["fExozodi"] * fSky)
        dictOut[sName] = float(np.ravel(cp.faRequiredExposureTime(
            [dictR], [dictBand], dictBand["fSignalToNoise"], dictMission))[0])
    dictOut["fLeakShareOfAstro"] = float(np.ravel(dictRates["faLeak"] / (
        dictRates["faLeak"] + dictRates["fZodi"] + dictRates["fExozodi"]))[0])
    return dictOut


def fdictModelOwnTerms(dictStar, dictA, sScenario, fLambdaM, dictParams):
    """Each mission constant and count rate of the model's own setup, beside AYO's."""
    dictModel, dictBand = fdictModelOwnSetup(dictStar, sScenario, dictParams)
    if dictModel is None:
        return {"t_science_detection_2ch": fdictModelDetection(dictStar, dictParams),
                "t_science_ayo": dictA["t_science"]}
    dictMission = dictParams["dictMission"]
    fT = dictBand["fOpticalThroughput"] * dictMission["fContaminationThroughput"] * \
        dictMission["fQuantumEfficiency"] * dictMission["fDetectiveQuantumEfficiency"] * \
        dictMission["fThroughputCalibration"]
    fLamDc = ffLamDArcsec(fLambdaM, dictMission["fDiameterM"] * dictMission["fCircumscribedRatio"])
    fOmega = ph.fnPhotometricApertureSolidAngle(fLambdaM, dictMission["fDiameterM"],
                                                dictMission["fApertureRadiusLamD"])
    fTsky = float(np.ravel(cp.faSkyThroughputAt(np.array([dictModel["fUpsilon"]]),
                                                dictMission))[0])
    fTskyFactor = fdictAyoInternal(dictA, fLambdaM)["fTskyOverTcore"] / (fTsky / dictModel["fUpsilon"])
    return dict(dictModel, levers=fdictLeverTimes(dictStar, dictBand, dictMission, fTskyFactor),
                fTskyFactorApplied=fTskyFactor, T_total=fT, T_total_ayo=ffThroughput(dictA),
                fLambdaNm=dictBand["fLambdaM"] * 1e9, SNR=dictBand["fSignalToNoise"],
                det_npix=dictBand["iNumPixels"], det_npix_ayo=dictA["det_npix"],
                fOmegaLamDCirc2=fOmega / fLamDc ** 2, fTsky=fTsky,
                fTskyOverUpsilon=fTsky / dictModel["fUpsilon"],
                fAreaM2=cp.fnCollectingAreaM2(dictMission),
                ratios=fdictRatios(dictModel, dictA, ["CR_p", "CR_bs", "CR_bz", "CR_bez",
                                                      "CR_bd", "t_science"]))


def fdictStarMatchedFlux(dictStar, fFluxRatio):
    """The star with its radius rescaled so the blackbody gives AYO's flux at this wavelength.

    Exozodi scales with the star's flux at fixed luminosity in both codes, so this also carries
    AYO's exozodi; it isolates the equations from the input photometry.
    """
    return dict(dictStar, fRadiusRsun=dictStar["fRadiusRsun"] / np.sqrt(fFluxRatio))


def fdictFormsResult(dictStar, dictA, fLambdaM, dictUsort):
    """The model's rates on the benchmark inputs, with ratios to AYO's."""
    dictForms = fdictModelRates(dictStar, fdictBandFor(dictA, fLambdaM), dictUsort)
    return dict(dictForms, fSepRatio=dictForms["fSepLamDCirc"] / dictA["sp"],
                ratios=fdictRatios(dictForms, dictA, ["CR_p", "CR_bs", "CR_bz", "CR_bez",
                                                      "CR_bd", "t_science"]))


def fdictOneScenario(dictStar, dictA, sScenario, fLambdaM, dictParams):
    """Internal AYO checks, the model's forms on AYO inputs, and the model's own constants."""
    dictMission = dictParams["dictMission"]
    dictUsort = fdictUsortMission(dictMission, dictA, fLambdaM)
    dictAstro = fdictAstroTerms(dictStar, dictA, fLambdaM, dictMission)
    dictMatched = fdictStarMatchedFlux(dictStar, dictAstro["F_star"])
    return dict(
        ayo=dictA, ayoInternal=fdictAyoInternal(dictA, fLambdaM),
        astroModelOverAyo=dictAstro,
        formsOnAyoInputs=fdictFormsResult(dictStar, dictA, fLambdaM, dictUsort),
        formsOnAyoInputsMatchedStar=fdictFormsResult(dictMatched, dictA, fLambdaM, dictUsort),
        modelOwnSetup=fdictModelOwnTerms(dictStar, dictA, sScenario, fLambdaM, dictParams))


def fnPrintSummary(dictOut):
    """One line per star and scenario: the ratios that carry the comparison."""
    for sScenario, dictStars in dictOut.items():
        print(f"\n== {sScenario}")
        for sStar, d in dictStars.items():
            dI, dAs = d["ayoInternal"], d["astroModelOverAyo"]
            dF = d["formsOnAyoInputsMatchedStar"]["ratios"]
            print(f"{sStar:<10} Tsky/Tcore={dI['fTskyOverTcore']:.2f} "
                  f"Fstar={dAs['F_star']:.3f} Fzodi={dAs['F_zodi']:.3f} "
                  f"Fexo={dAs['F_exozodi']:.3f} Cp={dAs['planet_contrast']:.3f} | forms: "
                  + " ".join(f"{k}={v:.3f}" for k, v in dF.items() if v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--char-xlsx", default="reference/ETC_cal_char.xlsx")
    p.add_argument("--detect-xlsx", default="reference/ETC_cal_detect.xlsx")
    p.add_argument("--catalog", default="../buildTargetCatalog/targetCatalog.csv")
    p.add_argument("--mission", default="../modelCoronagraph/missionParameters.json")
    p.add_argument("--out-json", default="output/etcBenchmarkTermByTerm.json")
    dictArgs = vars(p.parse_args())
    dictParams = json.load(open(dictArgs["mission"]))
    dictBooks = {"char": dictArgs["char_xlsx"], "detect": dictArgs["detect_xlsx"]}
    dictOut = {}
    for sScenario, (sBook, iCol, fLambdaM) in DICT_SCENARIOS.items():
        dictSheets = fdictReadWorkbook(dictBooks[sBook], iCol)
        dictCatalog = fdictReadCatalog(dictArgs["catalog"], list(dictSheets))
        dictOut[sScenario] = {s: fdictOneScenario(dictCatalog[s], dictSheets[s], sScenario,
                                                  fLambdaM, dictParams) for s in dictSheets}
    with open(dictArgs["out_json"], "w") as oFile:
        json.dump(dictOut, oFile, indent=1, default=float)
    fnPrintSummary(dictOut)


if __name__ == "__main__":
    main()
