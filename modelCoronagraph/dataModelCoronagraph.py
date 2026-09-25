#!/usr/bin/env python3
"""Emit the baseline HWO mission parameter set and the parametric DMVC coronagraph curves.

Every value here is transcribed from Stark et al. (2024, JATIS 10, 034006) Tables 1 and 2,
except the coronagraph core-throughput and contrast profiles. Those are outputs of detailed
coronagraph simulations that the papers do not distribute, so they are reconstructed from the
three published anchors for the DMVC: IWA 3.5 lambda/D, ~45 percent core throughput at wide
separation, and useful throughput down to ~1.5 lambda/D, with contrast floored at 1e-10.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from yieldlib import coronagraph as cg  # noqa: E402

F_YEAR_S = 365.25 * 86400.0
F_HOUR_S = 3600.0


S_CORONAGRAPH_REFERENCE = "reference/starkCoronagraphDmvc6.json"
S_EXOZODI_REFERENCE = "reference/starkExozodiHostsMaxLikelihood.json"
F_OFF_AXIS_ZODI = 1000.0
DICT_PINNED_EXOZODI = {
    "16537": {"sName": "eps Eri", "fZodi": 297.0}, "70497": {"sName": "tet Boo", "fZodi": 148.0},
    "84862": {"sName": "72 Her", "fZodi": 588.0}, "92043": {"sName": "110 Her", "fZodi": 235.0},
}
F_CIRCUMSCRIBED_RATIO = 8.0 / 6.7
F_APERTURE_FILL_FACTOR = 0.785


def fdictCoronagraphTableFromReference():
    """The DMVC6 core-throughput and raw-contrast curves digitized from Stark et al. (2024) Fig. 12.

    These replace the parametric stand-in that four hand-fitted constants used to define. The
    parametric form was fitted to points read off the figure by eye under a 0.75 relative
    tolerance, which is loose enough to accept a curve of the wrong SHAPE, and it was: optimistic
    on contrast by a factor of 5 at 2 lambda/D and by a factor of 2 from 4 to 10 lambda/D, and
    high on core throughput by 22 percent at 20 lambda/D. Returns None if the reference file is
    absent, in which case the parametric curves are used and the provenance says so.
    """
    sPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), S_CORONAGRAPH_REFERENCE)
    if not os.path.exists(sPath):
        return None
    with open(sPath) as oFile:
        return cg.fdictCoronagraphTable(json.load(oFile))


def fdictExozodiDistributionFromReference():
    """The LBTI HOSTS maximum-likelihood exozodi distribution digitized from Stark+2024 Fig. 9.

    Binned probabilities to 1000 zodis, renormalized to unit mass. The digitized histogram sums
    to 0.948 of the 10k x 500 draws the caption implies, but that shortfall is not mass beyond
    the axis: the last bins before 1000 zodis sit near the 10^2 plotting floor, and a 0.02 dex
    error in the log-count calibration alone accounts for 5 percent. Renormalizing reproduces the
    three-zodi median Stark quotes (2.98); putting the shortfall at 1000 zodis would move it to
    3.4. F_OFF_AXIS_ZODI therefore receives no mass here. Pinned stars (Stark Sec. 3.3: eps Eri 297, tet Boo 148, 72 Her 588 and
    110 Her 235 zodis) are listed separately in dictPinnedExozodi, keyed by HIP number.
    """
    sPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), S_EXOZODI_REFERENCE)
    with open(sPath) as oFile:
        dictRef = json.load(oFile)
    faProb = np.asarray(dictRef["faProbability"], dtype=float)
    return {"faEdgesZodi": dictRef["faEdgesZodi"], "faProbability": (faProb / faProb.sum()).tolist(),
            "fOffAxisZodi": F_OFF_AXIS_ZODI, "fMedianZodi": dictRef["dictSummary"]["fMedianZodi"],
            "sSource": dictRef["sSource"] + ", digitized by "
                       "explorations/digitiseStarkExozodiDistributionFigure.py"}


def fdictMissionParameters(fDiameterM, fExozodiLevel):
    """Baseline coronagraph-mission parameters, Stark et al. (2024) Tables 1 and 2.

    fDiameterM is the INSCRIBED diameter, which is how Stark et al. (2024) label every scenario
    ("6 m ID"). Two quantities do not follow that diameter, and treating them as if they did was
    the single largest error in this pipeline:

      * The published coronagraph curves are plotted against the CIRCUMSCRIBED diameter (Ref.
        stark2024 Sec. 6.1, which notes the DMVC6's apparent 3.5 lambda/D IWA is inflated for
        precisely this reason). Ref. stark2019 Sec. 6.2 adds that normalised to the inscribed
        pupil the DMVC and the monolithic vortex "would look nearly identical", and the monolithic
        vortex sits at ~3 lambda/D -- the same 1.19 ratio. Evaluating those curves at inscribed
        lambda/D put this pipeline's inner working angle 19% too far out in ANGLE.
      * The collecting area A is the full obscured primary, because Upsilon_c is normalised to the
        light entering the coronagraph from that whole aperture including the region outside the
        inscribed diameter (Ref. stark2019 Sec. 6.2). Using a circle of the inscribed diameter
        charged the Lyot stop's discard twice.

    F_APERTURE_FILL_FACTOR is the one number here that neither paper states: the fraction of the
    circumscribed circle a hex-segmented primary actually collects. 0.785 corresponds to the
    ~39.5 m^2 usually quoted for LUVOIR-B's 8 m circumscribed aperture. Yield goes as roughly the
    0.37 power of collecting area, verified in explorations/scanApertureGeometryConventions.py, so
    the plausible range 0.70-0.88 moves the yield by about +/-4%. It is an assumption, not a
    published value, and is the largest single unsourced input in this model.

    iRequiredDetections is 1. Two was tested, on the grounds that Stark budgets characterization
    only after orbit determination and cites Bruna et al. (2023) for two reflected-light
    detections sufficing: it lowers the yield from 20.4 to 18.8 without steepening C(tau) or
    strengthening the albedo penalty, so it does not explain the shortfall and Stark et al.
    (2024) drop the visit mandate in any case. The option is kept so the test can be repeated.

    bSkyThroughputFollowsCore makes the extended-source throughput T_sky, which multiplies the
    zodiacal and exozodiacal backgrounds, a function of separation as Stark et al. (2019) Eqs. 5-6
    define it, instead of its large-separation value everywhere (yieldlib.completeness.
    faSkyThroughputAt). Adopted 2026-09-24: with it the uncalibrated 6 m yield is 22.6 against
    the published 22.5, so the throughput calibration A03 fits falls from 1.62 to 0.99. The
    constant form had been charging planets near the inner working angle for background the
    coronagraph mask removes, and the calibration factor was absorbing it
    (explorations/whatIfCharacterizationTreatment.py, variant skyFollowsCore).
    """
    return {
        "fDiameterM": fDiameterM,
        "fCircumscribedRatio": F_CIRCUMSCRIBED_RATIO,
        "fApertureFillFactor": F_APERTURE_FILL_FACTOR,
        "fIwaLamD": cg.F_DEFAULT_IWA_LAMD,
        "fOwaLamD": cg.F_DEFAULT_OWA_LAMD,
        "fCoreThroughputMax": cg.F_DEFAULT_CORE_THROUGHPUT_MAX,
        "dictCoronagraphTable": fdictCoronagraphTableFromReference(),
        "fContrastFloor": cg.F_DEFAULT_CONTRAST_FLOOR,
        "fNoiseFloorDeltaMag": 26.5,
        "fContaminationThroughput": 0.95,
        "fApertureRadiusLamD": 0.7,
        "fGeometricAlbedo": 0.2,
        "fZodiMagArcsec2": 23.0,
        "fExozodiMagArcsec2": 22.0,
        "fExozodiLevel": fExozodiLevel,
        "fDarkCurrent": 3.0e-5,
        "fReadNoise": 0.0,
        "fClockInducedCharge": 1.3e-3,
        "fQuantumEfficiency": 0.9,
        "fDetectiveQuantumEfficiency": 0.75,
        "fTotalScienceTimeS": 2.0 * F_YEAR_S,
        "fSlewOverheadS": 1.0 * F_HOUR_S,
        "fWavefrontOverheadS": 2.7 * F_HOUR_S,
        "fWavefrontMultiplier": 1.1,
        "fExposureLimitS": 60.0 * 86400.0,
        "fThroughputCalibration": 1.0,
        "iMaxVisits": 6,
        "iRequiredDetections": 1,
        "bSkyThroughputFollowsCore": True,
        "sZodiModel": "stark2014",
        "sNoiseFloorModel": "denominator",
        "fNoiseFloorSignalToNoise": 7.0,
        "iNoiseFloorChannels": 2,
        "bTimeLimitIncludesOverheads": True,
        "dictExozodiDistribution": fdictExozodiDistributionFromReference(),
        "dictPinnedExozodi": DICT_PINNED_EXOZODI,
        "dictAlbedoDistribution": {"fMin": 0.08, "fMax": 0.32,
                                   "sSource": "Stark et al. 2024 Sec. 3.2: the adopted uniform "
                                              "distribution, mean 0.20, quoted as reducing the "
                                              "expected yield by about 12 percent"},
    }


def fdictBandParameters():
    """Detection channels (parallel SW and LW) and the characterization channel."""
    return {
        "listBandsDetection": [
            {"sName": "LW", "fLambdaM": 550e-9, "fOpticalThroughput": 0.34,
             "fBandwidthFraction": 0.20, "fSignalToNoise": 7.0, "iNumPixels": 4},
            {"sName": "SW", "fLambdaM": 450e-9, "fOpticalThroughput": 0.15,
             "fBandwidthFraction": 0.20, "fSignalToNoise": 7.0, "iNumPixels": 4},
        ],
        "dictBandCharacterization": {
            "sName": "IFS", "fLambdaM": 1000e-9, "fOpticalThroughput": 0.23,
            "fBandwidthFraction": 1.0 / 140.0, "fSignalToNoise": 5.0, "iNumPixels": 96},
        "listBandsCharacterization": flistCharacterizationOptions(),
    }


def flistCharacterizationOptions():
    """Bandpass options AYO chooses among for a water-vapour characterization.

    Digitized from the 20 percent bandwidth curve of Stark et al. (2024) Fig. 3: the continuum
    S/N needed for a strong H2O detection against the long-wavelength edge of the bandpass. A
    shorter edge needs more S/N but collects more stellar photons and sits at a smaller lambda/D,
    so the cheapest option differs star by star. Modelling only the 1000 nm option -- as an
    earlier version did -- makes characterization far too expensive for hard targets, which then
    fail the two-month cap and stop counting toward the yield.

    Pixel count scales with the diffraction-limited lenslet area, and the IFS throughput is held
    at its 1000 nm value for want of a published wavelength dependence.
    """
    listEdges = [(730e-9, 17.0), (763e-9, 15.0), (779e-9, 14.0), (824e-9, 13.0),
                 (842e-9, 12.0), (917e-9, 10.0), (930e-9, 9.0), (943e-9, 6.0),
                 (1000e-9, 5.0)]
    return [{"sName": f"IFS{int(f * 1e9)}", "fLambdaM": f, "fOpticalThroughput": 0.23,
             "fBandwidthFraction": 1.0 / 140.0, "fSignalToNoise": fSnr,
             "iNumPixels": max(int(round(96 * (f / 1000e-9) ** 2)), 4)}
            for f, fSnr in listEdges]


def fdictSelectionBoxes():
    """The canonical HabEx/LUVOIR EEC box and the redefined mass-selected box."""
    return {
        "canonical": {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
                      "fHzInnerAu": 0.95, "fHzOuterAu": 1.67},
        "redefined": {"sRadiusMode": "mass", "fMassLoEarth": 0.5, "fMassHiEarth": 2.0,
                      "fMassRadiusExponent": 0.27, "fHzInnerAu": 0.96, "fHzOuterAu": 1.20},
        "hzOnly": {"sRadiusMode": "canonical", "fRadiusMaxEarth": 1.4,
                   "fHzInnerAu": 0.96, "fHzOuterAu": 1.20},
    }


def fdfCoronagraphCurves(dictMission, iNumPoints):
    """Tabulate core throughput and raw contrast against separation in lambda/D."""
    faSep = np.logspace(np.log10(0.5), np.log10(dictMission["fOwaLamD"] * 1.2), iNumPoints)
    return pd.DataFrame({
        "fSeparationLamD": faSep,
        "fCoreThroughput": cg.faCoreThroughput(faSep, fIwaLamD=dictMission["fIwaLamD"],
                                               fOwaLamD=dictMission["fOwaLamD"],
                                               fThroughputMax=dictMission["fCoreThroughputMax"]),
        "fRawContrast": cg.faRawContrast(faSep, fContrastFloor=dictMission["fContrastFloor"],
                                         fOwaLamD=dictMission["fOwaLamD"]),
    })


def fdictParseArgs():
    """Command-line configuration for the mission and coronagraph model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diameter-m", type=float, default=6.0)
    p.add_argument("--exozodi-level", type=float, default=3.0)
    p.add_argument("--num-points", type=int, default=200)
    p.add_argument("--out-parameters", default="missionParameters.json")
    p.add_argument("--out-curves", default="coronagraphCurves.csv")
    return vars(p.parse_args())


def main():
    dictArgs = fdictParseArgs()
    dictMission = fdictMissionParameters(dictArgs["diameter_m"], dictArgs["exozodi_level"])
    dictOut = {"dictMission": dictMission, "dictBands": fdictBandParameters(),
               "dictBoxes": fdictSelectionBoxes(),
               "sProvenance": "Stark et al. 2024 JATIS 10 034006, Tables 1 and 2; coronagraph "
                              "profiles parametrized from published DMVC anchors."}
    with open(dictArgs["out_parameters"], "w") as oFile:
        json.dump(dictOut, oFile, indent=2)
    fdfCoronagraphCurves(dictMission, dictArgs["num_points"]).to_csv(
        dictArgs["out_curves"], index=False)
    print(json.dumps({"sWrote": dictArgs["out_parameters"],
                      "fIwaThroughput": float(cg.faCoreThroughput(dictMission["fIwaLamD"])),
                      "fPlateauThroughput": float(cg.faCoreThroughput(25.0))}, indent=2))


if __name__ == "__main__":
    main()
