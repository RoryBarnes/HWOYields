"""Unit tests for the per-star exozodi draw: the empirical HOSTS distribution and LBTI pinning."""

import json
import os
import sys

import numpy as np

S_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S_ROOT)
from yieldlib import survey as sv  # noqa: E402

S_REFERENCE = os.path.join(S_ROOT, "modelCoronagraph", "reference",
                           "starkExozodiHostsMaxLikelihood.json")
DICT_PINNED = {"16537": {"sName": "eps Eri", "fZodi": 297.0}}


def fdictMission(bDraw=True):
    """A mission dict carrying the digitized HOSTS distribution, renormalized as A02 does."""
    with open(S_REFERENCE) as oFile:
        dictRef = json.load(oFile)
    faProb = np.asarray(dictRef["faProbability"]) / np.sum(dictRef["faProbability"])
    return {"fExozodiLevel": 3.0, "bDrawExozodiLevels": bDraw, "dictPinnedExozodi": DICT_PINNED,
            "dictExozodiDistribution": {"faEdgesZodi": dictRef["faEdgesZodi"],
                                        "faProbability": faProb.tolist(),
                                        "fOffAxisZodi": 1000.0}}


def test_empirical_draw_reproduces_published_median():
    """Stark+2024 Sec. 3.3 quotes a median of three zodis for the HOSTS fit."""
    faLevels = sv.faDrawExozodiLevels(fdictMission(), 200000, 11)
    assert abs(np.median(faLevels) - 3.0) < 0.3


def test_empirical_draw_has_heavy_tail():
    """About a fifth of draws exceed 100 zodis (0.212 of the on-axis 0.948), against 0.2 percent
    for the lognormal stand-in this replaced."""
    faLevels = sv.faDrawExozodiLevels(fdictMission(), 200000, 12)
    assert 0.19 < np.mean(faLevels > 100.0) < 0.26


def test_pinned_star_gets_measured_level_when_drawing():
    faLevels = sv.faDrawExozodiLevels(fdictMission(), 3, 13, ["1", "16537", ""])
    assert faLevels[1] == 297.0


def test_fixed_level_pins_nothing():
    """The calibration run holds every star at the median, pinned stars included."""
    faLevels = sv.faDrawExozodiLevels(fdictMission(bDraw=False), 3, 13, ["1", "16537", ""])
    assert np.all(faLevels == 3.0)


def test_lognormal_form_still_accepted():
    dictMission = {"fExozodiLevel": 3.0, "dictExozodiDistribution":
                   {"fMedianZodi": 3.0, "fLogSigma": 1.2}}
    faLevels = sv.faDrawExozodiLevels(dictMission, 100000, 14)
    assert abs(np.median(faLevels) - 3.0) < 0.1
