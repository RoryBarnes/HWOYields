"""Shared figure style: the dataviz default categorical palette and recessive axes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S_SURFACE = "#fcfcfb"
S_INK_PRIMARY = "#0b0b0b"
S_INK_SECONDARY = "#52514e"
S_GRID = "#d9d8d4"
LIST_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DICT_BOX_COLOUR = {"canonical": LIST_SERIES[0], "redefined": LIST_SERIES[1],
                   "hzOnly": LIST_SERIES[2]}
DICT_BOX_LABEL = {"canonical": "Canonical EEC box", "redefined": "Redefined box",
                  "hzOnly": "Narrow HZ only"}
DICT_BOX_LABEL_LONG = {
    "canonical": "Canonical: 0.95-1.67 AU, $R\\leq1.4\\,R_\\oplus$",
    "redefined": "Redefined: 0.96-1.20 AU, 0.5-2 $M_\\oplus$",
    "hzOnly": "Narrow HZ only: 0.96-1.20 AU"}


def fnApplyStyle():
    """Apply the shared rcParams; call once at the top of every plot script."""
    plt.rcParams.update({
        "figure.facecolor": S_SURFACE, "axes.facecolor": S_SURFACE,
        "savefig.facecolor": S_SURFACE, "font.size": 9,
        "axes.edgecolor": S_INK_SECONDARY, "axes.labelcolor": S_INK_PRIMARY,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": S_INK_PRIMARY,
        "axes.grid": True, "grid.color": S_GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": S_INK_SECONDARY, "ytick.color": S_INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8,
        "figure.autolayout": False,
    })


def fnFinishAxes(oAxes, sXLabel, sYLabel, sTitle=None):
    """Label an axes object consistently and push the grid behind the data."""
    oAxes.set_xlabel(sXLabel)
    oAxes.set_ylabel(sYLabel)
    if sTitle:
        oAxes.set_title(sTitle, loc="left")
    oAxes.set_axisbelow(True)
    return oAxes
