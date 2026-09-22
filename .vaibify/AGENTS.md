# Project context: HWO exoEarth yield rederivation

## What this project is

An independent rederivation of the AYO yield formalism (Stark et al. 2014/2019/2024), built to
answer a counterfactual about the HWO baseline mission: what the yield would be for a
mass-selected planet sample (0.5–2 M⊕, R = M^0.27) in a narrowed habitable zone (0.96–1.20 AU
solar-equivalent, sqrt(L)-scaled).

## Things to know before changing anything

- **The calibration gate is load-bearing.** `calibrateToStark` fits one scalar throughput factor
  so the canonical box reproduces Stark's published 22.5 EECs. If that gate stops passing,
  nothing downstream should be reported. The uncalibrated agreement (24.5 vs 22.5) is the more
  meaningful statement, because yield depends only weakly on throughput.
- **The coronagraph is parametrized, not simulated.** Stark's Upsilon_c(x,y) and I(x,y) maps are
  not published. `yieldlib/coronagraph.py` reconstructs them from three published anchors. This
  is the project's single genuine gap; ratios between boxes are far more robust than absolutes.
- **Yield is NOT proportional to eta.** Characterization time is charged against the same budget
  (Stark 2024 §3.5 calls eta_Earth "actionable"), so the optimizer reallocates. Two unit tests
  in `tests/testOptimizer.py` pin this. Do not "simplify" by rescaling.
- **The Hipparcos supplement in A01 is not optional.** Gaia DR3 publishes no astrophysical
  parameters brighter than G≈3, which drops alpha Cen A/B, Procyon, tau Ceti and eps Eridani —
  the best targets in the sky. Without the supplement those losses get absorbed into the
  calibration factor and misreported as coronagraph error.
- **The occurrence posterior is prior-dominated in shape.** alpha and beta come from SAG13
  priors, not a re-fit of Kepler DR25. Do not describe it as an independent re-inference.

## Environment

`emcee` and `pytest` are pip-installed per session, not in the image. They belong in
`pythonPackages` in `vaibify.yml`.

## Reference material

Stark et al. 2024 = arXiv:2405.19418 (Tables 1 and 2 hold every mission parameter).
Stark et al. 2019 = arXiv:1904.11988 (exposure-time equations, coronagraph curves).
Use the `read-arxiv` skill rather than fetching PDFs.
