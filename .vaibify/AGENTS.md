# Project context: HWO exoEarth yield rederivation

## What this project is

An independent rederivation of the AYO yield formalism (Stark et al. 2014/2019/2024), built to
answer a counterfactual about the HWO baseline mission: what the yield would be for a
mass-selected planet sample (0.5–2 M⊕, R = M^0.27) in a narrowed habitable zone (0.96–1.20 AU
solar-equivalent, sqrt(L)-scaled).

## Things to know before changing anything

- **Nothing is fitted to Stark's results, and nothing should be.** There is no throughput
  calibration: `calibrateToStark` (A03) reports the expected 6 m yield with κ = 1 and records the
  factor a fit would need (`fFittedThroughputFactor`) as a diagnostic only. 22.5 is an output of
  AYO, not an input, and a fitted κ absorbs any near-uniform error in exposure time, which hides
  it. `--fit-throughput` exists for experiments; do not make it the default.
- **Follow Stark's documented method wherever it is stated**, even when a substitute agrees
  better with a published number. The albedo test is his per-visit flux test with both bounds
  (`sAlbedoMethod "perVisitThreshold"`, `bAlbedoBrightBound` in A02). It gives an albedo penalty
  of ~3.6% against his reported 12%; that shortfall is a reported finding, not something to tune.
- **The one departure from Stark's text is the η⊕ interval**, read as 68% (and 0.26 as the
  median) instead of the stated 86% (and mean), because his Fig. 10 requires it. It is declared
  in the report and paper, tested against its source by A15, and carries ~99% of the remaining
  disagreement with Fig. 10 (A14's 2×2). Keep it declared; do not bury it.
- **The coronagraph maps are the project's genuine gap.** Stark's Upsilon_c(x,y) and I(x,y) are
  not published; A02 uses his azimuthally averaged DMVC6 curves, digitized from the e-print's
  vector data, and T_sky(r) is our reconstruction. Ratios between boxes are far more robust than
  absolute yields.
- **Yield is NOT proportional to eta.** Characterization time is charged against the same budget
  (Stark 2024 §3.5 calls eta_Earth "actionable"), so the optimizer reallocates. Two unit tests
  in `tests/testOptimizer.py` pin this. Do not "simplify" by rescaling.
- **The occurrence posterior is prior-dominated in shape.** alpha and beta come from SAG13
  priors, not a re-fit of Kepler DR25. Do not describe it as an independent re-inference.
- **The open problem is characterization time**, too short for easy targets and too long for
  distant ones (first-18 mean 14.3 d vs Stark's 22 at 6 m). A13 shows no throughput setting
  fixes it; do not revisit throughput-shaped levers. It is a question for the AYO authors.
- **Yield-ratio uncertainties are 68% intervals** (16th–84th percentiles). The realized-yield
  "90% interval" is P5–P95. Do not mix the two up in prose.

## Pipeline notes

- A01 reads the real HPIC (Tuchow et al. 2024), which Stark et al. (2024) use; no Gaia or
  Hipparcos reconstruction is involved.
- A14 (`CompareModelVariants`) re-runs the chain for six variants against the adopted model:
  `albedoRecompute`, `etaInterval86` (every documented choice), `albedoRecomputeEta86`,
  `skyThroughputConstant`, `etaInterval86ConstantSky`, `brysonMixtureEta`. It re-runs A04
  internally and adds ~20 min to a full run.
- `vaibify-do generate-tests-deterministic` fails server-side; regenerate standards with
  `python3 generateStepTests.py --project ../.vaibify/projects/hwoYieldRederivation.json --repo-root ..`
  run from `explorations/`.
- `vaibify-do run-all-tests` runs the tests but does not record per-category results on the
  dashboard; record them with `vaibify-do run-test-category <step> sCategory=<category>`.
- Size `--processes`/`--workers` from `/sys/fs/cgroup/cpu.max`, not `nproc`.
- Our T_sky is Upsilon_c / EE(0.7 lambda/D) = 1.474 Upsilon_c (max 0.572 on the DMVC6), not
  the 0.678 that older text quoted (that came from the parametric fallback's 0.46 peak). A13's
  `skyThroughputShapes.json` tabulates it against the AYO-like shape; quote it from there.
- Any emcee run must seed numpy's GLOBAL RNG (`np.random.seed`) as well as the start positions,
  or its chain differs from run to run (A06 and A13 both had this bug).

## Documents

- The report (`doc/hwoYieldRederivationReport.pdf`) is built from `doc/reportTemplate.tex` and
  the paper (`doc/hwoYieldReproductionPaper.pdf`) from `doc/paperTemplate.tex`. Both go through
  `doc/buildReport.py`, which fills every `@@TOKEN@@` from step outputs and fails on an unknown
  one:
  - report: `cd doc && python3 buildReport.py`
  - paper: `cd doc && python3 buildReport.py --template paperTemplate.tex --out-tex hwoYieldReproductionPaper.tex`
- Write them as finished journal articles in American English: describe the model and evidence as
  they stand, not the history of how the project got there.

## Environment

`emcee` and `pytest` are pip-installed per session, not in the image. They belong in
`pythonPackages` in `vaibify.yml`.

## Reference material

Stark et al. 2024 = arXiv:2405.19418 (Tables 1 and 2 hold every mission parameter).
Stark et al. 2019 = arXiv:1904.11988 (exposure-time equations, coronagraph curves).
Stark et al. 2025 = arXiv:2502.18556 (AYO exposure-time calculator benchmark).
Use the `read-arxiv` skill rather than fetching PDFs.
