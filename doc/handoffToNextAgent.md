# Handoff: HWO exoEarth yield rederivation

**For:** the next agent picking this up
**Repo:** `/workspace/yields`, vaibify project `hwoYieldRederivation`
**State at handoff:** commit `4b0617b` plus uncommitted 2026-09-24 evening work (below); κ REMOVED; 22 of 24 independent published checks agree
**Sessions:** 2026-09-21 through 2026-09-24

---

## 0. Read first: state at the end of 2026-09-24 (supersedes §2–§4 where they conflict)

**Memory backup.** `~/.claude` is NOT on the persistent disk; a copy of the agent memory is in
`/workspace/.claudeMemoryBackup/`. If `~/.claude/projects/-workspace-yields/memory/` is empty
after a restart, copy it back.

**Term-by-term ETC comparison** (`explorations/compareEtcBenchmarkTermByTerm.py`). Stark+2025's ETC
benchmark is the USORT telescope (7.87/6.5 m, vortex coronagraph), NOT the 6 m EAC. Given the same
inputs the model's code reproduces AYO's per-observation times to 0.94–1.02 — the equations are
right. Form differences found: zodi latitude, noise floor, cap-with-overheads (all fixed, below);
leak normalization and T_sky shape (tested, rejected below); no binary stray-light term (not
implemented; ~10% of background for HIP 77052).

**Adopted in A02 (researcher-approved):** `sZodiModel "stark2014"` (Leinert ×0.69×f135(β); matches
AYO zodi 0.93–1.01), `sNoiseFloorModel "denominator"` (Δmag 26.5 floor as CR_nf in the
exposure-time equation), `bTimeLimitIncludesOverheads` (two-month limits include overheads).

**κ removed (researcher's decision).** No Stark paper has a throughput calibration; κ let every
near-uniform lever hide (`explorations/measureLeverTimeMultipliers.py`: at refitted κ each tested
lever changes per-planet times by a flat ~1.0 ± 0.05). A03 now adopts 1 and records the fitted
value (1.195) as a diagnostic only; `--fit-throughput` restores the old behaviour. A09 drops the
tuned "Calibrated 6 m yield" check when κ is not fitted.

**Run B (κ = 1) vs run A (κ fitted), only κ differs** (`explorations/compareKappaFittedVsUnity.py`;
run A snapshot in `explorations/output/runA_kappaFitted/`; current step outputs are run B):
planning yield 20.9 (−7%); realized median 18 (Stark 18), mean 21.04 (21.0), P(<12) 0.28 (0.29),
P25 6 m 0.32 (0.32), Fig. 10 red 16.8 (17.35), mode 12 (10); 22/24 independent checks agree.
Still failing: first-18 characterization time 14.3 d (22) and 2.05 d at 9 m (3.5); Fig. 11 hard
zeros beyond 15 pc and above 2 L☉. Yield ratio (deliverable) 0.239 (was 0.242).

**Levers #1/#2 rejected** (`explorations/whatIfLeakAndSkyThroughput.py`): leak ×1.78 (Stark 2019
Eq. 4 PSF_peak·Ω), PSF-convolved T_sky, AYO-vortex-shaped T_sky — library options
`sLeakNormalization`/`fLeakFactor`, `bSkyThroughputConvolved`, `dictSkyThroughputShape`, all off.
After refitting κ none moved char times, target mix or fixed-η level.

**Done 2026-09-24 evening: the κ degeneracy is closed.** All 56 design points evaluated
(`explorations/runCalibrationDegeneracyDesign.py`, then `fitCalibrationDegeneracy.py`,
`plotCalibrationDegeneracyCorner.py`, `summariseCalibrationDegeneracy.py`; outputs
`output/calibrationDegeneracy{,Summary}.json`, `calibrationDegeneracyCorner.pdf`). **No point in
the (κ, κ_c, f_leak, s_sky) box reaches Stark's first-18 characterization time**: over the 56
directly evaluated points it spans 3.1–16.9 d against his 22 d, and the Fig. 11 10–15 pc median
completeness spans 0.00–0.64 against his 0.73. Constrained on the yield alone the posterior gives
κ 1.47, κ_c 1.05, corr(ln κ, ln κ_c) −0.81 and a 6 m char time of 9.8 d [6.5, 12.7] — 2.2× too
fast; constrained on everything, nothing is reached and the yield falls to 16.4. Sensitivity
Sharpest form, free of emulator and prior bound: **of the 20
points (of 56) that reproduce the 22.5 yield to 10%, the first-18 char time spans only 7.0–11.6 d
against Stark's 22**, and 0.79–1.49 d at 9 m against 3.5 — no overlap. Characterization time can
be bought (the all-constraints posterior drives κ_c onto its 0.4 prior floor doing it) but only
by surrendering the yield. Sensitivity
singular values 31.7 / 2.31 / 1.50 / 0.90: the published observables constrain one parameter
combination, the κ/κ_c anticorrelation. Emulator leave-one-out RMS is 0.010–0.049 in ln against
0.10 tolerances, so this is not surrogate error. **Throughput-shaped levers are exhausted for the
characterization-time problem.**

**Done 2026-09-24 evening: the three one-variable experiments** (`explorations/
runOneVariableExperiments.py`, `summariseOneVariableExperiments.py`, outputs under
`output/oneVariableExperiments/`). Each changes ONE setting adopted because it matched a published
number, with every other setting, input and seed fixed and κ held at 1; unaffected stages reuse
the baseline step outputs. **The deliverable is robust:** yield ratio 0.2386 → 0.2316 (albedo,
−2.9%), 0.2363 (η, −1.0%), 0.2299 (T_sky, −3.6%), all far inside the 90% interval.
- *albedo* — Stark's per-visit method fits his level (17.75 vs 17.35) and Fig. 10 shape slightly
  better, but its albedo penalty is 3.5% against his reported 12%; "recompute" (8.6%) stays.
- *η interval* — the paper's stated 86% reading costs 1% on the ratio but wrecks the Fig. 10
  agreement (TV 0.038 → 0.132, mode 11 → 16, P(<12) 0.28 → 0.22). The adopted 68% reading moves
  the SHAPE, not the answer: report it as a stated limitation.
- *T_sky* — strongly load-bearing (planning yield −19%, level −16%, 20 fewer stars) and sourced
  from Stark 2019 Eqs. 5–6 rather than fitted; the best-justified of the three.

**The EEC yield peak (2026-09-25, `explorations/plotYieldPeakComparison.py`, figure
`output/oneVariableExperiments/figures/figYieldPeakComparison.pdf`, per-variant A07 figures
alongside it).** Build the PMF as the EXACT Poisson mixture over `faExpected_canonical`
(`yieldlib.yielddistribution`), not as a histogram of the 14400 realized draws: the histogram's
argmax bootstraps to [10, 16], the exact mixture's to [11, 12]. **Adopted model peaks at 11
against Stark's 10**, with mean 21.08 vs 21.0, median 18 vs 18, plateau 8–16 vs 8–14. A level scan
(shape held fixed, expected yields scaled) moves the peak one bin per ~5–10% of level — 0.85–0.90
gives peak 10, 1.05 gives 12 — so one bin is the resolution of this statistic and the peak should
now be reported as reproduced, not as an open discrepancy. The variants do not help: albedo
per-visit 12, η-86% 15, constant T_sky 9 (nominally closest, but only by dropping the level 16%).

**Combining the variants (2026-09-25).** Asked whether all three together would be the most
honest model, the answer is no, and the run shows it. Two variants move TOWARD the papers
(per-visit albedo, 86% η interval); constant T_sky moves AWAY, because T_sky(r) is Stark 2019
Eqs. 5–6, sourced physics. So `paperFaithful` (both paper-ward changes, T_sky(r) kept) and
`allThreeCombined` (plus T_sky(r) removed) were both run. **Following every documented Stark
choice at once makes agreement with his own Fig. 10 markedly worse**: peak 11 → 15 against his 10,
shape distance 0.038 → 0.128. `allThreeCombined` peaks at 12 — nearer his 10 in POSITION — but its
peak height is 0.051 against his 0.037 and its shape distance 0.149 is the worst of the six, so
peak position alone misleads; always quote the shape distance with it. Deliverable holds
throughout: worst case 0.2201, −7.7%, inside the 90% interval.

**Consequence worth putting in the report:** Stark's stated 86% η⊕ interval, pushed through his own
described method, does not reproduce his own Fig. 10. The adopted 68% reading therefore follows
his FIGURE against his TEXT — a finding about the paper rather than a fudge in this model, and a
stronger statement than the earlier "adopted to match the width" framing.

**The eta_Earth law, settled as far as published information allows (2026-09-25).** Stark's text
really does say "mean and 86% confidence interval eta = 0.26 +0.29/-0.14" (verified in the e-print
source; "86% confidence" occurs exactly once in the paper and he speaks in standard deviations
elsewhere). A06 anchors 0.26 at the GEOMETRIC CENTRE of [0.12, 0.55], i.e. treats it as the
MEDIAN, so the pipeline departs from his text in TWO places, not one. His actual law is a uniform
mixture of Bryson's two cases, not a lognormal, so his three stated facts are not inconsistent for
him even though no lognormal can hold them at once. A06 now has
`--eta-law {lognormal,brysonMixture}` (default lognormal, verified bit-identical) implementing his
own construction. **It does not rescue the peak: 7 against his 10, shape distance 0.173, the worst
of every variant.** The three laws bracket him — Bryson mixture 7, lognormal-68% 11, lognormal-86%
15, published 10 — and the adopted 68% lognormal is much the closest. Likely cause of the failure:
Bryson's published figure is for his own radius/Teff selection, Stark recomputed from the chains
for a different one, so rescaling fixes the mean but not the shape; the transferred mixture puts
59% of its mass in his quoted interval, matching neither reading. **Keep the 68% lognormal, show
all three in the report, ask Stark.** Deliverable robust throughout: 0.2201–0.2386 across all six
variants.

**Still unaudited:** aperture fill factor 0.785 is unsourced and now sets the absolute scale.

**Environment.** Container CPU is capped by cgroup (`/sys/fs/cgroup/cpu.max`), which `nproc`
ignores — read the cgroup and size `--workers` from it. Raised to **8 CPUs** on the evening of
2026-09-24 (`800000 100000`, `nproc` still says 9); process pools now genuinely parallelise.
Disk `/workspace` is back to 81% (18 GB free). **`pip install`s do not survive a container
restart** — `emcee` and `pytest` were gone after the 2026-09-24 restart and were reinstalled for
that session only; `EXOSIMS>=3.6.5` is still absent. They belong in `pythonPackages` in the
host's `vaibify.yml`. The
first `vaibify-do run-from-step` dispatch can hang silently: check `get-pipeline-state`
`bRunning` within a minute, kill the client by PID and rerun.

---

## 1. What the researcher asked for

Trace the ExoVista/AYO yield chain back to its raw planet-population assumptions, couple it to
emcee for a Bayesian prediction, and answer a specific counterfactual:

> How many observable planets would HWO find with **mass 0.5–2 M⊕**, **radii R = M^0.27**, around
> **FGK stars**, if the habitable zone were **0.96–1.20 AU for the Sun** and scaled as √L for
> other stars — for the **baseline HWO mission of Stark et al. (2024)**?

That is a *redefined* exoEarth-candidate box, narrower than the canonical one Stark uses
(0.95–1.67 AU, R up to 1.4 R⊕). The deliverable is the ratio of the redefined yield to the
canonical one, with uncertainties, plus a PDF report. The researcher is not a yield-estimation
specialist and has asked repeatedly for **thoroughness over speed**, because three separate times
early on an apparent agreement turned out to rest on errors that cancelled. Their standard for
reproducing Stark is **~10% agreement**, not exactness.

2026-09-24 additions to the request: (a) "try kappa" — find what the throughput calibration was
absorbing; (b) make exozodi a probabilistic variable of the pipeline; (c) put every
apples-to-apples published figure next to the model's version in the report; (d) per-star tables
of stellar properties, Earth zone (redefined box) and classic HZ, with zone-vs-zone statistics.
All four are done (§4).

## 2. The answer, as it currently stands

| quantity | value |
|---|---|
| η⊕, canonical box (median) | 0.255 |
| η⊕, redefined box (median) | 0.0520 |
| occurrence ratio (redefined/canonical) | 0.204 |
| **canonical EEC yield** (realized, marginalized) | median **19.3**, 90% interval [5.9, 51.6] |
| **redefined EEC yield** | median **4.6**, 90% interval [1.3, 15.5] |
| **yield ratio** | **0.245 +0.035 −0.025** |
| P(≥25), canonical / redefined | 35.7% / 1.2% |

The ratio has been stable across every revision (0.246 → 0.241 → 0.245). It exceeds the occurrence
ratio because the redefined box drops the faintest planets (checked, real).

## 3. Pipeline state

Twelve steps, `A01`–`A12`. Full run A02→A12 is about 80 minutes (A03 ~12, A04 ~17, A08 ~35).

| step | directory | what it does |
|---|---|---|
| A01 | `buildTargetCatalog` | reads the real HPIC |
| A02 | `modelCoronagraph` | mission parameters, digitized DMVC6, HOSTS exozodi distribution, **`bSkyThroughputFollowsCore: true`** |
| A03 | `calibrateToStark` | fits κ to 22.5 at fixed exozodi — now **κ = 0.988**, uncalibrated 22.62 |
| A04 | `computeCompleteness` | completeness on a **23-level exozodi grid** (canonical, redefined) + **200 exozodi draws** (`faZodiDraws`) |
| A05 | `optimizeSurvey` | survey at fixed exozodi AND per draw; penalties |
| A06 | `sampleOccurrencePosterior` | η posterior (unchanged) |
| A07 | `predictRedefinedYield` | η-grid per draw, η samples paired with draws, Poisson; fixed-η (Fig. 10 red) distribution too |
| A08 | `CompareApertureScaling` | 6–9 m, 100 draws each; P25, Fig. 12 PMFs, Fig. 14 per-planet char-time histograms |
| A09 | `VerifyAgainstPublished` | 23 independent checks |
| A10 | `CompareTargetCompleteness` | Fig. 11 comparison from A04's draws (representative = median-yield draw) |
| A11 | `TabulateEarthZones` | per-star Earth zone / classic HZ table + zone statistics (**new**) |
| A12 | `ComparePublishedFigures` | model versions of Stark Figs. 4, 7, 9, 10, 11, 12, 14, 15, 25 in the published axes (**new**); originals in `reference/` |

**Tests:** 78 library tests pass. Step test standards for A04–A12 are stale after this revision and
must be regenerated (`explorations/generateStepTests.py`; the server generator is broken, F11).
**PROOF:** `iProofLevel` 0; the only L1 blocker was the researcher's own verify click.

Library changes this session (all tested, all backward compatible by default):
- `completeness.py`: rates computed once per star at 1 zodi, exposure times per exozodi level
  (`fdictStarRates`, `fdictCompletenessAtZodi`); verified identical to the old code to 1e-15.
  `faSkyThroughputAt` implements T_sky(r).
- `optimizer.py`: one-sort equal-slope solver (identical to the 80-step bisection, 9–25× faster),
  Pareto pre-filter, optional per-star output incl. the allocation (visits, exposure).
- `exozodi.py` (new): grid, draws, interpolation, per-draw surveys (process pool), per-planet
  first-N characterization times.
- `yielddistribution.py` (new): Poisson-mixture PMF.

## 4. What changed on 2026-09-24, and the open problem

**κ explained.** Stark 2019 Eqs. 5–6 multiply zodi/exozodi by T_sky(x,y), a map (PSF convolved
with a uniform background). The model used its large-separation value 0.678 everywhere. As
T_sky(r) = 0.678 Υ(r)/Υmax the uncalibrated yield is 22.62 (was 18.65), κ 1.62 → 0.988
(`explorations/whatIfCharacterizationTreatment.py`, output `whatIfSkyThroughput.json`, both
variants). A split κ (detection vs spectra) cannot fix the char-time *ratio*, so it is rejected.

**Dust marginalized.** 200 draws with re-optimization per draw. Grid interpolation vs direct
evaluation: ≤0.24% on optimized yields (`explorations/validateExozodiGrid.py`). Exozodi penalty
**11.2%** (Stark 11.1%), albedo 8.4% (12%), fixed-η mean **18.30** (17.35, 5.5% high — inside the
researcher's 10%). P25 at 6/7/8/9 m: 36/54/67/77% incl σ_η (Stark 32/53/67/78), 8/52/88/99% excl
(6/49/89/99.5); crossing 7.09 m (7.1).

**Still failing (the open problem):** characterization time. First-18 mean 13.8 d at 6 m and
1.85 d at 9 m (Stark 22 and 3.5); mode 14 vs 10; Fig. 11 target mix (10–15 pc median completeness
0.35 vs 0.73; 15–20 pc 0.07 vs 0.28; L 1–2 0.08 vs 0.62). Earth-twin char times: τ Cet 0.21 d,
61 Vir 9.8, θ Boo 72, 110 Her 255 (exozodi-limited, ∝ d⁴). The model's char time is too steep in
target difficulty. Untested candidates, in order: (1) the per-EEC char time phase — the model takes
the best of 40 phases, the cheapest possible; (2) what "S/N per spectral bin" means in AYO (one
R=140 bin vs band); (3) exozodi surface brightness at the planet's separation rather than the
EEID (Stark 2019 Table 1 note d). Compare shapes with the Fig. 14 pair (per-planet histogram).

**Peak/level levers tested after the report (2026-09-24, all in `explorations/whatIfCharacterizationTreatment.py`):**
exozodi at the planet's separation (−1.3%), spectra at detection epoch (−0.8%), Stark's per-visit albedo test
(+5.6%, penalty 3.6%), per-star characterization cap (+3.5%). None adopted. `testDistantTargetReachability.py`
shows the 10–20 pc Fig. 11 targets are unreachable because 57% of their detectable planets need spectra
> 60 d; Stark+2025's ETC benchmark (spreadsheets in `explorations/reference/`) gives AYO's own spectra of
37–104 d at 12–18 pc, which this model reproduces. The open question — how Stark's survey completes those
stars — has been referred to the researcher to ask C. Stark. Do not tune the peak by hand meanwhile.

**Figures.** The report's §"Side by side" puts nine published figures next to A12's versions.
Figs. 10, 12, 15 agree closely in shape; Fig. 14 and Fig. 11 show the open problem.

**Earth zone vs classic HZ** (A11, each zone's own optimized survey, η at posterior medians):
expected yield 19.30 vs 4.63 (ratio 0.240; per-star median 0.241, 68% range 0.20–0.33). The ratio
rises with luminosity (0.21 below 0.3 L☉ to 0.33 above 5 L☉) and distance. The Earth-zone survey
observes fewer stars (171 vs 184) at higher completeness (0.54 vs 0.45).

## Previous open-problem notes (2026-09-23, kept for the record)

**The mode problem is real and is now explained (see the end of this section).** Stark 2024 Fig. 10 was
digitized from the e-print's vector PDF (`VerifyAgainstPublished/dataDigitiseStarkYieldDistribution.py`,
also `explorations/digitiseStarkYieldDistributionFigure.py`). The purple curve peaks at 10 with median 18.
An earlier claim here, that a 498-run histogram of that curve puts its mode anywhere in 7–21, was WRONG
and is retracted: each of Stark's 498 runs carries 1000 planet draws, so his mode is well determined.
A09's `dictModeDiagnostic` still reports that flawed range and should be replaced by a restored mode check.
The digitization reproduces Stark's text (red mean 17.35, P25 6% and ~32%). A09 now checks the
fixed-η⊕ mean (Fig. 10 red), the median, and P(yield < 12) instead, and reports the mode only as a
diagnostic. The "Sampling distribution mean (Fig. 4)" check was comparing against the wrong
quantity (the albedo-drawn yield); it now uses the calibrated yield and is marked tuned.

**The real gap was level, and most of it was exozodi.** Once the fixed-η⊕ level is matched, a plain Poisson
draw reproduces Stark's red curve (total-variation distance 0.038) and the realized median
(`explorations/compareYieldDistributionShape.py`). The η⊕ law was *not* the problem: reading 0.26 as a
mean looks better unscaled but is a compensating error (Stark's purple/red mean ratio 1.21 needs
⟨η⊕⟩ ≈ 0.29, which A06 already has).

Changes made:
- **Exozodi distribution.** The lognormal stand-in (median 3, ln σ 1.2) put 0.2% of stars above 100
  zodis. It is replaced by the LBTI HOSTS maximum-likelihood distribution digitized from Stark Fig. 9
  (`modelCoronagraph/reference/starkExozodiHostsMaxLikelihood.json`, digitizer
  `explorations/digitiseStarkExozodiDistributionFigure.py`). Its median is 2.98 zodis (Stark: 3), and
  21% of stars are above 100 zodis. The histogram sums to 0.948 of the implied 5M draws; this is
  renormalized, not put off-axis (off-axis gives a median of 3.4). Model exposure times do respond to
  exozodi (65% of background on yield-carrying stars, elasticity 0.59, `explorations/diagnoseExozodiSensitivity.py`),
  so the weak penalty was the draw, not the physics.
- **Pinned stars.** ε Eri 297, θ Boo 148, 72 Her (HPIC "w Her", HIP 84862) 588, 110 Her 235 zodis,
  applied only when drawing. They carry at most 0.47 EEC (`explorations/measurePinnedExozodiStarContribution.py`).
- **A03 calibrates with exozodi fixed** (`bExozodiDrawnDuringCalibration: false`); factor 1.664 → 1.622.
  Otherwise the calibration would absorb the new penalty.

Where it stands (A09: **15 of 17** independent checks agree):
- Pipeline (one exozodi draw, seed fixed): fixed-η⊕ yield **19.34** vs Stark **17.35** (11% high, tolerance 10%).
- **That seed is lucky.** Over 8 draws (`explorations/measureExozodiPenaltyAcrossDraws.py`) the mean is
  **18.86 ± 0.48** per draw (8.7% high), exozodi penalty **8.1%** vs 11.1%, albedo 8.7% vs 12%.
- P(yield < 12) is 0.14 vs 0.29; this follows mostly from the level.

**The η⊕ width explains the mode (`explorations/inferEtaLawFromStarkFigure10.py`).** With the model scaled
to Stark's red level, the lognormal η⊕ that best turns red into purple has median 0.26 and ln-σ 0.80
(total-variation 0.019; mode 11, median 17, P(<12) 0.288 vs published 10, 18, 0.291). Stark's quoted
0.26 +0.29/−0.14 read as a **68%** interval gives σ 0.76 and fits nearly as well (TV 0.024). Read as the
stated **86%**, which is what A06 does, it gives σ 0.52 and misses (TV 0.142, mode 15). Bryson 2021's
posterior chains are not published (the GitHub repo omits the `*_out` directories; stevepur.com holds
only the completeness contours). His Fig. 13 (digitized: `explorations/digitiseBrysonEtaEarthFigure.py`)
shows per-case ln-widths of about 0.8, consistent with the figure rather than with the 86% reading.
**Adopted 2026-09-23 at the researcher's direction** ("we don't need an exact reproduction, just ~10%"):
A06 now reads the interval at z = 1 (`--eta-interval-z`, default `F_Z_INTERVAL_DEFAULT`), giving ln-σ 0.76.
Result: the shape distance to Stark's purple curve went 0.157 → 0.057, and P(<12) went 0.14 → 0.22
(Stark 0.29). P25 at 7/8/9 m now agrees to 2–8%. **16 of 18** independent checks agree. The two misses
are the fixed-η⊕ level (19.34 vs 17.35) and the mode (15 vs 10, 25% tolerance). Our distribution's
top is flat within 10% from 10 to 19 EECs (Stark's: 8–15), so the mode offset is now mostly the 11%
level excess. With the model scaled to Stark's level, the same η⊕ law gives mode 12.

**Characterization-time diagnosis (2026-09-24).** Against Stark's Fig. 11, the survey allocates nothing to about half of the
L>1 and >15 pc targets. The cause is characterization cost: at R=140 near 1 µm the background is exozodi-dominated, so
τ_char ∝ d⁴ (Earth twin: 0.15 d at τ Cet, 57 d at θ Boo, 214 d at 110 Her). The easiest targets are too FAST instead:
the priority-first 18 EECs average 12.6 d at 6 m (Stark 22) and 1.3 d at 9 m (Stark 3.5).
- Ruled out, each by direct test: noise floor; CIC CR_sat definition; IFS pixels (96 = Stark's Table 2);
  first-18 ordering; the Fig. 11 metric; dust colour.
- Partial movers: characterizing at the detection phase, and removing κ from spectra. With spectra uncalibrated
  the planning yield SATURATES at 19.47 for any detection throughput (`explorations/whatIfCharacterizationTreatment.py`),
  so κ is what lets the survey reach 22.5, by speeding up spectra.
- Library switch: per-band `bApplyThroughputCalibration` (default true, pipeline unchanged).

**Report rewritten (2026-09-24).** `doc/reportTemplate.tex` is now organized around a probabilistic graphical model
(an influence diagram whose decision node is the allocation; actionable = has a path into it). The old
chronological report is in git history. `buildReport.py` injects every number, including the A09 table and the
exploration outputs listed in `DICT_EXTRA_PATHS`.

Next steps, in order:
0. **Find the characterization-time difference** (above). It should move the level, both penalties, the target
   mix and the Fig. 10 low tail together.
1. **Average over exozodi draws in A04/A05/A07** as Stark does (he uses 500), instead of one seed.
   With the current A04 this multiplies its ~6 min runtime by the number of draws, which is why it was
   not done without the researcher's go-ahead.
2. The remaining ~3 points each of exozodi and albedo penalty. Stark notes his albedo method
   likely *under*estimates the degradation, so an exact match there is not expected.

## 5. What has been tried: the full ledger

**29 hypotheses: 12 rejected on evidence, 17 real defects found and fixed.** Eight of the 17 moved
the result materially. The distinction between "rejected" and "real error but not the cause" is
load-bearing — a defect can be genuine, worth fixing, and irrelevant to the symptom that led to it.

### Rejected on evidence (12)

| hypothesis | evidence |
|---|---|
| Target-screening cap suppresses distant stars | Raising 1500 → 6500 changes the 9 m yield by 1.3% |
| A uniform yield bias explains the aperture miss | Stark's own 17.3/22.5 factor repairs 6 m and destroys 7/8/9 m |
| Characterization bandpass optimization missing | All nine options implemented; 1000 nm wins almost always; yield moved 0.01 |
| Two detections required for orbit determination | Lowers yield 20.4 → 18.8, leaves the relative slope at 0.96 |
| Model is background- rather than photon-limited | CR_p/CR_b ≈ 1.0–1.4, as a 1e-10 floor should give |
| One dominant source of scatter in ln τ | Freezing phase / radius / axis gives 1.12 / 0.96 / 0.96 vs 1.21 — three comparable terms |
| A separate coronagraph for characterization | Rejected on source: Stark 2019 uses one design for segmented off-axis |
| Equal-slope allocation is wrong | 0.00% shortfall against an exact dynamic-programming optimum on the real curves |
| Blackbody stellar flux is biased | +0.019 mag median against HPIC's measured V over 5300–7300 K; 0.979 against EXOSIMS |
| Stark's per-visit-threshold albedo method explains the weak penalty | Implemented: gives 3.3%, *weaker* than the re-derivation's 5.2% |
| Characterization need not succeed to count | Rejected on source: Stark 2019 Sec. 5.3 is explicit that it must |
| Visit count explains the weak albedo penalty | The penalty *rises* with visits (0.032 at 1 → 0.053 at 6); reasoning was backwards |

### Added 2026-09-24

| hypothesis | outcome |
|---|---|
| Constant T_sky (κ compensating) | **Real, the cause of κ**: T_sky(r) gives uncalibrated 22.62, κ 1.62 → 0.988 |
| One exozodi draw | **Real, fixed**: 200 draws, re-optimized per draw; exozodi penalty 11.2% |
| Split κ (detection vs spectra) | Rejected: a scalar on spectra cannot change the 6/9 m char-time ratio |

### Real defects found and fixed (17)

| defect | effect |
|---|---|
| Noise-floor screen discarded luminous stars | Admitted nothing above 4.6 L⊙ where Fig. 11 reaches 20 |
| **Characterization-time statistic** | Median over all detectable planets where the budget needs a mean over those counted. Took D^1.11 → D^1.88 |
| Coronagraph contrast floor applied universally | The 1e-10 floor was used inside the knee too |
| **Missing revisits** | Single-visit only. Raised uncalibrated yield 1.7×, took the factor 4.23 → 1.06 |
| Albedo distribution unmodelled | Implemented |
| Exozodi sampling unmodelled | Implemented |
| **Over-cap characterization charged in the cost** | 15 000 days charged at 20 pc for an observation never attempted. 10–15 pc completeness 0.000 → 0.532 |
| **Reconstructed target catalog** | HPIC was reachable all along; the download had returned a 169-byte redirect stub. The Gaia stand-in was short 26–34% of FGK dwarfs beyond 15 pc and ran ~170 K hot for K dwarfs. Replacing it *lowered* yield 19.5 → 14.6 and exposed a gate failure it had been masking |
| Missing T_sky on the backgrounds | Stark 2019 Eqs. 5–6 carry an extended-source throughput the point-source terms do not; backgrounds were 1.48× too bright |
| Wrong exozodi surface-brightness law | Used the constant-surface-brightness form Stark 2014 App. C explicitly abandoned, plus a radial term he tested and rejected |
| **Coronagraph curves read in the wrong aperture convention** | Curves are plotted against *circumscribed* D; scenarios are labelled by *inscribed* D. The IWA sat 19% too far out in angle. Yield 14.7 → 18.7, targets 127 → 149 |
| **Collecting area taken as the inscribed circle** | Υ_c is normalised to the full obscured primary, so the Lyot discard was charged twice |
| **Characterization charged at the detection phase** | Stark 2019 Sec. 7.2 optimizes the phase. Per-distance completeness vs Fig. 11 went 1.03/0.55/0.35/0.13 → 1.06/0.76/0.96/0.80 |
| Exozodi drawn unconditionally | The headline yield was always exozodi-sampled while being calibrated against a fixed-exozodi published number. Made the 11% exozodi penalty structurally unmeasurable (it read exactly 0.0000) |
| Parametric coronagraph curves | Replaced by the digitized published figure. Contrast had been optimistic by 5× at 2 λ/D and ~2× from 4–10 |
| **Characterization budget adjusted for the drawn albedo** | Stark 2024 Sec. 3.2 fixes it at A_G = 0.2. Albedo penalty 0.050 → 0.087, which closed both P₂₅ checks too |
| Bright-flux bound on the albedo draw | Physical (edge-on gibbous phase *is* small separation, and cos i uniform means those dominate) but worth only 0.002. Kept behind `bAlbedoBrightBound` |

## 6. Traps this project has already fallen into

Read these before forming a hypothesis. Each cost real time.

- **Compensating errors.** Four separate times an apparent agreement rested on two errors
  cancelling. Never accept a match without asking what would break it.
- **Comparing the wrong quantity.** The Fig. 11 comparison was for a long time against
  characterization-gated completeness when the figure's caption says "HZ completeness". Always
  confirm *which* published curve a number comes from.
- **Biased sub-samples in diagnostics.** The detection-rate-vs-albedo curve was first tallied over
  the top 60 stars by completeness — the saturated ones — and reported 0.94 against a survey mean
  of 0.56. Tally over the whole survey.
- **Read-off values.** Six coronagraph reference values read off a figure by eye were wrong by up
  to 3×, and passed only because the tolerances were 0.35 and 0.75. Digitize the vector content
  stream instead: `explorations/digitiseStarkCoronagraphFigure.py` and `digitiseStarkTargetFigure.py`
  both do this and work.
- **Micro-benchmarks.** A single-star timing profile reported the 16-phase case as slower than the
  30-phase case. Time the real function over a realistic block.
- **Monitor-tool notices were unreliable in the 2026-09-23 session.** Several reported completions
  with timestamps in the future, or files that did not exist. Verify with your own shell
  (`vaibify-do get-pipeline-state`, `ls` timestamps) before acting on one.
- **`pkill -f`** matches its own shell and kills the session. Twice. Use PIDs.
- **`until ! pgrep -f "<pattern>"` in a wrapper** whose own command line contains the pattern waits
  forever. Happened once; the job never started.

- **Stale `vaibify-do` clients block new runs (2026-09-24).** A `run-from-step` client stayed
  alive after its pipeline reported `[completed]`, and a later `run-selected-steps` sat silent
  without dispatching until the stale client was killed (by PID). Check `vaibify-do
  get-pipeline-state` (`bRunning`) and `ps` before assuming a run is going.
- **`pkill -f` killed the session's shell again (2026-09-24).** Third time. Use PIDs, always.
- **`create-step` capitalizes the directory** (`tabulateEarthZones` → `TabulateEarthZones`); the
  first run then fails preflight. Create the directory under the name the project JSON records.

## 7. Practical notes

- **`vaibify-do` works** (it did not for the first session; see `vaibifyIntegrationReport.md` F12).
  Run steps through it, not by calling scripts directly.
- **`generate-tests-deterministic` is broken server-side** for every step and category (F11). Use
  `explorations/generateStepTests.py` instead — it introspects the same declared outputs.
- **`kill-pipeline` is user-only.** Surface the request; do not retry.
- **Declared paths resolve step-relative**, not repo-relative, contradicting `scriptAuthoring.md`
  (F1).
- **Performance.** The inner loop is `fdictCompletenessTable`. A03 calls it 14×, A08 4×, A04 1×,
  which is 93% of runtime. `I_DEFAULT_CHARACTERIZATION_PHASES = 40` was verified against a
  200-phase reference (identical counted set, 0.18% median error). Do not raise it without
  re-measuring; do not lower it below 40, which is where counted planets start being lost.
- **Host dependencies.** `emcee`, `pytest` and `EXOSIMS>=3.6.5` are `pip install`ed in-container
  only. They must be added to `pythonPackages` in the host `vaibify.yml` to survive a rebuild.

## 8. Key files

| file | what |
|---|---|
| `doc/hwoYieldRederivationReport.pdf` | the scientific report, 24 pages, every number injected from step outputs |
| `doc/reportTemplate.tex` + `buildReport.py` | 66 `@@TOKEN@@` substitutions; no hardcoded results |
| `doc/vaibifyIntegrationReport.md` | findings F1–F12 for the vaibify maintainers |
| `yieldlib/` | shared physics: `physics`, `coronagraph`, `occurrence`, `completeness`, `optimizer`, `survey` |
| `explorations/` | 34 saved diagnostic scripts, every one re-runnable |
| `VerifyAgainstPublished/publishedVerification.json` | the 16 checks, with provenance and tolerance per check |

Ten checks comparing the coronagraph curves are marked `bTuned` — they are round-trip integrity
on the digitized table, not independent evidence, which is why the headline is **16** independent
checks rather than 26. That reduction was deliberate and is a real narrowing of what is claimed.

## 9. Independent cross-check

EXOSIMS 3.6.5 was used as an independent implementation. It shares no code and no scheduling
method. On identical stars and optics, restricted to FGK: local zodi **1.001**, exozodi **0.975**,
stellar photon rate **0.979**. It also *found* two of the defects above (the missing T_sky and the
exozodi law) by having terms this pipeline lacked, each then confirmed against Stark's own papers
rather than against EXOSIMS.

It cannot be run end to end here: its HPIC reader predates pandas 3's copy-on-write, and
`Nemati.calc_dMag_per_intTime` raises under this numpy — **EXOSIMS's own shipped sample script
fails at the same line**, so this is a version incompatibility, not a configuration error. See
`explorations/crossCheckRadiometryAgainstExosims.py`.
