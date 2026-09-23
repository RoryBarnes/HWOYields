# Handoff: HWO exoEarth yield rederivation

**For:** the next agent picking this up
**Repo:** `/workspace/yields`, vaibify project `hwoYieldRederivation`
**State at handoff:** commit `5023b17`, 15 of 16 independent published checks agree
**Sessions:** 2026-09-21 through 2026-09-23

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
early on an apparent agreement turned out to rest on errors that cancelled.

## 2. The answer, as it currently stands

| quantity | value |
|---|---|
| η⊕, canonical box (SAG13) | 0.256 |
| η⊕, redefined box | 0.0520 |
| occurrence ratio (redefined/canonical) | 0.203 |
| **canonical EEC yield** | median **21.7**, 90% interval [10.0, 42.1] |
| **redefined EEC yield** | median **5.3**, 90% interval [2.2, 12.1] |
| **yield ratio** | **0.246 +0.029 −0.024** |

The yield ratio exceeds the occurrence ratio (0.246 > 0.203) because the redefined box excludes
the smallest, faintest planets, so the ones that remain are individually easier to detect. That is
a real effect and not a bug — it was checked.

## 3. Pipeline state

Ten steps, `A01`–`A10`. **Runtime A03→A10 is 37.6 minutes** (was 127.6 before the performance work
in §7).

| step | directory | what it does | min |
|---|---|---|---|
| A01 | `buildTargetCatalog` | reads the real HPIC; validates blackbody flux against measured V | 0.2 |
| A02 | `modelCoronagraph` | mission parameters + digitized DMVC6 curves | 0.5 |
| A03 | `calibrateToStark` | bisects the throughput factor to hit the published 22.5 | 12.4 |
| A04 | `computeCompleteness` | per-star C(τ) over exposure and visit count | 7.2 |
| A05 | `optimizeSurvey` | AYO equal-slope allocation | 0.5 |
| A06 | `sampleOccurrencePosterior` | emcee posterior on SAG13 | 0.1 |
| A07 | `predictRedefinedYield` | the headline answer | 2.6 |
| A08 | `CompareApertureScaling` | 6/7/8/9 m out-of-sample predictions | 13.5 |
| A09 | `VerifyAgainstPublished` | the 16 independent checks | 0.1 |
| A10 | `CompareTargetCompleteness` | per-target comparison against digitized Fig. 11 | 1.4 |

**Tests:** 69 library tests (`pytest tests/`), 30 step test categories (all pass).
**PROOF:** `iProofLevel` 0. The *only* L1 blocker is `user-not-approved` on all ten steps — the
researcher's own verify click, which is theirs to give. Do not attempt it.
**Git:** local only, no remote configured. `reconcile-remote-state` has nothing to fetch.

## 4. The one open problem

**`Realized-yield distribution mode`: model 19.47 against a published 10** (tolerance 0.4). Every
other independent check agrees.

What makes this specific: the **mean (23.3 vs 21) and the σ (4.5 vs 5) already agree**. A mode at
half the mean means Stark's realized yield distribution is strongly **right-skewed**, while this
model's is close to symmetric. So this is a question about distribution *shape*, not level, and
level-shifting fixes will not touch it.

The realized distribution is built in `predictRedefinedYield` by Poisson-drawing planets per star
from the allocated completeness, then marginalising over the emcee η⊕ posterior. Candidate
explanations, none yet tested:

1. **The η⊕ posterior's shape.** Stark's σ_η⊕ is a Gaussian-ish grid over pessimistic/nominal/
   optimistic scenarios (his Sec. 4.1, a 3×3 grid spanning −1.5σ to +1.5σ). This pipeline
   marginalises over a full emcee posterior on the SAG13 power law, which is log-normal-ish and
   right-skewed in η⊕ but may be narrower in the left tail. A mode at 10 with a mean at 21 needs a
   long left tail in *yield*, i.e. substantial probability of η⊕ well below nominal.
2. **Poisson granularity per star.** With ~174 stars each contributing η·C ≈ 0.12 expected
   detections, the per-star counts are mostly 0 or 1; how that is aggregated could flatten or
   sharpen the mode.
3. **Whether the published "mode" is read from the same curve.** The 10 came from reading Stark's
   Fig. 7. Check which of his distributions it is — the one including σ_η⊕ is broader than the one
   excluding it, and the report may be comparing against the wrong one. **Check this first, it is
   cheap and it has bitten this project before** (see §6, the Fig. 11 metric).

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
- **`pkill -f`** matches its own shell and kills the session. Twice. Use PIDs.
- **`until ! pgrep -f "<pattern>"` in a wrapper** whose own command line contains the pattern waits
  forever. Happened once; the job never started.

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
