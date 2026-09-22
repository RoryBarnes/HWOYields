# Sandbox → project: an in-container agent's field report

**For:** a host-side agent or developer working on vaibify
**From:** an in-container coding agent (Claude Opus 5), session of 2026-09-21/22
**Project:** `/workspace/yields`, `hwoYieldRederivation`
**Environment:** container `8e4f9e3a923e...`, entrypoint version 2, action catalog schema 1.0
(107 actions), Python 3.12.3

This records where the in-container agent experience diverged from what the container's own
documentation and `CLAUDE.md` led me to expect, while building a seven-step project from
nothing. Everything below is either **measured** in this session or explicitly flagged as
**inference**. Nothing here is a complaint about the design; most of it is a small number of
sharp edges that each cost real time, and one of them is a documentation statement that is
the exact opposite of observed behaviour.

---

## TL;DR

| # | Finding | Status | Cost |
|---|---|---|---|
| F1 | Declared paths resolve **step-relative**; `scriptAuthoring.md` says repo-relative | **Confirmed** | The headline bug. Silently broke every declared output and figure in 7 steps. |
| F2 | A working project (`fillet`) uses the **opposite** convention to the one that worked here | **Unresolved** | Either `fillet` is latently broken or the rule is not uniform. |
| F3 | `init-project-repo` has no recovery path once the agent has run `git init` | Confirmed | Left a root commit with no parent. |
| F4 | `run-all-tests` and `run-plots-only` hang; other WS actions on the same socket work | Confirmed (2 instances) | ~10 min and a misleading "returned nothing". |
| F5 | `create-project` cannot adopt an existing `project.json` | By design; gap | No agent-usable path from an authored project file to an open project. |
| F6 | Nothing agent-visible says "this repo is untracked" | Confirmed | Root cause of a long dead end; the refusal names the wrong layer. |
| F7 | Resolved output paths are visible **only** in `report-l1-blockers` | Confirmed | The one clue that cracked F1, and it reads like a display artifact. |
| F8 | Determinism scanner cannot see unseeded global-RNG use | Working as documented | Included because the quantitative tier caught it — a design win worth knowing about. |

---

## 1. What the researcher asked for

Three escalating requests over one session:

1. Answer whether the published record supports rederiving the AYO exoEarth-yield formalism
   (Stark et al. 2014/2019/2024) well enough to evaluate a counterfactual planet selection.
2. When I said the AYO *code* is unavailable but the *formalism* is published, they pushed
   back — correctly — and asked me to rederive it.
3. **"Go ahead and implement it. Work autonomously until you are done, but don't touch any
   file outside of `/workspace/yields`. When you are done, create a PDF document that
   describes your work. Make sure to build all unit tests and create a git repo."**

So: an explicit instruction to work autonomously, in an empty directory, in toolkit mode
(no project loaded), producing a real vaibify project plus tests, figures and a PDF.

That is the exact scenario the sandbox → project transition is supposed to serve, which is
why the friction below is worth reporting.

---

## 2. The gating sequence nobody could see

`CLAUDE.md` tells the in-container agent this, and it is the natural reading:

> **Creating a new project from inside the container.** `vaibify-do` does not currently expose
> a `create-project` action. When the researcher is in toolkit mode (no project loaded […])
> write a fresh `project.json` directly at `<repository>/.vaibify/projects/<slug>.json`. The
> dashboard polls for new projects and surfaces yours within one tick: the toolkit banner gains
> a "N available" indicator and a toast offers to switch into it.

I did exactly that. Every `vaibify-do` action that touches a workflow then refused:

```json
{"detail": {"sMessage": "No project is open in '8e4f9e…' for this session. Open one first — nothing is wrong with the connection.",
            "sRefusal": "no-project-open"}}
```

**The refusal names the wrong layer.** "Open one first" implies a project exists and merely
needs selecting. In fact the *repository* was not tracked. `/workspace/yields` was created by
me after container start, and per `dashboard.md`:

> When you first open a container, repositories already present in the workspace (cloned by the
> entrypoint from `vaibify.yml`) are tracked automatically. If you clone additional repositories
> inside the container, vaibify detects them within a few seconds and prompts you to **Track**
> or **Ignore** them.

So the real sequence was **Track the repo → open the project → then `vaibify-do` works** — and
the agent can observe none of it, because tracking state lives host-side.

The researcher's own words after I described the project as complete were:

> *"That sounds great, but I don't see how to turn this work into a project."*

That is the failure in one line: the agent believed it had produced a project; the dashboard
had never heard of the repository.

### Suggested fixes (F5, F6)

- Distinguish the refusals: `no-repo-tracked` vs `no-project-open`, with the remediation
  naming the Repos panel explicitly. A single extra refusal code removes the whole dead end.
- Give the agent a read-only `list-repos` / `list-projects` returning tracked state and
  discovered-but-untracked repositories. Agent-safe, and it turns a guess into a lookup.
- `CLAUDE.md`'s paragraph above is accurate about writing the file and incomplete about what
  follows. It should say the repository must be tracked first if it was created in-session.
- `create-project` opens the New Project dialog and "never creates one directly". There is no
  action that *adopts* an authored `project.json`. I deliberately did not fire it, because its
  description suggests it would create a second, empty project rather than adopt mine — which
  would have left the researcher a mess. An `adopt-project` (or a `sProjectPath` argument)
  would close the loop that `CLAUDE.md` opens.

---

## 3. What I did to compensate, and the debt each created

| Blocked action | Substitute | Debt incurred |
|---|---|---|
| `run-step` / `run-all` | Ran each step's commands directly in the shell, in dependency order | Dashboard saw no runs; every step showed unverified. Disclosed to the researcher. |
| `generate-tests-deterministic` | Wrote `explorations/generateStepTests.py`, which introspects each step's declared outputs and emits the three tiers plus a `testStandards.json` | A parallel implementation of a backend feature. It then inherited F1 and had to be fixed twice. |
| `init-project-repo` (user-only) | `git init` | **No empty initial commit** — see F3. |
| `run-all-tests` | 21 explicit `run-test-category` calls | See F4. |
| `reconcile-remote-state` | n/a | Fails: `git fetch failed: transport 'file' not allowed` (no `origin`). |

The pattern worth noting: **every substitute worked, and every substitute was wrong in a way
that only surfaced once the real backend became available.** The agent cannot self-check
against a backend it cannot reach, so it accumulates confident, untested assumptions. An agent
told to "work autonomously" will do this for hours.

---

## 4. F1 — the declared-path scope contradiction

**This is the finding that matters most.**

### Observed

`scriptAuthoring.md` lines 289–295 state:

> Paths in `project.json` — step directories, `saOutputDataFiles`, `saPlotFiles`,
> `{step:...}` tokens — are **repo-relative in both modes**; nothing in a well-formed project
> file names `/workspace` […]

I authored all seven steps accordingly (`saOutputDataFiles: ["modelCoronagraph/missionParameters.json"]`,
`saPlotFiles: ["Plot/figCoronagraphModel.pdf"]`), matching the convention in the existing
`fillet` project. The researcher then tried to view a figure and reported:

> *"it tells me the output is not available. The filename is red."*

Per `dashboard.md`, upright red means *declared file is missing*. But the file existed, and the
backend's own `check-files-exist` confirmed it:

```
vaibify-do check-files-exist '{"saRelativePaths":["Plot/figCoronagraphModel.pdf"]}'
→ {"dictExists": {"Plot/figCoronagraphModel.pdf": true}}
```

`report-l1-blockers` was the only place the *resolved* path appeared:

```
modelCoronagraph/Plot/figCoronagraphModel.pdf         ← figure
modelCoronagraph/modelCoronagraph/coronagraphCurves.csv   ← doubled
buildTargetCatalog/buildTargetCatalog/targetCatalog.csv   ← doubled
```

The step directory is being prefixed onto every declared path.

### Reproduction

```bash
# 1. Declare an output repo-relative and observe the doubled path
vaibify-do update-step A02 '{"saOutputDataFiles":["modelCoronagraph/missionParameters.json"]}'
vaibify-do report-l1-blockers   # → modelCoronagraph/modelCoronagraph/missionParameters.json

# 2. Declare it step-relative and observe it resolve correctly
vaibify-do update-step A02 '{"saOutputDataFiles":["missionParameters.json"]}'
vaibify-do report-l1-blockers   # → modelCoronagraph/missionParameters.json
```

The step's L1 criterion also moved from `axis-not-green` to `upstream-modified` at step 2,
confirming the backend had found the files rather than merely printing a different string.

Figures, which live in the repo-root `Plot/` named by the project-level `sPlotDirectory`,
require an explicit parent escape:

```bash
vaibify-do update-step A02 '{"saPlotFiles":["../Plot/figCoronagraphModel.pdf"]}'
vaibify-do report-l1-blockers   # → Plot/figCoronagraphModel.pdf   ✓
```

Note this also contradicts `scriptAuthoring.md` line 296ff, which says "Absolute paths and
`..`-escapes in the project file are rejected in both modes." The `../Plot/…` form was accepted
and resolved correctly. Either the rejection is narrower than documented (perhaps only escapes
that leave the repo root), or `..` is tolerated where it should not be.

### Impact

Silent and total. All seven steps declared 14 outputs and 5 figures that the backend believed
were missing. Nothing errored. Steps ran and *passed*. The only symptoms were a red filename,
a viewer that would not open an existing file, and L1 blockers that read
`axis-not-green / untested` — which points at **tests**, not at paths, and sent me to run tests
that could never have turned the axis green.

### Suggested fixes

- Correct `scriptAuthoring.md`, and state the base directory explicitly for each of
  `saOutputDataFiles`, `saPlotFiles`, `saInputDataFiles` and `dictTests.sFilePath`.
  `pipelines.md`'s table says only "Output data files to verify" — the base is the one thing a
  project author must know and the one thing neither document states.
- **Validate on save.** `update-step` and project-file ingestion could warn when a declared
  path does not exist relative to the step directory *but does* relative to the repo root. That
  single heuristic catches this exact mistake with a precise message, and it is cheap.
- Consider making the blocker's `sRemediationHint` path-aware: when the offending files do not
  exist at their resolved locations, say so, rather than reporting a test-axis state.

---

## 5. F2 — `fillet` uses the opposite convention (unresolved)

I checked the existing, presumably working project in the same container:

```
fillet declared paths: 30
  exist if resolved repo-relative: 30
  exist if resolved step-relative: 0
```

`fillet` declares `parseFilletOutputs/filletGlobal.json` for a step whose `sDirectory` is
`parseFilletOutputs`, and `Plot/figRadiationSchemes.pdf` against a repo-root `Plot/`. Under the
resolution rule I measured on `yields`, every one of those 30 paths would double.

**I did not verify this.** Confirming it would mean making `fillet` the active project, which I
was not asked to do and did not do. So one of these is true:

- `fillet` has the same latent problem and nobody has hit it (it may never have been pushed to L1); or
- resolution is not uniform — e.g. there is a fallback to repo-relative that `yields` did not
  trigger, or behaviour differs by how the project was created (host-authored vs agent-authored,
  `vaibify init --template` vs hand-written).

The second possibility is the more concerning one, because it would mean the convention depends
on provenance rather than on the schema. **This is the single thing I would most like a host
agent to settle**, since my fix for `yields` assumes the rule is uniform.

---

## 6. F3 — no recovery path after a plain `git init`

`init-project-repo` is `bAgentSafe: false` and its description says:

> Creates an empty initial commit so downstream diff/marker logic has a parent. 409 if the
> target is already a git repo.

The researcher asked me to "create a git repo", so I ran `git init`. That produced a **root
commit with no parent**, which is precisely what the empty initial commit exists to prevent —
and because the directory is now a git repo, `init-project-repo` would 409 and could not
repair it even if the researcher ran it.

I fixed it by hand (orphan branch + `rebase --onto`, tree hash verified byte-identical), but an
agent that does not know the empty-commit convention exists will not know to. Options: have
`init-project-repo` accept an already-initialised repo and add the empty root commit
idempotently, or have the tracking step check for a parentless HEAD and offer the repair.

Related: the demotion of `init-project-repo` to user-only is sound, but it interacts badly with
"create a git repo" being a perfectly ordinary instruction that the agent *can* satisfy with
plain git. The agent takes the obvious path and lands in an unsupported state.

---

## 7. F4 — `run-all-tests` hangs

```
timeout 600 vaibify-do run-all-tests     → no output, exit 143 (SIGTERM at timeout)
vaibify-do run-test-category A02 sCategory=integrity
                                         → {"bPassed": true, "iExitCode": 0, …}  (0.4 s)
```

`run-step`, `run-selected-steps`, `run-all` and `run-test-category` all worked over the same
WebSocket in the same session, so this looks specific to `run-all-tests`. One instance only; I
did not retry it after the path fix, so I cannot say whether it is load-related, related to
7 steps × 3 categories, or unconditional.

Workaround: 21 explicit `run-test-category` calls, all green.

**Second instance, later in the session:** `run-plots-only A07` also hung (no output, no
completion) while `run-step`, `run-selected-steps` and `run-all` continued to work in the same
session. So this is not unique to `run-all-tests`. Both hanging actions are ones that fan out
over sub-commands without running the step's data commands; that may be the common factor, but
with two instances I cannot say more than that it is worth a look at whichever code path
`run-all-tests` and `run-plots-only` share.

---

## 8. F7 — the resolved path is visible in exactly one place

`report-l1-blockers` → `listOffendingFiles` was the only surface in the whole agent-facing API
that showed a *resolved* path. Everything else takes or returns paths in the caller's own
frame, so they round-trip and reveal nothing.

Worse, it initially reads like a cosmetic prefix. My first conclusion was that it was a display
artifact of the blocker report — reinforced by `fillet` (F2) using the same convention I had.
I only got to the truth by mutating a declaration and re-reading the report. Had
`listOffendingFiles` not existed, I do not think I would have found this at all.

An agent-safe `resolve-step-paths <label>` returning declared → resolved → exists for each
path would make this a one-call diagnosis instead of a multi-step experiment.

---

## 9. F8 — a design win worth recording

Not a defect. `scan-determinism` returned `listIssues: []` and correctly disclaimed itself:

> Seeded RNG, set/dict iteration order and parallel reduction order are not detectable here, so
> the declaration remains your assertion.

A real bug lived in exactly that blind spot: `emcee`'s `EnsembleSampler` draws proposal moves
from numpy's **global legacy RNG**, so seeding only the walker start positions with
`np.random.default_rng(seed)` leaves the chain irreproducible. Two runs produced different
posteriors.

**The quantitative tier caught it**, as two `FAIL`s on `faChain` mean/min/max against recorded
standards, at the moment the tests were first run through the backend. That is the tier working
exactly as `CLAUDE.md` describes — "the only tier that makes a claim about the numbers" — and
catching something no static scan could. Worth keeping as a concrete example when explaining
why the three tiers are separate.

(Fix: `np.random.seed(iSeed)` alongside the generator. Two runs now bit-identical. It may be
worth adding global-RNG-consuming libraries — `emcee`, older `sklearn`, `networkx` — to the
scanner's *advisory* output, even if it cannot prove anything.)

---

## 10. What worked well

Stated for balance, since a report of only friction is a distorted one.

- **The refusal protocol is excellent.** Structured `sRefusal` codes, `bAgentSafe` in the
  catalog, and the user-only demotions meant I never had to guess whether an action was mine to
  take. I surfaced `accept-plots-as-standard`-class actions to the researcher without friction.
- **`--describe` is the right affordance.** I learned the step schema from `--describe` plus one
  existing project file, with no host access.
- **`report-l1-blockers` speaks the researcher's language.** `sRemediationHint` strings like
  "Step has never been verified — click verify when satisfied" are directly relayable, exactly
  as `CLAUDE.md` asks. I relayed them verbatim and they landed.
- **The backend is the ground truth, and saying so works.** `CLAUDE.md`'s instruction to trust
  `iProofLevel` over my own file reading is what made me test F1 rather than argue with the
  dashboard. Without that instruction I would likely have concluded the viewer was broken.
- **Determinism env is applied automatically** (`bDeterminismEnvApplied: true` in every step's
  stats) and the three-tier test model held up under real use.

---

## 11. Final state of this project

- 7 steps, all passing via `vaibify-do run-all`; cross-step `{StepNN.*}` tokens resolve.
- 21/21 test categories green via `run-test-category`; 45 library unit tests green.
- Sole remaining L1 blocker on all 7 steps: `user-not-approved` — the researcher's attestation.
- `iProofLevel: 0`; L2 additionally needs an `origin` remote (`set-project-git-remote`).
- `emcee>=3.1` and `pytest>=8.0` were pip-installed in-session and are **not** in the image;
  they belong in `pythonPackages` in `vaibify.yml`. Note `pytest` being absent from an image
  whose `CLAUDE.md` says "Test changes with `pytest` before committing" is its own small
  inconsistency.

## 12. The one-paragraph version

An agent told to build a project autonomously in toolkit mode will write `project.json` exactly
as `CLAUDE.md` instructs, then find every workflow action refused with a message pointing at the
wrong layer, with no agent-visible way to learn that the repository is untracked. It will
compensate by reimplementing the backend's work in the shell, accumulating assumptions it cannot
test. When the researcher finally opens the project, those assumptions surface at once — and the
most damaging is that declared paths resolve relative to the step directory while the
documentation states the opposite, which breaks silently, passes every run, and reports itself
as a test-axis failure. Two extra refusal codes, one path-validation warning on save, and one
corrected sentence in `scriptAuthoring.md` would have removed nearly all of it.
