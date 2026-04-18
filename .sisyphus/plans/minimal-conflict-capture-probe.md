# Minimal Conflict-Capture Probe on `dynamic_selection_phase1`

## TL;DR
> **Summary**: Keep `dynamic_selection_phase1` as the active branch and add the smallest possible time-localized conflict-capture mechanism needed to target the remaining early-flanker failures that were **not** explained by subject heterogeneity.
> **Deliverables**:
> - one new bounded mechanism toggle/config layered on top of the current phase-1 baseline
> - smoke-run outputs for both `20-29` and `80-89`
> - mechanism-score analysis against the same locked 4 metrics
> - summary memo with pass/fail verdict for the minimal mechanism probe
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3/4 → Task 5 → Task 6

## Context
### Original Request
- Do **not** go back to the large DMC-like extension line.
- Use the single-subject heterogeneity result as a stopping point, not as a new tuning rabbit hole.
- Move to the **next minimal mechanism line**.
- Add only **one mechanism degree of freedom** if possible.
- Keep the workflow small, auditable, and comparable to the current `dynamic_selection_phase1` baseline.

### What We Learned From the Heterogeneity Probe
- The completed single-subject diagnostic workflow ended with:
  - `HETEROGENEITY-NOT-SUPPORTED`
- Interpretation:
  - subject-level `scale` variation can improve some distances,
  - but it does **not** recover the key flanker failures strongly enough,
  - especially the early error/capture regime.
- Therefore the next branch should test a **mechanism-level explanation**, not more subject heterogeneity.

### Locked Design Decision
- The next branch is a **diagnostic mechanism probe**, not a new full model family.
- We will add **one bounded conflict-capture term** and test whether it improves the current failure signature.
- We will keep all of the following unchanged unless explicitly required for plumbing:
  - Stage-2 readout logic
  - urgency-free decision policy
  - fitting objective shape
  - subject heterogeneity workflow

### Proposed Minimal Mechanism
- Add a single **time-localized incongruent capture term** that temporarily boosts the wrong / flanker-driven competition early in the trial.
- Candidate parameterization (locked unless Task 1 disproves feasibility):
  - `capture_strength`
  - `capture_midpoint_s`
  - `capture_tau_s`
- However, this plan’s core constraint is:
  - **only one effective free mechanism degree of freedom in the first pass**.
- So the implementation should prefer:
  - fixed `midpoint` and `tau`
  - tune / probe only `capture_strength`

## Work Objectives
### Core Objective
Determine whether a minimal early conflict-capture term, layered onto `dynamic_selection_phase1`, can improve the locked flanker diagnostics more convincingly than subject heterogeneity alone, without reviving the old large-complexity mechanism branch.

### Deliverables
- `code/scripts/run_minimal_conflict_capture_probe.py`
- `code/scripts/analyze_minimal_conflict_capture_probe.py`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/baseline_manifest.json`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/probe_config_grid.csv`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/<age_group>/capture_probe_metrics.csv`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/reaggregated/probe_scorecard.csv`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/reaggregated/success_bar.json`
- `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/minimal_conflict_capture_summary.md`

### Definition of Done (verifiable conditions with commands)
- A single command runs the bounded mechanism probe for both age groups and writes all outputs to a new dedicated result tree.
- The probe changes only the explicitly declared conflict-capture term; all other active `dynamic_selection_phase1` settings stay fixed.
- The analysis writes a locked 4-metric scorecard and a binary verdict:
  - `CAPTURE-PROBE-SUPPORTED`
  - `CAPTURE-PROBE-NOT-SUPPORTED`
- The summary memo explicitly compares:
  - current baseline
  - heterogeneity probe result
  - minimal conflict-capture probe result

### Must Have
- Reuse the existing Stage-2 cached evaluation path.
- Reuse the existing CAF / delta / conditional-error / tail utilities.
- Support both `20-29` and `80-89` in one workflow.
- Use a **bounded finite grid**, not free optimization.
- Keep result artifacts in a **new dedicated tree**.

### Must NOT Have
- No urgency branch revival.
- No DMC-like accumulator redesign.
- No new readout family.
- No full per-age-group refit.
- No per-subject fitting in this branch.
- No manual-plot-only decision making.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: **tests-after** using targeted pytest + command-line artifact verification
- QA policy: Every task must include a happy path and a failure / edge-case scenario
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: mechanism audit + bounded config scaffold + `20-29` probe
Wave 2: `80-89` probe + cross-age reaggregation + summary memo

### Dependency Matrix
| Task | Depends On | Blocks |
|---|---|---|
| 1 | none | 2, 3, 4 |
| 2 | 1 | 3, 4, 5 |
| 3 | 1, 2 | 5 |
| 4 | 1, 2 | 5 |
| 5 | 3, 4 | 6 |
| 6 | 5 | F1-F4 |

## TODOs

- [ ] 1. Audit the active phase-1 baseline and choose the exact minimal capture insertion point

  **What to do**: Inspect the current `dynamic_selection_phase1` path and identify the narrowest code location where an early incongruent-only conflict-capture term can be injected without altering unrelated mechanism behavior. Write a baseline manifest that locks all non-probe settings and documents the chosen insertion point.
  **Must NOT do**: Do not implement the mechanism yet. Do not broaden the design into a new family. Do not change any existing defaults.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_minimal_conflict_capture_probe.py --mode audit-baseline --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe` exits 0
  - [ ] `artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe/baseline_manifest.json` exists
  - [ ] manifest includes `insertion_point`, `non_probe_params`, and both age groups

- [ ] 2. Implement a bounded single-degree-of-freedom conflict-capture config scaffold

  **What to do**: Add the smallest mechanism configuration scaffold needed to turn the capture term on/off and probe a finite grid over `capture_strength`, while holding capture timing constants fixed. Write the probed grid to disk.
  **Must NOT do**: Do not create a multi-parameter search. Do not expose continuous optimization.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_minimal_conflict_capture_probe.py --mode write-grid --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe` exits 0
  - [ ] `probe_config_grid.csv` exists
  - [ ] grid has one tuned dimension (`capture_strength`) and fixed timing columns

- [ ] 3. Run the bounded capture probe for `20-29`

  **What to do**: Reuse the active phase-1 cached evaluation pipeline and run the bounded capture-strength probe for `20-29`, writing metric summaries for each grid point.
  **Must NOT do**: Do not touch readout, urgency, or subject-level fitting.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_minimal_conflict_capture_probe.py --mode simulate --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe` exits 0
  - [ ] `20-29/capture_probe_metrics.csv` exists
  - [ ] output records baseline row plus probe rows

- [ ] 4. Run the bounded capture probe for `80-89`

  **What to do**: Mirror Task 3 for `80-89` using the same probe structure and output schema.
  **Must NOT do**: Do not introduce age-group-specific mechanism rules beyond baseline artifact selection.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_minimal_conflict_capture_probe.py --mode simulate --age_group 80-89 --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe` exits 0
  - [ ] `80-89/capture_probe_metrics.csv` exists
  - [ ] output schema matches `20-29`

- [ ] 5. Build the cross-age scorecard and locked success bar

  **What to do**: Reaggregate the probe outputs across age groups and compute the same locked target mechanisms used in the heterogeneity probe:
  1. earliest incongruent CAF
  2. first delta quantile
  3. incongruent error-minus-correct RT
  4. incongruent conditional tail
  Compare baseline vs capture-probe against the human target and write `success_bar.json` plus `probe_scorecard.csv`.
  **Must NOT do**: Do not switch to visual-only evaluation. Do not silently redefine improvement rules.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/analyze_minimal_conflict_capture_probe.py --input_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe` exits 0
  - [ ] `reaggregated/probe_scorecard.csv` exists
  - [ ] `reaggregated/success_bar.json` exists
  - [ ] verdict is one of:
    - `CAPTURE-PROBE-SUPPORTED`
    - `CAPTURE-PROBE-NOT-SUPPORTED`

- [ ] 6. Write the summary memo and execution wrapper notes

  **What to do**: Write a concise technical memo that states:
  - this is a diagnostic minimal-mechanism probe
  - what exact mechanism was added
  - bounded grid and invariants
  - score deltas vs baseline
  - comparison to the prior heterogeneity result
  - final verdict
  **Must NOT do**: Do not over-claim. Do not present exploratory evidence as confirmatory.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] full workflow command exits 0
  - [ ] `minimal_conflict_capture_summary.md` exists
  - [ ] summary contains:
    - `diagnostic mechanism probe`
    - `dynamic_selection_phase1`
    - `capture_strength`
    - `HETEROGENEITY-NOT-SUPPORTED`
    - `CAPTURE-PROBE-`

## Final Verification Wave (MANDATORY)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit approval before completion.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Success Criteria
- The probe runs end-to-end for both age groups under a new dedicated result tree.
- Only the minimal bounded conflict-capture term is changed relative to the current phase-1 baseline.
- The branch produces a verdict that can directly inform whether to continue mechanism work or stop this direction.
- The summary memo makes it clear whether the minimal mechanism probe improves on the failed heterogeneity explanation.
