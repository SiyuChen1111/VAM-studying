# True Single-Subject Feasibility Test for `VGG + WW`

## TL;DR
> **Summary**: Test the `VGG + WW` framework the way we actually mean “single-subject”: fit the model on one subject’s own behavior data, evaluate on held-out trials from that same subject, and decide whether the framework is viable at the individual level at all.
> **Deliverables**:
> - one saved plan-aligned runner for true per-subject fitting
> - one saved plan-aligned analyzer for across-subject feasibility summarization
> - per-subject fit/eval artifacts for a bounded representative subject panel
> - a feasibility scorecard and final keep/drop recommendation for `VGG + WW`
> **Effort**: High
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3/4 → Task 5 → Task 6

## Context
### Original Request
- “single-subject” here means **the model really fits one subject’s data**, not a group-level fit with a tiny subject-specific scale tweak.
- The goal is not another probe layered on top of group parameters.
- The goal is to answer a framework question:
  - **Can `VGG + WW` fit individual subjects at all?**
- If true single-subject fitting still fails, we should seriously consider dropping `VGG + WW` rather than continuing to patch it.

### What We Have Already Ruled Out
- The heterogeneity probe ended with:
  - `HETEROGENEITY-NOT-SUPPORTED`
- Interpretation:
  - allowing small subject-level variation on top of a group-derived baseline does **not** rescue the failure pattern.
- The minimal conflict-capture probe ended with:
  - `CAPTURE-PROBE-NOT-SUPPORTED`
- Interpretation:
  - adding a tiny early capture mechanism patch on top of `dynamic_selection_phase1` does **not** rescue the failure pattern either.
- Therefore the next question is not “what tiny patch should we try next?”
- It is:
  - **Is the `VGG + WW` framework itself viable when trained/evaluated at the level of a single subject?**

### New Scientific Guidance To Respect
- Existing feedback says the current model limitations are concentrated in two behavior signatures:
  - the RT distribution is not right-skewed enough relative to human behavior
  - the model fails to express the relevant error-time structure (especially the conflict-related slow-error side, after earlier phases where it could not even produce errors)
- That implies this workflow must not stop at “did loss go down?”
- It must explicitly ask whether true single-subject fitting improves:
  - RT right-skew / heavy-tail behavior
  - error-vs-correct RT ordering in the direction expected from the subject’s own data
- The proposal itself is still considered reasonable; the point of this branch is to reassess the **modeling feasibility** of `VGG + WW`, not the general logic of the proposal.

### Locked Design Decision
- This branch is a **framework feasibility test**, not a mechanism probe.
- We are allowed to train the subject-specific model parameters on one subject’s own data.
- We are **not** allowed to broaden scope into a new architecture family yet.
- We should reuse as much of the current Stage-1 / Stage-2 pipeline as possible.

### Working Definition of “Single-Subject Feasibility”
- For each chosen subject:
  - build subject-specific train/test trial sets from that subject’s own trials
  - run the standard `VGG + WW` training path on that subject only
  - evaluate on held-out trials from that same subject
- We are not asking whether every subject is perfectly fit.
- We are asking whether the framework shows **credible, repeatable individual-level learnability** on a representative subject panel.
- In practice, that means a subject fit should not only optimize a scalar score, but should also move the model toward the subject’s own:
  - RT skew / heavy-tail structure
  - conflict-sensitive error-time signature

## Work Objectives
### Core Objective
Determine whether `VGG + WW` is worth keeping by running a true per-subject fit/evaluation workflow on a bounded representative panel of individual subjects and summarizing whether the framework can fit single-subject behavior in a stable, non-degenerate way.

### Deliverables
- `code/scripts/run_true_single_subject_feasibility.py`
- `code/scripts/analyze_true_single_subject_feasibility.py`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/panel_manifest.json`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/subject_panel.csv`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/<age_group>/user_<ID>/best_config.json`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/<age_group>/user_<ID>/best_model_params.npz`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/<age_group>/user_<ID>/subject_eval_summary.json`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/reaggregated/feasibility_scorecard.csv`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/reaggregated/feasibility_verdict.json`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/true_single_subject_feasibility_summary.md`

### Definition of Done (verifiable conditions with commands)
- A single command runs the full bounded subject-panel feasibility workflow and writes all outputs to a new dedicated result tree.
- The workflow fits the model **from subject-specific data**, not by only sweeping `scale` on top of group parameters.
- The analysis writes a subject-level feasibility scorecard and a binary final verdict:
  - `VGGWW-SINGLE-SUBJECT-FEASIBLE`
  - `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`
- The summary memo explicitly explains whether `VGG + WW` should be kept or deprioritized based on these single-subject results.

### Must Have
- Reuse the current Stage-1 / Stage-2 training pipeline where possible.
- Reuse existing behavior metrics already used in group-level evaluation.
- Support both `20-29` and `80-89` in one workflow.
- Use a **bounded representative subject panel**, not the entire cohort in the first pass.
- Keep result artifacts in a **new dedicated tree**.
- Explicitly report subject-level RT skew and error-vs-correct RT ordering, not just aggregate score.

### Must NOT Have
- No new model family.
- No return to DMC-like redesign in this branch.
- No urgency branch revival here.
- No hidden fallback to “group fit + subject scale sweep”.
- No manual-only evaluation without machine-readable verdict files.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: **tests-after** using targeted pytest + command-line artifact verification
- QA policy: Every task must include a happy path and a failure / edge-case scenario
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: panel definition + subject split scaffold + one subject per age group smoke fit
Wave 2: remaining subject fits + cross-subject analysis + final memo

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

- [ ] 1. Audit the current age-group training pipeline and define the true single-subject fitting contract

  **What to do**: Inspect the existing Stage-1 / Stage-2 age-group training pipeline and identify the smallest reusable path for fitting one subject’s own data. Write a manifest that locks the data contract, split assumptions, and the exact subject-level outputs we expect.
  **Must NOT do**: Do not train yet. Do not silently fallback to group-level parameter reuse.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_true_single_subject_feasibility.py --mode audit-baseline --output_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility` exits 0
  - [ ] `artifacts/results/repro_legacy_interim/true_single_subject_feasibility/panel_manifest.json` exists
  - [ ] manifest includes split contract, output contract, and both age groups

- [ ] 2. Build the representative subject panel and per-subject split scaffold

  **What to do**: Select a bounded representative subject panel (recommended first pass: 3 subjects per age group) and create deterministic subject-specific train/test split artifacts from each subject’s own trials.
  **Must NOT do**: Do not use cross-subject pooling inside a subject fit. Do not use test-only selection for feasibility.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_true_single_subject_feasibility.py --mode build-panel --output_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility` exits 0
  - [ ] `subject_panel.csv` exists
  - [ ] panel contains both age groups and explicit subject IDs
  - [ ] per-subject split metadata exists for every selected subject

- [ ] 3. Run true single-subject fitting for the first age group panel

  **What to do**: Fit the current `VGG + WW` pipeline on each selected `20-29` subject using only that subject’s own train split, then evaluate on that subject’s own held-out split. Write per-subject config, params, and eval summaries.
  **Must NOT do**: Do not reduce this to scale sweep only. Do not reuse another subject’s fit.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_true_single_subject_feasibility.py --mode fit --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility` exits 0
  - [ ] for each selected `20-29` subject, `best_config.json`, `best_model_params.npz`, and `subject_eval_summary.json` exist

- [ ] 4. Run true single-subject fitting for the second age group panel

  **What to do**: Mirror Task 3 for the selected `80-89` subjects using the same fitting and evaluation contract.
  **Must NOT do**: Do not introduce age-group-specific rescue logic beyond normal subject-specific training.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/run_true_single_subject_feasibility.py --mode fit --age_group 80-89 --output_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility` exits 0
  - [ ] for each selected `80-89` subject, `best_config.json`, `best_model_params.npz`, and `subject_eval_summary.json` exist

- [ ] 5. Build the cross-subject feasibility scorecard and binary framework verdict

  **What to do**: Reaggregate all per-subject eval summaries into one feasibility scorecard. The analysis must explicitly answer whether the framework shows credible subject-level learnability or whether it repeatedly fails/degenerates across the panel. In addition to the existing behavior metrics, the scorecard must make the right-skew and error-timing questions explicit.
  **Must NOT do**: Do not hide failed subjects. Do not reduce the final decision to a single lucky subject.

  **Recommended Agent Profile**:
  - Category: `deep`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] `python code/scripts/analyze_true_single_subject_feasibility.py --input_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility --output_root artifacts/results/repro_legacy_interim/true_single_subject_feasibility` exits 0
  - [ ] `reaggregated/feasibility_scorecard.csv` exists
  - [ ] `reaggregated/feasibility_verdict.json` exists
  - [ ] scorecard includes explicit columns for subject-level RT skew and error-vs-correct RT ordering / slow-error direction
  - [ ] verdict is one of:
    - `VGGWW-SINGLE-SUBJECT-FEASIBLE`
    - `VGGWW-SINGLE-SUBJECT-NOT-FEASIBLE`

- [ ] 6. Write the framework-retention summary memo

  **What to do**: Write a concise technical memo that states:
  - what “true single-subject” meant in this workflow
  - what subject panel was used
  - whether fits were stable and non-degenerate
  - whether held-out subject behavior was captured credibly
  - whether the subject-level fits improved right-skew / heavy-tail structure
  - whether the subject-level fits recovered the subject’s own error-time direction (including slow-error if present)
  - whether `VGG + WW` should be kept or deprioritized
  **Must NOT do**: Do not over-claim. Do not mask panel-wide failure with isolated anecdotes.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: `[]`

  **Acceptance Criteria**:
  - [ ] full workflow command exits 0
  - [ ] `true_single_subject_feasibility_summary.md` exists
  - [ ] summary contains:
    - `true single-subject`
    - `VGG + WW`
    - `HETEROGENEITY-NOT-SUPPORTED`
    - `CAPTURE-PROBE-NOT-SUPPORTED`
    - `VGGWW-SINGLE-SUBJECT-`

## Final Verification Wave (MANDATORY)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit approval before completion.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Success Criteria
- The workflow runs end-to-end for a bounded subject panel covering both age groups.
- Each subject fit is truly learned from that subject’s own data rather than from group parameters plus a tiny tweak.
- The analysis produces a panel-wide answer to the real framework question: whether `VGG + WW` is viable at the single-subject level.
- The final memo makes it explicit whether we should continue investing in `VGG + WW` or consider dropping it.
