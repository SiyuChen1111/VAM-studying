# Minimal Single-Subject Simulation on `dynamic_selection_phase1`

## TL;DR
> **Summary**: Keep `dynamic_selection_phase1` fixed as the only mechanism, move evaluation from group-level to a tightly bounded single-subject diagnostic workflow, and test whether subject heterogeneity naturally improves the current flanker mechanism failures without adding new dynamics.
> **Deliverables**:
> - alignment-safe subject manifest for both age groups
> - single-subject forward-simulation runner with subject-varying `scale` only
> - reaggregation + mechanism-score analysis outputs
> - summary memo with heterogeneity verdict
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3/4/5 → Task 6 → Task 7

## Context
### Original Request
- Keep `dynamic_selection_phase1` as the active branch.
- Stop the DMC-like extension line.
- Switch the next mechanism line to minimal single-subject simulation.
- Keep the model simple: no new readout, no urgency, no new DMC parameters, no complex per-subject fitting.
- First pass should use 6-10 subjects across both age groups, test-only filtering, and subject-varying `scale` only.

### Interview Summary
- Subject batch size is fixed at **8 total subjects** by default: **4 from `20-29` and 4 from `80-89`**.
- Selection uses **hybrid strata** within each age group, using:
  - earliest incongruent CAF
  - incongruent error-minus-correct RT
  - RT skewness
- Selection is diagnostic, not confirmatory: subjects are chosen from test data to expose heterogeneity structure.
- `scale` is the only subject-varying parameter.
- `scale` search uses an **age-group median center + small bounded grid**, not continuous optimization.
- Artifacts must go to a **new dedicated single-subject results tree**.
- Success bar: the reaggregated single-subject simulation must improve **at least 2 of 4** target mechanisms, and one of those improvements must be either **earliest incongruent CAF** or **incongruent error-minus-correct RT direction**.

### Metis Review (gaps addressed)
- Added an explicit **CSV↔NPZ alignment verification** step before any subject mask is applied.
- Locked this as a **diagnostic heterogeneity probe**, not a new modeling branch.
- Fixed comparator ambiguity by requiring both:
  - contextual comparison to the full age-group aggregate
  - primary success comparison against the selected-subject human aggregate
- Added explicit subject eligibility thresholds and exclusion logging for sparse-error cases.
- Locked reaggregation to a single weighting rule: **trial-weighted reaggregation**.
- Locked `scale` freedom to a bounded grid around a data-derived center; no optimizer-based subject calibration.

## Work Objectives
### Core Objective
Build the smallest executable single-subject forward-simulation workflow that reuses the existing `dynamic_selection_phase1` machinery and determines whether subject heterogeneity, rather than a more complex conflict mechanism, explains the remaining mismatch in CAF / delta / conditional-error structure.

### Deliverables
- `code/scripts/run_dynamic_selection_single_subject.py`
- `code/scripts/analyze_dynamic_selection_single_subject.py`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/subject_manifest.csv`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/selection_summary.csv`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/<age_group>/user_<id>/predictions.npz`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/<age_group>/user_<id>/summary.json`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/reaggregated/reaggregated_metrics.csv`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/reaggregated/success_bar.json`
- `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/heterogeneity_probe_summary.md`

### Definition of Done (verifiable conditions with commands)
- A single command generates the subject manifest, per-subject predictions, and per-subject summaries for exactly 8 selected subjects across both age groups.
- A single analysis command generates per-subject metrics, reaggregated metrics, success-bar scoring, and the final summary memo.
- All subject simulations preserve the same non-`scale` parameters as the active `dynamic_selection_phase1` baseline.
- Subject-specific `scale` values stay within the bounded grid defined in this plan.
- Sparse-error subjects are excluded with explicit reasons written to disk.
- Alignment mismatch between CSV rows and cached NPZ rows aborts the run with a clear error.

### Must Have
- Reuse `evaluate_cached_stage2_params()` for forward simulation.
- Reuse existing CAF / delta / conditional-error / tail utilities from `code/scripts/analyze_20_29_urgency_tie.py`.
- Support both age groups in the same workflow.
- Subject selection is deterministic and written to disk.
- Reaggregation is trial-weighted and explicitly labeled.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No new Stage-2 mechanism.
- No urgency branch revival.
- No new DMC-like parameters.
- No continuous per-subject optimization.
- No full per-subject fitting.
- No manual plot inspection as the primary pass/fail criterion.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: **tests-after** using existing pytest plus command-line artifact verification
- QA policy: Every task includes a happy path and a failure / edge-case scenario
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: baseline audit + alignment guard + subject-selection metrics + `20-29` runner foundation
Wave 2: `80-89` runner + reaggregation analysis + summary memo / command polish

### Dependency Matrix (full, all tasks)
| Task | Depends On | Blocks |
|---|---|---|
| 1 | none | 2, 3, 4, 5 |
| 2 | 1 | 4, 5, 6 |
| 3 | 1 | 4, 5, 6 |
| 4 | 1, 2, 3 | 6 |
| 5 | 1, 2, 3 | 6 |
| 6 | 2, 3, 4, 5 | 7 |
| 7 | 6 | F1-F4 |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → quick / unspecified-high / deep
- Wave 2 → 3 tasks → quick / unspecified-high / writing

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Audit active baseline artifacts and lock invariants

  **What to do**: Verify the active `dynamic_selection_phase1` baseline inputs for both `20-29` and `80-89`, locate the canonical cached NPZ + CSV pairs, and write a machine-readable baseline manifest that locks every non-`scale` parameter used by the single-subject workflow. If `80-89` is missing a directly reusable phase-1 artifact, materialize an equivalent baseline artifact using the same active mechanism settings before any single-subject work proceeds.
  **Must NOT do**: Do not alter any Stage-2 dynamics, do not introduce age-group-specific mechanism changes, and do not start subject filtering before the baseline manifest is written.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: bounded repo audit + manifest writing
  - Skills: `[]` - existing repo patterns are sufficient
  - Omitted: `['codebase-audit']` - broader architecture review is unnecessary here

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2, 3, 4, 5 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/run_20_29_dynamic_selection_smoke.py:65-79,157-198` - existing cached-eval orchestration and result writing pattern for `dynamic_selection_phase1`
  - API/Type: `code/scripts/train_age_groups_efficient.py:1172-1210` - `evaluate_cached_stage2_params()` accepts cached arrays + params for pure forward evaluation
  - Pattern: `code/scripts/vgg_wongwang_lim.py:40-112` - active dynamic-selection mechanism lives here; keep it unchanged
  - Data: `code/scripts/prepare_age_group_data.py:41-45,79-90` - prepared CSVs preserve `user_id` and age-group identity
  - Test: `tests/test_dynamic_selection_phase1.py` - regression coverage for dynamic-selection invariants

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode audit-baseline --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/baseline_manifest.json` exists
  - [ ] `python - <<'PY'
import json, pathlib
path = pathlib.Path('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/baseline_manifest.json')
data = json.loads(path.read_text())
assert set(data['age_groups']) == {'20-29','80-89'}
for age in data['age_groups'].values():
    assert 'non_scale_params' in age
    assert 'cache_npz' in age and 'csv_path' in age
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Baseline audit succeeds for both age groups
    Tool: Bash
    Steps: Run `python code/scripts/run_dynamic_selection_single_subject.py --mode audit-baseline --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Baseline manifest is written and contains both `20-29` and `80-89` entries with identical non-scale config field names.
    Evidence: .sisyphus/evidence/task-1-baseline-audit.txt

  Scenario: Missing age-group baseline aborts clearly
    Tool: Bash
    Steps: Run the same command against a temporary config that omits one age group baseline path.
    Expected: Process exits non-zero and writes a message naming the missing age group and missing artifact path.
    Evidence: .sisyphus/evidence/task-1-baseline-audit-error.txt
  ```

  **Commit**: NO | Message: `feat(sim): audit phase1 baselines` | Files: `code/scripts/run_dynamic_selection_single_subject.py`, `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/*`

- [ ] 2. Build alignment-safe CSV↔cache subject filter layer

  **What to do**: Add a thin loader/wrapper that reads the prepared test CSV, verifies exact row alignment against the cached NPZ inputs, preserves `user_id`, and applies deterministic row masks to produce subject-filtered cached dictionaries for downstream forward simulation. Alignment verification must fail closed on row-count mismatch or identity/order mismatch.
  **Must NOT do**: Do not rely on length-only validation, do not silently reorder rows, and do not duplicate the Stage-2 forward evaluator.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: high-precision data alignment logic with failure handling
  - Skills: `[]` - repo-local pattern reuse is enough
  - Omitted: `['refactor']` - this is a narrow wrapper, not a broad refactor

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 6 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/train_age_groups_efficient.py:1000-1026` - current cache validation only checks lengths; this task must tighten it
  - Pattern: `code/scripts/train_age_groups_efficient.py:113-122` - `attach_flanker_labels_from_csv` assumes row-order identity; preserve and harden that assumption
  - API/Type: `code/scripts/train_age_groups_efficient.py:985-997` - cached array key expectations
  - Data: `code/scripts/prepare_age_group_data.py` - source of `user_id` in prepared CSVs
  - Test: `tests/test_dynamic_selection_phase1.py` - extend test style to cover alignment-safe CSV enrichment

  **Acceptance Criteria** (agent-executable only):
  - [ ] `pytest tests/test_dynamic_selection_phase1.py -q` exits 0
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode verify-alignment --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
import pandas as pd
df = pd.read_csv('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/alignment_check_20-29.csv')
assert {'user_id','row_index','alignment_ok'}.issubset(df.columns)
assert df['alignment_ok'].all()
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Alignment-safe filtering preserves exact rows
    Tool: Bash
    Steps: Run `python code/scripts/run_dynamic_selection_single_subject.py --mode verify-alignment --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Alignment report is written with all `alignment_ok == True` and no row-count drift.
    Evidence: .sisyphus/evidence/task-2-alignment-check.txt

  Scenario: Row-order mismatch aborts with a clear error
    Tool: Bash
    Steps: Run the alignment mode against a deliberately permuted temporary CSV fixture.
    Expected: Process exits non-zero and reports `CSV_NPZ_ALIGNMENT_MISMATCH` with the first offending row index.
    Evidence: .sisyphus/evidence/task-2-alignment-check-error.txt
  ```

  **Commit**: YES | Message: `feat(sim): add alignment-safe cached subject filtering` | Files: `code/scripts/run_dynamic_selection_single_subject.py`, `tests/test_dynamic_selection_phase1.py`

- [ ] 3. Compute deterministic subject-selection manifest

  **What to do**: Compute subject-level diagnostic metrics from the test CSV for both age groups, apply eligibility thresholds, rank subjects within each age group using the locked hybrid strata, and write a deterministic manifest with 4 selected subjects per age group. Use these exact rules:
  - eligibility: at least 20 incongruent trials, at least 3 incongruent error trials, and at least 10 unique RT values in the test split
  - metrics: earliest incongruent CAF, incongruent error-minus-correct RT, RT skewness
  - standardize each metric within age group using z-scores after dropping ineligible subjects
  - compute `extreme_score = mean(abs(z_metric))`
  - select 4 subjects per age group by taking the top 4 `extreme_score` values while enforcing at least one subject from the lower half and one from the upper half of earliest incongruent CAF rank; break ties by higher incongruent error count, then lexicographic `user_id`
  - write excluded subjects with explicit exclusion reasons
  **Must NOT do**: Do not hand-pick subjects, do not use pooled cross-age ranking, and do not use any model-fit quantity in selection.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: deterministic metric extraction and manifest writing
  - Skills: `[]` - no external domain skill needed
  - Omitted: `['statistical-analysis']` - no inferential statistics are needed

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 6 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/analyze_20_29_urgency_tie.py:119-215` - reuse CAF / delta / error-RT / tail-summary utility style
  - Data: `code/scripts/prepare_age_group_data.py` - prepared CSV subject identity fields
  - Pattern: `code/scripts/create_matched_20_29_control_branch.py` - precedent for `user_id`-based filtering

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode select-subjects --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
import pandas as pd
df = pd.read_csv('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/subject_manifest.csv')
assert len(df) == 8
assert set(df['age_group']) == {'20-29','80-89'}
assert (df.groupby('age_group').size() == 4).all()
assert {'user_id','extreme_score','earliest_incongruent_caf','incongruent_error_minus_correct_rt','rt_skewness'}.issubset(df.columns)
PY` exits 0
  - [ ] `python - <<'PY'
import pandas as pd
df = pd.read_csv('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/excluded_subjects.csv')
assert {'user_id','age_group','reason'}.issubset(df.columns)
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Deterministic hybrid-strata manifest is produced
    Tool: Bash
    Steps: Run `python code/scripts/run_dynamic_selection_single_subject.py --mode select-subjects --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Exactly 8 selected subjects are written, split 4/4 by age group, with metric columns and deterministic tie-break behavior.
    Evidence: .sisyphus/evidence/task-3-subject-selection.txt

  Scenario: Sparse-error subject is excluded and logged
    Tool: Bash
    Steps: Run the same mode against a fixture where one candidate has fewer than 3 incongruent error trials.
    Expected: The subject is absent from `subject_manifest.csv` and appears in `excluded_subjects.csv` with reason `INSUFFICIENT_INCONGRUENT_ERRORS`.
    Evidence: .sisyphus/evidence/task-3-subject-selection-error.txt
  ```

  **Commit**: NO | Message: `feat(sim): add deterministic subject manifest generation` | Files: `code/scripts/run_dynamic_selection_single_subject.py`, `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/*`

- [ ] 4. Implement `20-29` single-subject forward simulation runner

  **What to do**: Reuse the active `dynamic_selection_phase1` baseline for `20-29`, derive the age-group `scale` center as `0.1 * median(subject_median_rt / age_group_median_rt)` over eligible `20-29` test subjects, construct a bounded 5-point grid `[center-0.02, center-0.01, center, center+0.01, center+0.02]` clipped to `[0.05, 0.15]`, run forward simulation for each selected subject across that grid, and keep the best `scale` by minimizing absolute error on the subject’s incongruent error-minus-correct RT; tie-break by lower absolute error on earliest incongruent CAF, then nearest-to-center `scale`.
  **Must NOT do**: Do not vary any non-`scale` parameter, do not optimize continuously, and do not change the stage-2 mechanism or readout.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: forward-simulation wiring must be precise and constraint-preserving
  - Skills: `[]` - existing repo APIs are sufficient
  - Omitted: `['pua']` - escalation is unnecessary for this bounded task

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6 | Blocked By: 1, 2, 3

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/run_20_29_dynamic_selection_smoke.py:157-198` - cached eval + summary writing pattern to clone
  - API/Type: `code/scripts/train_age_groups_efficient.py:1172-1210` - core forward evaluator to reuse
  - Pattern: `code/scripts/vgg_wongwang_lim.py:40-112` - preserve exact mechanism path; no changes beyond existing args
  - Output Convention: `code/scripts/analyze_20_29_dynamic_selection_smoke.py` - naming for predictions and summaries

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode simulate --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
import json, pathlib, pandas as pd
manifest = pd.read_csv('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/subject_manifest.csv')
for uid in manifest.loc[manifest.age_group=='20-29','user_id']:
    root = pathlib.Path(f'artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/20-29/user_{uid}')
    assert (root/'predictions.npz').exists()
    data = json.loads((root/'summary.json').read_text())
    assert data['age_group'] == '20-29'
    assert 0.05 <= data['selected_scale'] <= 0.15
    assert data['grid_size'] == 5
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: 20-29 subject simulations complete with bounded scale search
    Tool: Bash
    Steps: Run `python code/scripts/run_dynamic_selection_single_subject.py --mode simulate --age_group 20-29 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Each selected `20-29` subject receives `predictions.npz` and `summary.json`; all selected scales are inside the clipped 5-point grid.
    Evidence: .sisyphus/evidence/task-4-simulate-20-29.txt

  Scenario: Boundary-hit scale is flagged
    Tool: Bash
    Steps: Run the same mode against a fixture where the best `scale` lands on the grid boundary.
    Expected: `summary.json` includes `boundary_hit: true` and the run-level log counts that subject under `boundary_hits`.
    Evidence: .sisyphus/evidence/task-4-simulate-20-29-error.txt
  ```

  **Commit**: YES | Message: `feat(sim): add 20-29 single-subject forward simulation` | Files: `code/scripts/run_dynamic_selection_single_subject.py`

- [ ] 5. Implement `80-89` single-subject forward simulation runner

  **What to do**: Mirror Task 4 for `80-89` using the same mechanism, the same bounded-grid logic, and the same tie-break rules, but derive the `80-89` age-group median center from eligible `80-89` test subjects. Ensure output structure exactly parallels the `20-29` tree.
  **Must NOT do**: Do not introduce age-group-specific modeling rules beyond the age-group-specific median center and baseline artifact paths.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: same precision needs as Task 4
  - Skills: `[]` - no extra skill required
  - Omitted: `['codebase-audit']` - unnecessary breadth

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6 | Blocked By: 1, 2, 3

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/run_20_29_dynamic_selection_smoke.py` - baseline runner shape to generalize without mechanism drift
  - API/Type: `code/scripts/train_age_groups_efficient.py:1172-1210` - same forward evaluator as Task 4
  - Data: baseline manifest written in Task 1 - source of `80-89` baseline artifact paths

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode simulate --age_group 80-89 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
import json, pathlib, pandas as pd
manifest = pd.read_csv('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/manifest/subject_manifest.csv')
for uid in manifest.loc[manifest.age_group=='80-89','user_id']:
    root = pathlib.Path(f'artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/80-89/user_{uid}')
    assert (root/'predictions.npz').exists()
    data = json.loads((root/'summary.json').read_text())
    assert data['age_group'] == '80-89'
    assert 0.05 <= data['selected_scale'] <= 0.15
    assert data['grid_size'] == 5
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: 80-89 subject simulations complete with mirrored output structure
    Tool: Bash
    Steps: Run `python code/scripts/run_dynamic_selection_single_subject.py --mode simulate --age_group 80-89 --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Each selected `80-89` subject receives the same artifact schema as `20-29`, with bounded `scale` values and no mechanism drift.
    Evidence: .sisyphus/evidence/task-5-simulate-80-89.txt

  Scenario: Missing 80-89 baseline artifact aborts clearly
    Tool: Bash
    Steps: Run the same mode against a temporary manifest that points the `80-89` baseline to a nonexistent file.
    Expected: Process exits non-zero and reports the missing baseline artifact path for `80-89`.
    Evidence: .sisyphus/evidence/task-5-simulate-80-89-error.txt
  ```

  **Commit**: NO | Message: `feat(sim): add 80-89 single-subject forward simulation` | Files: `code/scripts/run_dynamic_selection_single_subject.py`

- [ ] 6. Build reaggregation, mechanism scoring, and success-bar analysis

  **What to do**: Implement analysis that reads all selected-subject prediction outputs, reconstructs trial-level data frames, computes per-subject and reaggregated metrics using the existing CAF / delta / conditional-error / tail utilities, and writes both contextual and primary comparisons:
  - contextual: full age-group human aggregate vs full age-group baseline simulation vs reaggregated selected-subject simulation
  - primary: selected-subject human aggregate vs reaggregated selected-subject simulation
  Lock the 4 target mechanisms to:
  1. earliest incongruent CAF accuracy
  2. first delta quantile
  3. incongruent error-minus-correct RT
  4. incongruent conditional tail summary
  Define improvement as a reduction in absolute distance to the selected-subject human aggregate. Mark success only if at least 2 metrics improve and one is metric 1 or 3. Write `success_bar.json` with per-metric before/after distances and final verdict.
  **Must NOT do**: Do not use visual judgment as a pass/fail rule, and do not switch aggregation weighting away from trial-weighted reaggregation.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this task locks the scientific interpretation and score computation
  - Skills: `[]` - repo-local analysis utilities are enough
  - Omitted: `['statistical-analysis']` - no inferential test is required

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7 | Blocked By: 2, 3, 4, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/analyze_20_29_dynamic_selection_smoke.py:56-77,161-192` - existing smoke-analysis reuse pattern
  - API/Type: `code/scripts/analyze_20_29_urgency_tie.py:119-215` - canonical metric utility implementations
  - Output Convention: `caf_congruent.csv`, `caf_incongruent.csv`, `delta_plot.csv`, `conditional_error_rt.csv`, `conditional_tail_summary.csv`

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/analyze_dynamic_selection_single_subject.py --input_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
import json, pathlib, pandas as pd
root = pathlib.Path('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/reaggregated')
assert (root/'reaggregated_metrics.csv').exists()
assert (root/'success_bar.json').exists()
data = json.loads((root/'success_bar.json').read_text())
assert data['weighting'] == 'trial_weighted'
assert data['target_metrics'] == ['earliest_incongruent_caf','first_delta','incongruent_error_minus_correct_rt','incongruent_conditional_tail']
assert data['verdict'] in {'HETEROGENEITY-SUPPORTED','HETEROGENEITY-NOT-SUPPORTED'}
assert len(data['metric_deltas']) == 4
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Reaggregation and success-bar scoring complete end-to-end
    Tool: Bash
    Steps: Run `python code/scripts/analyze_dynamic_selection_single_subject.py --input_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject`
    Expected: Reaggregated CSVs, the locked metric files, and `success_bar.json` are written; verdict is one of the two allowed labels.
    Evidence: .sisyphus/evidence/task-6-reaggregate.txt

  Scenario: Subject with undefined tail metric is excluded from tail scoring but logged
    Tool: Bash
    Steps: Run analysis on a fixture where one simulated subject has no valid incongruent error tail.
    Expected: Analysis completes, logs the exclusion in `metric_exclusions.csv`, and does not silently inject NaNs into `success_bar.json`.
    Evidence: .sisyphus/evidence/task-6-reaggregate-error.txt
  ```

  **Commit**: YES | Message: `feat(sim): add single-subject reaggregation and scoring` | Files: `code/scripts/analyze_dynamic_selection_single_subject.py`, `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/reaggregated/*`

- [ ] 7. Write the heterogeneity probe summary and execution wrappers

  **What to do**: Add final command wrappers / README-level notes for the single-subject workflow, write `heterogeneity_probe_summary.md`, and ensure the summary explicitly states: selection rules, exclusion counts, age-group centers, bounded `scale` grids, reaggregation weighting, metric deltas, and final verdict. The summary must explicitly label this run as a diagnostic heterogeneity probe rather than confirmation of a new mechanism.
  **Must NOT do**: Do not change scientific conclusions beyond what `success_bar.json` encodes, and do not present exploratory evidence as confirmatory.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: summary clarity and execution guidance
  - Skills: `[]` - the content is repo-specific
  - Omitted: `['doc-coauthoring']` - this is a bounded technical memo, not a broad documentation project

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: F1-F4 | Blocked By: 6

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/scripts/analyze_20_29_dynamic_selection_smoke.py` - summary-table / memo pattern
  - Pattern: `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md` - house style for concise technical handoff memos
  - Data: `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/reaggregated/success_bar.json` - sole source of final verdict

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python code/scripts/run_dynamic_selection_single_subject.py --mode full --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject && python code/scripts/analyze_dynamic_selection_single_subject.py --input_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject --output_root artifacts/results/repro_legacy_interim/dynamic_selection_single_subject` exits 0
  - [ ] `python - <<'PY'
from pathlib import Path
text = Path('artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/heterogeneity_probe_summary.md').read_text()
required = [
    'diagnostic heterogeneity probe',
    'trial-weighted',
    'earliest incongruent CAF',
    'incongruent error-minus-correct RT',
    'RT skewness',
    'HETEROGENEITY-'
]
for item in required:
    assert item in text
PY` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Full workflow produces summary memo with locked language
    Tool: Bash
    Steps: Run the full simulation + analysis commands end-to-end.
    Expected: `heterogeneity_probe_summary.md` is written and includes the diagnostic label, selection rules, scale-grid description, metric deltas, and final verdict.
    Evidence: .sisyphus/evidence/task-7-summary.txt

  Scenario: Missing success-bar file aborts summary generation
    Tool: Bash
    Steps: Run summary generation with `success_bar.json` removed from a temporary output root.
    Expected: Process exits non-zero and reports that summary generation requires `success_bar.json`.
    Evidence: .sisyphus/evidence/task-7-summary-error.txt
  ```

  **Commit**: NO | Message: `docs(sim): add heterogeneity probe summary` | Files: `code/scripts/run_dynamic_selection_single_subject.py`, `code/scripts/analyze_dynamic_selection_single_subject.py`, `artifacts/results/repro_legacy_interim/dynamic_selection_single_subject/heterogeneity_probe_summary.md`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1 after Wave 1 foundation work: `feat(sim): add single-subject selection and alignment-safe runner foundation`
- Commit 2 after Wave 2 analysis outputs and summary: `feat(sim): add single-subject reaggregation and heterogeneity diagnostics`
- Do not commit generated result artifacts unless the repository already tracks that output class.

## Success Criteria
- The workflow runs end-to-end for both age groups without touching the DMC-like branch.
- Selected-subject simulation outputs are reproducible from deterministic manifests.
- The summary memo states one of two explicit verdicts only:
  - `HETEROGENEITY-SUPPORTED`
  - `HETEROGENEITY-NOT-SUPPORTED`
- The verdict is derived from the locked 2-of-4 success rule, with at least one improvement in earliest incongruent CAF or incongruent error-minus-correct RT direction.
