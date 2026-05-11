# Current status / handoff note

**Updated:** 2026-05-07

Historical references to `results/...`, `checkpoints_age_groups*`, and `data_age_groups*` in older memos remain valid through compatibility symlinks. The canonical locations now live under `artifacts/results/`, `artifacts/checkpoints/`, and `data/`.

## Main research direction

We are studying age-related differences in LIM / Flanker decision behavior using:
- shared Stage 1 visual evidence
- age-specific Stage 2 Wong-Wang decision dynamics

The current scientific focus is on:
- RT distributions
- skewness / heavy tails
- congruency effects
- error-slower patterns
- whether age differences are reflected in decision-dynamics behavior and geometry

## Critical correction already made

Stage 2 choice supervision was changed from:
- `target_labels`

to:
- `response_labels`

This matters because the previous training setup pushed the model toward a more ideal-observer regime.

## Current age branches in use

### Young branch
- active comparison branch: `20-29 matched`
- canonical data root: `data/age_groups_matched/20-29`
- canonical output root: `artifacts/checkpoints/age_groups_matched/20-29/stage2`

### Old branch
- branch: `80-89`
- canonical data root: `data/age_groups/80-89`
- canonical output root: `artifacts/checkpoints/age_groups/80-89/stage2`

## What is currently trustworthy

### Strongest current evidence
- Human-side behavior figures in `organized/proposal_aligned_human_behavior/`
- Current-best response-supervision summaries in `organized/current_best_response_supervision/`

### Current bottom line
- The VGG + Wong-Wang line can be pushed into a measurable error regime.
- That regime is still not human-like enough in RT scale, tail behavior, and error-vs-correct RT direction.
- The project is therefore in consolidation mode rather than open-ended local WW patching.

### Current best Phase 18 branch-local result

- The strongest retained Phase 18 artifact is now `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.png`.
- It summarizes the best negative-ΔRT branch-local result from `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`, matched to `epoch 10`.
- That checkpoint is still scientifically important because it is the first **retained/reported branch-local result we currently surface** with negative error-minus-correct ΔRT under the DMC + variational-evidence → Wong-Wang combination.
- Key retained numbers for that epoch are:
  - `beh_opt = 0.7738`
  - raw exported `pred_mean = 0.4531s`
  - `t0_seconds = 0.25`
  - report-style display mean `(+t0) = 0.7031s`
  - `error_minus_correct_rt = -0.0176s`
- Interpret this branch as a **Phase 18 breakthrough but not a promoted final solution**: it shows the mechanism can produce fast errors, but the magnitude and overall RT regime are still not human-like enough to replace the later repo-wide negative-result constraints.

### Latest successor-screening verdict

The bounded successor-screening bundle under:

- `artifacts/results/rt_model_next_step/`

has already been executed to completion. Its authoritative synthesis memo is:

- `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`

That bundle reached the explicit final verdict:

- `NO_SUCCESSOR_BRANCH_CLEARED_GATES`

Meaning:

- `WW+t0` did not salvage the current WW line,
- RTNet-lite on cached logits collapsed RT center and failed the smoke gate,
- cached-logit LBA was mechanistically informative but still failed the locked promotion gates,
- no branch was promoted to full-panel or aligned single-subject successor validation.

### Latest HSFA repair verdict

An additional bounded HSFA repair bundle now exists under:

- `artifacts/results/rt_model_hsfa_v3_1/`

This bundle matters because it resolved the main ambiguity left by the earlier HSFA smoke branch under `artifacts/results/rt_model_hsfa_v3/`:

- `subject_mode='none'` is now a true forward-pass switch,
- feature-driven immediate stopping was blocked with hard minimum-step stopping,
- feature inputs were normalized and gated,
- calibrated-error objectives and seeded stochastic readout were added,
- and the repaired branch was re-evaluated on the locked aggregate smoke surface.

Its authoritative outputs are:

- `artifacts/results/rt_model_hsfa_v3_1/04_smoke_decision/hsfa_v3_1_smoke_decision_memo.md`
- `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`

That bundle reached the explicit repair verdict:

- `HSFA_V3_1_KILL_NEED_STAGE1_UNCERTAINTY_OR_BNN`

with final synthesis token:

- `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN`

Meaning:

- the original HSFA-v3 no-promotion result was not just a hidden implementation bug,
- the repaired feature-gated branches fixed RT-center collapse and early stopping,
- but every repaired variant still failed the locked aggregate promotion gates because response agreement and accuracy calibration remained too far from human,
- so the subject panel was correctly skipped.

### Active SPEA v1.1 calibration follow-up

There is now a live bounded calibration branch under:

- `artifacts/results/rt_model_semisup_spea_v1_1_calibration/`

This branch should be read as:

- a calibration-focused follow-up on the earlier SemiSupervisedSPEA partial result,
- not a new architecture-search family,
- and not an already-resolved scientific verdict.

Current completed work on this branch:

- protocol lock written to `00_protocol/spea_v1_1_calibration_protocol.md`
- calibrated trainer added at `code/scripts/train_age_group_semisup_spea_calibrated.py`
- explicit stop-time readout modes added in `code/scripts/stage2_spea_backend.py`
- targeted regression coverage added at `tests/test_spea_stop_time_coupling.py`
- replay artifact root created at `01_replay_prior/c0_v2_replay/`

Current interpretation:

- this is implementation-scaffolding progress only
- no new aggregate smoke or subject-panel verdict exists yet for SPEA v1.1
- do not summarize this branch as promoted, partial, or killed until its own decision artifacts exist

## Verified single-subject bundles from the latest session

The following result roots are now the most important verified artifacts for current single-subject work:

- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/`
  - clean WW single-subject RT+response-only workflow
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only_noise05/`
  - same WW single-subject workflow with `fixed_noise_ampa = 0.05`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/`
  - panel-level clean-vs-noise comparison CSVs, summaries, and interpretation notes for both `20-29` and `80-89`
- `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_accumrnn_aligned/`
  - AccumRNN single-subject workflow aligned to the same bounded panel / persisted subsets as the verified WW root
- `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/`
  - aligned WW vs AccumRNN per-subject export comparison bundle

### Current single-subject research judgment

- WW remains the more behaviorally coherent baseline.
- Increasing WW noise can induce an errorful regime, but the resulting error RT direction is still not human-like.
- The aligned single-subject comparison does not yet justify replacing WW with the current AccumRNN implementation.

Canonical narrative summary:
- `artifacts/results/organized/handoff/single_subject_rt_response_research_judgment_memo.md`

### Useful but lower-tier evidence
- `organized/legacy_interim_reference/figureA4_interim_trajectory_geometry.png`
  - This is a geometry preview from the earlier interim path
  - It should not be treated as the final corrected-supervision mechanism figure

## What is still missing

To produce a final corrected-supervision model comparison, we still need clean saved best-so-far model outputs that are trustworthy for:
- model RT distribution plots
- model skew / error-slower plots
- updated trajectory geometry / updated A4-like figure

What is **not** still missing inside the bounded successor screen:

- there is no pending promoted-branch run left to execute under `artifacts/results/rt_model_next_step/`
- the saved result of that plan is already a finished negative-result bundle
- the saved result of `artifacts/results/rt_model_hsfa_v3_1/` is also a finished negative-result bundle
- any further successor-branch work should start from a **new planning phase**, not by reopening the closed routing chain or the closed HSFA repair loop

## Immediate purpose of the organized folders

The `organized/` tree exists so that a later agent can quickly tell:
- which outputs are current
- which outputs are human-only
- which outputs are frozen summaries
- which outputs are older interim references

## Suggested restart point for a future agent

1. Read `artifacts/results/organized/README.md`
2. Read `artifacts/results/organized/handoff/HANDOFF_INDEX.md`
3. Read `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
4. Read `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
5. Read `artifacts/results/organized/handoff/supervisor_update_2026-04-08.md`
6. Read `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md`
7. Inspect `artifacts/results/organized/current_best_response_supervision/response_supervision_current_comparison.csv`
8. Treat `artifacts/results/organized/legacy_interim_reference/figureA4_interim_trajectory_geometry.png` as a preview, not a final result
9. If the task is about SPEA calibration follow-up, read `artifacts/results/rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md` and `.sisyphus/plans/semisup_spea_v1_1_calibration_agent_plan.md`

## File structure cleanup completed

A safe non-destructive file structure cleanup has been performed:

### What was done
- Verified the organized layer around 4 primary sections:
  - `organized/current_best_response_supervision/`
  - `organized/proposal_aligned_human_behavior/`
  - `organized/legacy_interim_reference/`
  - `organized/handoff/`
- Archived supplemental convenience mirrors under `organized/archive/` so they no longer compete with the main navigation:
  - `organized/archive/age_groups_raw/`
  - `organized/archive/standalone_figures/`
- Archived superseded response-supervision source snapshots under `artifacts/results/age_groups_response_supervision_interim/archive/`:
  - `figureRS1_response_supervision_eval05_summary.png`
  - `response_supervision_eval05_comparison.csv`
  - `figureRS3_agegroup_human_vs_model.png`
- Removed `.DS_Store` from `artifacts/results/organized/`
- Updated `organized/README.md` with the current folder map, evidence levels, archive note, and best entry points
- Updated `organized/FILE_MAPPING.md` with the current keep/copy/archive mapping and safe archive guidance

### Current organized structure
```
organized/
├── README.md                          # Master guide (updated)
├── FILE_MAPPING.md                    # Complete file-by-file mapping
├── archive/                           # Archived convenience mirrors
├── proposal_aligned_human_behavior/   # HUMAN-ONLY (highest trust)
├── current_best_response_supervision/ # CURRENT-BEST MODEL (medium trust)
├── legacy_interim_reference/          # LEGACY (reference only)
└── handoff/                           # This file
```

Archive contents:
```
organized/archive/
├── age_groups_raw/                    # Older organized mirror
└── standalone_figures/                # Superseded convenience figures
```

### What remains untouched
- All original source directories under `artifacts/results/` remain intact for script compatibility
- Original result files inside the primary organized folders remain in place
- Root-level compatibility aliases like `results/` remain outside this note’s cleanup scope unless a later symlink-removal pass is requested
