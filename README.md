# Flaner-rt-distribution-modeling

This repository studies **human reaction-time (RT) behavior** in the LIM / Flanker task using a **VGG-based visual front-end** and **decision-dynamics models**.

The main modeling line is:

- shared Stage-1 visual evidence model
- age-specific Stage-2 Wong-Wang (WW) decision dynamics
- group-level comparisons between `20-29` and `80-89`

The broader scientific goal is to understand whether the model can reproduce key human RT-distribution signatures, including:

- plausible RT scale
- right-skew / heavy tails
- congruency effects
- error-conditioned RT structure
- age-related differences in decision dynamics

---

## Current status

The project has already explored:

1. initial VGG drift-rate prototyping
2. formal VGG + WW age-group modeling
3. response-supervision correction
4. readout redesign experiments
5. behavior-balanced selector / eval redesign
6. lightweight WW objective tweaks
7. structured accumulator-RNN prototype
8. RT-distribution-shape losses
9. WW error-regime experiments

### Current bottom line

The strongest current result is that the WW line can be pushed into an **error regime**, but the resulting regime is still **not yet human-like**.

The best retained branch-local Phase 18 artifact for that breakthrough is now:

- `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.png`

It summarizes the DMC + variational-evidence → Wong-Wang branch under `artifacts/results/rt_model_dmc_var_ww/smoke_a5_s3_neg_drt/`, specifically the retained negative-ΔRT checkpoint matched to `epoch 10`, where:

- raw exported `pred_mean = 0.4531s`
- `t0_seconds = 0.25`
- report-style display mean `(+t0) = 0.7031s`
- `error_minus_correct_rt = -0.0176s`

Interpretation:

- this is the first **retained/reported branch-local result we currently surface** where the DMC + variational-evidence combination produces negative error-minus-correct ΔRT;
- it is a meaningful mechanism-level breakthrough for the WW line;
- it still does **not** qualify as a final promoted solution because the resulting RT regime remains too far from human in overall scale/tail behavior and in the magnitude of the fast-error effect.

The later bounded successor-screening program under:

- `artifacts/results/rt_model_next_step/`

has now been executed to completion and reached a negative result:

- `WW+t0` did not salvage the current WW line,
- RTNet-lite on cached logits collapsed RT center and failed the smoke gate,
- cached-logit LBA was mechanistically informative but still failed the locked promotion gates,
- therefore the final branch verdict is `NO_SUCCESSOR_BRANCH_CLEARED_GATES`.

Recent single-subject verification work refined that picture:

- a **clean RT+response-only** WW single-subject workflow is now available under
  `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/`
- a matched **fixed-noise WW probe** is available under
  `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only_noise05/`
- a **clean-vs-noise comparison summary** is available under
  `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/`
- an **aligned WW vs AccumRNN feasibility branch** is available under
  `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_accumrnn_aligned/`
- an **aligned WW vs AccumRNN single-subject export comparison** is available under
  `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/`

The current verified interpretation is that WW noise can induce an errorful regime, but that regime is still not human-like, and the aligned single-subject comparison does not yet show a stronger replacement model.

An additional bounded HSFA repair audit was completed under:

- `artifacts/results/rt_model_hsfa_v3_1/`

That bundle removed the main ambiguity left by the earlier HSFA smoke branch:

- `subject_mode='none'` is now a true forward-pass ablation,
- feature-driven immediate stopping was blocked with hard minimum-step stopping and a learnable feature gate,
- calibrated-error losses and seeded stochastic readout were added,
- and the repaired aggregate smoke was rerun under a locked protocol.

Its final result is still negative:

- aggregate smoke verdict: `HSFA_V3_1_KILL_NEED_STAGE1_UNCERTAINTY_OR_BNN`
- final synthesis token: `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN`

So the repository now has **two aligned negative-result constraints**:

1. the broader successor-screening bundle (`rt_model_next_step`) did not clear a new Stage-2 successor branch, and
2. the narrower HSFA-v3.1 repair audit showed that fixing the known Stage-2 implementation/objective mismatches still does not produce a promotable branch.

The repository is therefore in a **consolidation phase**, not an open-ended local patching phase.

An additional bounded follow-up is now active under:

- `artifacts/results/rt_model_semisup_spea_v1_1_calibration/`

This branch is **not** a new architecture-search program. It is a calibration-focused follow-up on the earlier SemiSupervisedSPEA partial result, with:

- locked protocol artifacts under `00_protocol/`
- a calibrated trainer entry point `code/scripts/train_age_group_semisup_spea_calibrated.py`
- explicit stop-time-coupled readout support in `code/scripts/stage2_spea_backend.py`
- targeted readout regression coverage in `tests/test_spea_stop_time_coupling.py`

Current status of that branch:

- Batch 1 implementation scaffolding is complete.
- A replay root now exists at `artifacts/results/rt_model_semisup_spea_v1_1_calibration/01_replay_prior/c0_v2_replay/`.
- There is **no new scientific promotion verdict yet**. Treat this as an active bounded calibration branch whose experiment batch is still in progress.

---

## Recommended entry points

If you are new to this repository, start here:

1. `docs/history/logs.md`
   - dated project timeline from the first VGG work through the current error-regime chain

2. `artifacts/results/organized/handoff/supervisor_update_2026-04-08.md`
   - concise supervisor-facing summary

3. `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md`
   - most important technical memo for the latest WW branch

4. `artifacts/results/organized/README.md`
   - guide to current result folders and evidence levels

5. `artifacts/results/organized/handoff/single_subject_rt_response_research_judgment_memo.md`
   - current single-subject WW clean/noise and aligned WW vs AccumRNN research judgment

6. `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
   - final HSFA-v3.1 repair judgment: repaired Stage-2 fixes still do not justify promotion

7. `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
   - final successor-screening verdict: no tested next-line branch cleared gates
8. `artifacts/results/rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`
   - active bounded calibration follow-up protocol for the current SPEA v1.1 branch

---

## Top-level structure

### Core docs
- `CLAUDE.md` — repo-specific agent operating guide for the current workflow
- `docs/history/logs.md` — dated research timeline
- `docs/project/research_plan.md` — age-group execution plan
- `docs/project/research_proposal_v4.md` — broader research proposal
- `docs/project/AGENTS.md` — project-specific agent knowledge base aligned to the current repo layout

### Main code
- canonical scripts now live under `code/scripts/`
- root-level `scripts/` is retained as a compatibility symlink
- key entry points include:
  - `code/scripts/vgg_wongwang_lim.py`
  - `code/scripts/train_age_groups_efficient.py`
  - `code/scripts/vgg_accumulator_rnn.py`
  - `code/scripts/vgg_accumulator_rnn_v2.py`
  - `code/scripts/train_age_group_accumrnn.py`
  - `code/scripts/run_matched_full_age_group_analysis.py`

### Path abstraction status
- Stage 2 path abstraction is partially implemented via:
  - `code/scripts/project_paths.py`
- The most active age-group WW scripts now use this layer for canonical roots.
- The new grouped layout exists, while root-level names are retained as compatibility symlinks.

### Data / checkpoints
- canonical grouped layout:
  - `data/age_groups/`
  - `data/age_groups_matched/`
  - `data/vam_data/`
  - `artifacts/checkpoints/age_groups/`
  - `artifacts/checkpoints/age_groups_matched/`
  - `artifacts/checkpoints/test/`
- compatibility symlinks retained at root:
  - `data_age_groups/`
  - `data_age_groups_matched/`
  - `vam_data/`
  - `checkpoints_age_groups/`
  - `checkpoints_age_groups_matched/`
  - `checkpoints_test/`

### Results
- canonical location:
  - `artifacts/results/`
- root-level `results/` is retained as a compatibility symlink
- curated navigation layer remains under:
  - `artifacts/results/organized/`

### Supporting material
- `docs/papers/` — reference PDFs
- `docs/notes/` — personal / informal notes
- `docs/project/` — project-level plans, guides, and repo notes
- `docs/history/` — historical timeline / logs
- `notebooks/legacy_root/` — older root-level notebooks moved out of the main path
- `logs/runtime_archive/` — archived runtime `.log` / `.pid` files
- `config/` — configuration-style root files such as `requirements.txt`
- `code/vam/` — canonical VAM mixed code/assets directory
- root-level `vam/` and `Kar/` are retained as compatibility symlinks during migration and should not be treated as the canonical code location

---

## Reproducibility note

This repository follows a script-based workflow and keeps:

- code
- data preparation
- analysis scripts
- result summaries
- handoff memos

so that the research trail remains interpretable.

For the current best interpretation of file structure and evidence levels, see:

- `artifacts/results/organized/README.md`
- `artifacts/results/organized/FILE_MAPPING.md`

For a root-file relocation map, also see:

- `docs/project/ROOT_LEVEL_PY_CLASSIFICATION.md`

For the large-directory migration strategy, see:

- `docs/project/REPO_RESTRUCTURING_PLAN.md`

---

## Current scientific takeaway

The strongest current finding is:

> A VGG + WW model can be pushed from a no-error regime into a measurable error regime, but the resulting regime is still too fast and directionally wrong relative to current human error-conditioned RT behavior.

For the retained Phase 18 branch-local figure illustrating that transition into a negative-ΔRT regime, use:

- `artifacts/results/rt_model_dmc_var_ww/rt_model_breakdown.png`

That figure is intentionally kept as the single top-level generated Phase 18 comparison artifact after cleanup; the other Phase 18 convenience PNGs were removed so later readers see only the retained breakdown view.

The most recent bounded next-step program adds one more concrete conclusion:

> Under the locked `rt_model_next_step` successor-screening contract, none of `WW+t0`, RTNet-lite, or cached-logit LBA cleared the required gates for promotion.

So the repo is not waiting on one more local branch comparison. The justified next move is:

- consolidate the finished negative-result bundle,
- treat `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md` as the authoritative successor-screening outcome,
- treat `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md` as the authoritative HSFA repair outcome,
- and begin a **new planning phase** centered on Stage-1 uncertainty / BNN-like feature sampling before any further successor-branch design work.
