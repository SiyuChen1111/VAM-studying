# Root-level Python classification

This document classifies **repository-root `.py` files only**. It is a cleanup and orientation aid, not a migration plan.

Scope:
- included: `.py` files directly under the repository root
- excluded: Python files under `vam/`, `Kar/`, `archive/`, and other subdirectories

Current rule:
- **do not move root-level Python files during the active response-supervision orchestrator run**
- the live workflow currently depends on root-relative script names and root-relative `logs/`, `results/`, and checkpoint paths

## Category summary

| Category | Meaning | Move now? |
|---|---|---|
| active | live orchestration / currently used response-supervision chain | No |
| training | model training / evaluation entrypoints still likely used directly | No |
| data-prep | dataset shaping, stimulus generation, preprocessing | Not now |
| analysis | reporting, comparison, inspection, visualization | Not now |
| legacy | older monitors, older experiment families, superseded entrypoints | Later candidate |
| utility | support modules, wrappers, test helpers, one-off fix scripts | Later candidate, but only after import/path audit |

## 1. Active

These files are part of the currently relevant response-supervision workflow and should stay at the repository root for now.

- `orchestrate_response_supervision_experiment.py` — live orchestrator entrypoint
- `train_age_groups_efficient.py` — training entrypoint launched by the orchestrator
- `freeze_response_supervision_current_best.py` — orchestrator refresh step
- `generate_response_supervision_interim_report.py` — orchestrator refresh/report step
- `generate_response_supervision_multipanel.py` — orchestrator refresh/report step
- `generate_response_supervision_agegroup_compare.py` — orchestrator refresh/report step
- `generate_proposal_aligned_behavior_figures.py` — proposal-aligned refresh/report step
- `vgg_wongwang_lim.py` — support module used by the active workflow

Why this category is frozen:
- the orchestrator launches several scripts by bare filename from the repository root
- the active chain reads and writes root-relative `logs/`, `results/`, `checkpoints_age_groups/`, and `checkpoints_age_groups_matched/` paths

## 2. Training

These are training or evaluation entrypoints that still look like direct operational scripts, even if they are not part of the live orchestrator chain.

- `train_stage1_classification.py`
- `train_stage2_rt_fitting.py`
- `train_age_group_model.py`
- `extract_age_group_logits_fast.py`
- `extract_age_group_logits.py`
- `evaluate_vgg_wongwang_lim.py`

Recommendation:
- keep at root until there is a deliberate script-entrypoint reorganization

## 3. Data-prep

These files shape data, build matched branches, or prepare stimuli and caches.

- `prepare_age_group_data.py`
- `preprocess_vam_data.py`
- `create_stimulus_mapping.py`
- `generate_stimulus_images.py`
- `generate_stimulus_numpy.py`
- `precompute_images.py`
- `add_image_paths.py`
- `update_80_89_data.py`
- `create_matched_20_29_control_branch.py`
- `create_matched_20_29_logits_subset.py`

Recommendation:
- good future candidates for a `scripts/data_prep/` style grouping
- do not move until imports, CLI assumptions, and relative file paths are audited

## 4. Analysis

These scripts primarily generate reports, figures, analyses, or inspection outputs.

- `run_age_group_post_analysis.py`
- `generate_interim_age_group_report.py`
- `analyze_human_data.py`
- `visualize_stimuli.py`

Recommendation:
- good future candidates for a `scripts/analysis/` or `scripts/reports/` grouping
- keep in place for now because some may still be run directly from the root

## 5. Legacy

These are the strongest candidates for later archival grouping because they appear to belong to older monitors, older experimental flows, or superseded model families.

- `monitor_response_supervision_safe2.py`
- `monitor_response_supervision_pipeline.py`
- `train_stage2_only.py`
- `train_age_group_stage2.py`
- `train_model_balanced.py`
- `AlexNet_BN_LSTM_backbone.py`
- `AlexNet_BN_LSTM_sup_2.py`
- `alexnet_lstm_rt.py`

Recommendation:
- best later candidates for a `legacy_scripts/` or `archive/scripts/` grouping
- still do not move blindly until references are checked

## 6. Utility

These are support files, wrappers, helpers, or one-off fix/test scripts that do not fit cleanly into the other categories.

- `vgg_wongwang_lim_data.py`
- `wong_wang.py`
- `test_imports.py`
- `run_vam.py`
- `reproduce_vam_guide.py`
- `fix_drift_rate_output.py`
- `add_visualization.py`

Recommendation:
- these may eventually split into `support/`, `wrappers/`, and `one_off_fixes/`
- do not move until import relationships are audited

## Protected during the live orchestrator run

While the current orchestrator run is active, treat the following as protected:

- `orchestrate_response_supervision_experiment.py`
- `train_age_groups_efficient.py`
- `freeze_response_supervision_current_best.py`
- `generate_response_supervision_interim_report.py`
- `generate_response_supervision_multipanel.py`
- `generate_response_supervision_agegroup_compare.py`
- `generate_proposal_aligned_behavior_figures.py`
- `vgg_wongwang_lim.py`
- `logs/response_supervision_orchestrator.log`
- `logs/train_80_89_response_supervision_orchestrated.log`
- `logs/train_20_29_matched_response_supervision_orchestrated.log`
- `results/response_supervision_orchestrator_done.txt`
- `checkpoints_age_groups/80-89/stage2/`
- `checkpoints_age_groups_matched/20-29/stage2/`

## Practical next step after the live run finishes

The safest next cleanup is:

1. keep `active` untouched
2. audit imports and subprocess calls for `training`, `data-prep`, `analysis`, and `utility`
3. move `legacy` only after confirming nothing live still references it
4. introduce wrapper entrypoints before moving any script that is currently called by filename

## Full root-level inventory by category

This is the complete root-level `.py` inventory covered by this document:

- active: `orchestrate_response_supervision_experiment.py`, `train_age_groups_efficient.py`, `freeze_response_supervision_current_best.py`, `generate_response_supervision_interim_report.py`, `generate_response_supervision_multipanel.py`, `generate_response_supervision_agegroup_compare.py`, `generate_proposal_aligned_behavior_figures.py`, `vgg_wongwang_lim.py`
- training: `train_stage1_classification.py`, `train_stage2_rt_fitting.py`, `train_age_group_model.py`, `extract_age_group_logits_fast.py`, `extract_age_group_logits.py`, `evaluate_vgg_wongwang_lim.py`
- data-prep: `prepare_age_group_data.py`, `preprocess_vam_data.py`, `create_stimulus_mapping.py`, `generate_stimulus_images.py`, `generate_stimulus_numpy.py`, `precompute_images.py`, `add_image_paths.py`, `update_80_89_data.py`, `create_matched_20_29_control_branch.py`, `create_matched_20_29_logits_subset.py`
- analysis: `run_age_group_post_analysis.py`, `generate_interim_age_group_report.py`, `analyze_human_data.py`, `visualize_stimuli.py`
- legacy: `monitor_response_supervision_safe2.py`, `monitor_response_supervision_pipeline.py`, `train_stage2_only.py`, `train_age_group_stage2.py`, `train_model_balanced.py`, `AlexNet_BN_LSTM_backbone.py`, `AlexNet_BN_LSTM_sup_2.py`, `alexnet_lstm_rt.py`
- utility: `vgg_wongwang_lim_data.py`, `wong_wang.py`, `test_imports.py`, `run_vam.py`, `reproduce_vam_guide.py`, `fix_drift_rate_output.py`, `add_visualization.py`
