# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope and priorities

This repository has two distinct code paths:

1. **Active research workflow**: `code/scripts/` for the age-group Flanker / LIM reaction-time modeling pipeline.
2. **Legacy / secondary subproject**: `code/vam/` for the original JAX/Flax VAM package.

For most tasks, start with the active workflow under `code/scripts/`, `tests/`, `docs/project/`, and `artifacts/results/organized/`.

De-prioritize `archive/` and `skills/` unless the task explicitly points there.

## Project status

Per `README.md`, this repo is in a **consolidation phase**. The current strongest result is that the VGG + Wong-Wang line can reach an error regime, but it is still not human-like enough. Before proposing new local tweaks to Wong-Wang, check the existing handoff memos and organized results to avoid repeating already-tested ideas.

There is now an additional hard stop condition from the bounded successor-screening bundle under `artifacts/results/rt_model_next_step/`: the saved final synthesis memo concludes `NO_SUCCESSOR_BRANCH_CLEARED_GATES`. Treat that as authoritative evidence that the tested `WW+t0`, RTNet-lite, and cached-logit LBA branches did not earn promotion, and that any further successor design work should start from a **new planning phase**, not by resuming that plan in place.

There is also now a narrower HSFA-specific hard stop condition from `artifacts/results/rt_model_hsfa_v3_1/`: the repaired HSFA-v3.1 bundle fixed the main Stage-2 implementation/objective mismatches, reran the locked aggregate smoke, and still ended in `HSFA_V3_1_KILL_NEED_STAGE1_UNCERTAINTY_OR_BNN` with final synthesis token `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN`. Treat that as authoritative evidence that more local HSFA Stage-2 tweaking is not the justified next move.

There is also now an **active bounded calibration follow-up** under `artifacts/results/rt_model_semisup_spea_v1_1_calibration/`. Treat this as a live protocolized follow-up on the earlier SemiSupervisedSPEA partial result, not as a new architecture-search family and not as an already-resolved scientific branch verdict.

Recommended orientation files:

- `README.md`
- `docs/history/logs.md`
- `artifacts/results/organized/README.md`
- `artifacts/results/organized/handoff/supervisor_update_2026-04-08.md`
- `artifacts/results/organized/handoff/error_regime_experiment_chain_memo.md`
- `artifacts/results/organized/handoff/single_subject_rt_response_research_judgment_memo.md`
- `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
- `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
- `artifacts/results/rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`

## Canonical paths

Use the grouped canonical directories, not the root compatibility symlinks.

- Data: `data/`
- Artifacts: `artifacts/`
- Code: `code/`

Important path helpers are centralized in `code/scripts/project_paths.py`.

Canonical roots defined there include:

- `data/age_groups/`
- `data/age_groups_matched/`
- `data/vam_data/`
- `artifacts/checkpoints/age_groups/`
- `artifacts/checkpoints/age_groups_matched/`
- `artifacts/checkpoints/test/`
- `artifacts/results/`

Root names like `scripts/`, `vam/`, `results/`, `data_age_groups/`, and `checkpoints_age_groups/` are compatibility symlinks and should not be treated as the primary layout.

Some older helper scripts still accept or expect those root compatibility names. When that happens, keep the script-specific invocation exactly as documented, but treat the grouped `code/`, `data/`, and `artifacts/` layout as the canonical source of truth.

## Working directories and commands

Many active scripts assume they are run from `code/scripts/`.

Typical setup:

```bash
cd /Users/siyu/Documents/GitHub/VAM-studying
source .venv/bin/activate
cd code/scripts
```

### Environment setup

The repository has a root virtualenv and a shared requirements file in `config/requirements.txt`.

```bash
cd /Users/siyu/Documents/GitHub/VAM-studying
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r config/requirements.txt
```

### Active age-group workflow

Run these from `code/scripts/`.

#### Data preparation

```bash
python prepare_age_group_data.py
python create_stimulus_mapping.py
```

#### Main age-group training

```bash
python train_age_groups_efficient.py
```

This script is the main entry point for the current workflow. It:

- reuses an existing Stage-1 visual model when available
- extracts logits for age groups
- trains Stage-2 Wong-Wang models
- performs scale search and comparison logic

#### Stage-1-from-scratch path

`train_stage1_classification.py` is not yet fully path-abstracted. When you run it from `code/scripts/`, keep using the compatibility-root arguments below unless you explicitly update the script first.

```bash
python train_stage1_classification.py --data_dir vam_data --output_dir checkpoints_age_groups/20-29/stage1 --epochs 50
```

#### Smoke / pipeline scripts

```bash
./run_test.sh
./run_pipeline.sh
```

`run_test.sh` is the quickest end-to-end smoke pipeline for the VGG → Wong-Wang flow. It runs Stage 1, Stage 2, and evaluation with reduced settings.

### Tests

Run from the repository root:

```bash
pytest tests/test_dynamic_selection_phase1.py tests/test_dynamic_selection_single_subject.py
```

Run a single test file:

```bash
pytest tests/test_dynamic_selection_single_subject.py
```

Run a single test case:

```bash
pytest tests/test_dynamic_selection_single_subject.py -k alignment
```

These tests are currently the most useful targeted validation for the dynamic selection / single-subject workflow.

### Legacy VAM subproject

Run these from `code/vam/` when working on the original JAX/Flax VAM package:

```bash
cd /Users/siyu/Documents/GitHub/VAM-studying/code/vam
python -m vam.training --project test --expt_name exp001
python -m vam.training --model_type task_opt --n_epochs 30
python -m vam.training --model_type binned_rt --n_rt_bins 5 --rt_bin 3
```

Dependency files for this subproject live alongside it, such as:

- `code/vam/training_requirements.txt`
- `code/vam/dev_requirements.txt`
- `code/vam/analysis_requirements.txt`

## Architecture overview

### 1. Active pipeline: visual evidence + decision dynamics

The main current workflow is a two-stage model for Flanker / LIM reaction-time behavior:

1. **Stage 1 visual model**
   - VGG-based image encoder / classifier
   - turns stimulus images into 4-class logits (`L/R/U/D`)
2. **Stage 2 decision model**
   - Wong-Wang-style decision dynamics over 4 competing alternatives
   - converts transformed logits into decision-time / RT behavior
3. **Evaluation and analysis**
   - compares RT-distribution shape, error structure, congruency effects, and age-group differences

The central implementation files are:

- `code/scripts/train_age_groups_efficient.py`
- `code/scripts/vgg_wongwang_lim.py`
- `code/scripts/run_matched_full_age_group_analysis.py`
- `code/scripts/run_age_group_post_analysis.py`

### 2. `vgg_wongwang_lim.py` is the main model-definition file

`code/scripts/vgg_wongwang_lim.py` defines the core Stage-2 interface and readout behavior.

At a high level the flow is:

- image input
- VGG16 feature extraction
- fully connected logits for the four response directions
- Stage-2 input transform (`ReLU(logits * scale)`)
- Wong-Wang multiclass competition
- RT / choice readout

This file also contains the newer behavioral mechanisms that many experiments modify:

- baseline Stage-2 input transform
- dynamic flanker suppression
- DMC-like early capture / later control behavior
- soft-hazard RT readout
- urgency-style RT readout
- minimal conflict-capture variants

If a task mentions dynamic selection, flanker suppression, DMC-like behavior, hazard readout, or urgency readout, start here.

### 3. `train_age_groups_efficient.py` is the main orchestration script

`code/scripts/train_age_groups_efficient.py` is the highest-value file for understanding the current workflow end to end.

It ties together:

- path resolution via `project_paths.py`
- stimulus/image dataset loading
- cached Stage-1 logits
- optional smoke-eval subset logic
- flanker-label attachment from CSVs
- Stage-2 fitting and scale search
- age-group comparison outputs

If you need to change the main training/evaluation workflow, inspect this file before touching helper scripts.

### 4. Tests focus on dynamic selection and alignment invariants

Current targeted tests live in:

- `tests/test_dynamic_selection_phase1.py`
- `tests/test_dynamic_selection_single_subject.py`

For the active SPEA v1.1 calibration follow-up, also treat these as targeted guardrails:

- `tests/test_spea_stochastic_readout.py`
- `tests/test_spea_stop_time_coupling.py`
- `tests/test_spea_stage1_sampling.py`
- `tests/test_spea_error_losses.py`
- `tests/test_spea_accumulator_rollout.py`

They verify behavior such as:

- dynamic input modifications when selection is disabled or enabled
- flanker suppression / DMC-like time-course behavior
- clipping behavior for scale grids
- handling of missing error-trial RT contrasts
- CSV/NPZ row-alignment checks

When changing dynamic-selection logic, these tests are the first guardrail to run.

### 5. Legacy JAX/Flax VAM is a separate subproject

`code/vam/` is not just a utility folder; it is a separate modeling stack using JAX/Flax and variational inference.

Important files there include:

- `code/vam/vam/training.py`
- `code/vam/vam/models.py`
- `code/vam/vam/config.py`
- `code/vam/vam/lba.py`
- `code/vam/manuscript/`

Use this path when the task is about reproducing or modifying the original VAM package, ELBO training, LBA internals, or manuscript-era analysis.

## Repository-specific working rules

- Prefer `code/scripts/` for current research changes unless the task explicitly targets `code/vam/`.
- Prefer canonical grouped paths from `project_paths.py` over hardcoded root symlink paths.
- Before proposing new experiments, read the organized result memos to avoid duplicating already-rejected Wong-Wang tweaks.
- Before proposing any new successor branch, read `artifacts/results/rt_model_next_step/06_synthesis/final_successor_branch_memo.md` so you do not reopen the already-closed `WW+t0` / RTNet-lite / cached-logit LBA screening loop.
- Before proposing any further HSFA local repair, read `artifacts/results/rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md` so you do not reopen the already-closed HSFA-v3.1 repair loop.
- If the task is about the active SPEA v1.1 calibration follow-up, start with:
  - `artifacts/results/rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`
  - `.sisyphus/plans/semisup_spea_v1_1_calibration_agent_plan.md`
  - `code/scripts/train_age_group_semisup_spea_calibrated.py`
  - `code/scripts/stage2_spea_backend.py`
  - `tests/test_spea_stop_time_coupling.py`
- Do not treat the active SPEA v1.1 calibration branch as scientifically resolved until it has written its own smoke decision and final synthesis outputs.
- For current single-subject work, treat these as the canonical verified roots unless the user explicitly requests a different branch:
  - `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only/`
  - `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_rt_response_only_noise05/`
  - `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_accumrnn_aligned/`
  - `artifacts/results/repro_legacy_interim/single_subject_model_export_comparison_rt_response_only_aligned/`
  - `artifacts/results/repro_legacy_interim/true_single_subject_feasibility_clean_vs_noise_comparison/`
- Treat the repository as script-driven and reproducibility-oriented: preserve clear script entry points and interpretable artifact locations.
- For UI-free research tasks, validation usually means targeted `pytest` plus running the smallest relevant script path rather than broad repo-wide test sweeps.
