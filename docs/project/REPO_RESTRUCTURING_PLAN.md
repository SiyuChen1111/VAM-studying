# Repo Restructuring Plan

**Status:** Stage 2 started — a shared path abstraction layer now exists, but large top-level directory moves should still wait until broader script coverage and CSV path cleanup are completed.  
**Date:** 2026-04-09

---

## 1. Purpose

This document defines a **safe future restructuring plan** for the repository.

The immediate goal is **not** to move all large top-level directories right now.
The immediate goal is to:

1. identify the desired target layout,
2. identify what currently prevents direct movement,
3. document which scripts and data artifacts depend on the current root-relative names,
4. specify a safe migration order.

This plan exists because the current workflow still depends heavily on root-relative paths such as:

- `data_age_groups/`
- `data_age_groups_matched/`
- `checkpoints_age_groups/`
- `checkpoints_age_groups_matched/`
- `vam_data/`
- `vam/`
- `results/`

---

## 2. Current high-level structure

Current important top-level directories:

- `scripts/`
- `docs/`
- `results/`
- `data_age_groups/`
- `data_age_groups_matched/`
- `checkpoints_age_groups/`
- `checkpoints_age_groups_matched/`
- `checkpoints_test/`
- `vam/`
- `vam_data/`
- `archive/`

This structure is now cleaner than before, but it is still not yet grouped into a canonical higher-level layout such as `code/`, `data/`, and `artifacts/`.

---

## 3. Proposed target layout

The desired future structure is:

```text
Project/
├── README.md
├── .gitignore
├── code/
│   ├── scripts/
│   └── vam/                # only after explicit split / migration plan
├── data/
│   ├── vam_data/
│   ├── age_groups/
│   └── age_groups_matched/
├── artifacts/
│   ├── checkpoints/
│   │   ├── age_groups/
│   │   ├── age_groups_matched/
│   │   └── test/
│   └── results/
├── docs/
├── notebooks/
├── logs/
└── archive/
```

### Intended mapping

| Current path | Proposed future path |
|---|---|
| `scripts/` | `code/scripts/` |
| `vam_data/` | `data/vam_data/` |
| `data_age_groups/` | `data/age_groups/` |
| `data_age_groups_matched/` | `data/age_groups_matched/` |
| `checkpoints_age_groups/` | `artifacts/checkpoints/age_groups/` |
| `checkpoints_age_groups_matched/` | `artifacts/checkpoints/age_groups_matched/` |
| `checkpoints_test/` | `artifacts/checkpoints/test/` |
| `results/` | `artifacts/results/` |
| `vam/` | leave in place for now; later evaluate split into code + assets |
| `archive/` | leave in place for now |

---

## 4. Why we should **not** move these directories yet

### 4.1 Root-relative path assumptions are still widespread

Examples already confirmed in code:

- `scripts/train_age_groups_efficient.py`
  - defaults to `data_age_groups` and `checkpoints_age_groups`
- `scripts/run_age_group_post_analysis.py`
  - reads `checkpoints_age_groups/...` and `data_age_groups/...`
- `scripts/prepare_age_group_data.py`
  - reads `vam_data/metadata.csv`
- `scripts/train_stage1_classification.py`
  - defaults to `vam_data`
- `scripts/run_vam.py`
  - expects a top-level `vam/`
- many orchestrator scripts point directly at `checkpoints_age_groups*` and `data_age_groups_matched`

### 4.2 Serialized CSV paths are an additional hidden dependency

The most important hidden risk is that some prepared CSV files already store literal paths like:

- `data_age_groups/.../stimulus_images/...`

This means that moving the directories would break not only scripts, but also path fields already written inside derived datasets.

### 4.3 `vam/` is mixed-content, not just code

The top-level `vam/` currently includes:

- the nested `vam/` Python package
- assets such as bird images / background images
- documentation and manuscript code

So it should **not** be moved wholesale under `code/` without a more careful split.

### 4.4 `archive/` is also mixed-content

`archive/` contains:

- old helper scripts
- deprecated Stage-2 records
- model assets
- historical reference materials

It should remain top-level until a dedicated archive design pass happens.

---

## 5. Scripts and assets most likely to break under immediate migration

### A. Data-path-sensitive scripts

These directly reference `data_age_groups`, `data_age_groups_matched`, or `vam_data`:

- `scripts/prepare_age_group_data.py`
- `scripts/train_age_group_model.py`
- `scripts/train_age_group_stage2.py`
- `scripts/train_age_groups_efficient.py`
- `scripts/train_age_group_accumrnn.py`
- `scripts/train_age_group_accumrnn_v2.py`
- `scripts/run_age_group_post_analysis.py`
- `scripts/run_matched_full_age_group_analysis.py`
- `scripts/generate_proposal_aligned_behavior_figures.py`
- `scripts/generate_response_supervision_agegroup_compare.py`
- `scripts/create_stimulus_mapping.py`
- `scripts/precompute_images.py`
- `scripts/update_80_89_data.py`
- `scripts/analyze_human_data.py`
- `scripts/train_stage1_classification.py`
- `scripts/evaluate_vgg_wongwang_lim.py`
- `scripts/visualize_stimuli.py`

### B. Checkpoint-path-sensitive scripts

These directly reference `checkpoints_age_groups*` or `checkpoints_test`:

- `scripts/train_age_groups_efficient.py`
- `scripts/train_age_group_model.py`
- `scripts/train_age_group_stage2.py`
- `scripts/extract_age_group_logits.py`
- `scripts/extract_age_group_logits_fast.py`
- `scripts/run_age_group_post_analysis.py`
- `scripts/run_matched_full_age_group_analysis.py`
- many orchestrator scripts:
  - `scripts/orchestrate_matched_20_29_*`
  - `scripts/orchestrate_ww_noise_probe.py`
  - `scripts/orchestrate_response_supervision_experiment.py`
  - `scripts/monitor_*`
  - `scripts/retrain_stage2_both.sh`
  - `scripts/supervise_age_groups_pipeline.sh`

### C. `vam/`-sensitive scripts

These directly assume root-level `vam/`:

- `scripts/run_vam.py`
- `scripts/train_stage1_classification.py` (via graphics / assets assumptions)
- `scripts/reproduce_vam_guide.py`
- several older helper scripts and archived utilities

---

## 6. Safe migration principle

The safest migration principle is:

> **abstract paths first, move directories second**

This means:

1. add a shared path-resolution layer,
2. update scripts to read paths from that layer,
3. fix serialized CSV path dependencies,
4. only then move the big directories.

---

## 7. Recommended migration order

### Stage 1 — No directory moves yet (safe now)

Do now:

- keep all big top-level directories where they are
- continue using the cleaned root layout already in place
- maintain `README.md`, `docs/`, `scripts/`, `results/organized/`, and `docs/history/logs.md` as the human-facing structure

### Stage 2 — Introduce path abstraction

Create one shared project-path module or config layer, for example:

- `scripts/project_paths.py`

Current progress:

- implemented: `scripts/project_paths.py`
- integrated into the current most active scripts:
  - `scripts/train_age_groups_efficient.py`
  - `scripts/run_age_group_post_analysis.py`
  - `scripts/run_matched_full_age_group_analysis.py`
  - `scripts/prepare_age_group_data.py`

This module should expose canonical variables such as:

- `DATA_AGE_GROUPS_ROOT`
- `DATA_AGE_GROUPS_MATCHED_ROOT`
- `CHECKPOINTS_AGE_GROUPS_ROOT`
- `CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT`
- `VAM_DATA_ROOT`
- `VAM_ROOT`
- `RESULTS_ROOT`

Then update scripts to stop hardcoding root-relative names directly.

### Stage 3 — Fix serialized dataset paths

Before moving any `data_*` directories, update the generated CSVs and any dependent code so that image paths are not stored with hardcoded root folder names.

Safer alternatives:

- store relative subpaths only
- or store stimulus IDs and reconstruct paths programmatically

### Stage 4 — Move data clusters together

Move these **together**, not separately:

- `data_age_groups/` + `data_age_groups_matched/`
- and ideally `vam_data/` in the same migration phase

Target:

- `data/age_groups/`
- `data/age_groups_matched/`
- `data/vam_data/`

### Stage 5 — Move checkpoint clusters together

Move these **together**:

- `checkpoints_age_groups/`
- `checkpoints_age_groups_matched/`
- `checkpoints_test/`

Target:

- `artifacts/checkpoints/age_groups/`
- `artifacts/checkpoints/age_groups_matched/`
- `artifacts/checkpoints/test/`

### Stage 6 — Move `results/`

Only after scripts and docs are already path-abstracted.

Target:

- `artifacts/results/`

This step requires a markdown-reference update pass because many memos reference `results/...` directly.

### Stage 7 — Re-evaluate `vam/` and `archive/`

Do not move either until everything above is stable.

For `vam/`, a future split may be better than a blind move:

- `code/vam/` for package code
- `data/vam_assets/` or `docs/papers/` for assets/docs

For `archive/`, keep it top-level unless there is a separate archival-structure design pass.

---

## 8. Recommended immediate action

### What is safe now

- Keep the current post-cleanup root layout
- Treat the current top-level big directories as **compatibility anchors**
- Use this document to explain why they remain where they are

### What should happen next if restructuring is pursued

The **next safe implementation step** is not a move.
It is:

> create a shared path abstraction layer and update the active scripts to use it.

Only after that should physical directory moves be attempted.

---

## 9. One-sentence summary

The repository now has a much cleaner root-level file layout, but the big top-level workflow directories (`data_*`, `checkpoints_*`, `vam_data`, `vam`, `results`) should remain where they are until path abstraction and CSV path cleanup are completed, because moving them now would likely break a large number of scripts and stored path references.
