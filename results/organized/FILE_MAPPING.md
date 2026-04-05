# File-by-file mapping: results directory

This document is the cleanup and duplication map for the current results layout.

It answers four practical questions:

1. What is the cleanest folder structure for the project **right now**?
2. Which files are the best entry points for understanding current progress?
3. Which files should be treated as legacy references only?
4. What, if anything, can be safely archived or removed?

The goal is to improve structure and traceability **without breaking active analysis or training workflows**.

---

## A. Proposed cleaned structure

Recommended current structure:

```text
results/
├── age_groups_interim/                         # canonical legacy/interim source directory
├── age_groups_response_supervision_interim/   # canonical current-best response-supervision source directory
├── age_groups_response_supervision_frozen/    # canonical frozen summary source directory
├── proposal_aligned_behavior/                 # canonical human-side proposal-aligned source directory
└── organized/                                 # curated navigation / interpretation layer
    ├── README.md
    ├── FILE_MAPPING.md
    ├── archive/
    ├── current_best_response_supervision/
    ├── proposal_aligned_human_behavior/
    ├── legacy_interim_reference/
    └── handoff/
```

Why this is the cleanest structure right now:
- it keeps original script output paths untouched
- it separates evidence by interpretation level
- it gives later agents a stable human-readable entry point
- it avoids risky moves or renames while still reducing confusion

Archived organized folders kept for provenance:
- `organized/archive/age_groups_raw/`
- `organized/archive/standalone_figures/`

---

## B. Duplication / cleanup map by current result directory

### 1. `results/proposal_aligned_behavior/`

Status:
- **Keep as-is** as the canonical source directory for proposal-aligned human behavior outputs.
- **Copy key files into organized layer** for easier reading.

| Source file | Organized copy | Action | Notes |
|---|---|---|---|
| `figureB2_human_multipanel_20_29_vs_80_89.png` | `organized/proposal_aligned_human_behavior/figureB2_human_multipanel_20_29_vs_80_89.png` | Keep + copy | Best visual entry point for human age comparison |
| `figureP1_human_rt_distributions.png` | `organized/proposal_aligned_human_behavior/figureP1_human_rt_distributions.png` | Keep + copy | Human RT distribution evidence |
| `figureP2_human_signature_summary.png` | `organized/proposal_aligned_human_behavior/figureP2_human_signature_summary.png` | Keep + copy | Human signature summary |
| `human_behavior_signature_summary.csv` | `organized/proposal_aligned_human_behavior/human_behavior_signature_summary.csv` | Keep + copy | Human behavioral summary table |
| `figureP3_model_summary.png` | `organized/proposal_aligned_human_behavior/figureP3_model_summary.png` | Keep + copy | Contextual model summary paired with proposal-aligned outputs |
| `proposal_aligned_figure_note.md` | `organized/proposal_aligned_human_behavior/proposal_aligned_figure_note.md` | Keep + copy | Figure interpretation note |
| `integrated_current_results_analysis.md` | `organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md` | Keep + copy | Best narrative overview after handoff |

Recommendation:
- Leave source directory untouched.
- Treat the organized copy as the preferred read layer.

### 2. `results/age_groups_response_supervision_interim/`

Status:
- **Keep as-is** as the canonical source directory for current-best response-supervision outputs.
- **Copy key files into organized layer**.
- Some files can be marked as lower-priority provenance artifacts.

| Source file | Organized copy | Action | Notes |
|---|---|---|---|
| `response_supervision_current_comparison.csv` | `organized/current_best_response_supervision/response_supervision_current_comparison.csv` | Keep + copy | Main current-best comparison table |
| `response_supervision_interim_memo.md` | `organized/current_best_response_supervision/response_supervision_interim_memo.md` | Keep + copy | Main model-side interpretation note |
| `figureRS1_response_supervision_summary.png` | `organized/current_best_response_supervision/figureRS1_response_supervision_summary.png` | Keep + copy | Core summary figure |
| `figureRS2_response_supervision_multipanel.png` | `organized/current_best_response_supervision/figureRS2_response_supervision_multipanel.png` | Keep + copy | Broader model comparison figure |
| `figureRS3A_agegroup_accuracy_human_vs_model.png` | `organized/current_best_response_supervision/figureRS3A_agegroup_accuracy_human_vs_model.png` | Keep + copy | Accuracy-specific comparison |
| `figureRS3B_agegroup_congruency_gap_human_vs_model.png` | `organized/current_best_response_supervision/figureRS3B_agegroup_congruency_gap_human_vs_model.png` | Keep + copy | Congruency-gap comparison |
| `figureRS3_agegroup_human_vs_model.png` | no organized copy | Archived in source dir | Older combined comparison figure; moved to `results/age_groups_response_supervision_interim/archive/` |
| `response_supervision_multipanel_summary.csv` | no organized copy | Leave untouched but document | Useful provenance table, not a primary entry point |
| `figureRS1_response_supervision_eval05_summary.png` | no organized copy | Archived in source dir | Earlier eval snapshot; moved to `results/age_groups_response_supervision_interim/archive/` |
| `response_supervision_eval05_comparison.csv` | no organized copy | Archived in source dir | Earlier eval snapshot; moved to `results/age_groups_response_supervision_interim/archive/` |

Recommendation:
- Keep primary source files in place.
- Prefer the organized copy for reading.
- The eval05 files and older combined summary figure are now grouped under `results/age_groups_response_supervision_interim/archive/`.

### 3. `results/age_groups_response_supervision_frozen/`

Status:
- **Keep as-is**.
- **Copy all files into organized layer**, because this directory is already small and clearly scoped.

| Source file | Organized copy | Action | Notes |
|---|---|---|---|
| `frozen_current_best_comparison.csv` | `organized/current_best_response_supervision/frozen_current_best_comparison.csv` | Keep + copy | Frozen comparison summary |
| `frozen_current_best_memo.md` | `organized/current_best_response_supervision/frozen_current_best_memo.md` | Keep + copy | Frozen interpretation note |
| `figureF1_frozen_current_best_behavior.png` | `organized/current_best_response_supervision/figureF1_frozen_current_best_behavior.png` | Keep + copy | Frozen behavior-level figure |

Recommendation:
- Leave untouched.
- Continue presenting these as frozen summaries nested under the current-best response-supervision view.

### 4. `results/age_groups_interim/`

Status:
- **Keep as-is** as the canonical legacy/interim source directory.
- **Copy only reference-worthy files** into the organized legacy layer.
- Treat the whole directory as legacy reference unless doing provenance tracing.

| Source file | Organized copy | Action | Notes |
|---|---|---|---|
| `figureA1_80_89_signatures.png` | `organized/legacy_interim_reference/figureA1_80_89_signatures.png` | Keep + copy | Historical signature figure |
| `figureA2_80_89_rt_distributions.png` | `organized/legacy_interim_reference/figureA2_80_89_rt_distributions.png` | Keep + copy | Historical RT figure |
| `figureA3_interim_vs_final_comparison.png` | `organized/legacy_interim_reference/figureA3_interim_vs_final_comparison.png` | Keep + copy | Historical comparison figure |
| `figureA4_interim_trajectory_geometry.png` | `organized/legacy_interim_reference/figureA4_interim_trajectory_geometry.png` | Keep + copy | Mechanism preview only, not final evidence |
| `figureA4_interim_trajectory_spread.csv` | `organized/legacy_interim_reference/figureA4_interim_trajectory_spread.csv` | Keep + copy | Geometry preview table |
| `three_way_comparison_current.csv` | `organized/legacy_interim_reference/three_way_comparison_current.csv` | Keep + copy | Historical comparison table |
| `three_way_comparison_memo.md` | `organized/legacy_interim_reference/three_way_comparison_memo.md` | Keep + copy | Historical interpretation note |
| `figureB1_80_89_multipanel_profile.png` | no organized copy | Leave untouched but document | Legacy output not currently surfaced in organized layer |
| `interim_vs_final_summary.csv` | no organized copy | Leave untouched but document | Provenance table |
| `interim_vs_final_summary.md` | no organized copy | Leave untouched but document | Provenance note |
| `current_stage_conclusion_memo.md` | no organized copy | Leave untouched but document | Contextual memo |

Recommendation:
- Keep source directory intact.
- Do not promote any file from here above human-side or corrected-supervision outputs.

### 5. `results/organized/`

Status:
- **Keep and expand** as the main navigation layer.

| Path | Action | Notes |
|---|---|---|
| `organized/README.md` | Update | Make evidence levels and entry points explicit |
| `organized/FILE_MAPPING.md` | Update | Central structure / cleanup guide |
| `organized/handoff/CURRENT_STATUS.md` | Keep | Best short operational restart note |
| `organized/handoff/agent_file_structure_cleanup_prompt.md` | Keep | Records the requested cleanup framing |
| `organized/.DS_Store` | Remove | Safe low-risk OS metadata cleanup |
| `organized/archive/age_groups_raw/` | Keep archived | Supplemental older mirror moved out of primary navigation |
| `organized/archive/standalone_figures/` | Keep archived | Convenience figures moved out of primary navigation |

---

## C. Best entry points for understanding current progress

Recommended reading order:

1. `results/organized/handoff/CURRENT_STATUS.md`
2. `results/organized/README.md`
3. `results/organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md`
4. `results/organized/proposal_aligned_human_behavior/figureB2_human_multipanel_20_29_vs_80_89.png`
5. `results/organized/proposal_aligned_human_behavior/figureP2_human_signature_summary.png`
6. `results/organized/current_best_response_supervision/response_supervision_current_comparison.csv`
7. `results/organized/current_best_response_supervision/response_supervision_interim_memo.md`

These are the best entry points because together they show:
- the project question,
- the current scientific status,
- the strongest human evidence,
- and the current-best model comparison,

without over-promoting older interim artifacts.

---

## D. Files that should be treated as legacy references only

Treat the following as **legacy reference only**:

- everything under `results/organized/legacy_interim_reference/`
- everything under `results/age_groups_interim/` unless doing provenance tracing
- `results/organized/archive/age_groups_raw/`
- `results/organized/archive/standalone_figures/`
- especially:
  - `figureA4_interim_trajectory_geometry.png`
  - `figureA4_interim_trajectory_spread.csv`
  - `figureA3_interim_vs_final_comparison.png`

Important interpretation rule:
- legacy geometry material is a **mechanism preview**, not final corrected-supervision evidence.

---

## E. Safe cleanup / archive suggestions

### Safe to remove now

| Path | Why removal is safe | Canonical replacement |
|---|---|---|
| `results/organized/.DS_Store` | OS-generated metadata file with no scientific or workflow value | none needed |

### Already archived safely

| Path | Why archive instead of delete | Canonical replacement |
|---|---|---|
| `results/age_groups_response_supervision_interim/archive/figureRS1_response_supervision_eval05_summary.png` | Older eval snapshot; superseded by current summary figure | `results/organized/current_best_response_supervision/figureRS1_response_supervision_summary.png` |
| `results/age_groups_response_supervision_interim/archive/response_supervision_eval05_comparison.csv` | Older eval snapshot; superseded by current comparison table | `results/organized/current_best_response_supervision/response_supervision_current_comparison.csv` |
| `results/age_groups_response_supervision_interim/archive/figureRS3_agegroup_human_vs_model.png` | Older combined view; split A/B figures are clearer in organized layer | `results/organized/current_best_response_supervision/figureRS3A_agegroup_accuracy_human_vs_model.png` and `figureRS3B_agegroup_congruency_gap_human_vs_model.png` |
| `results/organized/archive/age_groups_raw/` | Older organized mirror already moved out of primary navigation | `results/organized/README.md` + the four primary organized folders |
| `results/organized/archive/standalone_figures/` | Convenience figures already moved out of primary navigation | `results/organized/proposal_aligned_human_behavior/` and `results/organized/current_best_response_supervision/` |

### Leave untouched but document

These are not primary entry points, but they still may matter for provenance:

- `results/age_groups_response_supervision_interim/response_supervision_multipanel_summary.csv`
- `results/age_groups_interim/interim_vs_final_summary.csv`
- `results/age_groups_interim/interim_vs_final_summary.md`
- `results/age_groups_interim/current_stage_conclusion_memo.md`
- `results/age_groups_interim/figureB1_80_89_multipanel_profile.png`
- `results/organized/archive/age_groups_raw/`

Recommendation:
- do not delete these unless the user explicitly asks for a provenance-pruning pass
- if needed later, move them into a clearly named archive folder rather than removing them

---

## Final direct answers

### 1. What is the cleanest folder structure for the project right now?

Keep the four original result directories as canonical source/write locations, and treat `results/organized/` as the single curated navigation layer with four stable sections:
- `current_best_response_supervision/`
- `proposal_aligned_human_behavior/`
- `legacy_interim_reference/`
- `handoff/`

### 2. Which files are the best entry points for someone trying to understand current progress?

- `results/organized/handoff/CURRENT_STATUS.md`
- `results/organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md`
- `results/organized/proposal_aligned_human_behavior/figureB2_human_multipanel_20_29_vs_80_89.png`
- `results/organized/proposal_aligned_human_behavior/figureP2_human_signature_summary.png`
- `results/organized/current_best_response_supervision/response_supervision_current_comparison.csv`
- `results/organized/current_best_response_supervision/response_supervision_interim_memo.md`

### 3. Which files should be treated as legacy references only?

- everything in `results/organized/legacy_interim_reference/`
- everything in `results/organized/archive/age_groups_raw/`
- everything in `results/organized/archive/standalone_figures/`
- especially `figureA4_interim_trajectory_geometry.png`
- and, by provenance, the original `results/age_groups_interim/` directory

### 4. What, if anything, can be safely archived or removed?

Safely removed now:
- `results/organized/.DS_Store`

Safely archivable later, but best left untouched unless the user requests an archive pass:
- `results/age_groups_response_supervision_interim/response_supervision_multipanel_summary.csv`
- no further organized-folder archive move is required right now because `age_groups_raw/` and `standalone_figures/` are already archived under `results/organized/archive/`
