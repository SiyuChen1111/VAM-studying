# Current status / handoff note

**Updated:** 2026-04-02

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
- data root: `data_age_groups_matched/20-29`
- output root: `checkpoints_age_groups_matched/20-29/stage2`

### Old branch
- branch: `80-89`
- data root: `data_age_groups/80-89`
- output root: `checkpoints_age_groups/80-89/stage2`

## What is currently trustworthy

### Strongest current evidence
- Human-side behavior figures in `organized/proposal_aligned_human_behavior/`
- Current-best response-supervision summaries in `organized/current_best_response_supervision/`

### Useful but lower-tier evidence
- `organized/legacy_interim_reference/figureA4_interim_trajectory_geometry.png`
  - This is a geometry preview from the earlier interim path
  - It should not be treated as the final corrected-supervision mechanism figure

## What is still missing

To produce a final corrected-supervision model comparison, we still need clean saved best-so-far model outputs that are trustworthy for:
- model RT distribution plots
- model skew / error-slower plots
- updated trajectory geometry / updated A4-like figure

## Immediate purpose of the organized folders

The `organized/` tree exists so that a later agent can quickly tell:
- which outputs are current
- which outputs are human-only
- which outputs are frozen summaries
- which outputs are older interim references

## Suggested restart point for a future agent

1. Read `results/organized/README.md`
2. Inspect `organized/current_best_response_supervision/response_supervision_current_comparison.csv`
3. Inspect `organized/proposal_aligned_human_behavior/integrated_current_results_analysis.md`
4. Treat `organized/legacy_interim_reference/figureA4_interim_trajectory_geometry.png` as a preview, not a final result

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
- Archived superseded response-supervision source snapshots under `results/age_groups_response_supervision_interim/archive/`:
  - `figureRS1_response_supervision_eval05_summary.png`
  - `response_supervision_eval05_comparison.csv`
  - `figureRS3_agegroup_human_vs_model.png`
- Removed `.DS_Store` from `results/organized/`
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
- All original source directories (`age_groups/`, `age_groups_interim/`, `age_groups_response_supervision_interim/`, `age_groups_response_supervision_frozen/`, `proposal_aligned_behavior/`) remain intact for script compatibility
- Original result files inside the primary organized folders remain in place
- Root-level loose figures in `results/` remain outside this cleanup scope unless a later archive pass is requested
