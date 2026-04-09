# Archive / cleanup plan

## Goal

Reduce confusion in the current workspace without breaking active scripts or destroying provenance.

## Principles

1. **Do not move canonical source result directories used by scripts.**
2. Prefer **archiving or mirroring** over destructive deletion.
3. Only delete files that are clearly low-risk and reproducible.
4. Keep `figureA4_interim_trajectory_geometry.png` and its spread CSV.

## Safe cleanup actions

### A. Safe to remove immediately
- `.DS_Store` files in `results/` and `results/organized/`

Reason:
- OS-generated metadata
- zero scientific value
- fully reproducible / unnecessary

### B. Safe to archive (not delete)
Move these into an archive subfolder so they stop competing with current outputs:

#### From `results/age_groups_response_supervision_interim/`
- `figureRS1_response_supervision_eval05_summary.png`
- `response_supervision_eval05_comparison.csv`
- `figureRS3_agegroup_human_vs_model.png`

Reason:
- these were earlier versions superseded by later response-supervision outputs
- they still have provenance value
- they are no longer the preferred read path

### C. Keep as active/current

#### Current model-facing summaries
- `response_supervision_current_comparison.csv`
- `response_supervision_interim_memo.md`
- `figureRS1_response_supervision_summary.png`
- `figureRS2_response_supervision_multipanel.png`
- `figureRS3A_agegroup_accuracy_human_vs_model.png`
- `figureRS3B_agegroup_congruency_gap_human_vs_model.png`

#### Human behavioral analysis
- everything under `results/proposal_aligned_behavior/`

#### Legacy mechanism preview
- `results/age_groups_interim/figureA4_interim_trajectory_geometry.png`
- `results/age_groups_interim/figureA4_interim_trajectory_spread.csv`

#### Handoff / organization docs
- everything under `results/organized/`

## Python file cleanup guidance

Do **not** delete the many top-level experimental scripts blindly.

Instead, classify them into:
- active WW-main-path utilities
- active diagnostic utilities
- inactive historical prototypes

The new prototype / experiment scripts should be left in place for now because they still document what was tried.

## Recommended next archive pass

If a later cleanup pass is desired, the next safe target is to move clearly historical or failed prototype helpers into a dedicated `archive/prototypes/` folder, but only after a careful dependency check.
