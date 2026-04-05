# Results organization guide

This folder is a **non-destructive organization layer**. It exists to make the current project state understandable **without changing active script output paths**.

The original result directories remain canonical write targets:
- `results/age_groups_interim/`
- `results/age_groups_response_supervision_interim/`
- `results/age_groups_response_supervision_frozen/`
- `results/proposal_aligned_behavior/`

The `results/organized/` tree is a curated read layer grouped by **research meaning and evidence level**, not by the script that generated a file.

## Cleanest structure for the project right now

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

This is the recommended structure to use **right now** because it preserves traceability and keeps the evidence hierarchy visible.

## Evidence levels and how to read them

| Organized folder | What it means | Trust level | Use for |
|---|---|---:|---|
| `proposal_aligned_human_behavior/` | Human-only behavioral evidence aligned to the proposal | Highest current trust | Human age effects, RT distributions, skewness, congruency, error-slower patterns |
| `current_best_response_supervision/` | Current-best model-side summaries under response supervision | Medium | Behavior-level human-vs-model comparison |
| `legacy_interim_reference/` | Older interim outputs retained for context | Reference only | Historical comparison, geometry preview |
| `archive/` | Older convenience mirrors moved out of the main path | Archive only | Provenance and low-priority historical browsing |
| `handoff/` | Continuation notes for later agents | Operational | Fast project restart and orientation |

## Critical scientific status

### Human side
- Human-side figures are currently the **most trustworthy behavior-level outputs**.
- These are the best place to start if the question is about age-related behavioral signatures.

### Model side
- Stage 2 supervision was corrected from `target_labels` to `response_labels`.
- That correction matters scientifically, but the model still tends toward overly high accuracy.
- Treat current model outputs as **current-best summaries**, not final converged mechanism evidence.

### Mechanism / geometry side
- `legacy_interim_reference/figureA4_interim_trajectory_geometry.png` is still useful.
- It is a **preview only**, not the final corrected-supervision mechanism figure.
- A fully updated geometry result still requires clean saved best-so-far outputs under the corrected branch.

## Folder map

### `current_best_response_supervision/`
Use this first for the **best available model-facing comparison outputs**.

Typical contents:
- `response_supervision_current_comparison.csv`
- `response_supervision_interim_memo.md`
- `figureRS1_response_supervision_summary.png`
- `figureRS2_response_supervision_multipanel.png`
- `figureRS3A_agegroup_accuracy_human_vs_model.png`
- `figureRS3B_agegroup_congruency_gap_human_vs_model.png`
- frozen summary carryovers:
  - `frozen_current_best_comparison.csv`
  - `frozen_current_best_memo.md`
  - `figureF1_frozen_current_best_behavior.png`

Interpretation:
- Good for **current-best behavior-level comparison**
- Not sufficient for final parameter-level or mechanism-level claims

### `proposal_aligned_human_behavior/`
Use this first for the **best current human behavioral evidence**.

Typical contents:
- `figureB2_human_multipanel_20_29_vs_80_89.png`
- `figureP1_human_rt_distributions.png`
- `figureP2_human_signature_summary.png`
- `human_behavior_signature_summary.csv`
- `figureP3_model_summary.png`
- `proposal_aligned_figure_note.md`
- `integrated_current_results_analysis.md`

Interpretation:
- Best source for age-related slowing, skewness, congruency, and error-slower behavior
- Strongest current evidence for proposal-facing behavioral claims

### `legacy_interim_reference/`
Use this only for **historical context or preview material**.

Typical contents:
- `figureA1_80_89_signatures.png`
- `figureA2_80_89_rt_distributions.png`
- `figureA3_interim_vs_final_comparison.png`
- `figureA4_interim_trajectory_geometry.png`
- `figureA4_interim_trajectory_spread.csv`
- `three_way_comparison_current.csv`
- `three_way_comparison_memo.md`

Interpretation:
- Useful for reference and comparison against earlier interim work
- **Do not** treat these as final corrected-supervision evidence

### `handoff/`
Use this for quick continuation:
- `CURRENT_STATUS.md`
- `agent_file_structure_cleanup_prompt.md`

### `archive/`
Use this only for **older convenience mirrors that should not compete with the main navigation**.

Current archived contents:
- `archive/age_groups_raw/`
- `archive/standalone_figures/`

Interpretation:
- retained for provenance and occasional backtracking
- intentionally removed from the top-level read path

## Best entry points for understanding current progress

If someone wants the fastest accurate overview, use this order:

1. `handoff/CURRENT_STATUS.md`
2. `proposal_aligned_human_behavior/integrated_current_results_analysis.md`
3. `proposal_aligned_human_behavior/figureB2_human_multipanel_20_29_vs_80_89.png`
4. `proposal_aligned_human_behavior/figureP2_human_signature_summary.png`
5. `current_best_response_supervision/response_supervision_current_comparison.csv`
6. `current_best_response_supervision/response_supervision_interim_memo.md`
7. `legacy_interim_reference/figureA4_interim_trajectory_geometry.png` only as a preview reference

## What should be treated as legacy reference only

The following should be read as **legacy / interim reference only**:
- everything under `legacy_interim_reference/`
- especially `legacy_interim_reference/figureA4_interim_trajectory_geometry.png`
- the original mirrored source directory `results/age_groups_interim/`, unless you are tracing provenance

## Archived supplemental folders

The following older convenience mirrors were moved under `organized/archive/` so they no longer compete with the primary navigation structure:
- `archive/age_groups_raw/`
- `archive/standalone_figures/`

This keeps them available for provenance while making the main entry path cleaner for later readers and agents.

## Important caution

Do **not** mix the following as if they were equivalent evidence levels:
- human-side final behavioral evidence,
- current-best response-supervision summaries,
- older interim geometry preview outputs.

They answer related questions, but they are **not** the same tier of evidence.

## Cleanup and archive guidance

Safe low-risk cleanup:
- remove OS-generated metadata files such as `.DS_Store`

Already archived inside `organized/`:
- `results/organized/archive/age_groups_raw/`
- `results/organized/archive/standalone_figures/`

Already archived inside source result directories:
- `results/age_groups_response_supervision_interim/archive/figureRS1_response_supervision_eval05_summary.png`
- `results/age_groups_response_supervision_interim/archive/response_supervision_eval05_comparison.csv`
- `results/age_groups_response_supervision_interim/archive/figureRS3_agegroup_human_vs_model.png`

Safe archive candidates from original source directories:
- lower-priority provenance tables that are not primary entry points, such as:
  - `results/age_groups_response_supervision_interim/response_supervision_multipanel_summary.csv`

Recommendation:
- leave remaining provenance-only files in place for now,
- do not delete them destructively,
- document them as lower-priority provenance artifacts unless the user later wants a dedicated archive pass.

For the detailed keep/copy/archive map, see `FILE_MAPPING.md`.
