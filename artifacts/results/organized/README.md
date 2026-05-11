# Results organization guide

This folder is a **non-destructive organization layer**. It exists to make the current project state understandable **without changing active script output paths**.

Historical memos may still refer to `results/...` paths. In the current repo layout, read those as compatibility-path aliases for the canonical `artifacts/results/...` tree.

The original result directories remain canonical write targets:
- `artifacts/results/age_groups_interim/`
- `artifacts/results/age_groups_response_supervision_interim/`
- `artifacts/results/age_groups_response_supervision_frozen/`
- `artifacts/results/proposal_aligned_behavior/`

The `artifacts/results/organized/` tree is a curated read layer grouped by **research meaning and evidence level**, not by the script that generated a file.

## Cleanest structure for the project right now

```text
artifacts/results/
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

### Phase 18 DMC+Var→WW side
- The retained Phase 18 branch-local artifact root is `artifacts/results/rt_model_dmc_var_ww/`.
- The kept generated figure for this branch is `../rt_model_dmc_var_ww/rt_model_breakdown.png`.
- The same retained figure is also expected in PDF form as `../rt_model_dmc_var_ww/rt_model_breakdown.pdf`.
- That figure corresponds to the best retained negative-ΔRT branch-local result from `smoke_a5_s3_neg_drt`, matched to `epoch 10`.
- This branch matters because it is the first **retained/reported branch-local result we currently surface** where the DMC + variational-evidence → Wong-Wang combination produces negative error-minus-correct ΔRT.
- Treat it as a **mechanistically informative breakthrough, not a promoted final solution**: it shows the error-direction problem can be crossed, but the resulting RT regime still does not clear the repo-wide “human-like enough” gate.
- For retained behavior/result figures in this repo, figures should export paired `.png` + `.pdf` artifacts rather than PNG-only outputs.

### Successor-branch screening side
- The bounded `artifacts/results/rt_model_next_step/` program has already been executed.
- Its saved final verdict is `NO_SUCCESSOR_BRANCH_CLEARED_GATES`.
- Read that bundle as a **completed negative-result screen**, not as an active branch program waiting for one more promotion run.

### HSFA repair side
- The bounded `artifacts/results/rt_model_hsfa_v3_1/` repair bundle has also already been executed.
- Its saved aggregate repair verdict is `HSFA_V3_1_KILL_NEED_STAGE1_UNCERTAINTY_OR_BNN`, with final synthesis token `HSFA_V3_1_KILL_AND_START_STAGE1_UNCERTAINTY_PLAN`.
- Read that bundle as a **completed negative-result repair audit**, not as an active Stage-2 branch waiting for one more local fix.

### Active SPEA calibration side
- A live bounded follow-up now exists under `artifacts/results/rt_model_semisup_spea_v1_1_calibration/`.
- Its protocol lock is `../rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md`.
- Read this branch as an **active calibration follow-up** on the earlier SemiSupervisedSPEA partial result, not as a new architecture-search family.
- Batch 1 currently means protocol lock + trainer/readout/test scaffolding only; there is **no new scientific verdict yet** for SPEA v1.1.

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
- `HANDOFF_INDEX.md`
- `CURRENT_STATUS.md`
- `agent_file_structure_cleanup_prompt.md`
- `single_subject_rt_response_research_judgment_memo.md`

Interpretation:
- `HANDOFF_INDEX.md` explains which handoff files are current, authoritative, or historical.
- `CURRENT_STATUS.md` is the fastest operational restart note.
- `single_subject_rt_response_research_judgment_memo.md` is the current best single-subject judgment document for WW clean/noise and aligned WW vs AccumRNN.

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
2. `handoff/HANDOFF_INDEX.md`
3. `../rt_model_hsfa_v3_1/06_synthesis/hsfa_v3_1_final_memo.md`
4. `../rt_model_next_step/06_synthesis/final_successor_branch_memo.md`
5. `handoff/single_subject_rt_response_research_judgment_memo.md`
6. `../rt_model_semisup_spea_v1_1_calibration/00_protocol/spea_v1_1_calibration_protocol.md` when the task is about the active SPEA calibration follow-up
7. `../rt_model_dmc_var_ww/summary_smoke.md` when the task is specifically about the Phase 18 DMC+Var→WW breakthrough branch
8. `../rt_model_dmc_var_ww/rt_model_breakdown.png` for the retained Phase 18 comparison figure
9. `../rt_model_dmc_var_ww/rt_model_breakdown.pdf` for the matching retained vector/PDF export
10. `proposal_aligned_human_behavior/integrated_current_results_analysis.md`
11. `proposal_aligned_human_behavior/figureB2_human_multipanel_20_29_vs_80_89.png`
12. `proposal_aligned_human_behavior/figureP2_human_signature_summary.png`
13. `current_best_response_supervision/response_supervision_current_comparison.csv`
14. `current_best_response_supervision/response_supervision_interim_memo.md`
15. `legacy_interim_reference/figureA4_interim_trajectory_geometry.png` only as a preview reference

## What should be treated as legacy reference only

The following should be read as **legacy / interim reference only**:
- everything under `legacy_interim_reference/`
- especially `legacy_interim_reference/figureA4_interim_trajectory_geometry.png`
- the original mirrored source directory `artifacts/results/age_groups_interim/`, unless you are tracing provenance

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
- `artifacts/results/organized/archive/age_groups_raw/`
- `artifacts/results/organized/archive/standalone_figures/`

Already archived inside source result directories:
- `artifacts/results/age_groups_response_supervision_interim/archive/figureRS1_response_supervision_eval05_summary.png`
- `artifacts/results/age_groups_response_supervision_interim/archive/response_supervision_eval05_comparison.csv`
- `artifacts/results/age_groups_response_supervision_interim/archive/figureRS3_agegroup_human_vs_model.png`

Safe archive candidates from original source directories:
- lower-priority provenance tables that are not primary entry points, such as:
  - `artifacts/results/age_groups_response_supervision_interim/response_supervision_multipanel_summary.csv`

Recommendation:
- leave remaining provenance-only files in place for now,
- do not delete them destructively,
- document them as lower-priority provenance artifacts unless the user later wants a dedicated archive pass.

For the detailed keep/copy/archive map, see `FILE_MAPPING.md`.
