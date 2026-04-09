# Commit Plan

**Date:** 2026-04-09  
**Purpose:** provide a staged, low-risk plan for what to commit to GitHub after the repo restructuring.

---

## Guiding principle

The repository has just gone through two major changes at once:

1. scientific documentation consolidation
2. directory restructuring into a new canonical grouped layout

Because of that, the safest Git strategy is:

> **commit the canonical structure first, then add selected results, then decide separately what to do with heavy checkpoints and legacy archives.**

Do **not** try to commit everything in one huge step.

---

## Commit batch 1 — canonical structure and documentation (**recommended first**) 

This is the most important commit to make first.

### Include

#### Root
- `README.md`
- `.gitignore`

#### Docs
- `docs/project/AGENTS.md`
- `docs/project/REPRODUCE_VAM.md`
- `docs/project/REPRODUCTION_GUIDE.md`
- `docs/project/ROOT_LEVEL_PY_CLASSIFICATION.md`
- `docs/project/research_plan.md`
- `docs/project/research_proposal_v4.md`
- `docs/project/REPO_RESTRUCTURING_PLAN.md`
- `docs/project/MIGRATION_SUMMARY.md`
- `docs/project/COMMIT_PLAN.md`
- `docs/history/logs.md`

#### Canonical code layout
- `code/scripts/`
- `code/vam/`
- `code/Kar/`
- `config/`

#### Key curated results / memos
- `artifacts/results/organized/`
- `artifacts/results/age_groups_full_matched_compare/`

### Why this batch first

This batch gives GitHub visitors:
- a clean root
- a canonical code location
- a readable research timeline
- supervisor-facing documentation
- a curated results entry point

even before any heavy artifacts are considered.

---

## Commit batch 2 — selected canonical result families (**optional but reasonable**) 

Only do this if repo size remains acceptable.

### Include

- `artifacts/results/age_groups/`
- `artifacts/results/age_groups_response_supervision_interim/`
- `artifacts/results/age_groups_response_supervision_frozen/`
- `artifacts/results/proposal_aligned_behavior/`
- `artifacts/results/age_groups_interim/`

### Why this batch is optional

These are scientifically useful, but they are not as essential as the curated `organized/` layer plus the current matched full comparison.

If GitHub size or cleanliness is a concern, you can skip this batch initially.

---

## Commit batch 3 — canonical data roots (**optional, depends on repo policy**) 

### Include only if you want data under version control

- `data/age_groups/`
- `data/age_groups_matched/`
- `data/vam_data/`

### Caution

These are central to reproducibility, but they may be large and may contain generated/prepared files better handled outside Git in some workflows.

If your GitHub repo is meant to be lighter-weight, you may prefer to:
- keep only the preparation scripts
- document where the data should come from
- avoid committing the full prepared data payload

---

## Commit batch 4 — checkpoints (**usually not recommended as a default GitHub commit**) 

### Usually keep out of the main repo

- `artifacts/checkpoints/age_groups/`
- `artifacts/checkpoints/age_groups_matched/`
- `artifacts/checkpoints/test/`

### Why

These are often the largest and least Git-friendly parts of the repository.

Recommended alternatives:
- keep locally
- archive outside GitHub
- or publish selected files only (for example `best_config.json`) if needed

---

## Commit batch 5 — legacy archive content (**usually do not include by default**) 

### Usually keep local / archive-only

- `archive/checkpoints_experiments/`
- `archive/response_label_refit_backup/`
- `archive/stage2_deprecated_2026-03-31/`

### Why

These are provenance-preserving historical materials, but they are not the cleanest public-facing view of the repo.

Only include them if you explicitly want a full historical archive in GitHub.

---

## Do **not** commit by default

### Runtime / local-only artifacts
- `logs/runtime_archive/`
- `*.pid`
- `*.log`

### Tool / environment local state
- `.trae/`
- `.sisyphus/`
- `anaconda_projects/`
- `.venv/`

### Legacy notebook clutter
- `notebooks/legacy_root/`

### Large local model assets
- `archive/model_assets/`

### Compatibility symlinks

Prefer committing the **canonical directories**, not the root-level compatibility symlinks, unless you explicitly want those links in the repo.

Examples of symlinks that do **not** need to be the main committed interface:
- `scripts`
- `results`
- `data_age_groups`
- `data_age_groups_matched`
- `vam_data`
- `checkpoints_age_groups`
- `checkpoints_age_groups_matched`
- `checkpoints_test`
- `vam`
- `Kar`

The canonical layout is now under:
- `code/`
- `data/`
- `artifacts/`

---

## Suggested minimal first commit

If you want the safest and cleanest first GitHub push, start with:

- `README.md`
- `.gitignore`
- `docs/`
- `code/`
- `config/`
- `artifacts/results/organized/`
- `artifacts/results/age_groups_full_matched_compare/`

This gives you:
- the cleaned repo structure
- the project narrative
- the main code
- the key result summaries

without forcing you to decide immediately about large raw data and checkpoints.

---

## Practical note on current git status

Because the repository was restructured, Git currently sees many old tracked paths as:
- deleted (`D`)

and the new canonical locations as:
- untracked (`??`)

This is expected.

The right mental model is:

> old tracked paths were moved into new canonical directories.

So do **not** panic at the large delete list.

The correct next step is to add the new canonical paths intentionally, batch by batch.

---

## One-sentence recommendation

Start by committing the new canonical structure and curated documentation/results first; defer large checkpoints, full prepared data, and historical archives until you decide whether the GitHub repo should be lightweight or fully archival.
