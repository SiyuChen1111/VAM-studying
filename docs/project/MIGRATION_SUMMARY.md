# Migration Summary

**Date:** 2026-04-09  
**Status:** major directory migration completed with compatibility symlinks retained

---

## Purpose

This document records the current state of the repository after the major directory restructuring pass.

It answers four practical questions:

1. What is now the **canonical** directory layout?
2. Which old root-level names still exist as **compatibility symlinks**?
3. What was moved during the migration?
4. What remains to be done before the compatibility layer can be removed?

---

## 1. Canonical layout (now active)

The repository now uses the following canonical top-level structure:

```text
README.md
.gitignore

code/
├── scripts/
├── vam/
└── Kar/

data/
├── age_groups/
├── age_groups_matched/
└── vam_data/

artifacts/
├── checkpoints/
│   ├── age_groups/
│   ├── age_groups_matched/
│   └── test/
└── results/

docs/
notebooks/
logs/
archive/
```

This is now the **canonical internal structure**.

---

## 2. Compatibility layer (still present)

To avoid immediately breaking older scripts and serialized path references, the old root-level names still exist as **symlinks**.

### Current compatibility symlinks

| Root-level name | Points to |
|---|---|
| `scripts/` | `code/scripts/` |
| `data_age_groups/` | `data/age_groups/` |
| `data_age_groups_matched/` | `data/age_groups_matched/` |
| `vam_data/` | `data/vam_data/` |
| `checkpoints_age_groups/` | `artifacts/checkpoints/age_groups/` |
| `checkpoints_age_groups_matched/` | `artifacts/checkpoints/age_groups_matched/` |
| `checkpoints_test/` | `artifacts/checkpoints/test/` |
| `results/` | `artifacts/results/` |
| `vam/` | `code/vam/` |
| `Kar/` | `code/Kar/` |

These symlinks are deliberate and should be treated as a temporary compatibility layer.

---

## 3. What was moved

### Code

Moved into `code/`:

- all active root-level scripts → `code/scripts/`
- `vam/` → `code/vam/`
- `Kar/` → `code/Kar/`

### Data

Moved into `data/`:

- `data_age_groups/` → `data/age_groups/`
- `data_age_groups_matched/` → `data/age_groups_matched/`
- `vam_data/` → `data/vam_data/`

### Artifacts

Moved into `artifacts/`:

- `checkpoints_age_groups/` → `artifacts/checkpoints/age_groups/`
- `checkpoints_age_groups_matched/` → `artifacts/checkpoints/age_groups_matched/`
- `checkpoints_test/` → `artifacts/checkpoints/test/`
- `results/` → `artifacts/results/`

### Previously completed root cleanup (before this migration)

Already moved earlier:

- project docs → `docs/project/`
- timeline log → `docs/history/`
- PDFs → `docs/papers/`
- notes → `docs/notes/`
- legacy notebooks → `notebooks/legacy_root/`
- runtime logs / pid files → `logs/runtime_archive/`
- `requirements.txt` → `config/`

---

## 4. Path abstraction status

A shared path module now exists:

- `code/scripts/project_paths.py`

This is the basis for moving the workflow away from hardcoded root-relative paths.

### Scripts already updated to use `project_paths.py`

- `code/scripts/train_age_groups_efficient.py`
- `code/scripts/run_age_group_post_analysis.py`
- `code/scripts/run_matched_full_age_group_analysis.py`
- `code/scripts/prepare_age_group_data.py`

These are the first priority scripts because they sit on the active age-group WW path.

---

## 5. What remains before symlinks can be removed

The migration is structurally complete, but **not yet compatibility-complete**.

The remaining work is:

### A. Expand path abstraction coverage

Many scripts still directly reference:

- `data_age_groups*`
- `checkpoints_age_groups*`
- `vam_data`
- `vam/`
- `results/`

especially:

- orchestrator scripts
- monitor scripts
- extraction scripts
- figure-generation scripts

### B. Fix serialized CSV path dependencies

Some prepared CSVs already store literal paths like:

- `data_age_groups/.../stimulus_images/...`

These must be normalized before `data_age_groups*` symlinks can be safely removed.

### C. Re-check VAM asset assumptions

Some scripts assume a root-level `vam/` graphics directory. These need to be updated or abstracted before removing the `vam` symlink.

---

## 6. Current recommendation

### Safe now

- Use the **canonical directories** in new code and documentation:
  - `code/`
  - `data/`
  - `artifacts/`
- Keep the current root-level symlinks in place
- Continue migrating active scripts to `project_paths.py`

### Not safe yet

- Deleting the root-level compatibility symlinks
- Renaming the new canonical directories again
- Assuming all old workflow scripts are already updated

---

## 7. Practical meaning

The repo is now in a good intermediate state:

- **cleaner and more structured for GitHub / review**
- **still backward-compatible for the current workflow**

This is the intended result of the migration stage.

---

## 8. One-sentence summary

The repository has been physically migrated into a cleaner grouped layout (`code/`, `data/`, `artifacts/`, `docs/`, etc.), while the old root-level workflow names are temporarily retained as symlinks so that active scripts and serialized paths do not break during the transition.
