from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'README.md').exists() and (parent / '.gitignore').exists():
            return parent
    raise RuntimeError('Could not locate project root from project_paths.py')


PROJECT_ROOT = _find_project_root()
DATA_ROOT = PROJECT_ROOT / 'data'
ARTIFACTS_ROOT = PROJECT_ROOT / 'artifacts'
CODE_ROOT = PROJECT_ROOT / 'code'
DATA_AGE_GROUPS_ROOT = DATA_ROOT / "age_groups"
DATA_AGE_GROUPS_MATCHED_ROOT = DATA_ROOT / "age_groups_matched"
CHECKPOINTS_AGE_GROUPS_ROOT = ARTIFACTS_ROOT / "checkpoints" / "age_groups"
CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT = ARTIFACTS_ROOT / "checkpoints" / "age_groups_matched"
CHECKPOINTS_TEST_ROOT = ARTIFACTS_ROOT / "checkpoints" / "test"
RESULTS_ROOT = ARTIFACTS_ROOT / "results"
VAM_DATA_ROOT = DATA_ROOT / "vam_data"
VAM_ROOT = CODE_ROOT / "vam"


def rel_to_root(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def age_group_data_dir(age_group: str, matched: bool = False) -> Path:
    root = DATA_AGE_GROUPS_MATCHED_ROOT if matched else DATA_AGE_GROUPS_ROOT
    return root / age_group


def age_group_stage2_dir(age_group: str, matched: bool = False) -> Path:
    root = CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT if matched else CHECKPOINTS_AGE_GROUPS_ROOT
    return root / age_group / "stage2"


def age_group_stage1_dir(age_group: str, matched: bool = False) -> Path:
    root = CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT if matched else CHECKPOINTS_AGE_GROUPS_ROOT
    return root / age_group / "stage1"
