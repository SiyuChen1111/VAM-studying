import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from project_paths import (
    CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT,
    CHECKPOINTS_AGE_GROUPS_ROOT,
    RESULTS_ROOT,
    age_group_data_dir,
    age_group_stage2_dir,
    rel_to_root,
)
from train_age_groups_efficient import (
    DIRECTION_MAP,
    attach_flanker_labels_from_csv,
    evaluate_cached_stage2_params,
    set_random_seed,
    to_jsonable,
    validate_cached_stage2_inputs,
)


AGE_GROUPS = ("20-29", "80-89")
N_QUANTILES = 5
SUBJECTS_PER_GROUP = 4
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_single_subject"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-subject dynamic selection workflow entrypoint. "
            "Supports: audit-baseline, verify-alignment, select-subjects, simulate, full."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("audit-baseline", "verify-alignment", "select-subjects", "simulate", "full"),
        help=(
            "Operation mode. audit-baseline writes baseline manifest; verify-alignment validates CSV↔NPZ row identity; "
            "select-subjects writes deterministic subject manifest; simulate runs bounded per-subject scale search; "
            "full runs audit→alignment→selection→simulate for both age groups."
        ),
    )
    parser.add_argument(
        "--age_group",
        default=None,
        choices=AGE_GROUPS,
        help="Age group required for verify-alignment.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory to write workflow artifacts.",
    )
    parser.add_argument(
        "--baseline_inputs_json",
        default=None,
        help=(
            "Optional JSON file overriding baseline input paths for audit-baseline QA. "
            "If provided, it must include both age groups and explicit paths for cached NPZ/CSV and Stage-2 artifacts."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    return parser.parse_args()


def _stable_int_seed(value: str) -> int:
    # Stable across runs/processes.
    import zlib

    return int(zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF)


def _load_npz_params_no_scale(npz_path: Path) -> Dict[str, Any]:
    params_npz = np.load(npz_path)
    params = {k: params_npz[k] for k in params_npz.files if k != "scale"}
    return cast(Dict[str, Any], params)


def _load_stage2_config(stage2_cfg_path: Path) -> dict:
    cfg = _load_json(stage2_cfg_path)
    cfg = dict(cfg)
    cfg["_source_config_path"] = str(stage2_cfg_path)
    return cfg


def _concat_cached_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in sorted(keys):
        if key not in a or key not in b:
            raise ValueError(f"CACHED_CONCAT_KEY_MISMATCH: key={key} a={key in a} b={key in b}")
        va = a[key]
        vb = b[key]
        if not isinstance(va, np.ndarray) or not isinstance(vb, np.ndarray):
            raise ValueError(f"CACHED_CONCAT_NON_ARRAY: key={key} type_a={type(va)} type_b={type(vb)}")
        out[key] = np.concatenate([va, vb], axis=0)
    return out


def _filter_cached_by_mask(cached: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in cached.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.shape[0] != mask.shape[0]:
            raise ValueError(f"CACHED_MASK_SHAPE_MISMATCH: key={key} n={value.shape[0]} mask={mask.shape[0]}")
        out[key] = value[mask]
    return out


def _compute_earliest_incongruent_caf(rt_s: np.ndarray, correct: np.ndarray) -> float:
    df = pd.DataFrame({"rt": rt_s.astype(np.float32), "correct": correct.astype(bool)})
    df["bin"] = pd.qcut(df["rt"], q=N_QUANTILES, labels=False, duplicates="drop")
    earliest_bin = int(df["bin"].min())
    return float(df.loc[df["bin"] == earliest_bin, "correct"].mean())


def _compute_incongruent_error_minus_correct_rt(rt_s: np.ndarray, correct: np.ndarray, congruency: np.ndarray) -> float:
    incong = np.asarray(congruency, dtype=np.int64) == 1
    if not bool(np.any(incong)):
        return float("nan")
    rt_s = np.asarray(rt_s, dtype=np.float32)
    correct = np.asarray(correct, dtype=bool)
    err_vals = rt_s[incong & (~correct)]
    cor_vals = rt_s[incong & correct]
    if err_vals.size == 0 or cor_vals.size == 0:
        return float("nan")
    return float(err_vals.mean() - cor_vals.mean())


def _build_scale_grid(center: float) -> list[float]:
    raw = [center - 0.02, center - 0.01, center, center + 0.01, center + 0.02]
    clipped = [float(np.clip(x, 0.05, 0.15)) for x in raw]
    return clipped


def _load_train_test_cached_and_csv(age_group: str) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data_dir = age_group_data_dir(age_group)
    stage2_dir = age_group_stage2_dir(age_group)

    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Missing prepared train/test CSVs for age_group={age_group}: train={train_csv.exists()} test={test_csv.exists()}"
        )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    train_cached, test_cached = validate_cached_stage2_inputs(
        age_group,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    train_cached = attach_flanker_labels_from_csv(train_cached, str(train_csv))
    test_cached = attach_flanker_labels_from_csv(test_cached, str(test_csv))
    combined_cached = _concat_cached_dicts(train_cached, test_cached)

    # Fail closed on any alignment mismatch.
    build_alignment_report(combined_df, combined_cached)
    return combined_df, combined_cached


def simulate_age_group(*, age_group: str, output_root: Path, seed: int, device: str, choice_temperature: float) -> None:
    baseline = load_baseline_manifest(output_root)
    age_entry = baseline["age_groups"][age_group]
    stage2_cfg_path = Path(age_entry["stage2_config_path"])
    stage2_params_path = Path(age_entry["stage2_params_path"])
    if not stage2_cfg_path.exists():
        raise FileNotFoundError(f"Missing stage2 config for age_group={age_group}: {stage2_cfg_path}")
    if not stage2_params_path.exists():
        raise FileNotFoundError(f"Missing stage2 params for age_group={age_group}: {stage2_params_path}")

    stage2_cfg = _load_stage2_config(stage2_cfg_path)
    time_steps_raw = stage2_cfg.get("time_steps")
    if time_steps_raw is None:
        raise ValueError(f"STAGE2_CONFIG_MISSING_TIME_STEPS: age_group={age_group} path={stage2_cfg_path}")
    time_steps = int(time_steps_raw)
    params = _load_npz_params_no_scale(stage2_params_path)

    manifest_dir = output_root / "manifest"
    subject_manifest_path = manifest_dir / "subject_manifest.csv"
    if not subject_manifest_path.exists():
        raise FileNotFoundError(f"Missing subject manifest at {subject_manifest_path}. Run --mode select-subjects first.")
    subject_manifest = pd.read_csv(subject_manifest_path)
    subjects = subject_manifest.loc[subject_manifest["age_group"] == age_group, "user_id"].astype(str).tolist()
    if not subjects:
        raise ValueError(f"SIMULATE_NO_SUBJECTS_FOR_AGE_GROUP: age_group={age_group}")

    combined_df, combined_cached = _load_train_test_cached_and_csv(age_group)
    combined_df = combined_df.copy()
    combined_df["user_id"] = combined_df["user_id"].astype(str)

    # Derive scale center from the selection pool (train+test) to match the current deterministic manifest.
    age_group_median_rt = float(np.median(combined_df["response_time"].to_numpy(dtype=np.float32) / 1000.0))
    subject_medians = (
        combined_df.groupby("user_id", sort=True)["response_time"].median().to_numpy(dtype=np.float32) / 1000.0
    )
    center = 0.1 * float(np.median(subject_medians / max(age_group_median_rt, 1e-6)))
    grid = _build_scale_grid(center)

    out_age_dir = output_root / age_group
    out_age_dir.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict[str, Any]] = []
    boundary_hits: list[str] = []

    for uid in subjects:
        user_mask = (combined_df["user_id"].to_numpy(dtype=str) == uid)
        if not bool(np.any(user_mask)):
            exclusions.append({"age_group": age_group, "user_id": uid, "reason": "SUBJECT_NOT_FOUND_IN_POOL"})
            continue

        user_dir = out_age_dir / f"user_{uid}"
        user_dir.mkdir(parents=True, exist_ok=True)

        # Human metrics for selection.
        user_df = combined_df.loc[user_mask].copy()
        user_df["rt_s"] = user_df["response_time"].to_numpy(dtype=np.float32) / 1000.0
        user_df["correct"] = user_df["response_direction"].astype(str) == user_df["target_direction"].astype(str)
        user_df["incongruent"] = user_df["target_direction"].astype(str) != user_df["flanker_direction"].astype(str)
        incong_df = user_df.loc[user_df["incongruent"], ["rt_s", "correct"]]
        if incong_df.empty:
            exclusions.append({"age_group": age_group, "user_id": uid, "reason": "NO_INCONGRUENT_TRIALS"})
            continue
        human_earliest = _compute_earliest_incongruent_caf(
            incong_df["rt_s"].to_numpy(dtype=np.float32),
            incong_df["correct"].to_numpy(dtype=bool),
        )
        human_err_minus_cor = _compute_incongruent_error_minus_correct_rt(
            user_df["rt_s"].to_numpy(dtype=np.float32),
            user_df["correct"].to_numpy(dtype=bool),
            user_df["incongruent"].to_numpy(dtype=np.int64),
        )

        cached_user = _filter_cached_by_mask(combined_cached, user_mask)
        if len(cached_user.get("logits", [])) != len(user_df):
            raise ValueError(f"SIMULATE_SUBJECT_CACHE_LENGTH_MISMATCH: age_group={age_group} user_id={uid}")

        eval_rows: list[dict[str, Any]] = []
        for scale_index, scale in enumerate(grid):
            eval_seed = int(seed + _stable_int_seed(f"{age_group}:{uid}") % 100000 + scale_index)
            predictions, _ = evaluate_cached_stage2_params(
                params=params,
                scale=float(scale),
                time_steps=time_steps,
                cached=cached_user,
                device=device,
                choice_temperature=float(choice_temperature),
                rt_readout_mode="baseline",
                readout_config=None,
                selection_config=dynamic_selection_phase1_config(),
                random_seed=eval_seed,
                rt_shape_focus=False,
            )
            pred_rt = predictions["pred_rt"].astype(np.float32)
            pred_choice = predictions["pred_choice"].astype(np.int64)
            congruency = cached_user["congruency"].astype(np.int64)
            target_labels = cached_user["target_labels"].astype(np.int64)
            pred_correct = pred_choice == target_labels

            incong_mask = congruency == 1
            pred_earliest = _compute_earliest_incongruent_caf(pred_rt[incong_mask], pred_correct[incong_mask])
            pred_err_minus_cor = _compute_incongruent_error_minus_correct_rt(pred_rt, pred_correct, congruency)

            primary_abs_err = float("nan")
            if np.isfinite(human_err_minus_cor) and np.isfinite(pred_err_minus_cor):
                primary_abs_err = float(abs(pred_err_minus_cor - human_err_minus_cor))
            caf_abs_err = float(abs(pred_earliest - human_earliest))
            center_dist = float(abs(float(scale) - float(center)))
            eval_rows.append(
                {
                    "scale": float(scale),
                    "eval_seed": eval_seed,
                    "pred_earliest_incongruent_caf": float(pred_earliest),
                    "pred_incongruent_error_minus_correct_rt": float(pred_err_minus_cor),
                    "primary_abs_err": primary_abs_err,
                    "caf_abs_err": caf_abs_err,
                    "center_dist": center_dist,
                    "predictions": predictions,
                }
            )

        eval_df = pd.DataFrame([{k: v for k, v in row.items() if k != "predictions"} for row in eval_rows])
        # Select best scale using locked rules; fail closed if none valid.
        valid_primary = eval_df["primary_abs_err"].apply(np.isfinite)
        used_fallback = False
        if bool(valid_primary.any()):
            candidates = eval_df.loc[valid_primary].copy()
            candidates = candidates.sort_values(
                by=["primary_abs_err", "caf_abs_err", "center_dist", "scale"],
                ascending=[True, True, True, True],
                kind="mergesort",
            )
            chosen_scale = float(candidates.iloc[0]["scale"])
        else:
            # Fallback: no valid error-minus-correct metric; minimize CAF error then nearest-to-center.
            used_fallback = True
            candidates = eval_df.copy().sort_values(
                by=["caf_abs_err", "center_dist", "scale"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            chosen_scale = float(candidates.iloc[0]["scale"])

        boundary_hit = bool(np.isclose(chosen_scale, grid[0], atol=1e-12) or np.isclose(chosen_scale, grid[-1], atol=1e-12))
        if boundary_hit:
            boundary_hits.append(uid)

        # Persist the predictions for the chosen scale.
        chosen_row = next(r for r in eval_rows if float(r["scale"]) == float(chosen_scale))
        chosen_predictions = cast(Dict[str, np.ndarray], chosen_row["predictions"])
        np.savez_compressed(
            user_dir / "predictions.npz",
            pred_rt=chosen_predictions["pred_rt"],
            pred_choice=chosen_predictions["pred_choice"],
            choice_logits=chosen_predictions["choice_logits"],
            decision_times_class=chosen_predictions["decision_times_class"],
            true_rt=cached_user["rts"],
            target_labels=cached_user["target_labels"],
            response_labels=cached_user["response_labels"],
            flanker_labels=cached_user["flanker_labels"],
            congruency=cached_user["congruency"],
            **{
                k: v
                for k, v in chosen_predictions.items()
                if k not in {"pred_rt", "pred_choice", "choice_logits", "decision_times_class"}
            },
        )

        summary = {
            "age_group": age_group,
            "user_id": uid,
            "selection_pool": "train_plus_test",
            "stage2_config_path": safe_rel_to_root(stage2_cfg_path),
            "stage2_params_path": safe_rel_to_root(stage2_params_path),
            "time_steps": time_steps,
            "choice_temperature": float(choice_temperature),
            "seed": int(seed),
            "selection_config": dynamic_selection_phase1_config(),
            "age_group_median_rt": age_group_median_rt,
            "scale_center": float(center),
            "scale_grid": [float(x) for x in grid],
            "grid_size": 5,
            "selected_scale": float(chosen_scale),
            "boundary_hit": bool(boundary_hit),
            "used_fallback": bool(used_fallback),
            "human_metrics": {
                "earliest_incongruent_caf": float(human_earliest),
                "incongruent_error_minus_correct_rt": float(human_err_minus_cor),
            },
            "grid_evaluations": eval_df.to_dict(orient="records"),
        }
        (user_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))

    # Write run-level logs.
    if exclusions:
        pd.DataFrame(exclusions).to_csv(out_age_dir / "subject_exclusions.csv", index=False)
    run_summary = {
        "age_group": age_group,
        "selection_pool": "train_plus_test",
        "n_selected_subjects": int(len(subjects)),
        "n_excluded_subjects": int(len(exclusions)),
        "boundary_hits": boundary_hits,
        "n_boundary_hits": int(len(boundary_hits)),
        "scale_center": float(center),
        "scale_grid": [float(x) for x in grid],
    }
    (out_age_dir / "run_summary.json").write_text(json.dumps(to_jsonable(run_summary), indent=2))


def run_full(output_root: Path, seed: int, device: str, choice_temperature: float) -> None:
    audit_baseline(output_root=output_root, seed=seed, device=device, choice_temperature=choice_temperature)
    for age_group in AGE_GROUPS:
        verify_alignment(age_group, output_root)
    select_subjects(output_root)
    for age_group in AGE_GROUPS:
        simulate_age_group(
            age_group=age_group,
            output_root=output_root,
            seed=seed,
            device=device,
            choice_temperature=choice_temperature,
        )


def safe_rel_to_root(path: Path) -> str:
    try:
        return rel_to_root(path)
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_baseline_manifest(output_root: Path) -> dict:
    manifest_path = output_root / "manifest" / "baseline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing baseline manifest at {manifest_path}. Run --mode audit-baseline first."
        )
    return _load_json(manifest_path)


def locate_stage2_param_artifacts(age_group: str) -> Tuple[Path, Path]:
    """Return (best_config_path, best_model_params_path) for Stage-2.

    Prefer canonical age-group Stage-2 paths. If missing (observed for 20-29
    in some local checkouts), fall back to the closest repo-local archived
    artifact and surface that choice in the output manifest.
    """

    stage2_dir = age_group_stage2_dir(age_group)
    cfg = stage2_dir / "best_config.json"
    params = stage2_dir / "best_model_params.npz"
    if cfg.exists() and params.exists():
        return cfg, params

    archived_root = CHECKPOINTS_AGE_GROUPS_ROOT / "archive_stage2_pre_cached_restart" / age_group
    archived_cfg = archived_root / "best_config.pre_cached_restart.json"
    archived_params = archived_root / "best_model_params.pre_cached_restart.npz"
    if archived_cfg.exists() and archived_params.exists():
        return archived_cfg, archived_params

    matched_dir = CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT / age_group / "stage2"
    matched_cfg = matched_dir / "best_config.json"
    matched_params = matched_dir / "best_model_params.npz"
    if matched_cfg.exists() and matched_params.exists():
        return matched_cfg, matched_params

    missing = [str(p) for p in (cfg, params, archived_cfg, archived_params, matched_cfg, matched_params) if not p.exists()]
    raise FileNotFoundError(
        f"Could not locate Stage-2 best_config/best_model_params for age_group={age_group}. Missing candidates: {missing}"
    )


def load_baseline_inputs_override(path: Path) -> Dict[str, Dict[str, str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or "age_groups" not in payload:
        raise ValueError(f"BASELINE_INPUTS_INVALID_SCHEMA: expected top-level dict with 'age_groups' in {path}")
    age_groups = payload["age_groups"]
    if not isinstance(age_groups, dict):
        raise ValueError(f"BASELINE_INPUTS_INVALID_SCHEMA: 'age_groups' must be a dict in {path}")

    required_age_groups = set(AGE_GROUPS)
    provided = set(age_groups.keys())
    missing_groups = required_age_groups - provided
    if missing_groups:
        missing_str = ",".join(sorted(missing_groups))
        raise ValueError(
            f"BASELINE_INPUTS_MISSING_AGE_GROUP: missing={missing_str} provided={sorted(provided)} file={path}"
        )

    required_keys = {"cache_npz", "csv_path", "stage2_config_path", "stage2_params_path"}
    resolved: Dict[str, Dict[str, str]] = {}
    for age_group in AGE_GROUPS:
        entry = age_groups.get(age_group)
        if not isinstance(entry, dict):
            raise ValueError(f"BASELINE_INPUTS_INVALID_SCHEMA: age_group={age_group} entry must be a dict in {path}")
        missing_keys = required_keys - set(entry.keys())
        if missing_keys:
            missing_str = ",".join(sorted(missing_keys))
            raise ValueError(
                f"BASELINE_INPUTS_MISSING_KEY: age_group={age_group} missing={missing_str} file={path}"
            )
        resolved[age_group] = {k: str(entry[k]) for k in required_keys}

    return resolved


def compute_congruency_from_labels(target_labels: np.ndarray, flanker_labels: np.ndarray) -> np.ndarray:
    return (np.asarray(target_labels) != np.asarray(flanker_labels)).astype(np.int64)


def safe_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or np.allclose(values, values[0]):
        return 0.0
    skew = stats.skew(values, bias=False)
    return 0.0 if np.isnan(skew) else float(skew)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def compute_subject_metrics(test_df: pd.DataFrame, age_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (eligible_metrics_df, excluded_df)."""
    required_cols = [
        "user_id",
        "target_direction",
        "response_direction",
        "flanker_direction",
        "response_time",
    ]
    missing_cols = [c for c in required_cols if c not in test_df.columns]
    if missing_cols:
        raise ValueError(f"SUBJECT_SELECTION_MISSING_COLUMNS: age_group={age_group} missing={missing_cols}")

    df = test_df.copy()
    df["rt_s"] = df["response_time"].to_numpy(dtype=np.float32) / 1000.0
    df["correct"] = df["response_direction"].astype(str) == df["target_direction"].astype(str)
    df["incongruent"] = df["target_direction"].astype(str) != df["flanker_direction"].astype(str)

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for uid, user_df in df.groupby("user_id", sort=True):
        user_df = user_df.copy()
        incong = user_df.loc[user_df["incongruent"]]
        incong_n = int(len(incong))
        incong_errors_n = int((~incong["correct"]).sum())
        unique_rts = int(user_df["response_time"].nunique())

        reasons: list[str] = []
        if incong_n < 20:
            reasons.append("INSUFFICIENT_INCONGRUENT_TRIALS")
        if incong_errors_n < 3:
            reasons.append("INSUFFICIENT_INCONGRUENT_ERRORS")
        if unique_rts < 10:
            reasons.append("INSUFFICIENT_UNIQUE_RTS")

        if reasons:
            excluded.append({"user_id": uid, "age_group": age_group, "reason": ";".join(reasons)})
            continue

        # Metric 1: earliest incongruent CAF accuracy (first RT quantile bin in incongruent trials).
        incong = incong[["rt_s", "correct"]].copy()
        incong["bin"] = pd.qcut(incong["rt_s"], q=N_QUANTILES, labels=False, duplicates="drop")
        earliest_bin = int(incong["bin"].min())
        earliest_acc = float(incong.loc[incong["bin"] == earliest_bin, "correct"].mean())

        # Metric 2: incongruent error-minus-correct RT.
        incong_full = user_df.loc[user_df["incongruent"], ["rt_s", "correct"]]
        err_vals = incong_full.loc[~incong_full["correct"], "rt_s"].to_numpy(dtype=np.float32)
        cor_vals = incong_full.loc[incong_full["correct"], "rt_s"].to_numpy(dtype=np.float32)
        err_rt = float(err_vals.mean()) if err_vals.size else float("nan")
        cor_rt = float(cor_vals.mean()) if cor_vals.size else float("nan")
        err_minus_cor = err_rt - cor_rt if np.isfinite(err_rt) and np.isfinite(cor_rt) else float("nan")

        # Metric 3: RT skewness on the full test split for this subject.
        skew = safe_skew(user_df["rt_s"].to_numpy(dtype=np.float32))

        if not np.isfinite(earliest_acc) or not np.isfinite(err_minus_cor) or not np.isfinite(skew):
            excluded.append({
                "user_id": uid,
                "age_group": age_group,
                "reason": "UNDEFINED_METRIC",
            })
            continue

        rows.append(
            {
                "age_group": age_group,
                "user_id": str(uid),
                "earliest_incongruent_caf": earliest_acc,
                "incongruent_error_minus_correct_rt": err_minus_cor,
                "rt_skewness": skew,
                "incongruent_trials": incong_n,
                "incongruent_error_trials": incong_errors_n,
                "unique_rt_values": unique_rts,
            }
        )

    eligible = pd.DataFrame(rows)
    excluded_df = pd.DataFrame(excluded)
    return eligible, excluded_df


def select_subjects_for_age_group(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        raise ValueError("SUBJECT_SELECTION_NO_ELIGIBLE_SUBJECTS")
    if len(metrics) < SUBJECTS_PER_GROUP:
        raise ValueError(
            f"SUBJECT_SELECTION_INSUFFICIENT_ELIGIBLE: need={SUBJECTS_PER_GROUP} have={len(metrics)}"
        )

    df = metrics.copy()
    df["z_earliest_incongruent_caf"] = zscore(df["earliest_incongruent_caf"].to_numpy(dtype=np.float64))
    df["z_incongruent_error_minus_correct_rt"] = zscore(
        df["incongruent_error_minus_correct_rt"].to_numpy(dtype=np.float64)
    )
    df["z_rt_skewness"] = zscore(df["rt_skewness"].to_numpy(dtype=np.float64))
    df["extreme_score"] = (
        np.abs(df["z_earliest_incongruent_caf"]) +
        np.abs(df["z_incongruent_error_minus_correct_rt"]) +
        np.abs(df["z_rt_skewness"])
    ) / 3.0

    # Hybrid strata constraint based on earliest incongruent CAF rank.
    caf_rank = df["earliest_incongruent_caf"].rank(method="first", ascending=True)
    df["caf_rank"] = caf_rank.astype(int)
    midpoint = int(np.ceil(len(df) / 2.0))
    df["caf_half"] = np.where(df["caf_rank"] <= midpoint, "lower", "upper")

    # Deterministic ordering for selection.
    df = df.sort_values(
        by=["extreme_score", "incongruent_error_trials", "user_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    lower_best = df.loc[df["caf_half"] == "lower"].head(1)
    upper_best = df.loc[df["caf_half"] == "upper"].head(1)
    if lower_best.empty or upper_best.empty:
        raise ValueError("SUBJECT_SELECTION_STRATA_UNSATISFIABLE")

    chosen_ids = set()
    chosen_rows = []
    for piece in (lower_best, upper_best):
        row = piece.iloc[0]
        chosen_rows.append(row)
        chosen_ids.add(row["user_id"])

    for _, row in df.iterrows():
        if row["user_id"] in chosen_ids:
            continue
        chosen_rows.append(row)
        chosen_ids.add(row["user_id"])
        if len(chosen_rows) >= SUBJECTS_PER_GROUP:
            break

    selected = pd.DataFrame(chosen_rows).head(SUBJECTS_PER_GROUP)
    return selected.reset_index(drop=True)


def select_subjects(output_root: Path) -> tuple[Path, Path, Path]:
    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    selected_frames: list[pd.DataFrame] = []
    excluded_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for age_group in AGE_GROUPS:
        data_dir = age_group_data_dir(age_group)
        train_csv = data_dir / "train_data.csv"
        test_csv = data_dir / "test_data.csv"
        if not train_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(
                f"Missing prepared train/test CSVs for age_group={age_group}: train={train_csv.exists()} test={test_csv.exists()}"
            )
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        pool_df = pd.concat([train_df, test_df], ignore_index=True)
        test_users = int(test_df["user_id"].nunique())

        eligible, excluded = compute_subject_metrics(pool_df, age_group=age_group)
        excluded_frames.append(excluded)
        selected = select_subjects_for_age_group(eligible)
        selected_frames.append(selected)
        summary_rows.append(
            {
                "age_group": age_group,
                "pool_trials": int(len(pool_df)),
                "pool_users": int(pool_df["user_id"].nunique()),
                "test_only_users": test_users,
                "eligible_subjects": int(eligible["user_id"].nunique()) if not eligible.empty else 0,
                "excluded_subjects": int(excluded["user_id"].nunique()) if not excluded.empty else 0,
                "selected_subjects": int(selected["user_id"].nunique()),
                "selection_pool": "train_plus_test_fallback_due_to_test_only_insufficiency" if test_users < SUBJECTS_PER_GROUP else "train_plus_test",
            }
        )

    subject_manifest = pd.concat(selected_frames, ignore_index=True)
    if len(subject_manifest) != SUBJECTS_PER_GROUP * len(AGE_GROUPS):
        raise RuntimeError(f"SUBJECT_MANIFEST_SIZE_MISMATCH: got={len(subject_manifest)}")

    manifest_path = manifest_dir / "subject_manifest.csv"
    subject_manifest.to_csv(manifest_path, index=False)

    excluded_df = pd.concat(excluded_frames, ignore_index=True)
    if excluded_df.empty:
        excluded_df = pd.DataFrame.from_records([], columns=("user_id", "age_group", "reason"))
    excluded_path = manifest_dir / "excluded_subjects.csv"
    excluded_df.to_csv(excluded_path, index=False)

    selection_summary = pd.DataFrame(summary_rows)
    selection_summary_path = manifest_dir / "selection_summary.csv"
    selection_summary.to_csv(selection_summary_path, index=False)

    return manifest_path, excluded_path, selection_summary_path


def build_alignment_report(df: pd.DataFrame, cached: Dict[str, np.ndarray]) -> pd.DataFrame:
    if len(df) != len(cached["logits"]):
        raise ValueError(
            f"CSV_NPZ_ALIGNMENT_LENGTH_MISMATCH: csv={len(df)} npz={len(cached['logits'])}"
        )
    required_cols = [
        "user_id",
        "target_direction",
        "response_direction",
        "flanker_direction",
        "response_time",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV_NPZ_ALIGNMENT_MISSING_COLUMNS: missing={missing_cols}")

    target_from_csv = df["target_direction"].map(lambda x: DIRECTION_MAP[str(x)]).to_numpy(dtype=np.int64)
    response_from_csv = df["response_direction"].map(lambda x: DIRECTION_MAP[str(x)]).to_numpy(dtype=np.int64)
    flanker_from_csv = df["flanker_direction"].map(lambda x: DIRECTION_MAP[str(x)]).to_numpy(dtype=np.int64)
    congruency_from_csv = compute_congruency_from_labels(target_from_csv, flanker_from_csv)
    rts_from_csv = (df["response_time"].to_numpy(dtype=np.float32) / 1000.0).astype(np.float32)

    cached_target = np.asarray(cached["target_labels"], dtype=np.int64)
    cached_response = np.asarray(cached["response_labels"], dtype=np.int64)
    cached_congruency = np.asarray(cached["congruency"], dtype=np.int64)
    cached_rts = np.asarray(cached["rts"], dtype=np.float32)
    cached_flanker = np.asarray(
        cached.get("flanker_labels", np.full(len(cached_target), -1, dtype=np.int64)), dtype=np.int64
    )

    mismatch = {
        "target_labels": target_from_csv != cached_target,
        "response_labels": response_from_csv != cached_response,
        "flanker_labels": flanker_from_csv != cached_flanker,
        "congruency": congruency_from_csv != cached_congruency,
        "rts": ~np.isclose(rts_from_csv, cached_rts, atol=1e-6, rtol=1e-6),
    }
    mismatch_any = np.zeros(len(df), dtype=bool)
    for mask in mismatch.values():
        mismatch_any |= mask

    alignment_ok = ~mismatch_any
    report = pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(),
            "row_index": np.arange(len(df), dtype=np.int64),
            "alignment_ok": alignment_ok,
        }
    )

    if not bool(np.all(alignment_ok)):
        first_bad = int(np.flatnonzero(~alignment_ok)[0])
        uid = df.iloc[first_bad]["user_id"]
        bad_fields = [name for name, mask in mismatch.items() if bool(mask[first_bad])]
        raise ValueError(
            f"CSV_NPZ_ALIGNMENT_MISMATCH: row_index={first_bad} user_id={uid} fields={bad_fields}"
        )
    return report


def verify_alignment(age_group: str, output_root: Path) -> Path:
    baseline = load_baseline_manifest(output_root)
    age_entry = baseline["age_groups"][age_group]
    csv_path = Path(age_entry["csv_path"])
    cache_npz = Path(age_entry["cache_npz"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prepared CSV for alignment: {csv_path}")
    if not cache_npz.exists():
        raise FileNotFoundError(f"Missing cached NPZ for alignment: {cache_npz}")

    # Load the cached test arrays using the canonical validator.
    data_dir = age_group_data_dir(age_group)
    stage2_dir = age_group_stage2_dir(age_group)
    _, test_cached = validate_cached_stage2_inputs(
        age_group,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    test_cached = attach_flanker_labels_from_csv(test_cached, str(csv_path))

    df = pd.read_csv(csv_path)
    report = build_alignment_report(df, test_cached)

    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / f"alignment_check_{age_group}.csv"
    report.to_csv(out_path, index=False)
    return out_path


def load_non_scale_params(npz_path: Path) -> Dict[str, Any]:
    npz = np.load(npz_path)
    params: Dict[str, Any] = {}
    for key in npz.files:
        if key == "scale":
            continue
        params[key] = npz[key]
    return cast(Dict[str, Any], to_jsonable(params))


def find_existing_phase1_artifact_dir(age_group: str) -> Optional[Path]:
    # Known repo-local dynamic-selection smoke outputs for 20-29.
    candidates = [
        RESULTS_ROOT / "repro_legacy_interim" / f"dynamic_selection_{age_group.replace('-', '_')}_smoke" / "dynamic_selection_phase1",
        RESULTS_ROOT / "repro_legacy_interim" / f"dynamic_selection_{age_group}_smoke" / "dynamic_selection_phase1",
        RESULTS_ROOT / "repro_legacy_interim" / f"dynamic_selection_dmc_extension_{age_group.replace('-', '_')}_smoke" / "dynamic_selection_phase1",
        RESULTS_ROOT / "repro_legacy_interim" / f"dynamic_selection_dmc_extension_{age_group}_smoke" / "dynamic_selection_phase1",
    ]
    for cand in candidates:
        if (cand / "predictions.npz").exists() and (cand / "summary.json").exists():
            return cand
    return None


def dynamic_selection_phase1_config() -> Dict[str, Any]:
    # Must match the active repo-local phase-1 dynamic-selection settings.
    return {
        "selection_mode": "dynamic_flanker_suppression",
        "selection_strength": 0.35,
        "selection_midpoint_s": 0.18,
        "selection_tau_s": 0.06,
        "target_boost": 0.10,
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }


def materialize_phase1_artifact(
    *,
    age_group: str,
    target_dir: Path,
    stage2_cfg: dict,
    stage2_params_npz: Path,
    cached_npz: Path,
    cached_csv: Path,
    device: str,
    seed: int,
    choice_temperature: float,
) -> None:
    set_random_seed(int(seed))
    target_dir.mkdir(parents=True, exist_ok=True)

    # Validate canonical cached inputs and enrich with flanker_labels.
    data_dir = age_group_data_dir(age_group)
    stage2_dir = age_group_stage2_dir(age_group)
    _, test_cached = validate_cached_stage2_inputs(
        age_group,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    test_cached = attach_flanker_labels_from_csv(test_cached, str(cached_csv))

    # Load Stage-2 params from checkpoint artifacts; keep scale separate.
    params_npz = np.load(stage2_params_npz)
    params = {k: params_npz[k] for k in params_npz.files if k != "scale"}

    raw_scale = stage2_cfg.get("scale")
    if raw_scale is None:
        raise ValueError(f"Stage-2 config missing 'scale' for {age_group}: {stage2_cfg['_source_config_path']}")
    scale = float(raw_scale)
    if not np.isfinite(scale):
        raise ValueError(f"Stage-2 config scale is not finite for {age_group}: {scale}")
    raw_time_steps = stage2_cfg.get("time_steps")
    if raw_time_steps is None:
        raise ValueError(f"Stage-2 config missing 'time_steps' for {age_group}: {stage2_cfg['_source_config_path']}")
    time_steps = int(raw_time_steps)

    predictions, canonical_results = evaluate_cached_stage2_params(
        params=params,
        scale=scale,
        time_steps=time_steps,
        cached=test_cached,
        device=device,
        choice_temperature=choice_temperature,
        rt_readout_mode="baseline",
        readout_config=None,
        selection_config=dynamic_selection_phase1_config(),
        random_seed=int(seed),
        rt_shape_focus=False,
    )

    np.savez_compressed(
        target_dir / "predictions.npz",
        pred_rt=predictions["pred_rt"],
        pred_choice=predictions["pred_choice"],
        choice_logits=predictions["choice_logits"],
        decision_times_class=predictions["decision_times_class"],
        true_rt=test_cached["rts"],
        target_labels=test_cached["target_labels"],
        response_labels=test_cached["response_labels"],
        flanker_labels=test_cached["flanker_labels"],
        congruency=test_cached["congruency"],
        **{k: v for k, v in predictions.items() if k not in {"pred_rt", "pred_choice", "choice_logits", "decision_times_class"}},
    )

    summary = {
        "tag": "dynamic_selection_phase1",
        "age_group": age_group,
        "source": "materialized",
        "stage2_config_path": safe_rel_to_root(Path(stage2_cfg["_source_config_path"])),
        "stage2_params_path": safe_rel_to_root(stage2_params_npz),
        "cache_npz": safe_rel_to_root(cached_npz),
        "csv_path": safe_rel_to_root(cached_csv),
        "scale": scale,
        "time_steps": time_steps,
        "choice_temperature": float(choice_temperature),
        "selection_config": dynamic_selection_phase1_config(),
        "score": float(canonical_results["total_score"]),
        "model_accuracy": float(canonical_results["model_accuracy"]),
        "response_agreement": float(canonical_results["response_agreement"]),
        "model_congruency_rt_gap": float(canonical_results["model_congruency_rt_gap"]),
    }
    (target_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))


def ensure_phase1_artifact(
    *,
    age_group: str,
    output_root: Path,
    stage2_cfg: dict,
    stage2_params_npz: Path,
    cached_npz: Path,
    cached_csv: Path,
    device: str,
    seed: int,
    choice_temperature: float,
) -> Dict[str, Any]:
    target_dir = output_root / "baseline_artifacts" / age_group / "dynamic_selection_phase1"
    if (target_dir / "predictions.npz").exists() and (target_dir / "summary.json").exists():
        return {
            "decision": "reused_existing_in_output_root",
            "artifact_dir": safe_rel_to_root(target_dir),
        }

    existing = find_existing_phase1_artifact_dir(age_group)
    if existing is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing / "predictions.npz", target_dir / "predictions.npz")
        shutil.copy2(existing / "summary.json", target_dir / "summary.json")
        return {
            "decision": "copied_from_existing_repo_artifact",
            "source_artifact_dir": safe_rel_to_root(existing),
            "artifact_dir": safe_rel_to_root(target_dir),
        }

    materialize_phase1_artifact(
        age_group=age_group,
        target_dir=target_dir,
        stage2_cfg=stage2_cfg,
        stage2_params_npz=stage2_params_npz,
        cached_npz=cached_npz,
        cached_csv=cached_csv,
        device=device,
        seed=seed,
        choice_temperature=choice_temperature,
    )
    return {
        "decision": "materialized_equivalent_phase1_artifact",
        "artifact_dir": safe_rel_to_root(target_dir),
    }


def audit_baseline(
    output_root: Path,
    seed: int,
    device: str,
    choice_temperature: float,
    baseline_inputs_override: Optional[Path] = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    selection_config = dynamic_selection_phase1_config()

    override: Optional[Dict[str, Dict[str, str]]] = None
    if baseline_inputs_override is not None:
        override = load_baseline_inputs_override(baseline_inputs_override)

    age_group_entries: Dict[str, Any] = {}
    for age_group in AGE_GROUPS:
        if override is not None:
            entry = override[age_group]
            cache_npz = Path(entry["cache_npz"])
            csv_path = Path(entry["csv_path"])
            cfg_path = Path(entry["stage2_config_path"])
            params_path = Path(entry["stage2_params_path"])
            if not cache_npz.exists():
                raise FileNotFoundError(f"BASELINE_INPUTS_MISSING_FILE: age_group={age_group} cache_npz={cache_npz}")
            if not csv_path.exists():
                raise FileNotFoundError(f"BASELINE_INPUTS_MISSING_FILE: age_group={age_group} csv_path={csv_path}")
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"BASELINE_INPUTS_MISSING_FILE: age_group={age_group} stage2_config_path={cfg_path}"
                )
            if not params_path.exists():
                raise FileNotFoundError(
                    f"BASELINE_INPUTS_MISSING_FILE: age_group={age_group} stage2_params_path={params_path}"
                )
            # Keep the canonical prepared-data directory for validation (rt_stats/train split files).
            # Allow overriding the CSV path itself (e.g. permuted fixture) without having to mirror
            # the entire prepared-data directory.
            data_dir = age_group_data_dir(age_group)
            stage2_dir = cache_npz.parent
        else:
            data_dir = age_group_data_dir(age_group)
            stage2_dir = age_group_stage2_dir(age_group)

            cache_npz = stage2_dir / "test_logits.npz"
            csv_path = data_dir / "test_data.csv"

            cfg_path, params_path = locate_stage2_param_artifacts(age_group)

        # Strict baseline input invariants for later single-subject simulation.
        validate_cached_stage2_inputs(
            age_group,
            str(data_dir),
            str(stage2_dir / "train_logits.npz"),
            str(stage2_dir / "test_logits.npz"),
        )

        stage2_cfg = _load_json(cfg_path)
        stage2_cfg["_source_config_path"] = safe_rel_to_root(cfg_path)

        non_scale_params = load_non_scale_params(params_path)

        phase1_status = ensure_phase1_artifact(
            age_group=age_group,
            output_root=output_root,
            stage2_cfg=stage2_cfg,
            stage2_params_npz=params_path,
            cached_npz=cache_npz,
            cached_csv=csv_path,
            device=device,
            seed=seed,
            choice_temperature=choice_temperature,
        )

        age_group_entries[age_group] = {
            "cache_npz": safe_rel_to_root(cache_npz),
            "csv_path": safe_rel_to_root(csv_path),
            "non_scale_params": non_scale_params,
            "stage2_config_path": safe_rel_to_root(cfg_path),
            "stage2_params_path": safe_rel_to_root(params_path),
            "phase1_artifact": phase1_status,
        }

    manifest = {
        "entrypoint": "code/scripts/run_dynamic_selection_single_subject.py",
        "mode": "audit-baseline",
        "output_root": safe_rel_to_root(output_root),
        "seed": int(seed),
        "device": device,
        "choice_temperature": float(choice_temperature),
        "selection_config": selection_config,
        "age_groups": age_group_entries,
    }

    out_path = manifest_dir / "baseline_manifest.json"
    out_path.write_text(json.dumps(to_jsonable(manifest), indent=2))
    return out_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if args.mode == "audit-baseline":
        out = audit_baseline(
            output_root=output_root,
            seed=int(args.seed),
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
            baseline_inputs_override=None
            if args.baseline_inputs_json is None
            else Path(args.baseline_inputs_json),
        )
        print(f"Wrote baseline manifest: {out}")
        return

    if args.mode == "verify-alignment":
        if args.age_group is None:
            raise ValueError("--age_group is required for --mode verify-alignment")
        out = verify_alignment(str(args.age_group), output_root)
        print(f"Wrote alignment report: {out}")
        return

    if args.mode == "select-subjects":
        manifest_path, excluded_path, summary_path = select_subjects(output_root)
        print(f"Wrote subject manifest: {manifest_path}")
        print(f"Wrote excluded subjects: {excluded_path}")
        print(f"Wrote selection summary: {summary_path}")
        return

    if args.mode == "simulate":
        if args.age_group is None:
            raise ValueError("--age_group is required for --mode simulate")
        simulate_age_group(
            age_group=str(args.age_group),
            output_root=output_root,
            seed=int(args.seed),
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
        )
        print(f"Completed simulations for age_group={args.age_group} under {output_root}")
        return

    if args.mode == "full":
        run_full(
            output_root=output_root,
            seed=int(args.seed),
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
        )
        print(f"Completed full single-subject workflow under {output_root}")
        return

    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
