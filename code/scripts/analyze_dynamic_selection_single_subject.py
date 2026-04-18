import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from project_paths import RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir
from train_age_groups_efficient import (
    attach_flanker_labels_from_csv,
    evaluate_cached_stage2_params,
    to_jsonable,
    validate_cached_stage2_inputs,
)


AGE_GROUPS = ("20-29", "80-89")
N_QUANTILES = 5
DEFAULT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_single_subject"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze single-subject simulation outputs: reaggregate, score mechanisms, write success bar + memo."
    )
    parser.add_argument("--input_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def dynamic_selection_phase1_config() -> Dict[str, Any]:
    return {
        "selection_mode": "dynamic_flanker_suppression",
        "selection_strength": 0.35,
        "selection_midpoint_s": 0.18,
        "selection_tau_s": 0.06,
        "target_boost": 0.10,
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }


def safe_skew(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or np.allclose(values, values[0]):
        return 0.0
    skew = stats.skew(values, bias=False)
    return 0.0 if np.isnan(skew) else float(skew)


def compute_caf(df: pd.DataFrame, source: str, rt_col: str, correct_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition, [rt_col, correct_col]].copy()
        if subset.empty:
            continue
        subset["bin"] = pd.qcut(subset[rt_col], q=N_QUANTILES, labels=False, duplicates="drop")
        grouped = subset.groupby("bin", sort=True)
        for raw_bin, group in grouped:
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "bin_index": int(raw_bin) + 1,
                    "rt_min": float(group[rt_col].min()),
                    "rt_max": float(group[rt_col].max()),
                    "mean_rt": float(group[rt_col].mean()),
                    "accuracy": float(group[correct_col].mean()),
                    "n_trials": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def compute_delta(df: pd.DataFrame, source: str, rt_col: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition, [rt_col]].copy()
        subset["condition"] = condition
        subset["bin"] = pd.qcut(subset[rt_col], q=N_QUANTILES, labels=False, duplicates="drop")
        grouped = (
            subset.groupby(["condition", "bin"], sort=True)[rt_col]
            .mean()
            .reset_index()
            .rename(columns={rt_col: "mean_rt"})
        )
        pieces.append(grouped)
    combined = pd.concat(pieces, ignore_index=True)
    pivot = combined.pivot(index="bin", columns="condition", values="mean_rt").reset_index()
    pivot["quantile_index"] = pivot["bin"].astype(int) + 1
    pivot["source"] = source
    pivot["delta"] = pivot["incongruent"] - pivot["congruent"]
    out = pivot.rename(columns={"congruent": "mean_congruent_rt", "incongruent": "mean_incongruent_rt"})[
        ["source", "quantile_index", "mean_congruent_rt", "mean_incongruent_rt", "delta"]
    ].copy()
    return cast(pd.DataFrame, out)


def compute_conditional_error_rt(
    df: pd.DataFrame, source: str, rt_col: str, correct_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wide: Dict[str, Any] = {"source": source}
    rows: list[dict[str, Any]] = []
    for condition in ("congruent", "incongruent"):
        subset = df.loc[df["condition"] == condition].copy()
        correct_vals = subset.loc[subset[correct_col], rt_col].to_numpy()
        error_vals = subset.loc[~subset[correct_col], rt_col].to_numpy()
        correct_rt = float(correct_vals.mean()) if correct_vals.size else float("nan")
        error_rt = float(error_vals.mean()) if error_vals.size else float("nan")
        gap = error_rt - correct_rt if correct_vals.size and error_vals.size else float("nan")
        rows.append(
            {
                "source": source,
                "condition": condition,
                "correct_rt": correct_rt,
                "error_rt": error_rt,
                "error_minus_correct_rt": gap,
                "n_correct": int(correct_vals.size),
                "n_error": int(error_vals.size),
            }
        )
        wide[f"{condition}_correct_rt"] = correct_rt
        wide[f"{condition}_error_rt"] = error_rt
        wide[f"{condition}_error_minus_correct_rt"] = gap
        wide[f"{condition}_n_correct"] = int(correct_vals.size)
        wide[f"{condition}_n_error"] = int(error_vals.size)
    return pd.DataFrame([wide]), pd.DataFrame(rows)


def compute_tail_summary(df: pd.DataFrame, source: str, rt_col: str, correct_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in ("congruent", "incongruent"):
        for is_correct, label in ((True, "correct"), (False, "error")):
            subset = df.loc[(df["condition"] == condition) & (df[correct_col] == is_correct), rt_col].to_numpy()
            if subset.size == 0:
                q90 = q95 = q99 = skewness = float("nan")
            else:
                q90 = float(np.quantile(subset, 0.90))
                q95 = float(np.quantile(subset, 0.95))
                q99 = float(np.quantile(subset, 0.99))
                skewness = safe_skew(subset)
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "correctness": label,
                    "group": f"{condition}_{label}",
                    "q90": q90,
                    "q95": q95,
                    "q99": q99,
                    "skewness": skewness,
                    "n_trials": int(subset.size),
                }
            )
    return pd.DataFrame(rows)


def _extract_metric_scalar(
    *,
    caf_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    error_wide_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    metric: str,
) -> float:
    if metric == "earliest_incongruent_caf":
        sub = caf_df.loc[(caf_df["condition"] == "incongruent") & (caf_df["bin_index"] == 1)]
        return float(sub["accuracy"].iloc[0]) if not sub.empty else float("nan")
    if metric == "first_delta":
        sub = delta_df.loc[delta_df["quantile_index"] == 1]
        return float(sub["delta"].iloc[0]) if not sub.empty else float("nan")
    if metric == "incongruent_error_minus_correct_rt":
        return float(error_wide_df["incongruent_error_minus_correct_rt"].iloc[0])
    if metric == "incongruent_conditional_tail":
        sub = tail_df.loc[tail_df["group"] == "incongruent_error"]
        return float(sub["q95"].iloc[0]) if not sub.empty else float("nan")
    raise ValueError(f"Unknown metric: {metric}")


def _build_trial_df(
    *,
    rt_s: np.ndarray,
    choice: np.ndarray,
    target_labels: np.ndarray,
    congruency: np.ndarray,
    source: str,
) -> pd.DataFrame:
    rt_s = np.asarray(rt_s, dtype=np.float32)
    choice = np.asarray(choice, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    congruency = np.asarray(congruency, dtype=np.int64)
    df = pd.DataFrame(
        {
            "source": source,
            "rt_s": rt_s,
            "choice": choice,
            "target": target_labels,
            "congruency": congruency,
        }
    )
    df["condition"] = np.where(df["congruency"] == 1, "incongruent", "congruent")
    df["correct"] = df["choice"] == df["target"]
    return df


def _concat_cached_dicts(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in sorted(keys):
        if key not in a or key not in b:
            raise ValueError(f"CACHED_CONCAT_KEY_MISMATCH: key={key} a={key in a} b={key in b}")
        out[key] = np.concatenate([a[key], b[key]], axis=0)
    return out


def _load_train_test_cached_and_csv(age_group: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data_dir = age_group_data_dir(age_group)
    stage2_dir = age_group_stage2_dir(age_group)
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"
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
    return combined_df, _concat_cached_dicts(train_cached, test_cached)


def _load_test_cached_and_csv(age_group: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data_dir = age_group_data_dir(age_group)
    stage2_dir = age_group_stage2_dir(age_group)
    test_csv = data_dir / "test_data.csv"
    test_df = pd.read_csv(test_csv)
    _, test_cached = validate_cached_stage2_inputs(
        age_group,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    test_cached = attach_flanker_labels_from_csv(test_cached, str(test_csv))
    return test_df, test_cached


def _compute_contextual_metric_row(
    *,
    age_group: str,
    caf_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    error_wide_h: pd.DataFrame,
    error_wide_b: pd.DataFrame,
    error_wide_s: pd.DataFrame,
    tail_df: pd.DataFrame,
) -> dict[str, Any]:
    metric_names = [
        "earliest_incongruent_caf",
        "first_delta",
        "incongruent_error_minus_correct_rt",
        "incongruent_conditional_tail",
    ]
    out: dict[str, Any] = {"age_group": age_group}
    for metric in metric_names:
        human_val = _extract_metric_scalar(
            caf_df=caf_df.loc[caf_df["source"] == "human_full_age_group"],
            delta_df=delta_df.loc[delta_df["source"] == "human_full_age_group"],
            error_wide_df=error_wide_h,
            tail_df=tail_df.loc[tail_df["source"] == "human_full_age_group"],
            metric=metric,
        )
        baseline_val = _extract_metric_scalar(
            caf_df=caf_df.loc[caf_df["source"] == "baseline_full_age_group"],
            delta_df=delta_df.loc[delta_df["source"] == "baseline_full_age_group"],
            error_wide_df=error_wide_b,
            tail_df=tail_df.loc[tail_df["source"] == "baseline_full_age_group"],
            metric=metric,
        )
        sim_val = _extract_metric_scalar(
            caf_df=caf_df.loc[caf_df["source"] == "sim_selected"],
            delta_df=delta_df.loc[delta_df["source"] == "sim_selected"],
            error_wide_df=error_wide_s,
            tail_df=tail_df.loc[tail_df["source"] == "sim_selected"],
            metric=metric,
        )
        out[f"human_{metric}"] = float(human_val)
        out[f"baseline_{metric}"] = float(baseline_val)
        out[f"sim_selected_{metric}"] = float(sim_val)
        if np.isfinite(human_val) and np.isfinite(baseline_val) and np.isfinite(sim_val):
            out[f"baseline_distance_to_human_{metric}"] = float(abs(baseline_val - human_val))
            out[f"sim_selected_distance_to_human_{metric}"] = float(abs(sim_val - human_val))
        else:
            out[f"baseline_distance_to_human_{metric}"] = float("nan")
            out[f"sim_selected_distance_to_human_{metric}"] = float("nan")
    return out


def _filter_cached_by_mask(cached: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in cached.items():
        if value.shape[0] != mask.shape[0]:
            raise ValueError(f"CACHED_MASK_SHAPE_MISMATCH: key={key} n={value.shape[0]} mask={mask.shape[0]}")
        out[key] = value[mask]
    return out


def _load_params_and_time_steps(baseline: dict, age_group: str) -> Tuple[Dict[str, Any], int, float, Path, Path]:
    age_entry = baseline["age_groups"][age_group]
    cfg_path = Path(age_entry["stage2_config_path"])
    params_path = Path(age_entry["stage2_params_path"])
    cfg = _load_json(cfg_path)
    time_steps = int(cfg["time_steps"])
    scale = float(cfg["scale"]) if "scale" in cfg else 0.1
    params_npz = np.load(params_path)
    params = {k: params_npz[k] for k in params_npz.files if k != "scale"}
    return cast(Dict[str, Any], params), time_steps, scale, cfg_path, params_path


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reagg_dir = output_root / "reaggregated"
    reagg_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_root / "manifest" / "baseline_manifest.json"
    subject_manifest_path = output_root / "manifest" / "subject_manifest.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline manifest: {baseline_path}")
    if not subject_manifest_path.exists():
        raise FileNotFoundError(f"Missing subject manifest: {subject_manifest_path}")
    baseline = _load_json(baseline_path)
    subject_manifest = pd.read_csv(subject_manifest_path)
    subject_manifest["user_id"] = subject_manifest["user_id"].astype(str)
    selection_summary_path = output_root / "manifest" / "selection_summary.csv"
    if not selection_summary_path.exists():
        raise FileNotFoundError(f"Missing selection summary: {selection_summary_path}")
    selection_summary = pd.read_csv(selection_summary_path)

    # Build pooled (trial-weighted) trial frames across both age groups.
    pooled_frames: dict[str, list[pd.DataFrame]] = {
        "human_selected": [],
        "sim_selected": [],
        "baseline_selected": [],
    }
    summary_rows: list[dict[str, Any]] = []
    contextual_rows: list[dict[str, Any]] = []

    for age_group in AGE_GROUPS:
        subjects = subject_manifest.loc[subject_manifest["age_group"] == age_group, "user_id"].astype(str).tolist()
        if not subjects:
            continue
        age_dir = input_root / age_group
        # Load simulated outputs per subject.
        sim_parts: list[pd.DataFrame] = []
        human_parts: list[pd.DataFrame] = []
        for uid in subjects:
            pred_path = age_dir / f"user_{uid}" / "predictions.npz"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing per-subject predictions: {pred_path}")
            npz = np.load(pred_path)
            sim_parts.append(
                _build_trial_df(
                    rt_s=npz["pred_rt"],
                    choice=npz["pred_choice"],
                    target_labels=npz["target_labels"],
                    congruency=npz["congruency"],
                    source="sim_selected",
                )
            )
            human_parts.append(
                _build_trial_df(
                    rt_s=npz["true_rt"],
                    choice=npz["response_labels"],
                    target_labels=npz["target_labels"],
                    congruency=npz["congruency"],
                    source="human_selected",
                )
            )
        sim_df = pd.concat(sim_parts, ignore_index=True)
        human_df = pd.concat(human_parts, ignore_index=True)

        # Baseline comparison on the same selected trials (scale fixed at baseline config scale).
        combined_df, combined_cached = _load_train_test_cached_and_csv(age_group)
        combined_df = combined_df.copy()
        combined_df["user_id"] = combined_df["user_id"].astype(str)
        mask = combined_df["user_id"].isin(subjects).to_numpy(dtype=bool)
        cached_sel = _filter_cached_by_mask(combined_cached, mask)
        params, time_steps, baseline_scale, cfg_path, params_path = _load_params_and_time_steps(baseline, age_group)
        baseline_pred, _ = evaluate_cached_stage2_params(
            params=params,
            scale=float(baseline_scale),
            time_steps=int(time_steps),
            cached=cached_sel,
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
            rt_readout_mode="baseline",
            readout_config=None,
            selection_config=dynamic_selection_phase1_config(),
            random_seed=int(args.seed),
            rt_shape_focus=False,
        )
        baseline_df = _build_trial_df(
            rt_s=baseline_pred["pred_rt"],
            choice=baseline_pred["pred_choice"],
            target_labels=cached_sel["target_labels"],
            congruency=cached_sel["congruency"],
            source="baseline_selected",
        )

        test_df_full, test_cached_full = _load_test_cached_and_csv(age_group)
        human_full_df = _build_trial_df(
            rt_s=test_cached_full["rts"],
            choice=test_cached_full["response_labels"],
            target_labels=test_cached_full["target_labels"],
            congruency=test_cached_full["congruency"],
            source="human_full_age_group",
        )
        baseline_full_pred, _ = evaluate_cached_stage2_params(
            params=params,
            scale=float(baseline_scale),
            time_steps=int(time_steps),
            cached=test_cached_full,
            device=str(args.device),
            choice_temperature=float(args.choice_temperature),
            rt_readout_mode="baseline",
            readout_config=None,
            selection_config=dynamic_selection_phase1_config(),
            random_seed=int(args.seed),
            rt_shape_focus=False,
        )
        baseline_full_df = _build_trial_df(
            rt_s=baseline_full_pred["pred_rt"],
            choice=baseline_full_pred["pred_choice"],
            target_labels=test_cached_full["target_labels"],
            congruency=test_cached_full["congruency"],
            source="baseline_full_age_group",
        )

        pooled_frames["human_selected"].append(human_df)
        pooled_frames["sim_selected"].append(sim_df)
        pooled_frames["baseline_selected"].append(baseline_df)

        # Per-age-group metrics snapshot.
        caf = pd.concat(
            [
                compute_caf(human_df, "human_selected", "rt_s", "correct"),
                compute_caf(sim_df, "sim_selected", "rt_s", "correct"),
                compute_caf(baseline_df, "baseline_selected", "rt_s", "correct"),
            ],
            ignore_index=True,
        )
        delta = pd.concat(
            [
                compute_delta(human_df, "human_selected", "rt_s"),
                compute_delta(sim_df, "sim_selected", "rt_s"),
                compute_delta(baseline_df, "baseline_selected", "rt_s"),
            ],
            ignore_index=True,
        )
        err_wide_h, _ = compute_conditional_error_rt(human_df, "human_selected", "rt_s", "correct")
        err_wide_s, _ = compute_conditional_error_rt(sim_df, "sim_selected", "rt_s", "correct")
        err_wide_b, _ = compute_conditional_error_rt(baseline_df, "baseline_selected", "rt_s", "correct")
        tail = pd.concat(
            [
                compute_tail_summary(human_df, "human_selected", "rt_s", "correct"),
                compute_tail_summary(sim_df, "sim_selected", "rt_s", "correct"),
                compute_tail_summary(baseline_df, "baseline_selected", "rt_s", "correct"),
            ],
            ignore_index=True,
        )

        metrics = {
            "earliest_incongruent_caf": {
                "human_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "human_selected"],
                    delta_df=delta.loc[delta["source"] == "human_selected"],
                    error_wide_df=err_wide_h,
                    tail_df=tail.loc[tail["source"] == "human_selected"],
                    metric="earliest_incongruent_caf",
                ),
                "baseline_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "baseline_selected"],
                    delta_df=delta.loc[delta["source"] == "baseline_selected"],
                    error_wide_df=err_wide_b,
                    tail_df=tail.loc[tail["source"] == "baseline_selected"],
                    metric="earliest_incongruent_caf",
                ),
                "sim_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "sim_selected"],
                    delta_df=delta.loc[delta["source"] == "sim_selected"],
                    error_wide_df=err_wide_s,
                    tail_df=tail.loc[tail["source"] == "sim_selected"],
                    metric="earliest_incongruent_caf",
                ),
            },
            "first_delta": {
                "human_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "human_selected"],
                    delta_df=delta.loc[delta["source"] == "human_selected"],
                    error_wide_df=err_wide_h,
                    tail_df=tail.loc[tail["source"] == "human_selected"],
                    metric="first_delta",
                ),
                "baseline_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "baseline_selected"],
                    delta_df=delta.loc[delta["source"] == "baseline_selected"],
                    error_wide_df=err_wide_b,
                    tail_df=tail.loc[tail["source"] == "baseline_selected"],
                    metric="first_delta",
                ),
                "sim_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "sim_selected"],
                    delta_df=delta.loc[delta["source"] == "sim_selected"],
                    error_wide_df=err_wide_s,
                    tail_df=tail.loc[tail["source"] == "sim_selected"],
                    metric="first_delta",
                ),
            },
            "incongruent_error_minus_correct_rt": {
                "human_selected": float(err_wide_h["incongruent_error_minus_correct_rt"].iloc[0]),
                "baseline_selected": float(err_wide_b["incongruent_error_minus_correct_rt"].iloc[0]),
                "sim_selected": float(err_wide_s["incongruent_error_minus_correct_rt"].iloc[0]),
            },
            "incongruent_conditional_tail": {
                "human_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "human_selected"],
                    delta_df=delta.loc[delta["source"] == "human_selected"],
                    error_wide_df=err_wide_h,
                    tail_df=tail.loc[tail["source"] == "human_selected"],
                    metric="incongruent_conditional_tail",
                ),
                "baseline_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "baseline_selected"],
                    delta_df=delta.loc[delta["source"] == "baseline_selected"],
                    error_wide_df=err_wide_b,
                    tail_df=tail.loc[tail["source"] == "baseline_selected"],
                    metric="incongruent_conditional_tail",
                ),
                "sim_selected": _extract_metric_scalar(
                    caf_df=caf.loc[caf["source"] == "sim_selected"],
                    delta_df=delta.loc[delta["source"] == "sim_selected"],
                    error_wide_df=err_wide_s,
                    tail_df=tail.loc[tail["source"] == "sim_selected"],
                    metric="incongruent_conditional_tail",
                ),
            },
        }

        contextual_caf = pd.concat(
            [
                compute_caf(human_full_df, "human_full_age_group", "rt_s", "correct"),
                compute_caf(baseline_full_df, "baseline_full_age_group", "rt_s", "correct"),
                compute_caf(sim_df, "sim_selected", "rt_s", "correct"),
            ],
            ignore_index=True,
        )
        contextual_delta = pd.concat(
            [
                compute_delta(human_full_df, "human_full_age_group", "rt_s"),
                compute_delta(baseline_full_df, "baseline_full_age_group", "rt_s"),
                compute_delta(sim_df, "sim_selected", "rt_s"),
            ],
            ignore_index=True,
        )
        contextual_err_h, _ = compute_conditional_error_rt(human_full_df, "human_full_age_group", "rt_s", "correct")
        contextual_err_b, _ = compute_conditional_error_rt(baseline_full_df, "baseline_full_age_group", "rt_s", "correct")
        contextual_err_s, _ = compute_conditional_error_rt(sim_df, "sim_selected", "rt_s", "correct")
        contextual_tail = pd.concat(
            [
                compute_tail_summary(human_full_df, "human_full_age_group", "rt_s", "correct"),
                compute_tail_summary(baseline_full_df, "baseline_full_age_group", "rt_s", "correct"),
                compute_tail_summary(sim_df, "sim_selected", "rt_s", "correct"),
            ],
            ignore_index=True,
        )
        contextual_rows.append(
            _compute_contextual_metric_row(
                age_group=age_group,
                caf_df=contextual_caf,
                delta_df=contextual_delta,
                error_wide_h=contextual_err_h,
                error_wide_b=contextual_err_b,
                error_wide_s=contextual_err_s,
                tail_df=contextual_tail,
            )
        )

        summary_rows.append(
            {
                "age_group": age_group,
                "n_trials_selected": int(len(human_df)),
                "baseline_scale": float(baseline_scale),
                "stage2_config_path": str(cfg_path),
                "stage2_params_path": str(params_path),
                "selection_pool": str(
                    selection_summary.loc[selection_summary["age_group"] == age_group, "selection_pool"].iloc[0]
                ),
                **{
                    f"human_{k}": float(v["human_selected"]) for k, v in metrics.items()
                },
                **{
                    f"baseline_{k}": float(v["baseline_selected"]) for k, v in metrics.items()
                },
                **{
                    f"sim_{k}": float(v["sim_selected"]) for k, v in metrics.items()
                },
            }
        )

    reaggregated_metrics = pd.DataFrame(summary_rows)
    (reagg_dir / "reaggregated_metrics.csv").write_text(reaggregated_metrics.to_csv(index=False))
    pd.DataFrame(contextual_rows).to_csv(reagg_dir / "contextual_metrics.csv", index=False)

    # Pooled success bar (trial-weighted, across all selected trials).
    human_all = pd.concat(pooled_frames["human_selected"], ignore_index=True)
    sim_all = pd.concat(pooled_frames["sim_selected"], ignore_index=True)
    baseline_all = pd.concat(pooled_frames["baseline_selected"], ignore_index=True)

    caf_all = pd.concat(
        [
            compute_caf(human_all, "human_selected", "rt_s", "correct"),
            compute_caf(baseline_all, "baseline_selected", "rt_s", "correct"),
            compute_caf(sim_all, "sim_selected", "rt_s", "correct"),
        ],
        ignore_index=True,
    )
    delta_all = pd.concat(
        [
            compute_delta(human_all, "human_selected", "rt_s"),
            compute_delta(baseline_all, "baseline_selected", "rt_s"),
            compute_delta(sim_all, "sim_selected", "rt_s"),
        ],
        ignore_index=True,
    )
    err_wide_h_all, _ = compute_conditional_error_rt(human_all, "human_selected", "rt_s", "correct")
    err_wide_b_all, _ = compute_conditional_error_rt(baseline_all, "baseline_selected", "rt_s", "correct")
    err_wide_s_all, _ = compute_conditional_error_rt(sim_all, "sim_selected", "rt_s", "correct")
    tail_all = pd.concat(
        [
            compute_tail_summary(human_all, "human_selected", "rt_s", "correct"),
            compute_tail_summary(baseline_all, "baseline_selected", "rt_s", "correct"),
            compute_tail_summary(sim_all, "sim_selected", "rt_s", "correct"),
        ],
        ignore_index=True,
    )

    caf_congruent_path = reagg_dir / "caf_congruent.csv"
    caf_incongruent_path = reagg_dir / "caf_incongruent.csv"
    caf_all.loc[caf_all["condition"] == "congruent"].to_csv(caf_congruent_path, index=False)
    caf_all.loc[caf_all["condition"] == "incongruent"].to_csv(caf_incongruent_path, index=False)
    delta_all.to_csv(reagg_dir / "delta_plot.csv", index=False)
    pd.concat([err_wide_h_all, err_wide_b_all, err_wide_s_all], ignore_index=True).to_csv(
        reagg_dir / "conditional_error_rt.csv", index=False
    )
    tail_all.to_csv(reagg_dir / "conditional_tail_summary.csv", index=False)

    target_metrics = [
        "earliest_incongruent_caf",
        "first_delta",
        "incongruent_error_minus_correct_rt",
        "incongruent_conditional_tail",
    ]
    metric_deltas: list[dict[str, Any]] = []
    metric_exclusions: list[dict[str, Any]] = []

    for metric in target_metrics:
        human_val = _extract_metric_scalar(
            caf_df=caf_all.loc[caf_all["source"] == "human_selected"],
            delta_df=delta_all.loc[delta_all["source"] == "human_selected"],
            error_wide_df=err_wide_h_all,
            tail_df=tail_all.loc[tail_all["source"] == "human_selected"],
            metric=metric,
        )
        baseline_val = _extract_metric_scalar(
            caf_df=caf_all.loc[caf_all["source"] == "baseline_selected"],
            delta_df=delta_all.loc[delta_all["source"] == "baseline_selected"],
            error_wide_df=err_wide_b_all,
            tail_df=tail_all.loc[tail_all["source"] == "baseline_selected"],
            metric=metric,
        )
        sim_val = _extract_metric_scalar(
            caf_df=caf_all.loc[caf_all["source"] == "sim_selected"],
            delta_df=delta_all.loc[delta_all["source"] == "sim_selected"],
            error_wide_df=err_wide_s_all,
            tail_df=tail_all.loc[tail_all["source"] == "sim_selected"],
            metric=metric,
        )
        excluded = not (np.isfinite(human_val) and np.isfinite(baseline_val) and np.isfinite(sim_val))
        if excluded:
            metric_exclusions.append(
                {
                    "metric": metric,
                    "reason": "NON_FINITE_COMPARISON_VALUE",
                    "human": None if not np.isfinite(human_val) else float(human_val),
                    "baseline": None if not np.isfinite(baseline_val) else float(baseline_val),
                    "sim": None if not np.isfinite(sim_val) else float(sim_val),
                }
            )
            metric_deltas.append(
                {
                    "metric": metric,
                    "excluded": True,
                    "human": float(human_val),
                    "baseline": float(baseline_val),
                    "sim": float(sim_val),
                    "before_distance": None,
                    "after_distance": None,
                    "improved": False,
                }
            )
            continue
        before = float(abs(float(baseline_val) - float(human_val)))
        after = float(abs(float(sim_val) - float(human_val)))
        direction_match = None
        if metric == "incongruent_error_minus_correct_rt":
            direction_match = bool(np.sign(sim_val) == np.sign(human_val))
        metric_deltas.append(
            {
                "metric": metric,
                "excluded": False,
                "human": float(human_val),
                "baseline": float(baseline_val),
                "sim": float(sim_val),
                "before_distance": before,
                "after_distance": after,
                "improved": bool(after < before),
                "direction_match": direction_match,
            }
        )

    improved = [m for m in metric_deltas if (not m["excluded"]) and m["improved"]]
    improved_names = {m["metric"] for m in improved}
    passes_two_of_four = len(improved) >= 2
    passes_critical = ("earliest_incongruent_caf" in improved_names) or any(
        (m["metric"] == "incongruent_error_minus_correct_rt") and bool(m.get("direction_match"))
        for m in improved
    )
    verdict = "HETEROGENEITY-SUPPORTED" if (passes_two_of_four and passes_critical) else "HETEROGENEITY-NOT-SUPPORTED"

    metric_exclusions_df = pd.DataFrame(metric_exclusions)
    if metric_exclusions_df.empty:
        metric_exclusions_df = pd.DataFrame.from_records([], columns=("metric", "reason", "human", "baseline", "sim"))
    metric_exclusions_df.to_csv(reagg_dir / "metric_exclusions.csv", index=False)

    success_bar = {
        "weighting": "trial_weighted",
        "target_metrics": target_metrics,
        "n_trials_selected_total": int(len(human_all)),
        "metric_deltas": metric_deltas,
        "critical_metric_rule": "At least one improved metric must be earliest_incongruent_caf or incongruent_error_minus_correct_rt with matching human direction.",
        "verdict": verdict,
    }
    (reagg_dir / "success_bar.json").write_text(json.dumps(to_jsonable(success_bar), indent=2))

    success_bar_path = reagg_dir / "success_bar.json"
    if not success_bar_path.exists():
        raise FileNotFoundError(f"Summary generation requires success_bar.json at {success_bar_path}")
    success_bar = _load_json(success_bar_path)
    run_summaries: dict[str, dict[str, Any]] = {}
    for age_group in AGE_GROUPS:
        run_summary_path = input_root / age_group / "run_summary.json"
        if run_summary_path.exists():
            run_summaries[age_group] = _load_json(run_summary_path)

    selection_rules = [
        "eligibility: >=20 incongruent trials, >=3 incongruent error trials, >=10 unique RT values",
        "metrics: earliest incongruent CAF, incongruent error-minus-correct RT, RT skewness",
        "ranking: mean absolute within-age-group z-score (extreme_score)",
        "strata: at least one subject from lower and upper halves of earliest incongruent CAF rank",
        "tie-breaks: higher incongruent error count, then lexicographic user_id",
    ]

    memo_lines = [
        "# Heterogeneity probe summary",
        "",
        "This is a **diagnostic heterogeneity probe** on `dynamic_selection_phase1` (no new mechanism).",
        "",
        f"**Weighting**: trial-weighted (n={success_bar['n_trials_selected_total']} selected trials)",
        "",
        "## Selection rules",
        *[f"- {rule}" for rule in selection_rules],
        "",
        "Subject selection was deterministic and used earliest incongruent CAF, incongruent error-minus-correct RT, and **RT skewness** as the hybrid strata metrics.",
        "",
        "## Selection / exclusion accounting",
    ]
    for _, row in selection_summary.iterrows():
        memo_lines.append(
            f"- {row['age_group']}: pool=`{row['selection_pool']}`, pool_users={int(row['pool_users'])}, test_only_users={int(row['test_only_users'])}, selected={int(row['selected_subjects'])}, excluded={int(row['excluded_subjects'])}"
        )
    memo_lines.extend([
        "",
        "## Age-group centers and bounded scale grids",
    ])
    for age_group, data in run_summaries.items():
        memo_lines.append(
            f"- {age_group}: center={float(data['scale_center']):.6f}, grid={list(data['scale_grid'])}, boundary_hits={int(data['n_boundary_hits'])}"
        )
    memo_lines.extend([
        "",
        "## Target mechanisms",
        "- earliest incongruent CAF",
        "- first delta quantile",
        "- incongruent error-minus-correct RT",
        "- incongruent conditional tail (q95 of incongruent-error RT)",
        "",
        "## Contextual comparison outputs",
        "- `reaggregated/contextual_metrics.csv` compares full age-group human aggregate vs full age-group baseline simulation vs reaggregated selected-subject simulation.",
        "- `reaggregated/reaggregated_metrics.csv` contains the primary selected-subject human vs selected-subject simulation comparison.",
        "",
        "## Success bar",
        f"**Verdict**: `{verdict}`",
        "",
        "| Metric | Before distance (baseline→human) | After distance (single-subject→human) | Improved |",
        "|---|---:|---:|:---:|",
    ])
    for item in metric_deltas:
        if item["excluded"]:
            memo_lines.append(f"| {item['metric']} | excluded | excluded |  |")
        else:
            memo_lines.append(
                f"| {item['metric']} | {item['before_distance']:.6g} | {item['after_distance']:.6g} | {'yes' if item['improved'] else 'no'} |"
            )
    memo_lines.append("")
    if metric_exclusions:
        memo_lines.append("## Metric exclusions")
        for item in metric_exclusions:
            memo_lines.append(f"- {item['metric']}: {item['reason']}")
        memo_lines.append("")
    if any("fallback" in str(pool) for pool in selection_summary["selection_pool"].tolist()):
        memo_lines.append("## Deviation note")
        memo_lines.append(
            "- Test-only selection could not satisfy the required 4 subjects per age group with the available prepared test splits, so the workflow used a documented pooled train+test fallback for subject selection and simulation."
        )
        memo_lines.append("")
    memo_lines.append("This memo reports the verdict encoded by `reaggregated/success_bar.json` only.")
    (output_root / "heterogeneity_probe_summary.md").write_text("\n".join(memo_lines))


if __name__ == "__main__":
    main()
