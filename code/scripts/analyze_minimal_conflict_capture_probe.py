import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from project_paths import RESULTS_ROOT


AGE_GROUPS = ("20-29", "80-89")
DEFAULT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "minimal_conflict_capture_probe"
EVIDENCE_ROOT = Path(".sisyphus") / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze minimal conflict-capture probe outputs and write scorecard, success bar, and summary."
    )
    parser.add_argument("--input_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output_root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.ndarray,)):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _write_evidence(task: int, slug: str, payload: dict) -> Path:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_ROOT / f"task-{task}-{slug}.json"
    out.write_text(json.dumps(_to_jsonable(payload), indent=2))
    return out


def _safe_rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


TARGET_METRICS = [
    "earliest_incongruent_caf",
    "first_delta",
    "incongruent_error_minus_correct_rt",
    "incongruent_conditional_tail",
]


def _load_age_metrics(input_root: Path, age_group: str) -> pd.DataFrame:
    path = input_root / age_group / "capture_probe_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-age capture metrics: {path}")
    df = pd.read_csv(path)
    if "probe_tag" not in df.columns:
        df["probe_tag"] = np.where(np.isclose(df["capture_strength"], 0.0), "baseline", df["capture_strength"].map(lambda x: f"capture_strength_{float(x):.2f}"))
    if "is_baseline" not in df.columns:
        df["is_baseline"] = np.isclose(df["capture_strength"], 0.0)
    return df


def _weighted_probe_row(per_age_rows: pd.DataFrame) -> Dict[str, float]:
    weights = per_age_rows["n_trials_total"].to_numpy(dtype=np.float64)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("PROBE_ROW_ZERO_TOTAL_WEIGHT")
    out: Dict[str, float] = {}
    for metric in TARGET_METRICS:
        out[f"human_{metric}"] = float(np.average(per_age_rows[f"human_{metric}"].to_numpy(dtype=np.float64), weights=weights))
        out[f"probe_{metric}"] = float(np.average(per_age_rows[f"pred_{metric}"].to_numpy(dtype=np.float64), weights=weights))
    out["n_trials_total"] = total_weight
    out["total_score"] = float(np.average(per_age_rows["total_score"].to_numpy(dtype=np.float64), weights=weights))
    out["model_accuracy"] = float(np.average(per_age_rows["model_accuracy"].to_numpy(dtype=np.float64), weights=weights))
    out["response_agreement"] = float(np.average(per_age_rows["response_agreement"].to_numpy(dtype=np.float64), weights=weights))
    out["model_congruency_rt_gap"] = float(np.average(per_age_rows["model_congruency_rt_gap"].to_numpy(dtype=np.float64), weights=weights))
    return out


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    reagg_dir = output_root / "reaggregated"
    reagg_dir.mkdir(parents=True, exist_ok=True)

    grid_path = input_root / "probe_config_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing probe config grid: {grid_path}")
    grid_df = pd.read_csv(grid_path)
    baseline_row = grid_df.loc[grid_df["is_baseline"] == True]
    if baseline_row.empty:
        raise ValueError("PROBE_GRID_MISSING_BASELINE_ROW")
    baseline_tag = str(baseline_row.iloc[0]["probe_tag"])

    heterogeneity_success_path = (
        RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_single_subject" / "reaggregated" / "success_bar.json"
    )
    heterogeneity_verdict = None
    if heterogeneity_success_path.exists():
        heterogeneity_verdict = _load_json(heterogeneity_success_path).get("verdict")

    age_metric_frames = {age_group: _load_age_metrics(input_root, age_group) for age_group in AGE_GROUPS}
    score_rows = []
    baseline_metrics = None
    for _, row in grid_df.iterrows():
        probe_tag = str(row["probe_tag"])
        per_age_rows = []
        for age_group, age_df in age_metric_frames.items():
            match = age_df.loc[age_df["probe_tag"] == probe_tag]
            if match.empty:
                raise FileNotFoundError(f"Missing probe_tag={probe_tag} row in {age_group}/capture_probe_metrics.csv")
            per_age_rows.append(match.iloc[0].to_dict())
        metrics = _weighted_probe_row(pd.DataFrame(per_age_rows))
        if bool(row["is_baseline"]):
            baseline_metrics = metrics
        score_rows.append(
            {
                "probe_tag": probe_tag,
                "is_baseline": bool(row["is_baseline"]),
                "capture_strength": float(row["capture_strength"]),
                "capture_midpoint_s": float(row["capture_midpoint_s"]),
                "capture_tau_s": float(row["capture_tau_s"]),
                **metrics,
            }
        )
    if baseline_metrics is None:
        raise RuntimeError("BASELINE_METRICS_NOT_COMPUTED")

    scorecard = pd.DataFrame(score_rows)
    for metric in TARGET_METRICS:
        scorecard[f"distance_to_human_{metric}"] = (scorecard[f"probe_{metric}"] - scorecard[f"human_{metric}"]).abs()
    scorecard["total_distance_to_human"] = scorecard[[f"distance_to_human_{m}" for m in TARGET_METRICS]].sum(axis=1)
    scorecard_path = reagg_dir / "probe_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)

    candidates = scorecard.loc[~scorecard["is_baseline"]].copy()
    candidates = candidates.sort_values(
        by=["total_distance_to_human", "capture_strength", "probe_tag"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    best = candidates.iloc[0]
    metric_deltas = []
    improved_count = 0
    critical_ok = False
    for metric in TARGET_METRICS:
        before = float(abs(baseline_metrics[f"probe_{metric}"] - baseline_metrics[f"human_{metric}"]))
        after = float(abs(best[f"probe_{metric}"] - best[f"human_{metric}"]))
        improved = bool(after < before)
        direction_match = None
        if metric == "incongruent_error_minus_correct_rt":
            direction_match = bool(np.sign(best[f"probe_{metric}"]) == np.sign(best[f"human_{metric}"]))
        if improved:
            improved_count += 1
        if metric == "earliest_incongruent_caf" and improved:
            critical_ok = True
        if metric == "incongruent_error_minus_correct_rt" and improved and bool(direction_match):
            critical_ok = True
        metric_deltas.append(
            {
                "metric": metric,
                "human": float(best[f"human_{metric}"]),
                "baseline": float(baseline_metrics[f"probe_{metric}"]),
                "probe": float(best[f"probe_{metric}"]),
                "before_distance": before,
                "after_distance": after,
                "improved": improved,
                "direction_match": direction_match,
            }
        )
    verdict = "CAPTURE-PROBE-SUPPORTED" if improved_count >= 2 and critical_ok else "CAPTURE-PROBE-NOT-SUPPORTED"
    success_bar = {
        "weighting": "trial_weighted",
        "selected_probe_tag": str(best["probe_tag"]),
        "capture_strength": float(best["capture_strength"]),
        "capture_midpoint_s": float(best["capture_midpoint_s"]),
        "capture_tau_s": float(best["capture_tau_s"]),
        "target_metrics": TARGET_METRICS,
        "metric_deltas": metric_deltas,
        "critical_metric_rule": "At least one improved metric must be earliest_incongruent_caf or incongruent_error_minus_correct_rt with matching human direction.",
        "verdict": verdict,
    }
    success_bar_path = reagg_dir / "success_bar.json"
    success_bar_path.write_text(json.dumps(success_bar, indent=2))

    ev5 = _write_evidence(
        5,
        "analyze-scorecard",
        {
            "command": "python code/scripts/analyze_minimal_conflict_capture_probe.py --input_root <input_root> --output_root <output_root>",
            "input_root": _safe_rel_to_root(input_root),
            "output_root": _safe_rel_to_root(output_root),
            "probe_scorecard": _safe_rel_to_root(scorecard_path),
            "success_bar": _safe_rel_to_root(success_bar_path),
            "verdict": verdict,
        },
    )
    print(f"Wrote evidence: {ev5}")

    summary_lines = [
        "# Minimal conflict-capture probe summary",
        "",
        "This is a **diagnostic mechanism probe** layered on top of `dynamic_selection_phase1`.",
        "",
        f"Previous heterogeneity result: `{heterogeneity_verdict or 'UNKNOWN'}`",
        "",
        "## Explicit comparison",
        "- **Current baseline**: `dynamic_selection_phase1` with no conflict-capture term (`capture_strength=0.0`).",
        f"- **Heterogeneity probe result**: `{heterogeneity_verdict or 'UNKNOWN'}`; subject-level scale variation alone did not rescue the locked flanker diagnostics.",
        f"- **Minimal conflict-capture probe result**: `{verdict}` with best bounded setting `{success_bar['selected_probe_tag']}`.",
        "",
        "## Bounded probe design",
        "- only one effective free mechanism degree of freedom was probed: `capture_strength`",
        f"- fixed timing constants: `capture_midpoint_s={success_bar['capture_midpoint_s']}` and `capture_tau_s={success_bar['capture_tau_s']}`",
        "- Stage-2 readout remained `baseline` and no urgency branch was introduced",
        "",
        "## Best probe",
        f"- probe tag: `{success_bar['selected_probe_tag']}`",
        f"- capture_strength: `{success_bar['capture_strength']}`",
        f"- verdict: `{verdict}`",
        "",
        "## Metric deltas vs baseline",
        "| Metric | Before distance (baseline→human) | After distance (capture-probe→human) | Improved |",
        "|---|---:|---:|:---:|",
    ]
    for item in metric_deltas:
        summary_lines.append(
            f"| {item['metric']} | {item['before_distance']:.6g} | {item['after_distance']:.6g} | {'yes' if item['improved'] else 'no'} |"
        )
    summary_lines.extend(
        [
            "",
            "## Execution",
            "- Runner: `python code/scripts/run_minimal_conflict_capture_probe.py --mode full --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe`",
            "- Analysis: `python code/scripts/analyze_minimal_conflict_capture_probe.py --input_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe --output_root artifacts/results/repro_legacy_interim/minimal_conflict_capture_probe`",
            "",
            "This memo is a diagnostic mechanism probe summary and should not be treated as confirmatory evidence for a new model family.",
        ]
    )
    summary_path = output_root / "minimal_conflict_capture_summary.md"
    summary_path.write_text("\n".join(summary_lines))
    ev6 = _write_evidence(
        6,
        "summary-memo",
        {
            "summary": _safe_rel_to_root(summary_path),
            "verdict": verdict,
            "selected_probe_tag": success_bar["selected_probe_tag"],
            "capture_strength": success_bar["capture_strength"],
        },
    )
    print(f"Wrote evidence: {ev6}")


if __name__ == "__main__":
    main()
