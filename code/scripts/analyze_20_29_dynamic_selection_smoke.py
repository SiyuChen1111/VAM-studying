import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_20_29_urgency_tie import (
    compute_caf,
    compute_conditional_error_rt,
    compute_delta,
    compute_tail_summary,
)
from project_paths import RESULTS_ROOT


DEFAULT_RUN_DIR = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_20_29_smoke"
SOURCE_LABELS = {
    "human": "Human",
    "urgency_baseline": "Urgency baseline",
    "dynamic_selection_phase1": "Dynamic selection",
    "dynamic_selection_dmc_extension": "Dynamic selection + DMC-like",
}
SOURCE_COLORS = {
    "human": "#4C78A8",
    "urgency_baseline": "#F58518",
    "dynamic_selection_phase1": "#54A24B",
    "dynamic_selection_dmc_extension": "#B279A2",
}


def detect_source_order(run_dir: Path) -> list[str]:
    source_order = ["human", "urgency_baseline", "dynamic_selection_phase1"]
    if (run_dir / "dynamic_selection_dmc_extension" / "predictions.npz").exists():
        source_order.append("dynamic_selection_dmc_extension")
    return source_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dynamic-selection smoke comparison against urgency baseline.")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    return parser.parse_args()


def load_summary(run_dir: Path, tag: str) -> dict:
    with (run_dir / tag / "summary.json").open() as handle:
        return json.load(handle)


def load_predictions(run_dir: Path, tag: str) -> dict:
    npz = np.load(run_dir / tag / "predictions.npz")
    return {key: npz[key] for key in npz.files}


def build_trial_df(source: str, predictions: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "source": source,
        "true_rt": predictions["true_rt"].astype(np.float32),
        "pred_rt": predictions["pred_rt"].astype(np.float32),
        "target_label": predictions["target_labels"].astype(np.int64),
        "response_label": predictions["response_labels"].astype(np.int64),
        "pred_choice": predictions["pred_choice"].astype(np.int64),
        "congruency": predictions["congruency"].astype(np.int64),
    })
    df["condition"] = df["congruency"].map(lambda value: "congruent" if value == 0 else "incongruent")
    df["human_correct"] = df["response_label"] == df["target_label"]
    df["pred_correct"] = df["pred_choice"] == df["target_label"]
    return df


def save_outputs(run_dir: Path, caf_df: pd.DataFrame, delta_df: pd.DataFrame, error_wide: pd.DataFrame, tail_df: pd.DataFrame) -> None:
    for condition in ("congruent", "incongruent"):
        caf_df.loc[caf_df["condition"] == condition].to_csv(run_dir / f"caf_{condition}.csv", index=False)
    delta_df.to_csv(run_dir / "delta_plot.csv", index=False)
    error_wide.to_csv(run_dir / "conditional_error_rt.csv", index=False)
    tail_df.to_csv(run_dir / "conditional_tail_summary.csv", index=False)


def plot_caf(run_dir: Path, caf_df: pd.DataFrame, source_order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, condition in zip(axes, ("congruent", "incongruent")):
        subset = caf_df.loc[caf_df["condition"] == condition]
        for source in source_order:
            src = subset.loc[subset["source"] == source].sort_values("bin_index")
            if src.empty:
                continue
            ax.plot(src["bin_index"], src["accuracy"], marker="o", linewidth=2, color=SOURCE_COLORS[source], label=SOURCE_LABELS[source])
        ax.set_title(condition.capitalize())
        ax.set_xlabel("RT quantile bin")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional accuracy function")
    fig.savefig(run_dir / "caf_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_delta(run_dir: Path, delta_df: pd.DataFrame, source_order: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for source in source_order:
        src = delta_df.loc[delta_df["source"] == source].sort_values("quantile_index")
        if src.empty:
            continue
        ax.plot(src["quantile_index"], src["delta"], marker="o", linewidth=2, color=SOURCE_COLORS[source], label=SOURCE_LABELS[source])
    ax.axhline(0.0, color="black", alpha=0.5)
    ax.set_xlabel("RT quantile")
    ax.set_ylabel("Incongruent − congruent RT (s)")
    ax.set_title("Delta plot")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(run_dir / "delta_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_error(run_dir: Path, error_long: pd.DataFrame, source_order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
    x = np.arange(len(source_order))
    width = 0.35
    for ax, condition in zip(axes, ("congruent", "incongruent")):
        subset = error_long.loc[error_long["condition"] == condition].set_index("source")
        correct_vals = [subset.loc[source, "correct_rt"] for source in source_order]
        error_vals = [subset.loc[source, "error_rt"] for source in source_order]
        ax.bar(x - width / 2, correct_vals, width, label="Correct", color="#4C78A8")
        ax.bar(x + width / 2, error_vals, width, label="Error", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABELS[source] for source in source_order], rotation=20, ha="right")
        ax.set_title(condition.capitalize())
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional error RT structure")
    fig.savefig(run_dir / "conditional_error_rt_plot.png", bbox_inches="tight")
    plt.close(fig)


def plot_tail(run_dir: Path, tail_df: pd.DataFrame, source_order: list[str]) -> None:
    groups = ["congruent_correct", "congruent_error", "incongruent_correct", "incongruent_error"]
    x = np.arange(len(groups))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    center = (len(source_order) - 1) / 2.0
    for source_idx, source in enumerate(source_order):
        subset = tail_df.loc[tail_df["source"] == source].set_index("group")
        q95_vals = [subset.loc[group, "q95"] for group in groups]
        skew_vals = [subset.loc[group, "skewness"] for group in groups]
        offset = (source_idx - center) * width
        axes[0].bar(x + offset, q95_vals, width, label=SOURCE_LABELS[source], color=SOURCE_COLORS[source])
        axes[1].bar(x + offset, skew_vals, width, label=SOURCE_LABELS[source], color=SOURCE_COLORS[source])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([group.replace("_", "\n") for group in groups])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("q95 by condition/correctness")
    axes[1].set_title("Skewness by condition/correctness")
    axes[0].legend(frameon=False)
    fig.suptitle("Conditional tail summary")
    fig.savefig(run_dir / "conditional_tail_plot.png", bbox_inches="tight")
    plt.close(fig)


def build_comparison_table(summaries: dict, caf_df: pd.DataFrame, delta_df: pd.DataFrame, error_wide: pd.DataFrame, tail_df: pd.DataFrame, mechanism_sources: list[str]) -> pd.DataFrame:
    human_caf = caf_df.loc[caf_df["source"] == "human"]
    human_delta = delta_df.loc[delta_df["source"] == "human"].sort_values("quantile_index")
    human_error = error_wide.loc[error_wide["source"] == "human"].iloc[0]
    rows = []
    for source in mechanism_sources:
        summary = summaries[source]
        caf_subset = caf_df.loc[caf_df["source"] == source]
        delta_subset = delta_df.loc[delta_df["source"] == source].sort_values("quantile_index")
        error_row = error_wide.loc[error_wide["source"] == source].iloc[0]
        tail_subset = tail_df.loc[tail_df["source"] == source].set_index("group")
        rows.append({
            "mechanism": source,
            "score": float(summary["score"]),
            "frac_at_ceiling": float(summary["frac_at_ceiling"]),
            "coverage_score": float(summary["coverage_score"]),
            "pred_q95": float(summary["pred_q95"]),
            "pred_q99": float(summary["pred_q99"]),
            "model_congruency_rt_gap": float(summary["model_congruency_rt_gap"]),
            "rt_shape_score": float(summary["rt_shape_score"]),
            "pred_skewness": float(summary["pred_skewness"]),
            "caf_incongruent_fast_bin_accuracy": float(caf_subset.loc[caf_subset["condition"] == "incongruent"].sort_values("bin_index").iloc[0]["accuracy"]),
            "human_caf_incongruent_fast_bin_accuracy": float(human_caf.loc[human_caf["condition"] == "incongruent"].sort_values("bin_index").iloc[0]["accuracy"]),
            "delta_first_quantile": float(delta_subset.iloc[0]["delta"]),
            "human_delta_first_quantile": float(human_delta.iloc[0]["delta"]),
            "incongruent_error_minus_correct_rt": float(error_row["incongruent_error_minus_correct_rt"]),
            "human_incongruent_error_minus_correct_rt": float(human_error["incongruent_error_minus_correct_rt"]),
            "incongruent_correct_q95": float(tail_subset.loc["incongruent_correct", "q95"]),
            "incongruent_error_q95": float(tail_subset.loc["incongruent_error", "q95"]),
            "selection_mode": summary["selection_config"].get("selection_mode", "baseline"),
        })
    return pd.DataFrame(rows)


def write_memo(run_dir: Path, comparison_df: pd.DataFrame) -> None:
    urgency = comparison_df.loc[comparison_df["mechanism"] == "urgency_baseline"].iloc[0]
    dynamic = comparison_df.loc[comparison_df["mechanism"] == "dynamic_selection_phase1"].iloc[0]
    has_extension = "dynamic_selection_dmc_extension" in set(comparison_df["mechanism"])
    candidate = comparison_df.loc[comparison_df["mechanism"] == "dynamic_selection_dmc_extension"].iloc[0] if has_extension else dynamic
    candidate_label = "Dynamic-selection + DMC-like extension" if has_extension else "Dynamic-selection"

    candidate_caf_error = abs(candidate["caf_incongruent_fast_bin_accuracy"] - candidate["human_caf_incongruent_fast_bin_accuracy"])
    dynamic_caf_error = abs(dynamic["caf_incongruent_fast_bin_accuracy"] - dynamic["human_caf_incongruent_fast_bin_accuracy"])
    candidate_delta_error = abs(candidate["delta_first_quantile"] - candidate["human_delta_first_quantile"])
    dynamic_delta_error = abs(dynamic["delta_first_quantile"] - dynamic["human_delta_first_quantile"])
    candidate_error_rt_error = abs(candidate["incongruent_error_minus_correct_rt"] - candidate["human_incongruent_error_minus_correct_rt"])
    dynamic_error_rt_error = abs(dynamic["incongruent_error_minus_correct_rt"] - dynamic["human_incongruent_error_minus_correct_rt"])

    caf_improved_vs_phase1 = candidate_caf_error < dynamic_caf_error
    delta_improved_vs_phase1 = candidate_delta_error < dynamic_delta_error
    error_rt_improved_vs_phase1 = candidate_error_rt_error < dynamic_error_rt_error

    improvements = []
    regressions = []
    if caf_improved_vs_phase1:
        improvements.append("earliest incongruent CAF accuracy moved closer to human than phase 1")
    elif has_extension:
        regressions.append("earliest incongruent CAF accuracy did not improve over phase 1")

    if delta_improved_vs_phase1:
        improvements.append("early delta-plot direction/magnitude moved closer to human than phase 1")
    elif has_extension:
        regressions.append("early delta-plot realism regressed relative to phase 1")

    if error_rt_improved_vs_phase1:
        improvements.append("incongruent conditional error RT became less pathological than phase 1")
    elif has_extension:
        regressions.append("incongruent conditional error RT did not improve over phase 1")

    severe_regression = (
        candidate["frac_at_ceiling"] > 0.05
        or candidate["coverage_score"] < urgency["coverage_score"] * 0.5
        or candidate["model_congruency_rt_gap"] < 0.0
    )
    if has_extension and caf_improved_vs_phase1 and delta_improved_vs_phase1 and error_rt_improved_vs_phase1 and not severe_regression:
        verdict = "Yes — adding a minimal DMC-like automatic/control extension improves the flanker-specific diagnostics enough to justify promotion in smoke testing."
        recommendation = "Promote the dynamic-selection + DMC-like extension as the new active branch."
    elif has_extension:
        verdict = "No — the minimal DMC-like automatic/control extension does not improve the primary flanker-specific diagnostics enough over the current dynamic-selection phase 1 reference."
        recommendation = "Keep the phase-1 dynamic-selection branch active and stop tuning this particular DMC-like extension path until a different conflict-processing mechanism is proposed."
    elif improvements and not severe_regression:
        verdict = "Yes — the minimal dynamic-selection mechanism improves flanker-specific diagnostics over the current urgency baseline in smoke testing."
        recommendation = "Promote dynamic attentional selection to the next stage before attempting a DMC-like prototype."
    else:
        verdict = "No — the minimal dynamic-selection mechanism does not yet improve the flanker-specific diagnostics enough over the current urgency baseline."
        recommendation = "Move to a DMC-like minimal prototype next."

    memo = f"""# Mechanism pivot decision memo

## Bottom line

{verdict}

## Phase 1 comparison

- Urgency baseline earliest incongruent CAF accuracy: `{urgency['caf_incongruent_fast_bin_accuracy']:.4f}`
- Dynamic-selection earliest incongruent CAF accuracy: `{dynamic['caf_incongruent_fast_bin_accuracy']:.4f}`
- {candidate_label} earliest incongruent CAF accuracy: `{candidate['caf_incongruent_fast_bin_accuracy']:.4f}`
- Human earliest incongruent CAF accuracy: `{candidate['human_caf_incongruent_fast_bin_accuracy']:.4f}`

- Urgency baseline first delta quantile: `{urgency['delta_first_quantile']:.4f}` s
- Dynamic-selection first delta quantile: `{dynamic['delta_first_quantile']:.4f}` s
- {candidate_label} first delta quantile: `{candidate['delta_first_quantile']:.4f}` s
- Human first delta quantile: `{candidate['human_delta_first_quantile']:.4f}` s

- Urgency baseline incongruent error-minus-correct RT: `{urgency['incongruent_error_minus_correct_rt']:.4f}` s
- Dynamic-selection incongruent error-minus-correct RT: `{dynamic['incongruent_error_minus_correct_rt']:.4f}` s
- {candidate_label} incongruent error-minus-correct RT: `{candidate['incongruent_error_minus_correct_rt']:.4f}` s
- Human incongruent error-minus-correct RT: `{candidate['human_incongruent_error_minus_correct_rt']:.4f}` s

## Mechanism-oriented reading

 - Improvements observed: `{improvements}`
 - Regressions observed: `{regressions}`
- {candidate_label} ceiling fraction: `{candidate['frac_at_ceiling']:.4f}`
- {candidate_label} q95 / q99: `{candidate['pred_q95']:.4f}` / `{candidate['pred_q99']:.4f}`
- {candidate_label} selection mode: `{candidate['selection_mode']}`

## Recommendation

{recommendation}
"""
    (run_dir / "mechanism_pivot_decision_memo.md").write_text(memo)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    source_order = detect_source_order(run_dir)
    mechanism_sources = [source for source in source_order if source != "human"]

    urgency_predictions = load_predictions(run_dir, "urgency_baseline")
    dynamic_predictions = load_predictions(run_dir, "dynamic_selection_phase1")
    summaries = {
        "urgency_baseline": load_summary(run_dir, "urgency_baseline"),
        "dynamic_selection_phase1": load_summary(run_dir, "dynamic_selection_phase1"),
    }
    extension_predictions = None
    if "dynamic_selection_dmc_extension" in mechanism_sources:
        extension_predictions = load_predictions(run_dir, "dynamic_selection_dmc_extension")
        summaries["dynamic_selection_dmc_extension"] = load_summary(run_dir, "dynamic_selection_dmc_extension")

    human_df = build_trial_df("human", urgency_predictions)
    urgency_df = build_trial_df("urgency_baseline", urgency_predictions)
    dynamic_df = build_trial_df("dynamic_selection_phase1", dynamic_predictions)
    extension_df = build_trial_df("dynamic_selection_dmc_extension", extension_predictions) if extension_predictions is not None else None

    caf_frames = [
        compute_caf(human_df, "human", "true_rt", "human_correct"),
        compute_caf(urgency_df, "urgency_baseline", "pred_rt", "pred_correct"),
        compute_caf(dynamic_df, "dynamic_selection_phase1", "pred_rt", "pred_correct"),
    ]
    delta_frames = [
        compute_delta(human_df, "human", "true_rt"),
        compute_delta(urgency_df, "urgency_baseline", "pred_rt"),
        compute_delta(dynamic_df, "dynamic_selection_phase1", "pred_rt"),
    ]
    if extension_df is not None:
        caf_frames.append(compute_caf(extension_df, "dynamic_selection_dmc_extension", "pred_rt", "pred_correct"))
        delta_frames.append(compute_delta(extension_df, "dynamic_selection_dmc_extension", "pred_rt"))
    caf_df = pd.concat(caf_frames, ignore_index=True)
    delta_df = pd.concat(delta_frames, ignore_index=True)

    error_wide_frames = []
    error_long_frames = []
    error_inputs = [
        ("human", human_df, "true_rt", "human_correct"),
        ("urgency_baseline", urgency_df, "pred_rt", "pred_correct"),
        ("dynamic_selection_phase1", dynamic_df, "pred_rt", "pred_correct"),
    ]
    if extension_df is not None:
        error_inputs.append(("dynamic_selection_dmc_extension", extension_df, "pred_rt", "pred_correct"))
    for source, df, rt_col, correct_col in error_inputs:
        wide, long = compute_conditional_error_rt(df, source, rt_col, correct_col)
        error_wide_frames.append(wide)
        error_long_frames.append(long)
    error_wide = pd.concat(error_wide_frames, ignore_index=True)
    error_long = pd.concat(error_long_frames, ignore_index=True)

    tail_frames = [
        compute_tail_summary(human_df, "human", "true_rt", "human_correct"),
        compute_tail_summary(urgency_df, "urgency_baseline", "pred_rt", "pred_correct"),
        compute_tail_summary(dynamic_df, "dynamic_selection_phase1", "pred_rt", "pred_correct"),
    ]
    if extension_df is not None:
        tail_frames.append(compute_tail_summary(extension_df, "dynamic_selection_dmc_extension", "pred_rt", "pred_correct"))
    tail_df = pd.concat(tail_frames, ignore_index=True)

    save_outputs(run_dir, caf_df, delta_df, error_wide, tail_df)
    plot_caf(run_dir, caf_df, source_order)
    plot_delta(run_dir, delta_df, source_order)
    plot_error(run_dir, error_long, source_order)
    plot_tail(run_dir, tail_df, source_order)

    comparison_df = build_comparison_table(summaries, caf_df, delta_df, error_wide, tail_df, mechanism_sources)
    comparison_df.to_csv(run_dir / "mechanism_pivot_comparison_table.csv", index=False)
    comparison_df.to_csv(run_dir / "mechanism_extension_comparison_table.csv", index=False)
    write_memo(run_dir, comparison_df)
    (run_dir / "mechanism_extension_decision_memo.md").write_text((run_dir / "mechanism_pivot_decision_memo.md").read_text())


if __name__ == "__main__":
    main()
