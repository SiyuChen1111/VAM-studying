import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import torch
import torch.nn.functional as F

from run_age_group_post_analysis import (
    build_matched_sets,
    build_model,
    ensure_dirs,
    load_stage2_artifacts,
    pca_fit,
    project,
    set_apa_style,
    summarize_group,
)


INTERIM_RESULTS_DIR = Path("results/age_groups_interim")
DIR_MAP = {"L": 0, "R": 1, "U": 2, "D": 3}


def ensure_interim_dirs():
    ensure_dirs()
    INTERIM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_best_completed_scale_from_log(log_path: Path):
    text = log_path.read_text(errors="replace")
    pattern = re.compile(
        r"\[(?P<age>20-29) scale \d+/\d+\] Finished in [^|]+\| Score=(?P<score>[0-9.]+), "
        r"PredMean=(?P<pred_mean>[0-9.]+)s, Acc=(?P<model_acc>[0-9.]+), Cong=(?P<model_cong>[0-9.]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("No completed scale summaries found in 20-29 log")

    best = None
    best_score = -np.inf
    for match in matches:
        score = float(match.group("score"))
        if score > best_score:
            best_score = score
            best = {
                "age_group": match.group("age"),
                "status": "interim_best_completed_scale",
                "best_score": score,
                "model_mean_rt": float(match.group("pred_mean")),
                "model_accuracy": float(match.group("model_acc")),
                "model_congruency_rt_gap": float(match.group("model_cong")),
            }

    scale_pattern = re.compile(
        r"\[(20-29) scale (?P<idx>\d+)/\d+\] Finished in [^|]+\| Score=(?P<score>[0-9.]+), "
        r"PredMean=(?P<pred_mean>[0-9.]+)s, Acc=(?P<model_acc>[0-9.]+), Cong=(?P<model_cong>[0-9.]+)"
    )
    scale_lookup = {}
    scale_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for match in scale_pattern.finditer(text):
        idx = int(match.group("idx"))
        scale_lookup[float(match.group("score"))] = scale_values[idx - 1]
    if best is None:
        raise ValueError("Failed to extract best completed-scale summary from 20-29 log")
    best["best_scale"] = scale_lookup.get(best["best_score"], np.nan)

    progress_match = re.findall(r"\[20-29 scale (?P<scale>\d+/\d+)\] Epoch (?P<epoch>\d+/\d+)", text)
    if progress_match:
        best["current_progress"] = f"scale {progress_match[-1][0]}, epoch {progress_match[-1][1]}"
    else:
        best["current_progress"] = "progress unavailable"
    return best


def build_80_89_formal_summary():
    artifact = load_stage2_artifacts("80-89")
    group_df, _, summary = summarize_group(artifact)
    summary["status"] = "final"
    return group_df, summary


def build_20_29_interim_summary():
    log_path = Path("logs/train_20_29_cached_unbuffered.log")
    summary = parse_best_completed_scale_from_log(log_path)
    with open("data_age_groups/20-29/rt_stats.json", "r") as f:
        human_stats = json.load(f)
    df = pd.read_csv("data_age_groups/20-29/test_data.csv")
    target = df["target_direction"].map(lambda x: DIR_MAP[x])
    response = df["response_direction"].map(lambda x: DIR_MAP[x])
    flanker = df["flanker_direction"].map(lambda x: DIR_MAP[x])
    congruency = (target != flanker).astype(int)

    summary.update(
        {
            "human_mean_rt": float(human_stats["mean"]),
            "human_median_rt": float(human_stats["median"]),
            "human_skew": float(human_stats["skewness"]),
            "human_accuracy": float((target == response).mean()),
            "human_congruency_rt_gap": float(
                df.loc[congruency == 1, "response_time"].mean() / 1000.0
                - df.loc[congruency == 0, "response_time"].mean() / 1000.0
            ),
        }
    )
    return summary


def make_80_89_signature_plot(summary):
    metrics = [
        ("Mean RT (s)", summary["human_mean_rt"], summary["model_mean_rt"]),
        ("Median RT (s)", summary["human_median_rt"], summary["model_median_rt"]),
        ("RT skewness", summary["human_skew"], summary["model_skew"]),
        ("Accuracy", summary["human_accuracy"], summary["model_accuracy"]),
        ("Congruency gap (s)", summary["human_congruency_rt_gap"], summary["model_congruency_rt_gap"]),
    ]
    labels = [m[0] for m in metrics]
    human_vals = [m[1] for m in metrics]
    model_vals = [m[2] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - width / 2, human_vals, width, label="Human", color="#A0A0A0")
    ax.bar(x + width / 2, model_vals, width, label="Model", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Figure A1. 80-89 human vs model behavioral signatures")
    ax.legend(frameon=False)
    fig.savefig(INTERIM_RESULTS_DIR / "figureA1_80_89_signatures.png", bbox_inches="tight")
    plt.close(fig)


def make_interim_comparison_plot(interim_20, final_80):
    rows = [
        {
            "age_group": "20-29",
            "status": interim_20["status"],
            "best_scale": interim_20["best_scale"],
            "best_score": interim_20["best_score"],
            "model_mean_rt": interim_20["model_mean_rt"],
            "human_mean_rt": interim_20["human_mean_rt"],
            "model_accuracy": interim_20["model_accuracy"],
            "human_accuracy": interim_20["human_accuracy"],
            "model_congruency_rt_gap": interim_20["model_congruency_rt_gap"],
            "human_congruency_rt_gap": interim_20["human_congruency_rt_gap"],
            "note": interim_20["current_progress"],
        },
        {
            "age_group": "80-89",
            "status": final_80["status"],
            "best_scale": final_80["best_scale"],
            "best_score": final_80["best_score"],
            "model_mean_rt": final_80["model_mean_rt"],
            "human_mean_rt": final_80["human_mean_rt"],
            "model_accuracy": final_80["model_accuracy"],
            "human_accuracy": final_80["human_accuracy"],
            "model_congruency_rt_gap": final_80["model_congruency_rt_gap"],
            "human_congruency_rt_gap": final_80["human_congruency_rt_gap"],
            "note": "formal Stage 2 result",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(INTERIM_RESULTS_DIR / "interim_vs_final_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), constrained_layout=True)
    metrics = [
        ("mean_rt", "Mean RT (s)"),
        ("accuracy", "Accuracy"),
        ("congruency_rt_gap", "Congruency gap (s)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width / 2, df[f"human_{metric}"], width, label="Human", color="#A0A0A0")
        ax.bar(x + width / 2, df[f"model_{metric}"], width, label="Model", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{age}\n({status})" for age, status in zip(df["age_group"], df["status"])])
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.suptitle("Figure A3. Interim 20-29 versus final 80-89 comparison")
    fig.savefig(INTERIM_RESULTS_DIR / "figureA3_interim_vs_final_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return df


def make_80_89_rt_distribution_plot(group_df_80):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    human_cong = group_df_80.loc[group_df_80["congruency"] == 0, "response_time"] / 1000.0
    human_incong = group_df_80.loc[group_df_80["congruency"] == 1, "response_time"] / 1000.0
    model_cong = group_df_80.loc[group_df_80["congruency"] == 0, "pred_rt"]
    model_incong = group_df_80.loc[group_df_80["congruency"] == 1, "pred_rt"]
    hist_bins = np.linspace(0, 2, 81)
    kde_x = np.linspace(0, 2, 400)

    def draw_panel(ax, series_a, series_b, title):
        ax.hist(series_a, bins=hist_bins, density=True, alpha=0.18, color="#4C78A8")
        ax.hist(series_b, bins=hist_bins, density=True, alpha=0.18, color="#E45756")
        if len(series_a) > 1:
            kde_a = gaussian_kde(series_a)
            ax.plot(kde_x, kde_a(kde_x), color="#4C78A8", linewidth=2.2, linestyle="-", label="Congruent")
        if len(series_b) > 1:
            kde_b = gaussian_kde(series_b)
            ax.plot(kde_x, kde_b(kde_x), color="#E45756", linewidth=2.2, linestyle="-", label="Incongruent")
        ax.set_title(title)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 2)
        ax.legend(frameon=False)

    draw_panel(axes[0], human_cong, human_incong, "Human RT distributions")
    draw_panel(axes[1], model_cong, model_incong, "Model RT distributions")

    fig.suptitle("Figure A2. 80-89 congruent vs incongruent RT distributions (Human vs Model, step-hist + KDE)")
    fig.savefig(INTERIM_RESULTS_DIR / "figureA2_80_89_rt_distributions.png", bbox_inches="tight")
    plt.close(fig)


def make_80_89_multipanel_style_figure(group_df_80):
    df = group_df_80.copy()
    df["human_rt"] = df["response_time"] / 1000.0
    cond_order = ["Congruent", "Incongruent"]
    cond_colors = {"Congruent": "#4C78A8", "Incongruent": "#E45756"}
    point_alpha = 0.35

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("Figure B1. Human and model RT profile for 80-89")

    for col, source in enumerate(["human", "model"]):
        title = "Human" if source == "human" else "Stage2 model"
        axes[0, col].set_title(title)

    def add_row_label(ax, label, title):
        ax.text(-0.34, 1.12, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
        ax.text(-0.18, 1.12, title, transform=ax.transAxes, fontsize=12, va="top")

    def format_p_value(p_value):
        if p_value < 0.001:
            return "p < .001"
        return f"p = {p_value:.3f}".replace("0.", ".")

    def add_significance_if_needed(ax, p_value, y_top):
        if not np.isfinite(p_value) or p_value >= 0.05:
            return
        x1, x2 = 0, 1
        y = y_top * 1.04 if y_top > 0 else 0.05
        h = y_top * 0.03 if y_top > 0 else 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#555555", linewidth=1.0)
        ax.text((x1 + x2) / 2, y + h * 1.1, format_p_value(p_value), ha="center", va="bottom", fontsize=10)

    def bar_with_group_points(ax, overall, grouped, ylabel, ylim=None, p_value=None):
        x = np.arange(len(cond_order))
        vals = [overall.get(cond, np.nan) for cond in cond_order]
        colors = [cond_colors[c] for c in cond_order]
        ax.bar(x, vals, color=colors, alpha=0.35, width=0.65)
        for i, cond in enumerate(cond_order):
            pts = grouped.get(cond, [])
            if len(pts):
                jitter = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else np.array([0.0])
                ax.scatter(np.full(len(pts), i) + jitter, pts, color=colors[i], s=18, alpha=point_alpha, edgecolors="white", linewidths=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_order)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        y_top = max([v for v in vals if np.isfinite(v)] + [0])
        add_significance_if_needed(ax, p_value, y_top)

    human_rt = df.groupby("condition")["human_rt"].mean()
    model_rt = df.groupby("condition")["pred_rt"].mean()
    human_rt_groups = {cond: grp.groupby("stimulus_layout")["human_rt"].mean().tolist() for cond, grp in df.groupby("condition")}
    model_rt_groups = {cond: grp.groupby("stimulus_layout")["pred_rt"].mean().tolist() for cond, grp in df.groupby("condition")}
    human_rt_p = stats.mannwhitneyu(df.loc[df["condition"] == "Congruent", "human_rt"], df.loc[df["condition"] == "Incongruent", "human_rt"], alternative="two-sided").pvalue
    model_rt_p = stats.mannwhitneyu(df.loc[df["condition"] == "Congruent", "pred_rt"], df.loc[df["condition"] == "Incongruent", "pred_rt"], alternative="two-sided").pvalue
    add_row_label(axes[0, 0], "b", "RT")
    bar_with_group_points(axes[0, 0], human_rt.to_dict(), human_rt_groups, "Average RT (s)", p_value=human_rt_p)
    bar_with_group_points(axes[0, 1], model_rt.to_dict(), model_rt_groups, "Average RT (s)", p_value=model_rt_p)

    add_row_label(axes[1, 0], "c", "RT distributions")
    xs = np.linspace(0, 2, 400)
    for ax, source_col in [(axes[1, 0], "human_rt"), (axes[1, 1], "pred_rt")]:
        for cond in cond_order:
            series = df.loc[df["condition"] == cond, source_col].to_numpy()
            color = cond_colors[cond]
            ax.hist(series, bins=np.linspace(0, 2, 81), density=True, alpha=0.18, color=color)
            if len(series) > 1:
                kde = gaussian_kde(series)
                ax.plot(xs, kde(xs), color=color, linewidth=2.2, label=cond)
        ax.set_xlim(0, 2)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
    axes[1, 0].legend(frameon=False)

    add_row_label(axes[2, 0], "d", "Skew of RT distributions")
    human_skew = df.groupby("condition")["human_rt"].apply(pd.Series.skew)
    model_skew = df.groupby("condition")["pred_rt"].apply(pd.Series.skew)
    human_skew_groups = {cond: grp.groupby("stimulus_layout")["human_rt"].apply(pd.Series.skew).replace([np.inf, -np.inf], np.nan).dropna().tolist() for cond, grp in df.groupby("condition")}
    model_skew_groups = {cond: grp.groupby("stimulus_layout")["pred_rt"].apply(pd.Series.skew).replace([np.inf, -np.inf], np.nan).dropna().tolist() for cond, grp in df.groupby("condition")}
    human_skew_p = np.nan
    if len(human_skew_groups.get("Congruent", [])) > 0 and len(human_skew_groups.get("Incongruent", [])) > 0:
        human_skew_p = stats.mannwhitneyu(human_skew_groups.get("Congruent", []), human_skew_groups.get("Incongruent", []), alternative="two-sided").pvalue
    model_skew_p = np.nan
    if len(model_skew_groups.get("Congruent", [])) > 0 and len(model_skew_groups.get("Incongruent", [])) > 0:
        model_skew_p = stats.mannwhitneyu(model_skew_groups.get("Congruent", []), model_skew_groups.get("Incongruent", []), alternative="two-sided").pvalue
    bar_with_group_points(axes[2, 0], human_skew.to_dict(), human_skew_groups, "Skewness", p_value=human_skew_p)
    bar_with_group_points(axes[2, 1], model_skew.to_dict(), model_skew_groups, "Skewness", p_value=model_skew_p)

    fig.savefig(INTERIM_RESULTS_DIR / "figureB1_80_89_multipanel_profile.png", bbox_inches="tight")
    plt.close(fig)


def make_interim_trajectory_plot(group_df_80, interim_20, artifact_80):
    log_path = Path("logs/train_20_29_cached_unbuffered.log")
    log_text = log_path.read_text(errors="replace")
    best_scale = float(interim_20["best_scale"])
    scale_token = f"scale={best_scale:.3f}"
    if scale_token not in log_text:
        return None

    stage2_dir_20 = Path("checkpoints_age_groups/20-29/stage2")
    test_logits_20 = np.load(stage2_dir_20 / "test_logits.npz")
    test_df_20 = pd.read_csv("data_age_groups/20-29/test_data.csv")
    df_20 = test_df_20.copy()
    df_20["target_dir_idx"] = df_20["target_direction"].map(lambda x: DIR_MAP[x])
    df_20["flanker_dir_idx"] = df_20["flanker_direction"].map(lambda x: DIR_MAP[x])
    df_20["response_dir_idx"] = df_20["response_direction"].map(lambda x: DIR_MAP[x])
    df_20["correct"] = (df_20["target_dir_idx"] == df_20["response_dir_idx"]).astype(int)
    df_20["congruency"] = (df_20["target_dir_idx"] != df_20["flanker_dir_idx"]).astype(int)
    df_20["condition"] = df_20["congruency"].map(lambda x: "Congruent" if x == 0 else "Incongruent")

    model_20 = build_model({"time_steps": 111, "scale": best_scale}, artifact_80["params"])
    state_dict = model_20.state_dict()
    state_dict["scale"] = torch.tensor(best_scale, dtype=torch.float32)
    model_20.load_state_dict(state_dict, strict=False)
    model_20.eval()

    x20 = torch.tensor(test_logits_20["logits"].astype(np.float32), dtype=torch.float32)
    with torch.no_grad():
        scaled20 = F.relu(x20 * model_20.state_dict()["scale"])
        _, traj20, _ = model_20.ww.inference(scaled20)

    group_df_20 = df_20.copy()
    group_df_20["pred_rt"] = 0.0
    group_df_20["pred_choice"] = 0
    inf20 = {"trajectory": traj20.cpu().numpy()}

    matched = build_matched_sets({"20-29": group_df_20, "80-89": group_df_80})
    required_keys = [("20-29", "Congruent"), ("20-29", "Incongruent"), ("80-89", "Congruent"), ("80-89", "Incongruent")]
    if not all(key in matched for key in required_keys):
        return None

    x80 = artifact_80["test_logits"].astype(np.float32)
    model_80 = build_model(artifact_80["best_config"], artifact_80["params"])
    with torch.no_grad():
        scaled80 = F.relu(torch.tensor(x80, dtype=torch.float32) * model_80.state_dict()["scale"])
        _, traj80, _ = model_80.ww.inference(scaled80)
    inf80 = {"trajectory": traj80.cpu().numpy()}

    state_blocks = []
    mean_trajs = {}
    spread_rows = []
    inference_by_group = {"20-29": inf20, "80-89": inf80}
    colors = {
        ("20-29", "Congruent"): "#4C78A8",
        ("20-29", "Incongruent"): "#72B7B2",
        ("80-89", "Congruent"): "#F58518",
        ("80-89", "Incongruent"): "#E45756",
    }
    for age, condition in required_keys:
        idx = matched[(age, condition)].index.to_numpy()
        traj = inference_by_group[age]["trajectory"][idx]
        state_blocks.append(traj.reshape(-1, traj.shape[-1]))
        mean_traj = traj.mean(axis=0)
        mean_trajs[(age, condition)] = mean_traj
        spread = np.linalg.norm(traj - mean_traj[None, :, :], axis=2).mean()
        spread_rows.append({"age_group": age, "condition": condition, "mean_state_space_spread": float(spread)})

    all_states = np.concatenate(state_blocks, axis=0)
    mean, comps = pca_fit(all_states)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for key, mean_traj in mean_trajs.items():
        proj = project(mean_traj, mean, comps)
        axes[0].plot(proj[:, 0], proj[:, 1], label=f"{key[0]} {key[1]}", color=colors[key], linewidth=2)
        axes[0].scatter(proj[0, 0], proj[0, 1], color=colors[key], s=18)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("Interim trajectory geometry")
    axes[0].legend(frameon=False)

    spread_df = pd.DataFrame(spread_rows)
    x = np.arange(len(spread_df))
    bar_colors = [colors[(str(row["age_group"]), str(row["condition"]))] for _, row in spread_df.iterrows()]
    labels = [f"{row['age_group']}\n{row['condition']}" for _, row in spread_df.iterrows()]
    axes[1].bar(x, spread_df["mean_state_space_spread"], color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean trajectory spread")
    axes[1].set_title("Matched-trial spread")

    fig.suptitle("Figure A4. Interim trajectory geometry using 20-29 current-best scale and 80-89 formal fit")
    fig.savefig(INTERIM_RESULTS_DIR / "figureA4_interim_trajectory_geometry.png", bbox_inches="tight")
    plt.close(fig)
    spread_df.to_csv(INTERIM_RESULTS_DIR / "figureA4_interim_trajectory_spread.csv", index=False)
    return spread_df


def write_memo(interim_20, final_80, geometry_available):
    memo = f"""# Current-stage conclusion memo

## Scope
This memo summarizes what can be concluded before the 20-29 Stage 2 run fully finishes.

## Current result status
- 80-89: formal Stage 2 result available
- 20-29: Stage 2 still running; best completed-scale interim result available

## 80-89 formal result
- Best scale: {final_80['best_scale']:.3f}
- Best score: {final_80['best_score']:.4f}
- Human mean RT: {final_80['human_mean_rt']:.3f}s
- Model mean RT: {final_80['model_mean_rt']:.3f}s
- Human accuracy: {final_80['human_accuracy']:.4f}
- Model accuracy: {final_80['model_accuracy']:.4f}
- Human congruency RT gap: {final_80['human_congruency_rt_gap']:.4f}s
- Model congruency RT gap: {final_80['model_congruency_rt_gap']:.4f}s

Interpretation: the model captures the approximate congruency RT gap and maintains very high accuracy, but remains far too fast overall and fails to reproduce the heavy right-skew expected in older-adult behavior.

## 20-29 interim result
- Best completed-scale interim scale: {interim_20['best_scale']:.3f}
- Best completed-scale interim score: {interim_20['best_score']:.4f}
- Human mean RT: {interim_20['human_mean_rt']:.3f}s
- Model mean RT: {interim_20['model_mean_rt']:.3f}s
- Human accuracy: {interim_20['human_accuracy']:.4f}
- Model accuracy: {interim_20['model_accuracy']:.4f}
- Human congruency RT gap: {interim_20['human_congruency_rt_gap']:.4f}s
- Model congruency RT gap: {interim_20['model_congruency_rt_gap']:.4f}s
- Live progress note: {interim_20['current_progress']}

Interpretation: the same broad pathology is already visible in the young group—accuracy is too close to ceiling and RT remains too fast—although the temporal mismatch appears less severe than in the old group.

## Cross-group interim pattern
Across both groups, the fitted model captures relative conflict structure more successfully than the absolute temporal regime of human decisions. The current Stage 2 solutions appear to live in a too-efficient decision regime: highly accurate, conflict-sensitive, but insufficiently slow and insufficiently heavy-tailed.

## Most important implication
The most valuable next analysis is not yet a strong age-mechanism claim, but a diagnosis of why the shared Stage 1 + current Stage 2 fitting setup systematically produces overly fast and overly accurate behavior.

## Immediate next step after 20-29 finishes
Run the full research-plan-aligned comparison in the required order:
1. behavior / human signatures
2. parameter comparison
3. mechanism / trajectory geometry

## Interim visual assets generated now
- Figure A1. 80-89 human vs model behavioral signatures
- Figure A2. 80-89 RT distribution comparison (x-axis constrained to 0-2 s)
- Figure A3. Interim 20-29 versus final 80-89 comparison
"""
    if geometry_available:
        memo += "- Figure A4. Interim trajectory geometry using 20-29 current-best scale and 80-89 formal fit\n"
    (INTERIM_RESULTS_DIR / "current_stage_conclusion_memo.md").write_text(memo)


def main():
    set_apa_style()
    ensure_interim_dirs()

    group_df_80, summary_80 = build_80_89_formal_summary()
    artifact_80 = load_stage2_artifacts("80-89")
    interim_20 = build_20_29_interim_summary()

    make_80_89_signature_plot(summary_80)

    make_80_89_rt_distribution_plot(group_df_80)
    make_80_89_multipanel_style_figure(group_df_80)

    comparison_df = make_interim_comparison_plot(interim_20, summary_80)
    geometry_df = make_interim_trajectory_plot(group_df_80, interim_20, artifact_80)
    write_memo(interim_20, summary_80, geometry_df is not None)
    comparison_df.to_markdown(INTERIM_RESULTS_DIR / "interim_vs_final_summary.md", index=False)
    print(f"Saved interim outputs to {INTERIM_RESULTS_DIR}")


if __name__ == "__main__":
    main()
