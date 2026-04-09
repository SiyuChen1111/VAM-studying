import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from project_paths import DATA_AGE_GROUPS_ROOT, CHECKPOINTS_AGE_GROUPS_ROOT, RESULTS_ROOT
from vgg_wongwang_lim import WWWrapper


AGE_GROUPS = ["20-29", "80-89"]
RESULTS_DIR = RESULTS_ROOT / "age_groups"
DT_DEFAULT = 10
DIR_MAP = {"L": 0, "R": 1, "U": 2, "D": 3}


def set_apa_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_stage2_artifacts(age_group: str):
    stage2_dir = CHECKPOINTS_AGE_GROUPS_ROOT / age_group / "stage2"
    data_dir = DATA_AGE_GROUPS_ROOT / age_group

    with open(stage2_dir / "best_config.json", "r") as f:
        best_config = json.load(f)

    params_npz = np.load(stage2_dir / "best_model_params.npz")
    params = {k: params_npz[k] for k in params_npz.files}

    test_logits_npz = np.load(stage2_dir / "test_logits.npz")
    test_df = pd.read_csv(data_dir / "test_data.csv")

    if len(test_df) != len(test_logits_npz["logits"]):
        raise ValueError(f"Length mismatch for {age_group}: test csv vs logits")

    with open(data_dir / "rt_stats.json", "r") as f:
        human_stats = json.load(f)

    return {
        "age_group": age_group,
        "best_config": best_config,
        "params": params,
        "test_logits": test_logits_npz["logits"].astype(np.float32),
        "test_rts": test_logits_npz["rts"].astype(np.float32),
        "test_rts_normalized": test_logits_npz["rts_normalized"].astype(np.float32),
        "test_df": test_df.copy(),
        "human_stats": human_stats,
    }


def build_model(best_config, params):
    time_steps = int(best_config["time_steps"])
    model = WWWrapper(n_classes=4, dt=DT_DEFAULT, time_steps=time_steps)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in params:
            state_dict[key] = torch.tensor(params[key], dtype=torch.float32)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_dir_idx"] = out["target_direction"].map(lambda x: DIR_MAP.get(x, -1))
    out["flanker_dir_idx"] = out["flanker_direction"].map(lambda x: DIR_MAP.get(x, -1))
    out["response_dir_idx"] = out["response_direction"].map(lambda x: DIR_MAP.get(x, -1))
    out["correct"] = (out["target_dir_idx"] == out["response_dir_idx"]).astype(int)
    out["congruency"] = (out["target_dir_idx"] != out["flanker_dir_idx"]).astype(int)
    out["condition"] = out["congruency"].map(lambda x: "Congruent" if x == 0 else "Incongruent")
    return out


def run_inference(model: WWWrapper, logits: np.ndarray):
    x = torch.tensor(logits, dtype=torch.float32)
    with torch.no_grad():
        scale_tensor = model.state_dict()["scale"]
        scaled = F.relu(x * scale_tensor)
        decision_times_class, trajectory, threshold = model.ww.inference(scaled)
        final_rt, pred_choice = decision_times_class.min(dim=1)
    return {
        "decision_times_class": decision_times_class.cpu().numpy(),
        "trajectory": trajectory.cpu().numpy(),
        "threshold": float(threshold.detach().cpu().item()),
        "pred_rt": final_rt.cpu().numpy(),
        "pred_choice": pred_choice.cpu().numpy(),
    }


def summarize_group(artifact):
    df = enrich_df(artifact["test_df"])
    model = build_model(artifact["best_config"], artifact["params"])
    inf = run_inference(model, artifact["test_logits"])

    df["pred_rt"] = inf["pred_rt"]
    df["pred_choice"] = inf["pred_choice"]
    df["pred_correct"] = (df["pred_choice"] == df["target_dir_idx"]).astype(int)

    summary = {
        "age_group": artifact["age_group"],
        "best_scale": float(artifact["best_config"]["scale"]),
        "best_score": float(artifact["best_config"]["score"]),
        "human_mean_rt": float(df["response_time"].mean() / 1000.0),
        "model_mean_rt": float(df["pred_rt"].mean()),
        "human_median_rt": float(df["response_time"].median() / 1000.0),
        "model_median_rt": float(np.median(df["pred_rt"])),
        "human_skew": float(pd.Series(df["response_time"] / 1000.0).skew()),
        "model_skew": float(pd.Series(df["pred_rt"]).skew()),
        "human_accuracy": float(df["correct"].mean()),
        "model_accuracy": float(df["pred_correct"].mean()),
        "human_congruency_rt_gap": float(
            df.loc[df["congruency"] == 1, "response_time"].mean() / 1000.0
            - df.loc[df["congruency"] == 0, "response_time"].mean() / 1000.0
        ),
        "model_congruency_rt_gap": float(
            df.loc[df["congruency"] == 1, "pred_rt"].mean()
            - df.loc[df["congruency"] == 0, "pred_rt"].mean()
        ),
        "threshold": inf["threshold"],
        "noise_ampa": float(artifact["params"]["ww.noise_ampa"]),
        "J_ext": float(artifact["params"]["ww.J_ext"]),
        "I_0": float(artifact["params"]["ww.I_0"]),
        "J_self_mean": float(np.mean(np.diag(artifact["params"]["ww.J_matrix"]))),
        "J_offdiag_mean": float(
            np.mean(artifact["params"]["ww.J_matrix"][~np.eye(4, dtype=bool)])
        ),
    }

    return df, inf, summary


def build_matched_sets(group_dfs):
    key_cols = ["stimulus_layout", "target_direction", "flanker_direction", "congruency"]
    matched = {}
    young = group_dfs["20-29"]
    old = group_dfs["80-89"]

    for congruency, label in [(0, "Congruent"), (1, "Incongruent")]:
        y = young[young["congruency"] == congruency].copy()
        o = old[old["congruency"] == congruency].copy()
        pairs = []
        o_groups = o.groupby(key_cols)
        o_group_keys = o_groups.groups
        for key, y_grp in y.groupby(key_cols):
            if key not in o_group_keys:
                continue
            o_grp = o_groups.get_group(key)
            n = min(len(y_grp), len(o_grp))
            if n == 0:
                continue
            y_sel = y_grp.iloc[:n].copy()
            o_sel = o_grp.iloc[:n].copy()
            pairs.append((y_sel, o_sel))
        if pairs:
            matched[("20-29", label)] = pd.concat([p[0] for p in pairs]).sort_index()
            matched[("80-89", label)] = pd.concat([p[1] for p in pairs]).sort_index()
    return matched


def pca_fit(states: np.ndarray):
    mean = states.mean(axis=0, keepdims=True)
    centered = states - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2].T
    return mean.squeeze(0), comps


def project(states: np.ndarray, mean: np.ndarray, comps: np.ndarray):
    return (states - mean) @ comps


def make_parameter_plot(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    params = ["noise_ampa", "threshold", "J_ext", "I_0", "J_self_mean", "J_offdiag_mean"]
    titles = ["Noise", "Threshold", "J_ext", "I_0", "J_self", "J_offdiag"]
    for ax, param, title in zip(axes.flat, params, titles):
        ax.bar(summary_df["age_group"], summary_df[param], color=["#4C78A8", "#F58518"])
        ax.set_title(title)
    fig.suptitle("Figure 1. Stage-2 parameter comparison by age group")
    fig.savefig(RESULTS_DIR / "figure1_parameter_comparison.png", bbox_inches="tight")
    plt.close(fig)


def make_signature_plot(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = [
        ("mean_rt", "Mean RT (s)"),
        ("skew", "RT skewness"),
        ("accuracy", "Accuracy"),
        ("congruency_rt_gap", "Congruency RT gap (s)"),
    ]
    for ax, (name, ylabel) in zip(axes.flat, metrics):
        x = np.arange(len(summary_df))
        width = 0.35
        ax.bar(x - width / 2, summary_df[f"human_{name}"], width, label="Human", color="#A0A0A0")
        ax.bar(x + width / 2, summary_df[f"model_{name}"], width, label="Model", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["age_group"])
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Figure 2. Human signatures versus fitted model by age group")
    fig.savefig(RESULTS_DIR / "figure2_human_signatures.png", bbox_inches="tight")
    plt.close(fig)


def make_rt_distribution_plot(group_dfs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, age in zip(axes, AGE_GROUPS):
        human = group_dfs[age]["response_time"] / 1000.0
        model = group_dfs[age]["pred_rt"]
        ax.hist(human, bins=50, density=True, alpha=0.45, label="Human", color="#A0A0A0")
        ax.hist(model, bins=50, density=True, alpha=0.45, label="Model", color="#4C78A8")
        ax.set_title(age)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure 3. RT distributions by age group")
    fig.savefig(RESULTS_DIR / "figure3_rt_distributions.png", bbox_inches="tight")
    plt.close(fig)


def make_geometry_plot(group_dfs, inference_by_group):
    matched = build_matched_sets(group_dfs)
    required_keys = [("20-29", "Congruent"), ("20-29", "Incongruent"), ("80-89", "Congruent"), ("80-89", "Incongruent")]
    if not all(k in matched for k in required_keys):
        return None

    state_blocks = []
    mean_trajs = {}
    spread_rows = []
    labels = []

    for age, condition in required_keys:
        idx = matched[(age, condition)].index.to_numpy()
        traj = inference_by_group[age]["trajectory"][idx]
        state_blocks.append(traj.reshape(-1, traj.shape[-1]))
        mean_traj = traj.mean(axis=0)
        mean_trajs[(age, condition)] = mean_traj
        spread = np.linalg.norm(traj - mean_traj[None, :, :], axis=2).mean()
        spread_rows.append({"age_group": age, "condition": condition, "mean_state_space_spread": float(spread)})
        labels.append((age, condition))

    all_states = np.concatenate(state_blocks, axis=0)
    mean, comps = pca_fit(all_states)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {
        ("20-29", "Congruent"): "#4C78A8",
        ("20-29", "Incongruent"): "#72B7B2",
        ("80-89", "Congruent"): "#F58518",
        ("80-89", "Incongruent"): "#E45756",
    }
    for key, mean_traj in mean_trajs.items():
        proj = project(mean_traj, mean, comps)
        axes[0].plot(proj[:, 0], proj[:, 1], label=f"{key[0]} {key[1]}", color=colors[key], linewidth=2)
        axes[0].scatter(proj[0, 0], proj[0, 1], color=colors[key], s=18)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("State-space mean trajectories")
    axes[0].legend(frameon=False)

    spread_df = pd.DataFrame(spread_rows)
    x = np.arange(len(spread_df))
    bar_colors = []
    labels = []
    for _, row in spread_df.iterrows():
        age = str(row["age_group"])
        condition = str(row["condition"])
        bar_colors.append(colors[(age, condition)])
        labels.append(f"{age}\n{condition}")
    axes[1].bar(x, spread_df["mean_state_space_spread"], color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean trajectory spread")
    axes[1].set_title("State-space spread by age and congruency")

    fig.suptitle("Figure 4. Trajectory geometry under matched congruent and incongruent trials")
    fig.savefig(RESULTS_DIR / "figure4_trajectory_geometry.png", bbox_inches="tight")
    plt.close(fig)

    spread_df.to_csv(RESULTS_DIR / "trajectory_spread_summary.csv", index=False)
    return spread_df


def write_summary(summary_df: pd.DataFrame, geometry_df):
    lines = ["# Age-Group Post-Stage2 Analysis", ""]
    lines.append("## Parameter Summary")
    lines.append(str(summary_df.to_markdown(index=False)))
    lines.append("")
    if geometry_df is not None:
        lines.append("## Trajectory Geometry Summary")
        lines.append(str(geometry_df.to_markdown(index=False)))
        lines.append("")
    lines.append("## Generated Figures")
    lines.append("- Figure 1. Stage-2 parameter comparison by age group")
    lines.append("- Figure 2. Human signatures versus fitted model by age group")
    lines.append("- Figure 3. RT distributions by age group")
    if geometry_df is not None:
        lines.append("- Figure 4. Trajectory geometry under matched congruent and incongruent trials")
    (RESULTS_DIR / "analysis_summary.md").write_text("\n".join(lines))


def main():
    set_apa_style()
    ensure_dirs()

    group_dfs = {}
    inference_by_group = {}
    summary_rows = []

    for age in AGE_GROUPS:
        artifact = load_stage2_artifacts(age)
        df, inf, summary = summarize_group(artifact)
        group_dfs[age] = df
        inference_by_group[age] = inf
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "parameter_and_signature_summary.csv", index=False)

    make_parameter_plot(summary_df)
    make_signature_plot(summary_df)
    make_rt_distribution_plot(group_dfs)
    geometry_df = make_geometry_plot(group_dfs, inference_by_group)
    write_summary(summary_df, geometry_df)

    print(f"Saved post-analysis outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
