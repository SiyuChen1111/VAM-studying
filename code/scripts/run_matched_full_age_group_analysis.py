import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import gaussian_kde

from project_paths import (
    CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT,
    CHECKPOINTS_AGE_GROUPS_ROOT,
    DATA_AGE_GROUPS_MATCHED_ROOT,
    DATA_AGE_GROUPS_ROOT,
    RESULTS_ROOT,
    rel_to_root,
)
from vgg_wongwang_lim import WWWrapper


DT_DEFAULT = 10
DIR_MAP = {"L": 0, "R": 1, "U": 2, "D": 3}
DEFAULT_AGE_SPECS = {
    "20-29 matched": {
        "stage2_dir": CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT / "20-29" / "stage2",
        "data_dir": DATA_AGE_GROUPS_MATCHED_ROOT / "20-29",
        "color": "#4C78A8",
        "light": "#B9CCE8",
    },
    "80-89": {
        "stage2_dir": CHECKPOINTS_AGE_GROUPS_ROOT / "80-89" / "stage2",
        "data_dir": DATA_AGE_GROUPS_ROOT / "80-89",
        "color": "#F58518",
        "light": "#F3C7A2",
    },
}


def set_style():
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


def ensure_dirs(results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)


def load_stage2_artifacts(age_label: str, age_specs):
    spec = age_specs[age_label]
    stage2_dir = spec["stage2_dir"]
    data_dir = spec["data_dir"]
    with open(stage2_dir / "best_config.json", "r") as f:
        best_config = json.load(f)
    params_npz = np.load(stage2_dir / "best_model_params.npz")
    params = {k: params_npz[k] for k in params_npz.files}
    test_logits_npz = np.load(stage2_dir / "test_logits.npz")
    test_df = pd.read_csv(data_dir / "test_data.csv")
    if len(test_df) != len(test_logits_npz["logits"]):
        raise ValueError(f"Length mismatch for {age_label}: test csv vs logits")
    return {
        "age_group": age_label,
        "best_config": best_config,
        "params": params,
        "test_logits": test_logits_npz["logits"].astype(np.float32),
        "test_df": test_df.copy(),
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
    out["human_rt"] = out["response_time"] / 1000.0
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


def summarize_group(age_label: str, age_specs):
    artifact = load_stage2_artifacts(age_label, age_specs)
    df = enrich_df(artifact["test_df"])
    model = build_model(artifact["best_config"], artifact["params"])
    inf = run_inference(model, artifact["test_logits"])
    df["pred_rt"] = inf["pred_rt"]
    df["pred_choice"] = inf["pred_choice"]
    df["pred_correct"] = (df["pred_choice"] == df["target_dir_idx"]).astype(int)
    summary = {
        "age_group": age_label,
        "best_scale": float(artifact["best_config"]["scale"]),
        "best_score": float(artifact["best_config"]["score"]),
        "human_mean_rt": float(df["human_rt"].mean()),
        "model_mean_rt": float(df["pred_rt"].mean()),
        "human_median_rt": float(df["human_rt"].median()),
        "model_median_rt": float(np.median(df["pred_rt"])),
        "human_skew": float(pd.Series(df["human_rt"]).skew()),
        "model_skew": float(pd.Series(df["pred_rt"]).skew()),
        "human_accuracy": float(df["correct"].mean()),
        "model_accuracy": float(df["pred_correct"].mean()),
        "response_agreement": float((df["pred_choice"] == df["response_dir_idx"]).mean()),
        "human_congruency_rt_gap": float(df.loc[df["congruency"] == 1, "human_rt"].mean() - df.loc[df["congruency"] == 0, "human_rt"].mean()),
        "model_congruency_rt_gap": float(df.loc[df["congruency"] == 1, "pred_rt"].mean() - df.loc[df["congruency"] == 0, "pred_rt"].mean()),
        "human_error_minus_correct": float(df.loc[df["correct"] == 0, "human_rt"].mean() - df.loc[df["correct"] == 1, "human_rt"].mean()),
        "model_error_minus_correct": float(df.loc[df["pred_correct"] == 0, "pred_rt"].mean() - df.loc[df["pred_correct"] == 1, "pred_rt"].mean()) if (df["pred_correct"] == 0).any() else np.nan,
        "model_error_count": int((df["pred_correct"] == 0).sum()),
        "human_error_count": int((df["correct"] == 0).sum()),
    }
    return df, inf, summary


def build_matched_sets(group_dfs):
    key_cols = ["stimulus_layout", "target_direction", "flanker_direction", "congruency"]
    matched = {}
    young = group_dfs["20-29 matched"]
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
            pairs.append((y_grp.iloc[:n].copy(), o_grp.iloc[:n].copy()))
        if pairs:
            matched[("20-29 matched", label)] = pd.concat([p[0] for p in pairs]).sort_index()
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


def make_kde_figure(group_dfs, results_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    xs = np.linspace(0, 2.0, 400)
    line = {"Congruent": "--", "Incongruent": "-"}
    for row, age in enumerate(["20-29 matched", "80-89"]):
        human_df = group_dfs[age]
        model_df = group_dfs[age]
        for col, source in enumerate(["Human", "Model"]):
            ax = axes[row, col]
            value_col = "human_rt" if source == "Human" else "pred_rt"
            for cond, color in [("Congruent", "#4C78A8"), ("Incongruent", "#F58518")]:
                series = human_df.loc[human_df["condition"] == cond, value_col].to_numpy()
                if len(series) > 1:
                    ax.plot(xs, gaussian_kde(series)(xs), color=color, linestyle=line[cond], linewidth=2.2, label=cond)
            ax.set_title(f"{age} — {source}")
            ax.set_xlim(0, 2)
            ax.set_xlabel("RT (s)")
            ax.set_ylabel("Density")
        axes[row, 0].legend(frameon=False)
    fig.suptitle("Figure KDE. Human vs model RT distributions by age and congruency")
    fig.savefig(results_dir / "figure_kde_human_model_rt.png", bbox_inches="tight")
    plt.close(fig)


def make_a4_geometry_figure(group_dfs, inference_by_group, results_dir: Path):
    matched = build_matched_sets(group_dfs)
    required_keys = [
        ("20-29 matched", "Congruent"),
        ("20-29 matched", "Incongruent"),
        ("80-89", "Congruent"),
        ("80-89", "Incongruent"),
    ]
    if not all(k in matched for k in required_keys):
        return None

    state_blocks = []
    mean_trajs = {}
    spread_rows = []
    colors = {
        ("20-29 matched", "Congruent"): "#4C78A8",
        ("20-29 matched", "Incongruent"): "#72B7B2",
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
    axes[0].set_title("State-space mean trajectories")
    axes[0].legend(frameon=False)

    spread_df = pd.DataFrame(spread_rows)
    x = np.arange(len(spread_df))
    labels = [f"{row['age_group']}\n{row['condition']}" for _, row in spread_df.iterrows()]
    bar_colors = [colors[(str(row['age_group']), str(row['condition']))] for _, row in spread_df.iterrows()]
    axes[1].bar(x, spread_df["mean_state_space_spread"], color=bar_colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean trajectory spread")
    axes[1].set_title("State-space spread by age and congruency")

    fig.suptitle("Figure A4. Trajectory geometry under matched congruent and incongruent trials")
    fig.savefig(results_dir / "figureA4_matched_trajectory_geometry.png", bbox_inches="tight")
    plt.close(fig)
    spread_df.to_csv(results_dir / "figureA4_matched_trajectory_spread.csv", index=False)
    return spread_df


def write_summary(summary_df: pd.DataFrame, geometry_df, results_dir: Path):
    lines = ["# Full Matched Age-Group Analysis", ""]
    lines.append("## Parameter and behavioral summary")
    lines.append(str(summary_df.to_markdown(index=False)))
    lines.append("")
    if geometry_df is not None:
        lines.append("## Trajectory geometry summary")
        lines.append(str(geometry_df.to_markdown(index=False)))
        lines.append("")
    lines.extend([
        "## Generated figures",
        "- Figure KDE. Human vs model RT distributions by age and congruency",
        "- Figure A4. Trajectory geometry under matched congruent and incongruent trials",
    ])
    (results_dir / "analysis_summary.md").write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--young_label', default='20-29 matched')
    parser.add_argument('--old_label', default='80-89')
    parser.add_argument('--young_stage2_dir', default=rel_to_root(CHECKPOINTS_AGE_GROUPS_MATCHED_ROOT / '20-29' / 'stage2'))
    parser.add_argument('--young_data_dir', default=rel_to_root(DATA_AGE_GROUPS_MATCHED_ROOT / '20-29'))
    parser.add_argument('--old_stage2_dir', default=rel_to_root(CHECKPOINTS_AGE_GROUPS_ROOT / '80-89' / 'stage2'))
    parser.add_argument('--old_data_dir', default=rel_to_root(DATA_AGE_GROUPS_ROOT / '80-89'))
    parser.add_argument('--results_dir', default=rel_to_root(RESULTS_ROOT / 'age_groups_full_matched_compare'))
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    age_specs = {
        args.young_label: {
            'stage2_dir': Path(args.young_stage2_dir),
            'data_dir': Path(args.young_data_dir),
        },
        args.old_label: {
            'stage2_dir': Path(args.old_stage2_dir),
            'data_dir': Path(args.old_data_dir),
        },
    }
    set_style()
    ensure_dirs(results_dir)
    group_dfs = {}
    inference_by_group = {}
    summary_rows = []
    for age in [args.young_label, args.old_label]:
        df, inf, summary = summarize_group(age, age_specs)
        group_dfs[age] = df
        inference_by_group[age] = inf
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "parameter_and_signature_summary.csv", index=False)
    make_kde_figure(group_dfs, results_dir)
    geometry_df = make_a4_geometry_figure(group_dfs, inference_by_group, results_dir)
    write_summary(summary_df, geometry_df, results_dir)
    print(f"Saved matched full age-group analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
