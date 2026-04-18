import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir

from train_age_groups_efficient import (
    compute_human_stats_from_rts,
    evaluate_cached_stage2_params,
    infer_predictions_from_params,
    set_random_seed,
    train_stage2_with_scale,
    validate_cached_stage2_inputs,
)


DEFAULT_AGE_GROUP = "20-29"
DEFAULT_STAGE2_DIR = age_group_stage2_dir(DEFAULT_AGE_GROUP)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "repro_legacy_interim" / "readout_mode_20_29_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-horizon 20-29 readout-mode sweep with anti-pileup enabled.")
    parser.add_argument("--age_group", default=DEFAULT_AGE_GROUP)
    parser.add_argument("--data_dir", default=str(age_group_data_dir(DEFAULT_AGE_GROUP)))
    parser.add_argument("--train_logits_path", default=str(DEFAULT_STAGE2_DIR / "train_logits.npz"))
    parser.add_argument("--test_logits_path", default=str(DEFAULT_STAGE2_DIR / "test_logits.npz"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rt_readout_modes", default="baseline,soft_hazard,urgency")
    parser.add_argument("--time_steps_factor", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_subset", type=int, default=12000)
    parser.add_argument("--test_subset", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--lambda_rt", type=float, default=1.0)
    parser.add_argument("--lambda_choice", type=float, default=3.0)
    parser.add_argument("--lambda_cong", type=float, default=0.3)
    parser.add_argument("--lambda_pileup", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=12.0)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--urgency_type", choices=["collapsing_bound", "additive_urgency"], default="additive_urgency")
    parser.add_argument("--urgency_start", type=float, default=0.80)
    parser.add_argument("--urgency_slope", type=float, default=0.25)
    parser.add_argument("--urgency_floor", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def subset_cached_inputs(cached: Dict[str, np.ndarray], n_rows: int, rng: np.random.Generator) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    if n_rows >= len(cached["logits"]):
        idx = np.arange(len(cached["logits"]))
        return cached, idx
    idx = np.sort(rng.choice(len(cached["logits"]), size=n_rows, replace=False))
    return {k: v[idx] for k, v in cached.items()}, idx


def load_human_stats(data_dir: Path) -> dict:
    with (data_dir / "rt_stats.json").open() as f:
        return json.load(f)


def build_readout_config(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "dt_ms": 10.0,
        "choice_temperature": float(args.choice_temperature),
    }
    if mode == "soft_hazard":
        config.update({
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "eps": float(args.eps),
        })
    elif mode == "urgency":
        config.update({
            "urgency_type": args.urgency_type,
            "urgency_start": float(args.urgency_start),
            "urgency_slope": float(args.urgency_slope),
            "urgency_floor": float(args.urgency_floor),
        })
    return config


def run_mode_probe(
    mode: str,
    args: argparse.Namespace,
    human_stats: dict,
    train_cached: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
    time_steps: int,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    readout_config = build_readout_config(args, mode)
    train_seed = int(args.seed)
    eval_seed = int(args.seed + 1)
    results, score, params, selection_details = train_stage2_with_scale(
        scale=args.scale,
        time_steps=time_steps,
        logits=train_cached["logits"],
        rts=train_cached["rts"],
        rts_normalized=train_cached["rts_normalized"],
        target_labels=train_cached["target_labels"],
        response_labels=train_cached["response_labels"],
        congruency=train_cached["congruency"],
        flanker_labels=train_cached.get("flanker_labels"),
        human_stats=human_stats,
        epochs=args.epochs,
        lambda_rt=args.lambda_rt,
        lambda_choice=args.lambda_choice,
        lambda_cong=args.lambda_cong,
        lambda_pileup=args.lambda_pileup,
        choice_temperature=args.choice_temperature,
        rt_readout_mode=mode,
        readout_config=readout_config,
        random_seed=train_seed,
        eval_random_seed=eval_seed,
        device=args.device,
        log_prefix=f"[{mode}] ",
    )
    predictions, canonical_results = evaluate_cached_stage2_params(
        params=params,
        scale=args.scale,
        time_steps=time_steps,
        cached=test_cached,
        device=args.device,
        choice_temperature=args.choice_temperature,
        rt_readout_mode=mode,
        readout_config=readout_config,
        random_seed=eval_seed,
        rt_shape_focus=False,
    )
    pred_rt = predictions["pred_rt"]
    ceiling = (time_steps - 1) * 0.01
    row = {
        "mode": mode,
        "score": float(canonical_results["total_score"]),
        "best_epoch": float(selection_details.get("best_epoch", np.nan)) if selection_details else np.nan,
        "pred_mean": float(pred_rt.mean()),
        "pred_median": float(np.median(pred_rt)),
        "pred_q95": float(np.quantile(pred_rt, 0.95)),
        "pred_q99": float(np.quantile(pred_rt, 0.99)),
        "n_at_ceiling": int(np.sum(np.isclose(pred_rt, ceiling, atol=1e-6))),
        "frac_at_ceiling": float(np.mean(np.isclose(pred_rt, ceiling, atol=1e-6))),
        "learned_threshold": float(np.asarray(params["ww.threshold"]).item()),
        "learned_noise_ampa": float(np.asarray(params["ww.noise_ampa"]).item()),
        "learned_J_ext": float(np.asarray(params["ww.J_ext"]).item()),
        "learned_I_0": float(np.asarray(params["ww.I_0"]).item()),
        "model_accuracy": float(canonical_results["model_accuracy"]),
        "response_agreement": float(canonical_results["response_agreement"]),
        "model_congruency_rt_gap": float(canonical_results["model_congruency_rt_gap"]),
        "rt_shape_score": float(canonical_results["rt_shape_score"]),
        "pred_error_rt": float(canonical_results["pred_error_rt"]),
        "pred_correct_rt": float(canonical_results["pred_correct_rt"]),
        "error_minus_correct_rt": float(canonical_results["error_minus_correct_rt"]),
        "human_error_rt": float(canonical_results["human_error_rt"]),
        "human_correct_rt": float(canonical_results["human_correct_rt"]),
        "human_error_minus_correct_rt": float(canonical_results["human_error_minus_correct_rt"]),
        "pred_skewness": float(canonical_results["pred_skewness"]),
        "true_skewness": float(canonical_results["true_skewness"]),
        "quantile_score": float(canonical_results["quantile_score"]),
        "coverage_score": float(canonical_results["coverage_score"]),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
    }
    return row, readout_config, predictions


def save_mode_artifacts(
    out_dir: Path,
    mode: str,
    row: Dict[str, Any],
    readout_config: Dict[str, Any],
    predictions: Dict[str, np.ndarray],
) -> None:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "mode_summary.json").write_text(json.dumps({**row, "readout_config": readout_config}, indent=2))
    np.savez_compressed(mode_dir / "predictions.npz", **predictions)


def save_summary_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    colors = {
        "baseline": "#4C78A8",
        "soft_hazard": "#E45756",
        "urgency": "#54A24B",
    }
    for mode in ["baseline", "soft_hazard", "urgency"]:
        subset = df[df["mode"] == mode]
        if subset.empty:
            continue
        color = colors[mode]
        axes[0].bar(mode, float(np.asarray(subset["frac_at_ceiling"], dtype=float)[0]), color=color)
        axes[1].bar(mode, float(np.asarray(subset["pred_skewness"], dtype=float)[0]), color=color)
        axes[2].bar(mode, float(np.asarray(subset["error_minus_correct_rt"], dtype=float)[0]), color=color)
        axes[3].bar(mode, float(np.asarray(subset["model_congruency_rt_gap"], dtype=float)[0]), color=color)

    axes[0].set_title("Ceiling mass")
    axes[0].set_ylabel("Fraction at ceiling")
    axes[1].set_title("Predicted RT skewness")
    axes[1].set_ylabel("Skewness")
    axes[2].set_title("Error - Correct RT")
    axes[2].set_ylabel("Seconds")
    axes[3].set_title("Congruency RT gap")
    axes[3].set_ylabel("Seconds")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("20-29 fixed-horizon readout sweep (time_steps_factor=2.0, anti-pileup on)")
    fig.savefig(out_dir / "readout_sweep_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    modes = [m.strip() for m in args.rt_readout_modes.split(",") if m.strip()]
    if any(mode not in {"baseline", "soft_hazard", "urgency"} for mode in modes):
        raise ValueError("Only baseline, soft_hazard, and urgency are supported in this sweep.")

    data_dir = Path(args.data_dir)
    human_stats = load_human_stats(data_dir)
    base_time_steps = int(np.ceil(human_stats["percentile_99"] * 100))
    time_steps = int(np.ceil(base_time_steps * args.time_steps_factor))

    train_cached, test_cached = validate_cached_stage2_inputs(
        args.age_group,
        str(data_dir),
        args.train_logits_path,
        args.test_logits_path,
    )
    train_subset, train_idx = subset_cached_inputs(train_cached, args.train_subset, rng)
    test_subset, test_idx = subset_cached_inputs(test_cached, args.test_subset, rng)

    rows: List[Dict[str, Any]] = []
    for mode in modes:
        row, readout_config, predictions = run_mode_probe(
            mode=mode,
            args=args,
            human_stats=human_stats,
            train_cached=train_subset,
            test_cached=test_subset,
            time_steps=time_steps,
        )
        common = {
            "age_group": args.age_group,
            "scale": args.scale,
            "time_steps_factor": args.time_steps_factor,
            "base_time_steps": base_time_steps,
            "time_steps": time_steps,
            "ceiling_rt": (time_steps - 1) * 0.01,
            "train_subset_n": len(train_subset["logits"]),
            "test_subset_n": len(test_subset["logits"]),
        }
        full_row = {**common, **row}
        rows.append(full_row)
        save_mode_artifacts(out_dir, mode, full_row, readout_config, predictions)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "readout_sweep_results.csv", index=False)
    manifest = {
        "age_group": args.age_group,
        "scale": args.scale,
        "time_steps_factor": args.time_steps_factor,
        "base_time_steps": base_time_steps,
        "time_steps": time_steps,
        "epochs": args.epochs,
        "lambda_pileup": args.lambda_pileup,
        "rt_readout_modes": modes,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "train_indices_path": str(out_dir / "train_subset_indices.npy"),
        "test_indices_path": str(out_dir / "test_subset_indices.npy"),
        "device": args.device,
        "results_csv": str(out_dir / "readout_sweep_results.csv"),
    }
    np.save(out_dir / "train_subset_indices.npy", train_idx)
    np.save(out_dir / "test_subset_indices.npy", test_idx)
    (out_dir / "readout_sweep_manifest.json").write_text(json.dumps(manifest, indent=2))
    save_summary_plot(df, out_dir)


if __name__ == "__main__":
    main()
