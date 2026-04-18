import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_age_groups_efficient import (
    infer_predictions_from_params,
    train_stage2_with_scale,
    validate_cached_stage2_inputs,
)
from vgg_wongwang_lim import WWWrapper
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-29 learnable-WW time_steps sweep.")
    parser.add_argument("--age_group", default="20-29")
    parser.add_argument("--data_dir", default="data_age_groups/20-29")
    parser.add_argument("--train_logits_path", default="checkpoints_age_groups/20-29/stage2/train_logits.npz")
    parser.add_argument("--test_logits_path", default="checkpoints_age_groups/20-29/stage2/test_logits.npz")
    parser.add_argument(
        "--legacy_config_path",
        default="archive/response_label_refit_backup/80-89/best_config.target_supervision.json",
    )
    parser.add_argument(
        "--legacy_params_path",
        default="archive/response_label_refit_backup/80-89/best_model_params.target_supervision.npz",
    )
    parser.add_argument("--output_dir", default="artifacts/results/repro_legacy_interim/learnable_ww_20_29_sweep")
    parser.add_argument("--time_steps_factors", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_subset", type=int, default=12000)
    parser.add_argument("--test_subset", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--lambda_rt", type=float, default=1.0)
    parser.add_argument("--lambda_choice", type=float, default=3.0)
    parser.add_argument("--lambda_cong", type=float, default=0.3)
    parser.add_argument("--lambda_pileup", type=float, default=0.0)
    parser.add_argument("--anti_pileup_lambda", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_human_stats(data_dir: Path) -> dict:
    with (data_dir / "rt_stats.json").open() as f:
        return json.load(f)


def load_legacy_params(config_path: Path, params_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with config_path.open() as f:
        config = json.load(f)
    params_npz = np.load(params_path)
    params = {k: params_npz[k] for k in params_npz.files}
    return config, params


def subset_cached_inputs(cached: dict[str, np.ndarray], n_rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if n_rows >= len(cached["logits"]):
        return cached
    idx = rng.choice(len(cached["logits"]), size=n_rows, replace=False)
    return {k: v[idx] for k, v in cached.items()}


def run_inference_only_probe(
    legacy_params: dict[str, np.ndarray],
    scale: float,
    time_steps: int,
    logits: np.ndarray,
    device: str,
    choice_temperature: float,
) -> Dict[str, Any]:
    model = WWWrapper(n_classes=4, dt=10, time_steps=time_steps)
    state = model.state_dict()
    for key, value in legacy_params.items():
        if key in state:
            state[key] = torch.tensor(value, dtype=torch.float32)
    state["scale"] = torch.tensor(scale, dtype=torch.float32)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = infer_predictions_from_params(
            params={k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if k != "scale"},
            scale=scale,
            time_steps=time_steps,
            logits=logits,
            device=device,
            choice_temperature=choice_temperature,
        )
    pred_rt = pred["pred_rt"]
    ceiling = (time_steps - 1) * 0.01
    return {
        "mode": "inference_only",
        "pred_mean": float(pred_rt.mean()),
        "pred_median": float(np.median(pred_rt)),
        "pred_q95": float(np.quantile(pred_rt, 0.95)),
        "pred_q99": float(np.quantile(pred_rt, 0.99)),
        "n_at_ceiling": int(np.sum(np.isclose(pred_rt, ceiling, atol=1e-6))),
        "frac_at_ceiling": float(np.mean(np.isclose(pred_rt, ceiling, atol=1e-6))),
    }


def run_learnable_probe(
    human_stats: dict,
    train_cached: dict[str, np.ndarray],
    test_cached: dict[str, np.ndarray],
    scale: float,
    time_steps: int,
    epochs: int,
    choice_temperature: float,
    lambda_rt: float,
    lambda_choice: float,
    lambda_cong: float,
    lambda_pileup: float,
    device: str,
    mode_name: str,
) -> Dict[str, Any]:
    results, score, params, selection_details = train_stage2_with_scale(
        scale=scale,
        time_steps=time_steps,
        logits=train_cached["logits"],
        rts=train_cached["rts"],
        rts_normalized=train_cached["rts_normalized"],
        target_labels=train_cached["target_labels"],
        response_labels=train_cached["response_labels"],
        congruency=train_cached["congruency"],
        flanker_labels=train_cached.get("flanker_labels"),
        human_stats=human_stats,
        epochs=epochs,
        lambda_rt=lambda_rt,
        lambda_choice=lambda_choice,
        lambda_cong=lambda_cong,
        lambda_pileup=lambda_pileup,
        choice_temperature=choice_temperature,
        device=device,
        log_prefix=f"[{mode_name} factor {time_steps}] ",
    )
    pred = infer_predictions_from_params(
        params=params,
        scale=scale,
        time_steps=time_steps,
        logits=test_cached["logits"],
        device=device,
        choice_temperature=choice_temperature,
    )
    pred_rt = pred["pred_rt"]
    ceiling = (time_steps - 1) * 0.01
    return {
        "mode": mode_name,
        "score": float(score),
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
        "model_accuracy": float(results["model_accuracy"]),
        "response_agreement": float(results["response_agreement"]),
        "model_congruency_rt_gap": float(results["model_congruency_rt_gap"]),
        "rt_shape_score": float(results["rt_shape_score"]),
    }


def save_summary_plot(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for mode, color, label in [
        ("inference_only", "#4C78A8", "Inference-only"),
        ("learnable_ww", "#E45756", "Learnable WW"),
        ("learnable_ww_anti_pileup", "#54A24B", "Learnable WW + anti-pileup"),
    ]:
        subset = df[df["mode"] == mode].copy()
        if subset.empty:
            continue
        subset = subset.iloc[np.argsort(np.asarray(subset["time_steps_factor"], dtype=float))]
        axes[0].plot(subset["time_steps_factor"], subset["frac_at_ceiling"], marker="o", color=color, label=label)
        axes[1].plot(subset["time_steps_factor"], subset["pred_q99"], marker="o", color=color, label=label)
        axes[2].plot(subset["time_steps_factor"], subset["pred_mean"], marker="o", color=color, label=label)

    axes[0].set_title("Ceiling mass vs time_steps_factor")
    axes[0].set_xlabel("time_steps_factor")
    axes[0].set_ylabel("Fraction at ceiling")

    axes[1].set_title("Predicted q99 vs time_steps_factor")
    axes[1].set_xlabel("time_steps_factor")
    axes[1].set_ylabel("Predicted q99 (s)")

    axes[2].set_title("Predicted mean RT vs time_steps_factor")
    axes[2].set_xlabel("time_steps_factor")
    axes[2].set_ylabel("Predicted mean RT (s)")

    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    fig.suptitle("20-29 time_steps sweep with learnable Wong-Wang probe")
    fig.savefig(out_dir / "time_steps_sweep_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    factors = [float(x.strip()) for x in args.time_steps_factors.split(",") if x.strip()]
    data_dir = Path(args.data_dir)
    human_stats = load_human_stats(data_dir)
    base_time_steps = int(np.ceil(human_stats["percentile_99"] * 100))
    legacy_config, legacy_params = load_legacy_params(Path(args.legacy_config_path), Path(args.legacy_params_path))

    train_cached, test_cached = validate_cached_stage2_inputs(
        args.age_group,
        str(data_dir),
        args.train_logits_path,
        args.test_logits_path,
    )
    train_subset = subset_cached_inputs(train_cached, args.train_subset, rng)
    test_subset = subset_cached_inputs(test_cached, args.test_subset, rng)

    rows: List[Dict[str, Any]] = []
    for factor in factors:
        time_steps = int(np.ceil(base_time_steps * factor))
        common = {
            "age_group": args.age_group,
            "scale": args.scale,
            "base_time_steps": base_time_steps,
            "time_steps_factor": factor,
            "time_steps": time_steps,
            "ceiling_rt": (time_steps - 1) * 0.01,
            "train_subset_n": len(train_subset["logits"]),
            "test_subset_n": len(test_subset["logits"]),
        }
        inf_row = run_inference_only_probe(
            legacy_params=legacy_params,
            scale=args.scale,
            time_steps=time_steps,
            logits=test_subset["logits"],
            device=args.device,
            choice_temperature=args.choice_temperature,
        )
        rows.append({**common, **inf_row})

        learnable_row = run_learnable_probe(
            human_stats=human_stats,
            train_cached=train_subset,
            test_cached=test_subset,
            scale=args.scale,
            time_steps=time_steps,
            epochs=args.epochs,
            choice_temperature=args.choice_temperature,
            lambda_rt=args.lambda_rt,
            lambda_choice=args.lambda_choice,
            lambda_cong=args.lambda_cong,
            lambda_pileup=args.lambda_pileup,
            device=args.device,
            mode_name="learnable_ww",
        )
        rows.append({**common, **learnable_row})

        anti_pileup_row = run_learnable_probe(
            human_stats=human_stats,
            train_cached=train_subset,
            test_cached=test_subset,
            scale=args.scale,
            time_steps=time_steps,
            epochs=args.epochs,
            choice_temperature=args.choice_temperature,
            lambda_rt=args.lambda_rt,
            lambda_choice=args.lambda_choice,
            lambda_cong=args.lambda_cong,
            lambda_pileup=args.anti_pileup_lambda,
            device=args.device,
            mode_name="learnable_ww_anti_pileup",
        )
        rows.append({**common, **anti_pileup_row})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "time_steps_sweep_results.csv", index=False)

    summary = {
        "age_group": args.age_group,
        "scale": args.scale,
        "base_time_steps": base_time_steps,
        "legacy_reference_scale": float(legacy_config["scale"]),
        "epochs": args.epochs,
        "lambda_pileup": args.lambda_pileup,
        "anti_pileup_lambda": args.anti_pileup_lambda,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "device": args.device,
        "time_steps_factors": factors,
        "results_csv": str(out_dir / "time_steps_sweep_results.csv"),
    }
    (out_dir / "time_steps_sweep_manifest.json").write_text(json.dumps(summary, indent=2))
    save_summary_plot(df, out_dir)


if __name__ == "__main__":
    main()
