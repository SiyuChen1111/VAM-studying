import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from project_paths import RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir
from train_age_groups_efficient import (
    attach_flanker_labels_from_csv,
    evaluate_cached_stage2_params,
    set_random_seed,
    train_stage2_with_scale,
    validate_cached_stage2_inputs,
)


AGE_GROUP = "20-29"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "repro_legacy_interim" / "dynamic_selection_20_29_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-29 dynamic-selection smoke comparison against urgency baseline, with optional DMC-like extension.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--time_steps_factor", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_subset", type=int, default=3000)
    parser.add_argument("--test_subset", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    parser.add_argument("--lambda_rt", type=float, default=1.0)
    parser.add_argument("--lambda_choice", type=float, default=3.0)
    parser.add_argument("--lambda_cong", type=float, default=0.3)
    parser.add_argument("--lambda_pileup", type=float, default=1.0)
    parser.add_argument("--urgency_type", default="additive_urgency")
    parser.add_argument("--urgency_start", type=float, default=0.60)
    parser.add_argument("--urgency_slope", type=float, default=0.25)
    parser.add_argument("--urgency_floor", type=float, default=0.0)
    parser.add_argument("--selection_strength", type=float, default=0.35)
    parser.add_argument("--selection_midpoint_s", type=float, default=0.18)
    parser.add_argument("--selection_tau_s", type=float, default=0.06)
    parser.add_argument("--target_boost", type=float, default=0.10)
    parser.add_argument("--include_dmc_extension", action="store_true")
    parser.add_argument("--auto_strength", type=float, default=0.18)
    parser.add_argument("--auto_peak_s", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def subset_cached_inputs(cached: Dict[str, np.ndarray], n_rows: int, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    if n_rows >= len(cached["logits"]):
        idx = np.arange(len(cached["logits"]))
        return cached, idx
    idx = np.sort(rng.choice(len(cached["logits"]), size=n_rows, replace=False))
    return {key: value[idx] for key, value in cached.items()}, idx


def load_human_stats(data_dir: Path) -> dict:
    with (data_dir / "rt_stats.json").open() as handle:
        return json.load(handle)


def prepare_cached_inputs(seed: int, train_subset_n: int, test_subset_n: int) -> Tuple[dict, dict, np.ndarray, np.ndarray, dict, Path]:
    data_dir = age_group_data_dir(AGE_GROUP)
    stage2_dir = age_group_stage2_dir(AGE_GROUP)
    train_cached, test_cached = validate_cached_stage2_inputs(
        AGE_GROUP,
        str(data_dir),
        str(stage2_dir / "train_logits.npz"),
        str(stage2_dir / "test_logits.npz"),
    )
    train_cached = attach_flanker_labels_from_csv(train_cached, str(data_dir / "train_data.csv"))
    test_cached = attach_flanker_labels_from_csv(test_cached, str(data_dir / "test_data.csv"))
    rng = np.random.default_rng(seed)
    train_subset, train_idx = subset_cached_inputs(train_cached, train_subset_n, rng)
    test_subset, test_idx = subset_cached_inputs(test_cached, test_subset_n, rng)
    return train_subset, test_subset, train_idx, test_idx, load_human_stats(data_dir), data_dir


def save_run_artifacts(
    out_dir: Path,
    tag: str,
    row: Dict[str, Any],
    predictions: Dict[str, np.ndarray],
    test_cached: Dict[str, np.ndarray],
) -> None:
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(row, indent=2))
    np.savez_compressed(
        run_dir / "predictions.npz",
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


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cached, test_cached, train_idx, test_idx, human_stats, _ = prepare_cached_inputs(
        seed=args.seed,
        train_subset_n=args.train_subset,
        test_subset_n=args.test_subset,
    )
    base_time_steps = int(np.ceil(human_stats["percentile_99"] * 100))
    time_steps = int(np.ceil(base_time_steps * args.time_steps_factor))

    urgency_config = {
        "dt_ms": 10.0,
        "choice_temperature": float(args.choice_temperature),
        "urgency_type": args.urgency_type,
        "urgency_start": float(args.urgency_start),
        "urgency_slope": float(args.urgency_slope),
        "urgency_floor": float(args.urgency_floor),
    }
    dynamic_selection_config = {
        "selection_mode": "dynamic_flanker_suppression",
        "selection_strength": float(args.selection_strength),
        "selection_midpoint_s": float(args.selection_midpoint_s),
        "selection_tau_s": float(args.selection_tau_s),
        "target_boost": float(args.target_boost),
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }
    dmc_extension_config = {
        "selection_mode": "dynamic_flanker_dmc_like",
        "selection_strength": float(args.selection_strength),
        "selection_midpoint_s": float(args.selection_midpoint_s),
        "selection_tau_s": float(args.selection_tau_s),
        "target_boost": float(args.target_boost),
        "auto_strength": float(args.auto_strength),
        "auto_peak_s": float(args.auto_peak_s),
        "selection_apply_to": "incongruent_only",
        "dt_ms": 10.0,
    }

    run_specs = [
        ("urgency_baseline", None),
        ("dynamic_selection_phase1", dynamic_selection_config),
    ]
    if args.include_dmc_extension:
        run_specs.append(("dynamic_selection_dmc_extension", dmc_extension_config))

    rows = []
    for index, (tag, selection_config) in enumerate(run_specs):
        train_seed = int(args.seed + index * 100)
        eval_seed = int(args.seed + index * 100 + 1)
        results, score, params, selection_details = train_stage2_with_scale(
            scale=args.scale,
            time_steps=time_steps,
            logits=train_cached["logits"],
            rts=train_cached["rts"],
            rts_normalized=train_cached["rts_normalized"],
            target_labels=train_cached["target_labels"],
            response_labels=train_cached["response_labels"],
            congruency=train_cached["congruency"],
            flanker_labels=train_cached["flanker_labels"],
            human_stats=human_stats,
            epochs=args.epochs,
            lambda_rt=args.lambda_rt,
            lambda_choice=args.lambda_choice,
            lambda_cong=args.lambda_cong,
            lambda_pileup=args.lambda_pileup,
            choice_temperature=args.choice_temperature,
            rt_readout_mode="urgency",
            readout_config=urgency_config,
            selection_config=selection_config,
            random_seed=train_seed,
            eval_random_seed=eval_seed,
            device=args.device,
            log_prefix=f"[{tag}] ",
        )
        predictions, canonical_results = evaluate_cached_stage2_params(
            params=params,
            scale=args.scale,
            time_steps=time_steps,
            cached=test_cached,
            device=args.device,
            choice_temperature=args.choice_temperature,
            rt_readout_mode="urgency",
            readout_config=urgency_config,
            selection_config=selection_config,
            random_seed=eval_seed,
            rt_shape_focus=False,
        )
        pred_rt = predictions["pred_rt"]
        row = {
            "tag": tag,
            "age_group": AGE_GROUP,
            "scale": args.scale,
            "time_steps_factor": args.time_steps_factor,
            "base_time_steps": base_time_steps,
            "time_steps": time_steps,
            "ceiling_rt": (time_steps - 1) * 0.01,
            "train_subset_n": len(train_cached["logits"]),
            "test_subset_n": len(test_cached["logits"]),
            "score": float(canonical_results["total_score"]),
            "best_epoch": float(selection_details.get("best_epoch", np.nan)) if selection_details else np.nan,
            "pred_mean": float(pred_rt.mean()),
            "pred_median": float(np.median(pred_rt)),
            "pred_q95": float(np.quantile(pred_rt, 0.95)),
            "pred_q99": float(np.quantile(pred_rt, 0.99)),
            "n_at_ceiling": int(np.sum(np.isclose(pred_rt, (time_steps - 1) * 0.01, atol=1e-6))),
            "frac_at_ceiling": float(np.mean(np.isclose(pred_rt, (time_steps - 1) * 0.01, atol=1e-6))),
            "model_accuracy": float(canonical_results["model_accuracy"]),
            "response_agreement": float(canonical_results["response_agreement"]),
            "model_congruency_rt_gap": float(canonical_results["model_congruency_rt_gap"]),
            "rt_shape_score": float(canonical_results["rt_shape_score"]),
            "pred_error_rt": float(canonical_results["pred_error_rt"]),
            "pred_correct_rt": float(canonical_results["pred_correct_rt"]),
            "error_minus_correct_rt": float(canonical_results["error_minus_correct_rt"]),
            "pred_skewness": float(canonical_results["pred_skewness"]),
            "quantile_score": float(canonical_results["quantile_score"]),
            "coverage_score": float(canonical_results["coverage_score"]),
            "urgency_config": urgency_config,
            "selection_config": selection_config or {"selection_mode": "baseline"},
            "train_seed": train_seed,
            "eval_seed": eval_seed,
        }
        rows.append(row)
        save_run_artifacts(out_dir, tag, row, predictions, test_cached)

    pd.DataFrame(rows).to_csv(out_dir / "smoke_comparison_results.csv", index=False)
    manifest = {
        "age_group": AGE_GROUP,
        "scale": args.scale,
        "time_steps_factor": args.time_steps_factor,
        "lambda_pileup": args.lambda_pileup,
        "choice_temperature": args.choice_temperature,
        "urgency_config": urgency_config,
        "dynamic_selection_config": dynamic_selection_config,
        "dmc_extension_config": dmc_extension_config if args.include_dmc_extension else None,
        "train_subset_indices_preview": train_idx[:10].tolist(),
        "test_subset_indices_preview": test_idx[:10].tolist(),
        "runs": [tag for tag, _ in run_specs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
