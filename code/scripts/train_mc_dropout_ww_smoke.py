"""
MC Dropout → Wong-Wang Synthesis Smoke Test

Implements Improvement 1: Monte Carlo Dropout on Stage 1 VGG to introduce
epistemic uncertainty into the logit space.

Variants:
  M1: MC mean logits → WW (baseline MC)
  M2: Variance-augmented logits (mean + β·√var·ε) → WW
  M3: Sample-level WW aggregation (each MC sample → WW → aggregate)

Hypothesis:
  M1 showed MC mean ≈ eval logits (corr 0.999+) — dropout noise cancels in mean.
  M2 uses variance to add trial-level noise proportional to model uncertainty.
  M3 propagates full dropout stochasticity through WW and aggregates decisions.

Run from project root:
  source .venv/bin/activate && cd /Users/siyu/Documents/GitHub/VAM-studying
  python code/scripts/train_mc_dropout_ww_smoke.py \
    --age_group 20-29 \
    --data_root data/age_groups_matched \
    --output_root artifacts/results/rt_model_mc_dropout_ww/smoke \
    --variant all \
    --mc_samples 30 --epochs_ww 15 \
    --smoke_eval --smoke_max_trials 256 \
    --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cache_vgg_stage2_features import load_stage1_model
from project_paths import PROJECT_ROOT
from train_age_groups_efficient import (
    DIRECTION_MAP,
    StimulusDataset,
    compute_human_stats_from_rts,
    evaluate_joint_behavior,
    fit_stage2_from_logits,
    to_jsonable,
)
from vgg_wongwang_lim import VGGFeatureExtractor


# ── utilities ────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_behavior_balanced_subset(
    dataset: StimulusDataset,
    max_trials: int,
    seed: int,
) -> List[int]:
    rng = np.random.RandomState(seed)
    n_total = len(dataset)
    if n_total <= max_trials:
        return list(range(n_total))
    target_arr = dataset.target_labels
    response_arr = dataset.response_labels
    congruency_arr = dataset.congruency
    correct_mask = target_arr == response_arr
    error_mask = ~correct_mask
    cong_mask = congruency_arr == 0
    incong_mask = congruency_arr == 1
    indices: List[int] = []
    n_per = max(1, max_trials // 8)
    for correct in (True, False):
        mask = correct_mask if correct else error_mask
        for cong in (True, False):
            cmask = cong_mask if cong else incong_mask
            stratum = np.where(mask & cmask)[0]
            if len(stratum) > 0:
                chosen = rng.choice(stratum, size=min(n_per, len(stratum)), replace=False)
                indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices[:max_trials]


def _attach_flanker(bundle: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "flanker_labels" not in bundle or (bundle["flanker_labels"] == -1).all():
        bundle["flanker_labels"] = bundle["congruency"].copy()
    return bundle


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


# ── logit bundling ───────────────────────────────────────────────────
def mc_bundle_logits(
    model: VGGFeatureExtractor,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    n_samples: int,
    device: str,
    seed: int,
    return_samples: bool = False,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()
    all_logits, all_var = [], []
    all_target, all_response, all_rt, all_cong, all_flanker = [], [], [], [], []
    all_samples: List[np.ndarray] = [] if return_samples else []

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        mean_logits, var_logits, samples = model.mc_forward(
            images, n_samples=n_samples, seed=seed + batch_idx,
        )
        all_logits.append(mean_logits.cpu().numpy())
        all_var.append(var_logits.cpu().numpy())
        all_target.append(batch["target_label"].cpu().numpy())
        all_response.append(batch["response_label"].cpu().numpy())
        all_rt.append(batch["rt"].cpu().numpy())
        all_cong.append(batch["congruency"].cpu().numpy())
        if return_samples:
            all_samples.append(samples.cpu().numpy())  # [S, B, C]
        start = batch_idx * batch_size
        end = min(start + batch_size, len(dataset))
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'flanker_labels'):
            fl = np.asarray(dataset.dataset.flanker_labels[start:end])
        else:
            fl = np.full(end - start, -1, dtype=np.int64)
        all_flanker.append(fl)
        if (batch_idx + 1) % 10 == 0:
            print(f"  MC bundle batch {batch_idx + 1}/{len(loader)}")

    bundle = {
        "logits": np.concatenate(all_logits, axis=0).astype(np.float32),
        "logits_var": np.concatenate(all_var, axis=0).astype(np.float32),
        "target_labels": np.concatenate(all_target).astype(np.int64),
        "response_labels": np.concatenate(all_response).astype(np.int64),
        "rts": np.concatenate(all_rt).astype(np.float32),
        "rts_normalized": np.concatenate(all_rt).astype(np.float32),
        "congruency": np.concatenate(all_cong).astype(np.int64),
        "flanker_labels": np.concatenate(all_flanker).astype(np.int64),
    }
    if return_samples:
        bundle["mc_samples"] = np.concatenate(all_samples, axis=1).astype(np.float32)  # [S, N, C]
    return bundle


def eval_bundle_logits(
    model: VGGFeatureExtractor,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()
    all_logits, all_target, all_response, all_rt, all_cong, all_flanker = [], [], [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_target.append(batch["target_label"].cpu().numpy())
            all_response.append(batch["response_label"].cpu().numpy())
            all_rt.append(batch["rt"].cpu().numpy())
            all_cong.append(batch["congruency"].cpu().numpy())
            start = batch_idx * batch_size
            end = min(start + batch_size, len(dataset))
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'flanker_labels'):
                fl = np.asarray(dataset.dataset.flanker_labels[start:end])
            else:
                fl = np.full(end - start, -1, dtype=np.int64)
            all_flanker.append(fl)
    return {
        "logits": np.concatenate(all_logits, axis=0).astype(np.float32),
        "target_labels": np.concatenate(all_target).astype(np.int64),
        "response_labels": np.concatenate(all_response).astype(np.int64),
        "rts": np.concatenate(all_rt).astype(np.float32),
        "rts_normalized": np.concatenate(all_rt).astype(np.float32),
        "congruency": np.concatenate(all_cong).astype(np.int64),
        "flanker_labels": np.concatenate(all_flanker).astype(np.int64),
    }


# ── M2: variance-augmented logits ────────────────────────────────────
def augment_with_variance(
    bundle: Dict[str, np.ndarray],
    beta: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Add per-trial noise scaled by MC logit variance."""
    rng = np.random.RandomState(seed)
    logits = bundle["logits"].copy()
    logits_var = bundle["logits_var"]
    # normalize variance per dimension
    var_dim_mean = logits_var.mean(axis=0, keepdims=True) + 1e-8
    var_norm = logits_var / var_dim_mean
    noise_scale = beta * np.sqrt(np.maximum(var_norm, 0.0))
    noise = rng.randn(*logits.shape).astype(np.float32) * noise_scale
    augmented = dict(bundle)
    augmented["logits"] = (logits + noise).astype(np.float32)
    augmented["logits_var"] = bundle["logits_var"]  # keep original var
    return augmented


# ── M3: sample-level expansion ───────────────────────────────────────
def expand_mc_samples(
    bundle: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Expand MC samples into N× batch for sample-level WW training."""
    samples = bundle["mc_samples"]  # [S, B, C]
    S, B, C = samples.shape
    expanded = {
        "logits": samples.transpose(1, 0, 2).reshape(S * B, C).astype(np.float32),
        "target_labels": np.tile(bundle["target_labels"], S).astype(np.int64),
        "response_labels": np.tile(bundle["response_labels"], S).astype(np.int64),
        "rts": np.tile(bundle["rts"], S).astype(np.float32),
        "rts_normalized": np.tile(bundle["rts_normalized"], S).astype(np.float32),
        "congruency": np.tile(bundle["congruency"], S).astype(np.int64),
        "flanker_labels": np.tile(bundle["flanker_labels"], S).astype(np.int64),
        "logits_var": np.tile(bundle["logits_var"], (S, 1)).astype(np.float32),
        "_mc_n_samples": S,
        "_mc_original_n": B,
    }
    return expanded


# ── WW training helper ───────────────────────────────────────────────
_WW_CONFIG = {
    "rt_readout_mode": "soft_index",
    "readout_config": {
        "dt_ms": 10.0, "sigma_s": 0.05, "choice_temperature": 0.10,
        "t0_mode": "fixed_global", "t0_seconds": 0.15,
    },
    "selection_config": {"transform_mode": "softplus_centered", "softplus_center": 0.1},
    "t0_mode": "fixed_global",
    "t0_seconds": 0.15,
    "behavioral_loss_config": {
        "lambda_error_rate": 1.0, "lambda_error_sign": 1.0,
        "lambda_accuracy": 0.5, "lambda_response_nll": 1.0, "lambda_rt_mse": 1.0,
    },
}


def train_ww_on_bundle(
    train_bundle: Dict[str, np.ndarray],
    test_bundle: Dict[str, np.ndarray],
    epochs_ww: int,
    device: str,
    seed: int,
    output_dir: str,
    label: str,
    scales: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], float]:
    if scales is None:
        scales = np.linspace(0.10, 0.35, 4)
    human_stats = compute_human_stats_from_rts(train_bundle["rts"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{label}] Training WW ({epochs_ww} epochs, {len(train_bundle['logits'])} train trials)...")
    t0 = time.perf_counter()
    result = fit_stage2_from_logits(
        age_group="20-29", output_dir=str(out), human_stats=human_stats,
        train_cached=train_bundle, test_cached=test_bundle,
        device=device, scales=scales, epochs=epochs_ww,
        lambda_rt=1.0, lambda_choice=2.0, lambda_cong=0.3,
        choice_temperature=0.10,
        **_WW_CONFIG,
        random_seed=seed, eval_random_seed=seed + 1,
    )
    elapsed = time.perf_counter() - t0
    beh_opt = float(result["results"]["behavior_optimal_score"])
    print(f"  [{label}] Done in {elapsed:.1f}s | beh_opt={beh_opt:.4f} | "
          f"acc={result['results']['model_accuracy']:.4f} | "
          f"pred_mean={result['results']['pred_mean']:.3f}s | "
          f"ΔRT={result['results']['error_minus_correct_rt']:.4f}")
    return result, elapsed


# ── main experiment ──────────────────────────────────────────────────
def run_all_variants(
    *,
    model: VGGFeatureExtractor,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    variants: List[str],
    epochs_ww: int,
    mc_samples: int,
    var_beta: float,
    device: str,
    seed: int,
    output_dir: str,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Bundle MC logits (shared across variants) ──
    print(f"Generating MC Dropout logits (n_samples={mc_samples})...")
    t0 = time.perf_counter()
    need_samples = "m3" in variants
    train_mc = mc_bundle_logits(model, train_dataset, batch_size=64, n_samples=mc_samples,
                                device=device, seed=seed, return_samples=need_samples)
    test_mc = mc_bundle_logits(model, test_dataset, batch_size=64, n_samples=mc_samples,
                               device=device, seed=seed + 1, return_samples=need_samples)
    train_mc = _attach_flanker(train_mc)
    test_mc = _attach_flanker(test_mc)
    mc_time = time.perf_counter() - t0
    print(f"  MC logits generated in {mc_time:.1f}s")

    # ── Step 2: Eval-mode logits (baseline) ──
    print("Generating eval-mode logits (baseline)...")
    t0 = time.perf_counter()
    train_eval = eval_bundle_logits(model, train_dataset, batch_size=128, device=device)
    test_eval = eval_bundle_logits(model, test_dataset, batch_size=128, device=device)
    train_eval = _attach_flanker(train_eval)
    test_eval = _attach_flanker(test_eval)
    eval_time = time.perf_counter() - t0
    print(f"  Eval logits generated in {eval_time:.1f}s")

    # ── Step 3: Diagnostics ──
    print("\n--- Logit Distribution Diagnostics ---")
    for mode, tlog, telog in [("MC mean", train_mc["logits"], test_mc["logits"]),
                                 ("Eval", train_eval["logits"], test_eval["logits"])]:
        print(f"  {mode}: train mean={tlog.mean(axis=0).round(3)}, test mean={telog.mean(axis=0).round(3)}")
    print(f"  MC logit var (mean across dims): train={train_mc['logits_var'].mean():.6f}, "
          f"test={test_mc['logits_var'].mean():.6f}")

    from scipy.stats import pearsonr
    for split_name, mc_log, eval_log in [("train", train_mc["logits"], train_eval["logits"]),
                                           ("test", test_mc["logits"], test_eval["logits"])]:
        r_vals = [pearsonr(mc_log[:min(2000, len(mc_log)), c],
                           eval_log[:min(2000, len(eval_log)), c])[0]
                  for c in range(4)]
        print(f"  {split_name} MC-vs-Eval correlation: {[f'{r:.3f}' for r in r_vals]}")

    # ── Step 4: Save bundles ──
    np.savez_compressed(out / "train_mc_logits.npz", **train_mc)
    np.savez_compressed(out / "test_mc_logits.npz", **test_mc)
    np.savez_compressed(out / "train_eval_logits.npz", **train_eval)
    np.savez_compressed(out / "test_eval_logits.npz", **test_eval)

    # ── Step 5: Run each variant ──
    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_times: Dict[str, float] = {}

    scales = np.linspace(0.10, 0.35, 4)

    # --- Baseline (eval) ---
    if "eval" in variants:
        result, elapsed = train_ww_on_bundle(
            train_eval, test_eval, epochs_ww, device, seed,
            str(out / "eval_ww"), "EVAL", scales=scales,
        )
        all_metrics["eval"] = {"name": "Baseline (Eval)", "results": result["results"]}
        all_times["eval"] = elapsed

    # --- M1: MC mean ---
    if "m1" in variants:
        result, elapsed = train_ww_on_bundle(
            train_mc, test_mc, epochs_ww, device, seed,
            str(out / "m1_mc_mean"), "M1", scales=scales,
        )
        all_metrics["m1"] = {"name": "M1 (MC mean)", "results": result["results"]}
        all_times["m1"] = elapsed

    # --- M2: Variance-augmented ---
    if "m2" in variants:
        train_m2 = augment_with_variance(train_mc, beta=var_beta, seed=seed + 100)
        test_m2 = test_mc  # evaluate on clean mean logits
        result, elapsed = train_ww_on_bundle(
            train_m2, test_m2, epochs_ww, device, seed + 100,
            str(out / "m2_var_aug"), f"M2 (β={var_beta})", scales=scales,
        )
        all_metrics["m2"] = {"name": f"M2 (var-aug, β={var_beta})", "results": result["results"]}
        all_times["m2"] = elapsed

    # --- M3: Sample-level expansion ---
    if "m3" in variants:
        train_m3 = expand_mc_samples(train_mc)
        test_m3 = test_mc  # evaluate on mean logits (sample-level eval below)
        result, elapsed = train_ww_on_bundle(
            train_m3, test_m3, epochs_ww, device, seed + 200,
            str(out / "m3_sample_ww"), f"M3 (sample-expanded ×{mc_samples})", scales=scales,
        )
        all_metrics["m3"] = {"name": f"M3 (sample-WW)", "results": result["results"]}
        all_times["m3"] = elapsed

    # ── Step 6: Human reference ──
    test_arr = test_mc
    human_has_errors = (test_arr["response_labels"] != test_arr["target_labels"]).any()
    human_ref = {
        "acc": float((test_arr["response_labels"] == test_arr["target_labels"]).mean()),
        "resp_agree": float((test_arr["response_labels"] == test_arr["target_labels"]).mean()),
        "ΔRT": float(
            test_arr["rts"][test_arr["response_labels"] != test_arr["target_labels"]].mean()
            - test_arr["rts"][test_arr["response_labels"] == test_arr["target_labels"]].mean()
        ) if human_has_errors else float("nan"),
        "pred_mean": float(test_arr["rts"].mean()),
        "cong_gap": float(
            test_arr["rts"][test_arr["congruency"] == 1].mean()
            - test_arr["rts"][test_arr["congruency"] == 0].mean()
        ) if (test_arr["congruency"] == 1).any() and (test_arr["congruency"] == 0).any() else 0.0,
    }

    # ── Step 7: Comparison table ──
    row_specs = [
        ("behavior_optimal_score", "beh_opt"),
        ("model_accuracy", "acc"),
        ("response_agreement", "resp_agree"),
        ("error_minus_correct_rt", "ΔRT"),
        ("pred_mean", "pred_mean"),
        ("model_congruency_rt_gap", "cong_gap"),
    ]

    print(f"\n{'='*90}")
    print(f"  MC Dropout Variant Comparison")
    print(f"{'='*90}")

    metric_names = [rs[0] for rs in row_specs]
    col_width = 22
    header = f"  {'Metric':<24}"
    for key in ["eval", "m1", "m2", "m3"]:
        if key in all_metrics:
            header += f"{all_metrics[key]['name']:<{col_width}}"
    header += f"{'Human':<15}"
    print(header)
    print(f"  {'─'*24}{'─'*col_width * len(all_metrics)}{'─'*15}")

    for metric_key, label in row_specs:
        row = f"  {label:<24}"
        for vkey in ["eval", "m1", "m2", "m3"]:
            if vkey in all_metrics:
                val = all_metrics[vkey]["results"].get(metric_key, float("nan"))
                if isinstance(val, float):
                    row += f"{val:<{col_width}.4f}"
                else:
                    row += f"{str(val):<{col_width}}"
        hu_val = human_ref.get(label, "—")
        if isinstance(hu_val, float):
            row += f"{hu_val:<15.4f}"
        else:
            row += f"{str(hu_val):<15}"
        print(row)

    # ── Step 8: Save comparison ──
    _write_json(out / "comparison.json", {
        "variants": all_metrics,
        "human_ref": human_ref,
        "times": all_times,
        "mc_samples": mc_samples,
        "var_beta": var_beta,
        "epochs_ww": epochs_ww,
        "device": device,
        "seed": seed,
    })
    _write_json(out / "run_complete.json", {
        "status": "completed",
        "variants_tested": list(all_metrics.keys()),
    })

    print(f"\nResults saved to: {out}")
    return {"metrics": all_metrics, "human_ref": human_ref}


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC Dropout → Wong-Wang smoke test")
    parser.add_argument("--age_group", default="20-29", choices=["20-29", "80-89"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--variant", default="all",
                        choices=["m1", "m2", "m3", "eval", "all"],
                        help="Which variant(s) to run")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--var_beta", type=float, default=1.0,
                        help="M2 noise scale: noise = β * sqrt(var)")
    parser.add_argument("--epochs_ww", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--smoke_eval", action="store_true")
    parser.add_argument("--smoke_max_trials", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    data_dir = data_root / args.age_group
    train_csv = data_dir / "train_data.csv"
    test_csv = data_dir / "test_data.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing CSVs in {data_dir}")

    print(f"Loading datasets from {data_dir}...")
    train_full = StimulusDataset(str(train_csv))
    test_full = StimulusDataset(str(test_csv))
    print(f"  Full train: {len(train_full)}, Full test: {len(test_full)}")

    if args.smoke_eval:
        train_idx = _build_behavior_balanced_subset(train_full, args.smoke_max_trials, args.seed)
        test_idx = _build_behavior_balanced_subset(test_full, args.smoke_max_trials, args.seed + 1)
        train_ds = Subset(train_full, train_idx)
        test_ds = Subset(test_full, test_idx)
        print(f"  Smoke subset: train={len(train_ds)}, test={len(test_ds)}")
    else:
        train_ds = train_full
        test_ds = test_full

    # Ensure flanker_labels on full dataset
    for ds in (train_full, test_full):
        if hasattr(ds, 'flanker_labels') and ds.flanker_labels is None:
            df = ds.data
            if "flanker_direction" in df.columns:
                ds.flanker_labels = df["flanker_direction"].map(
                    lambda x: DIRECTION_MAP.get(x, -1)).to_numpy(dtype=np.int64)

    print(f"Loading Stage 1 VGG model on {args.device}...")
    model = load_stage1_model(args.device)
    print(f"  Dropout rate: {model.classifier[2].p}")

    variant_arg = args.variant
    variants = ["eval", "m1", "m2", "m3"] if variant_arg == "all" else [variant_arg]
    if variant_arg != "all" and "eval" not in variants:
        variants.insert(0, "eval")  # always include baseline

    run_all_variants(
        model=model,
        train_dataset=train_ds,
        test_dataset=test_ds,
        variants=variants,
        epochs_ww=args.epochs_ww,
        mc_samples=args.mc_samples,
        var_beta=args.var_beta,
        device=args.device,
        seed=args.seed,
        output_dir=str(output_root),
    )


if __name__ == "__main__":
    main()
