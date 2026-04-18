import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
import pandas as pd

from analyze_dynamic_selection_single_subject import (
    _build_trial_df,
    _extract_metric_scalar,
    compute_caf,
    compute_conditional_error_rt,
    compute_delta,
    compute_tail_summary,
)
from project_paths import RESULTS_ROOT, age_group_data_dir, age_group_stage2_dir
from run_dynamic_selection_single_subject import (
    AGE_GROUPS,
    dynamic_selection_phase1_config,
    locate_stage2_param_artifacts,
    safe_rel_to_root,
)
from train_age_groups_efficient import (
    attach_flanker_labels_from_csv,
    evaluate_cached_stage2_params,
    to_jsonable,
    validate_cached_stage2_inputs,
)


DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "repro_legacy_interim" / "minimal_conflict_capture_probe"
CAPTURE_MIDPOINT_S = 0.05
CAPTURE_TAU_S = 0.03
CAPTURE_STRENGTH_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
EVIDENCE_ROOT = Path(".sisyphus") / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded minimal conflict-capture probe on top of the active dynamic_selection_phase1 baseline. "
            "Supports: audit-baseline, write-grid, simulate, full."
        )
    )
    parser.add_argument("--mode", required=True, choices=("audit-baseline", "write-grid", "simulate", "full"))
    parser.add_argument("--age_group", default=None, choices=AGE_GROUPS)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--choice_temperature", type=float, default=0.10)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _write_evidence(task: int, slug: str, payload: dict) -> Path:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_ROOT / f"task-{task}-{slug}.json"
    out.write_text(json.dumps(to_jsonable(payload), indent=2))
    return out


def _load_npz_params_no_scale(npz_path: Path) -> Dict[str, Any]:
    params_npz = np.load(npz_path)
    return cast(Dict[str, Any], {k: params_npz[k] for k in params_npz.files if k != "scale"})


def _load_test_cached(age_group: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
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


def _build_capture_config(*, capture_strength: float) -> Dict[str, Any]:
    config = dynamic_selection_phase1_config().copy()
    config.update(
        {
            "capture_strength": float(capture_strength),
            "capture_midpoint_s": float(CAPTURE_MIDPOINT_S),
            "capture_tau_s": float(CAPTURE_TAU_S),
        }
    )
    return config


def _metric_bundle(*, predictions: Dict[str, np.ndarray], cached: Dict[str, np.ndarray]) -> Dict[str, float]:
    human_df = _build_trial_df(
        rt_s=cached["rts"],
        choice=cached["response_labels"],
        target_labels=cached["target_labels"],
        congruency=cached["congruency"],
        source="human",
    )
    pred_df = _build_trial_df(
        rt_s=predictions["pred_rt"],
        choice=predictions["pred_choice"],
        target_labels=cached["target_labels"],
        congruency=cached["congruency"],
        source="probe",
    )
    caf = pd.concat(
        [
            compute_caf(human_df, "human", "rt_s", "correct"),
            compute_caf(pred_df, "probe", "rt_s", "correct"),
        ],
        ignore_index=True,
    )
    delta = pd.concat(
        [
            compute_delta(human_df, "human", "rt_s"),
            compute_delta(pred_df, "probe", "rt_s"),
        ],
        ignore_index=True,
    )
    err_h, _ = compute_conditional_error_rt(human_df, "human", "rt_s", "correct")
    err_p, _ = compute_conditional_error_rt(pred_df, "probe", "rt_s", "correct")
    tail = pd.concat(
        [
            compute_tail_summary(human_df, "human", "rt_s", "correct"),
            compute_tail_summary(pred_df, "probe", "rt_s", "correct"),
        ],
        ignore_index=True,
    )
    metric_names = [
        "earliest_incongruent_caf",
        "first_delta",
        "incongruent_error_minus_correct_rt",
        "incongruent_conditional_tail",
    ]
    out: Dict[str, float] = {}
    for metric in metric_names:
        out[f"human_{metric}"] = float(
            _extract_metric_scalar(
                caf_df=caf.loc[caf["source"] == "human"],
                delta_df=delta.loc[delta["source"] == "human"],
                error_wide_df=err_h,
                tail_df=tail.loc[tail["source"] == "human"],
                metric=metric,
            )
        )
        out[f"pred_{metric}"] = float(
            _extract_metric_scalar(
                caf_df=caf.loc[caf["source"] == "probe"],
                delta_df=delta.loc[delta["source"] == "probe"],
                error_wide_df=err_p,
                tail_df=tail.loc[tail["source"] == "probe"],
                metric=metric,
            )
        )
    return out


def audit_baseline(output_root: Path, *, seed: int, device: str, choice_temperature: float) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    age_group_entries: Dict[str, Any] = {}
    insertion_point = (
        "code/scripts/vgg_wongwang_lim.py:84-92 in build_dynamic_stage2_input, "
        "where capture_pulse_t contributes to flanker_multiplier_t for dynamic_flanker_suppression"
    )
    for age_group in AGE_GROUPS:
        data_dir = age_group_data_dir(age_group)
        stage2_dir = age_group_stage2_dir(age_group)
        cfg_path, params_path = locate_stage2_param_artifacts(age_group)
        validate_cached_stage2_inputs(
            age_group,
            str(data_dir),
            str(stage2_dir / "train_logits.npz"),
            str(stage2_dir / "test_logits.npz"),
        )
        cfg = _load_json(cfg_path)
        age_group_entries[age_group] = {
            "test_csv_path": safe_rel_to_root(data_dir / "test_data.csv"),
            "test_cache_npz": safe_rel_to_root(stage2_dir / "test_logits.npz"),
            "stage2_config_path": safe_rel_to_root(cfg_path),
            "stage2_params_path": safe_rel_to_root(params_path),
            "scale": float(cfg["scale"]),
            "time_steps": int(cfg["time_steps"]),
        }
    manifest = {
        "entrypoint": "code/scripts/run_minimal_conflict_capture_probe.py",
        "mode": "audit-baseline",
        "output_root": safe_rel_to_root(output_root),
        "seed": int(seed),
        "device": str(device),
        "choice_temperature": float(choice_temperature),
        "selection_config": dynamic_selection_phase1_config(),
        "insertion_point": insertion_point,
        "non_probe_params": {
            "rt_readout_mode": "baseline",
            "capture_midpoint_s": float(CAPTURE_MIDPOINT_S),
            "capture_tau_s": float(CAPTURE_TAU_S),
            "capture_strength_free_dimension": "capture_strength",
        },
        "age_groups": age_group_entries,
    }
    out_path = output_root / "baseline_manifest.json"
    out_path.write_text(json.dumps(to_jsonable(manifest), indent=2))
    ev = _write_evidence(
        1,
        "audit-baseline",
        {
            "command": "python code/scripts/run_minimal_conflict_capture_probe.py --mode audit-baseline --output_root <output_root>",
            "output_root": safe_rel_to_root(output_root),
            "baseline_manifest": safe_rel_to_root(out_path),
        },
    )
    print(f"Wrote evidence: {ev}")
    return out_path


def write_grid(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, strength in enumerate(CAPTURE_STRENGTH_GRID):
        rows.append(
            {
                "probe_tag": "baseline" if idx == 0 else f"capture_strength_{float(strength):.2f}",
                "is_baseline": bool(idx == 0),
                "capture_strength": float(strength),
                "capture_midpoint_s": float(CAPTURE_MIDPOINT_S),
                "capture_tau_s": float(CAPTURE_TAU_S),
            }
        )
    out_path = output_root / "probe_config_grid.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    ev = _write_evidence(
        2,
        "write-grid",
        {
            "command": "python code/scripts/run_minimal_conflict_capture_probe.py --mode write-grid --output_root <output_root>",
            "output_root": safe_rel_to_root(output_root),
            "probe_config_grid": safe_rel_to_root(out_path),
            "tuned_dimension": "capture_strength",
            "fixed_columns": ["probe_tag", "is_baseline", "capture_midpoint_s", "capture_tau_s"],
            "n_rows": int(len(rows)),
        },
    )
    print(f"Wrote evidence: {ev}")
    return out_path


def _load_manifest_and_grid(output_root: Path) -> Tuple[dict, pd.DataFrame]:
    manifest_path = output_root / "baseline_manifest.json"
    grid_path = output_root / "probe_config_grid.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing baseline manifest: {manifest_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing probe config grid: {grid_path}")
    return _load_json(manifest_path), pd.read_csv(grid_path)


def simulate_age_group(output_root: Path, *, age_group: str, seed: int, device: str, choice_temperature: float) -> Path:
    manifest, grid_df = _load_manifest_and_grid(output_root)
    age_entry = manifest["age_groups"][age_group]
    stage2_cfg_path = Path(age_entry["stage2_config_path"])
    stage2_params_path = Path(age_entry["stage2_params_path"])
    cfg = _load_json(stage2_cfg_path)
    baseline_scale = float(cfg["scale"])
    time_steps = int(cfg["time_steps"])
    params = _load_npz_params_no_scale(stage2_params_path)
    _, test_cached = _load_test_cached(age_group)
    age_dir = output_root / age_group
    age_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    grid_records = grid_df.to_dict(orient="records")
    n_trials_total = int(len(test_cached["logits"]))
    for row_idx, row in enumerate(grid_records):
        capture_strength = float(row["capture_strength"])
        selection_config = _build_capture_config(capture_strength=capture_strength)
        eval_seed = int(seed)
        predictions, canonical_results = evaluate_cached_stage2_params(
            params=params,
            scale=baseline_scale,
            time_steps=time_steps,
            cached=test_cached,
            device=device,
            choice_temperature=float(choice_temperature),
            rt_readout_mode="baseline",
            readout_config=None,
            selection_config=selection_config,
            random_seed=eval_seed,
            rt_shape_focus=False,
        )
        metric_bundle = _metric_bundle(predictions=predictions, cached=test_cached)
        rows.append(
            {
                "age_group": age_group,
                "probe_tag": str(row["probe_tag"]),
                "is_baseline": bool(row["is_baseline"]),
                "capture_strength": capture_strength,
                "capture_midpoint_s": float(row["capture_midpoint_s"]),
                "capture_tau_s": float(row["capture_tau_s"]),
                "scale": baseline_scale,
                "time_steps": time_steps,
                "n_trials_total": n_trials_total,
                "total_score": float(canonical_results["total_score"]),
                "model_accuracy": float(canonical_results["model_accuracy"]),
                "response_agreement": float(canonical_results["response_agreement"]),
                "model_congruency_rt_gap": float(canonical_results["model_congruency_rt_gap"]),
                **metric_bundle,
            }
        )

    out_path = age_dir / "capture_probe_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    task_num = 3 if age_group == "20-29" else 4
    ev = _write_evidence(
        task_num,
        f"simulate-{age_group}",
        {
            "command": f"python code/scripts/run_minimal_conflict_capture_probe.py --mode simulate --age_group {age_group} --output_root <output_root>",
            "output_root": safe_rel_to_root(output_root),
            "age_group": age_group,
            "capture_probe_metrics": safe_rel_to_root(out_path),
            "n_rows": int(len(rows)),
        },
    )
    print(f"Wrote evidence: {ev}")
    return out_path


def run_full(output_root: Path, *, seed: int, device: str, choice_temperature: float) -> None:
    audit_baseline(output_root, seed=seed, device=device, choice_temperature=choice_temperature)
    write_grid(output_root)
    for age_group in AGE_GROUPS:
        simulate_age_group(output_root, age_group=age_group, seed=seed, device=device, choice_temperature=choice_temperature)

    # Plan requirement: a single command should write all outputs under the dedicated result tree.
    # We invoke the analysis script as the final step so --mode full is end-to-end.
    analysis_cmd = [
        sys.executable,
        "code/scripts/analyze_minimal_conflict_capture_probe.py",
        "--input_root",
        str(output_root),
        "--output_root",
        str(output_root),
    ]
    completed = subprocess.run(analysis_cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Analysis command failed with exit code {completed.returncode}: {' '.join(analysis_cmd)}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.mode == "audit-baseline":
        out = audit_baseline(output_root, seed=int(args.seed), device=str(args.device), choice_temperature=float(args.choice_temperature))
        print(f"Wrote baseline manifest: {out}")
        return
    if args.mode == "write-grid":
        out = write_grid(output_root)
        print(f"Wrote probe config grid: {out}")
        return
    if args.mode == "simulate":
        if args.age_group is None:
            raise ValueError("--age_group is required for --mode simulate")
        out = simulate_age_group(output_root, age_group=str(args.age_group), seed=int(args.seed), device=str(args.device), choice_temperature=float(args.choice_temperature))
        print(f"Wrote capture probe metrics: {out}")
        return
    if args.mode == "full":
        run_full(output_root, seed=int(args.seed), device=str(args.device), choice_temperature=float(args.choice_temperature))
        print(f"Completed full minimal conflict-capture workflow under {output_root}")
        return
    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
