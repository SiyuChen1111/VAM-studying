import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_run(run_dir: Path):
    predictions = np.load(run_dir / 'predictions_smoke.npz')
    with open(run_dir / 'config_smoke.json', 'r') as f:
        config = json.load(f)
    with open(run_dir / 'metrics_smoke.json', 'r') as f:
        metrics = json.load(f)
    subset_meta_path = run_dir / 'smoke_eval_subset_meta.json'
    ranking_summary_path = run_dir / 'checkpoint_ranking_summary.json'
    trajectory_path = run_dir / 'trajectory_samples.npz'
    trajectory = np.load(trajectory_path) if trajectory_path.exists() else None
    return {
        'run_dir': run_dir,
        'pred_rt': predictions['pred_rt'].astype(np.float32),
        'pred_choice': predictions['pred_choice'].astype(np.int64),
        'true_rt': predictions['true_rt'].astype(np.float32),
        'target_labels': predictions['target_labels'].astype(np.int64),
        'response_labels': predictions['response_labels'].astype(np.int64),
        'congruency': predictions['congruency'].astype(np.int64),
        'config': config,
        'metrics': metrics,
        'subset_meta': json.loads(subset_meta_path.read_text()) if subset_meta_path.exists() else None,
        'ranking_summary': json.loads(ranking_summary_path.read_text()) if ranking_summary_path.exists() else None,
        'trajectory': {
            'traj': trajectory['traj'].astype(np.float32),
            'pred_choice': trajectory['pred_choice'].astype(np.int64),
            'target_labels': trajectory['target_labels'].astype(np.int64),
            'response_labels': trajectory['response_labels'].astype(np.int64),
            'pred_rt': trajectory['pred_rt'].astype(np.float32),
            'true_rt': trajectory['true_rt'].astype(np.float32),
            'congruency': trajectory['congruency'].astype(np.int64),
            'dt_ms': int(trajectory['dt_ms']),
        } if trajectory is not None else None,
    }


def describe_selected_checkpoint(run):
    scale = run['config'].get('scale')
    epoch = run['config'].get('best_epoch')
    if epoch is None:
        return f"scale={scale:.4f}" if scale is not None else 'unknown'
    return f"scale={scale:.4f}, epoch={epoch}"


def has_defined_metric(value) -> bool:
    return not np.isnan(float(value))


def ranking_tradeoff_visible(run) -> bool:
    ranking_summary = run.get('ranking_summary')
    if not ranking_summary:
        return False
    candidates = ranking_summary.get('candidate_checkpoints', [])
    if len(candidates) < 2:
        return False
    for i, left in enumerate(candidates):
        left_metrics = left.get('metrics', {})
        for right in candidates[i + 1:]:
            right_metrics = right.get('metrics', {})
            rt_shape_delta = float(left_metrics.get('rt_shape_score', np.nan)) - float(right_metrics.get('rt_shape_score', np.nan))
            agreement_delta = float(left_metrics.get('response_agreement', np.nan)) - float(right_metrics.get('response_agreement', np.nan))
            congruency_delta = float(left_metrics.get('congruency_score', np.nan)) - float(right_metrics.get('congruency_score', np.nan))
            if np.isnan(rt_shape_delta) or np.isnan(agreement_delta) or np.isnan(congruency_delta):
                continue
            if (rt_shape_delta > 0 and (agreement_delta < 0 or congruency_delta < 0)) or (rt_shape_delta < 0 and (agreement_delta > 0 or congruency_delta > 0)):
                return True
    return False


def summarize_behavior(pred_rt, pred_choice, true_rt, target_labels, response_labels, congruency):
    pred_correct = pred_choice == target_labels
    human_correct = response_labels == target_labels
    pred_error_rt = float(pred_rt[~pred_correct].mean()) if (~pred_correct).any() else float('nan')
    pred_correct_rt = float(pred_rt[pred_correct].mean()) if pred_correct.any() else float('nan')
    human_error_rt = float(true_rt[~human_correct].mean()) if (~human_correct).any() else float('nan')
    human_correct_rt = float(true_rt[human_correct].mean()) if human_correct.any() else float('nan')
    pred_cong = float(pred_rt[congruency == 0].mean()) if (congruency == 0).any() else float('nan')
    pred_incong = float(pred_rt[congruency == 1].mean()) if (congruency == 1).any() else float('nan')
    human_cong = float(true_rt[congruency == 0].mean()) if (congruency == 0).any() else float('nan')
    human_incong = float(true_rt[congruency == 1].mean()) if (congruency == 1).any() else float('nan')
    return {
        'pred_mean': float(pred_rt.mean()),
        'pred_median': float(np.median(pred_rt)),
        'pred_q05': float(np.quantile(pred_rt, 0.05)),
        'pred_skew': float(stats.skew(pred_rt)),
        'human_mean': float(true_rt.mean()),
        'human_median': float(np.median(true_rt)),
        'human_skew': float(stats.skew(true_rt)),
        'pred_tail_q95_q50': float(np.quantile(pred_rt, 0.95) - np.quantile(pred_rt, 0.50)),
        'human_tail_q95_q50': float(np.quantile(true_rt, 0.95) - np.quantile(true_rt, 0.50)),
        'pred_error_rt': pred_error_rt,
        'pred_correct_rt': pred_correct_rt,
        'pred_error_minus_correct': float(pred_error_rt - pred_correct_rt),
        'human_error_rt': human_error_rt,
        'human_correct_rt': human_correct_rt,
        'human_error_minus_correct': float(human_error_rt - human_correct_rt),
        'pred_cong_rt': pred_cong,
        'pred_incong_rt': pred_incong,
        'pred_cong_gap': float(pred_incong - pred_cong),
        'human_cong_rt': human_cong,
        'human_incong_rt': human_incong,
        'human_cong_gap': float(human_incong - human_cong),
        'response_agreement': float((pred_choice == response_labels).mean()),
        'model_accuracy': float((pred_choice == target_labels).mean()),
        'human_accuracy': float((response_labels == target_labels).mean()),
    }


def summarize_trajectory(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trajectory = run.get('trajectory')
    if trajectory is None:
        return None
    traj = trajectory['traj']
    pred_choice = trajectory['pred_choice']
    target_labels = trajectory['target_labels']
    congruency = trajectory['congruency']
    pred_correct = pred_choice == target_labels
    winner_gap = np.partition(traj, -2, axis=2)[:, :, -1] - np.partition(traj, -2, axis=2)[:, :, -2]
    summary = {
        'shape': list(traj.shape),
        'dt_ms': int(trajectory['dt_ms']),
        'winner_gap_final_mean': float(winner_gap[:, -1].mean()),
        'winner_gap_final_correct': float(winner_gap[pred_correct, -1].mean()) if pred_correct.any() else float('nan'),
        'winner_gap_final_error': float(winner_gap[~pred_correct, -1].mean()) if (~pred_correct).any() else float('nan'),
        'winner_gap_final_congruent': float(winner_gap[congruency == 0, -1].mean()) if (congruency == 0).any() else float('nan'),
        'winner_gap_final_incongruent': float(winner_gap[congruency == 1, -1].mean()) if (congruency == 1).any() else float('nan'),
        'correct_error_distance': float(np.abs(traj[pred_correct].mean(axis=0) - traj[~pred_correct].mean(axis=0)).mean()) if pred_correct.any() and (~pred_correct).any() else float('nan'),
        'congruency_distance': float(np.abs(traj[congruency == 0].mean(axis=0) - traj[congruency == 1].mean(axis=0)).mean()) if (congruency == 0).any() and (congruency == 1).any() else float('nan'),
    }
    interpretability_flags = [
        summary['winner_gap_final_mean'] > 0.01,
        not np.isnan(summary['correct_error_distance']) and summary['correct_error_distance'] > 0.005,
        not np.isnan(summary['congruency_distance']) and summary['congruency_distance'] > 0.005,
    ]
    summary['interpretable'] = bool(any(interpretability_flags))
    return summary


def make_rt_hist(output_path: Path, baseline, candidate, candidate_label: str):
    plt.figure(figsize=(8, 5))
    plt.hist(baseline['true_rt'], bins=24, density=True, alpha=0.35, label='Human', color='#808080')
    plt.hist(baseline['pred_rt'], bins=24, density=True, alpha=0.35, label='Baseline', color='#4C78A8')
    plt.hist(candidate['pred_rt'], bins=24, density=True, alpha=0.35, label=candidate_label, color='#F58518')
    plt.xlabel('RT (s)')
    plt.ylabel('Density')
    plt.title('Smoke RT distributions')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_error_correct_plot(output_path: Path, baseline_summary, candidate_summary, candidate_label: str):
    labels = ['Correct', 'Error']
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 5))
    plt.bar(x - width, [baseline_summary['human_correct_rt'], baseline_summary['human_error_rt']], width=width, label='Human', color='#808080')
    plt.bar(x, [baseline_summary['pred_correct_rt'], baseline_summary['pred_error_rt']], width=width, label='Baseline', color='#4C78A8')
    plt.bar(x + width, [candidate_summary['pred_correct_rt'], candidate_summary['pred_error_rt']], width=width, label=candidate_label, color='#F58518')
    plt.xticks(x, labels)
    plt.ylabel('Mean RT (s)')
    plt.title('Error vs correct RT')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_congruency_plot(output_path: Path, baseline_summary, candidate_summary, candidate_label: str):
    labels = ['Congruent', 'Incongruent']
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(7, 5))
    plt.bar(x - width, [baseline_summary['human_cong_rt'], baseline_summary['human_incong_rt']], width=width, label='Human', color='#808080')
    plt.bar(x, [baseline_summary['pred_cong_rt'], baseline_summary['pred_incong_rt']], width=width, label='Baseline', color='#4C78A8')
    plt.bar(x + width, [candidate_summary['pred_cong_rt'], candidate_summary['pred_incong_rt']], width=width, label=candidate_label, color='#F58518')
    plt.xticks(x, labels)
    plt.ylabel('Mean RT (s)')
    plt.title('Congruent vs incongruent RT')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_trajectory_condition_plot(output_path: Path, trajectory: Dict[str, Any], title: str, left_mask: np.ndarray, right_mask: np.ndarray, left_label: str, right_label: str):
    traj = trajectory['traj']
    dt_ms = trajectory['dt_ms']
    time_axis = np.arange(traj.shape[1]) * dt_ms / 1000.0
    plt.figure(figsize=(8, 5))
    if left_mask.any():
        plt.plot(time_axis, traj[left_mask].mean(axis=(0, 2)), label=left_label, color='#4C78A8')
    if right_mask.any():
        plt.plot(time_axis, traj[right_mask].mean(axis=(0, 2)), label=right_label, color='#F58518')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean accumulator state')
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_winner_gap_plot(output_path: Path, trajectory: Dict[str, Any]):
    traj = trajectory['traj']
    dt_ms = trajectory['dt_ms']
    time_axis = np.arange(traj.shape[1]) * dt_ms / 1000.0
    sorted_traj = np.sort(traj, axis=2)
    gap = sorted_traj[:, :, -1] - sorted_traj[:, :, -2]
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, gap.mean(axis=0), color='#54A24B')
    plt.xlabel('Time (s)')
    plt.ylabel('Winner - runner-up')
    plt.title('Candidate winner gap over time')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_recommendation(baseline_summary, candidate_summary):
    mean_scale_ok = abs(candidate_summary['pred_mean'] - baseline_summary['pred_mean']) / max(baseline_summary['pred_mean'], 1e-6) <= 0.10
    median_scale_ok = abs(candidate_summary['pred_median'] - baseline_summary['pred_median']) / max(baseline_summary['pred_median'], 1e-6) <= 0.10
    lower_tail_ok = candidate_summary['pred_q05'] >= baseline_summary['pred_q05'] - 0.02
    scale_gate = mean_scale_ok and median_scale_ok and lower_tail_ok
    agreement_ok = candidate_summary['response_agreement'] >= baseline_summary['response_agreement'] - 0.02
    sensible_congruency = candidate_summary['pred_cong_gap'] > 0
    later_errors = candidate_summary['pred_error_minus_correct'] >= baseline_summary['pred_error_minus_correct'] - 0.02
    more_right_skewed = candidate_summary['pred_skew'] >= baseline_summary['pred_skew'] or candidate_summary['pred_tail_q95_q50'] >= baseline_summary['pred_tail_q95_q50']
    worth_scaling = all([scale_gate, agreement_ok, sensible_congruency, later_errors, more_right_skewed])
    return {
        'mean_scale_ok': mean_scale_ok,
        'median_scale_ok': median_scale_ok,
        'lower_tail_ok': lower_tail_ok,
        'scale_gate': scale_gate,
        'more_right_skewed': more_right_skewed,
        'later_errors': later_errors,
        'sensible_congruency': sensible_congruency,
        'agreement_ok': agreement_ok,
        'worth_scaling': worth_scaling,
    }


def write_summary(output_path: Path, baseline, candidate, baseline_summary, candidate_summary, checks, candidate_label: str, candidate_trajectory_summary: Optional[Dict[str, Any]]):
    baseline_subset = baseline.get('subset_meta') or {}
    candidate_subset = candidate.get('subset_meta') or {}
    same_checkpoint = (
        baseline['config'].get('scale') == candidate['config'].get('scale')
        and baseline['config'].get('best_epoch') == candidate['config'].get('best_epoch')
    )
    baseline_error_defined = has_defined_metric(baseline_summary['pred_error_minus_correct'])
    candidate_error_defined = has_defined_metric(candidate_summary['pred_error_minus_correct'])
    baseline_tradeoff = ranking_tradeoff_visible(baseline)
    candidate_tradeoff = ranking_tradeoff_visible(candidate)
    lines = [
        '# RT Readout Smoke Summary',
        '',
        f"- Baseline run: `{baseline['run_dir']}`",
        f"- {candidate_label} run: `{candidate['run_dir']}`",
        f"- Baseline checkpoint: {describe_selected_checkpoint(baseline)}",
        f"- {candidate_label} checkpoint: {describe_selected_checkpoint(candidate)}",
        '',
        f'## Baseline vs {candidate_label}',
        '',
        f"- Mean RT: baseline={baseline_summary['pred_mean']:.4f}, {candidate_label}={candidate_summary['pred_mean']:.4f}, human={baseline_summary['human_mean']:.4f}",
        f"- Median RT: baseline={baseline_summary['pred_median']:.4f}, {candidate_label}={candidate_summary['pred_median']:.4f}, human={baseline_summary['human_median']:.4f}",
        f"- Predicted skewness: baseline={baseline_summary['pred_skew']:.4f}, {candidate_label}={candidate_summary['pred_skew']:.4f}, human={baseline_summary['human_skew']:.4f}",
        f"- Tail spread q95-q50: baseline={baseline_summary['pred_tail_q95_q50']:.4f}, {candidate_label}={candidate_summary['pred_tail_q95_q50']:.4f}, human={baseline_summary['human_tail_q95_q50']:.4f}",
        f"- Error minus correct RT: baseline={baseline_summary['pred_error_minus_correct']:.4f}, {candidate_label}={candidate_summary['pred_error_minus_correct']:.4f}, human={baseline_summary['human_error_minus_correct']:.4f}",
        f"- Congruency gap: baseline={baseline_summary['pred_cong_gap']:.4f}, {candidate_label}={candidate_summary['pred_cong_gap']:.4f}, human={baseline_summary['human_cong_gap']:.4f}",
        f"- Response agreement: baseline={baseline_summary['response_agreement']:.4f}, {candidate_label}={candidate_summary['response_agreement']:.4f}",
        '',
        '## Eval subset diagnostics',
        '',
        f"- Baseline subset mode: {baseline_subset.get('subset_mode', baseline['config'].get('smoke_eval_mode', 'unknown'))}",
        f"- {candidate_label} subset mode: {candidate_subset.get('subset_mode', candidate['config'].get('smoke_eval_mode', 'unknown'))}",
        f"- Baseline human-error trials: {baseline_subset.get('actual_human_error_trials', 'n/a')}",
        f"- {candidate_label} human-error trials: {candidate_subset.get('actual_human_error_trials', 'n/a')}",
        f"- Baseline congruent / incongruent: {baseline_subset.get('congruent_trials', 'n/a')} / {baseline_subset.get('incongruent_trials', 'n/a')}",
        f"- {candidate_label} congruent / incongruent: {candidate_subset.get('congruent_trials', 'n/a')} / {candidate_subset.get('incongruent_trials', 'n/a')}",
        f"- Baseline balance constraints satisfied: {baseline_subset.get('balance_constraints_satisfied', 'n/a')}",
        f"- {candidate_label} balance constraints satisfied: {candidate_subset.get('balance_constraints_satisfied', 'n/a')}",
        '',
        '## Hard gates',
        '',
        f"- Mean RT stays near baseline scale: {checks['mean_scale_ok']}",
        f"- Median RT stays near baseline scale: {checks['median_scale_ok']}",
        f"- Lower tail does not collapse earlier: {checks['lower_tail_ok']}",
        f"- RT-scale gate passed: {checks['scale_gate']}",
        '',
        '## Decision',
        '',
        f"- More right-skewed than baseline: {checks['more_right_skewed']}",
        f"- Error RT shifts later: {checks['later_errors']}",
        f"- Congruency gap remains sensible: {checks['sensible_congruency']}",
        f"- Response agreement does not materially collapse: {checks['agreement_ok']}",
        f"- Baseline error_minus_correct_rt defined: {baseline_error_defined}",
        f"- {candidate_label} error_minus_correct_rt defined: {candidate_error_defined}",
        f"- Baseline and {candidate_label} selected the same checkpoint: {same_checkpoint}",
        f"- Ranking tradeoff visible (baseline): {baseline_tradeoff}",
        f"- Ranking tradeoff visible ({candidate_label}): {candidate_tradeoff}",
        f"- Ranking tradeoff visible overall: {baseline_tradeoff or candidate_tradeoff}",
    ]
    if candidate_trajectory_summary is not None:
        lines.extend([
            '',
            '## Candidate trajectory summary',
            '',
            f"- Trajectory shape: {candidate_trajectory_summary['shape']}",
            f"- dt_ms: {candidate_trajectory_summary['dt_ms']}",
            f"- Winner gap final mean: {candidate_trajectory_summary['winner_gap_final_mean']:.4f}",
            f"- Winner gap final correct / error: {candidate_trajectory_summary['winner_gap_final_correct']:.4f} / {candidate_trajectory_summary['winner_gap_final_error']:.4f}",
            f"- Winner gap final congruent / incongruent: {candidate_trajectory_summary['winner_gap_final_congruent']:.4f} / {candidate_trajectory_summary['winner_gap_final_incongruent']:.4f}",
            f"- Correct-vs-error trajectory distance: {candidate_trajectory_summary['correct_error_distance']:.4f}",
            f"- Congruency trajectory distance: {candidate_trajectory_summary['congruency_distance']:.4f}",
            f"- Trajectories interpretable: {candidate_trajectory_summary['interpretable']}",
        ])
    lines.extend([
        f"- Worth scaling up: {checks['worth_scaling']}",
        f"- Conclusion: {'worth scaling' if checks['worth_scaling'] else 'reject and stay baseline'}",
    ])
    output_path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', required=True)
    parser.add_argument('--candidate_dir', required=True)
    parser.add_argument('--candidate_label', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    baseline = load_run(Path(args.baseline_dir))
    candidate = load_run(Path(args.candidate_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = summarize_behavior(
        baseline['pred_rt'],
        baseline['pred_choice'],
        baseline['true_rt'],
        baseline['target_labels'],
        baseline['response_labels'],
        baseline['congruency'],
    )
    candidate_summary = summarize_behavior(
        candidate['pred_rt'],
        candidate['pred_choice'],
        candidate['true_rt'],
        candidate['target_labels'],
        candidate['response_labels'],
        candidate['congruency'],
    )
    candidate_trajectory_summary = summarize_trajectory(candidate)
    checks = build_recommendation(baseline_summary, candidate_summary)

    make_rt_hist(output_dir / 'rt_hist_smoke.png', baseline, candidate, args.candidate_label)
    make_error_correct_plot(output_dir / 'error_vs_correct_smoke.png', baseline_summary, candidate_summary, args.candidate_label)
    make_congruency_plot(output_dir / 'cong_vs_incong_smoke.png', baseline_summary, candidate_summary, args.candidate_label)
    if candidate['trajectory'] is not None:
        pred_correct = candidate['trajectory']['pred_choice'] == candidate['trajectory']['target_labels']
        congruent = candidate['trajectory']['congruency'] == 0
        incongruent = candidate['trajectory']['congruency'] == 1
        make_trajectory_condition_plot(
            output_dir / 'traj_correct_vs_error_smoke.png',
            candidate['trajectory'],
            f'{args.candidate_label}: correct vs error trajectory',
            pred_correct,
            ~pred_correct,
            'Correct',
            'Error',
        )
        make_trajectory_condition_plot(
            output_dir / 'traj_cong_vs_incong_smoke.png',
            candidate['trajectory'],
            f'{args.candidate_label}: congruent vs incongruent trajectory',
            congruent,
            incongruent,
            'Congruent',
            'Incongruent',
        )
        make_winner_gap_plot(output_dir / 'traj_winner_gap_smoke.png', candidate['trajectory'])
    write_summary(output_dir / 'summary_smoke.md', baseline, candidate, baseline_summary, candidate_summary, checks, args.candidate_label, candidate_trajectory_summary)


if __name__ == '__main__':
    main()
