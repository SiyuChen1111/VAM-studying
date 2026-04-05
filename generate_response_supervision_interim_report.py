import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path('results/age_groups_response_supervision_interim')


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_partial_best(age_group: str, output_root: str, human_mean_rt: float):
    partial_path = Path(output_root) / age_group / 'stage2' / 'partial_best' / 'best_config.partial.json'
    meta_path = Path(output_root) / age_group / 'stage2' / 'partial_best' / 'best_checkpoint_meta.json'
    if not partial_path.exists():
        raise FileNotFoundError(f'Missing partial best config: {partial_path}')
    cfg = json.loads(partial_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        'age_group': age_group,
        'status': f"response_supervision_best_scale{cfg['scale']:.1f}_epoch{cfg['epoch']}",
        'scale': float(cfg['scale']),
        'score': float(cfg['score']),
        'rt_score': float(cfg['results']['rt_score']),
        'model_accuracy': float(cfg['results']['model_accuracy']),
        'human_accuracy': float(cfg['results']['human_accuracy']),
        'model_congruency_rt_gap': float(cfg['results']['model_congruency_rt_gap']),
        'human_congruency_rt_gap': float(cfg['results']['human_congruency_rt_gap']),
        'human_mean_rt': human_mean_rt,
        'model_mean_rt': float(cfg['results']['pred_mean']),
        'note': meta.get('supervision_type', 'response_labels'),
    }


def make_summary_plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), constrained_layout=True)
    metrics = [
        ('score', 'Total score'),
        ('rt_score', 'RT score'),
        ('model_accuracy', 'Model accuracy'),
        ('model_congruency_rt_gap', 'Model congruency gap (s)'),
    ]
    x = np.arange(len(df))
    colors = ['#7B9BD1', '#E8A87C']
    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(x, df[metric], color=colors[:len(df)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df['age_group'], rotation=15)
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('Figure RS1. Current partial-best response-supervision summary')
    fig.savefig(OUT_DIR / 'figureRS1_response_supervision_summary.png', bbox_inches='tight')
    plt.close(fig)


def write_memo(df: pd.DataFrame):
    young = df[df['age_group'] == '20-29'].iloc[0]
    old = df[df['age_group'] == '80-89'].iloc[0]
    memo = f"""# Response-supervision interim memo

## Current best-so-far checkpoints from `partial_best`

### 20-29 matched
- checkpoint = {young['status']}
- score = {young['score']:.4f}
- rt_score = {young['rt_score']:.4f}
- model accuracy = {young['model_accuracy']:.4f}
- human accuracy = {young['human_accuracy']:.4f}
- model congruency RT gap = {young['model_congruency_rt_gap']:.4f}
- human congruency RT gap = {young['human_congruency_rt_gap']:.4f}
- model mean RT = {young['model_mean_rt']:.3f} s
- human mean RT = {young['human_mean_rt']:.3f} s

### 80-89
- checkpoint = {old['status']}
- score = {old['score']:.4f}
- rt_score = {old['rt_score']:.4f}
- model accuracy = {old['model_accuracy']:.4f}
- human accuracy = {old['human_accuracy']:.4f}
- model congruency RT gap = {old['model_congruency_rt_gap']:.4f}
- human congruency RT gap = {old['human_congruency_rt_gap']:.4f}
- model mean RT = {old['model_mean_rt']:.3f} s
- human mean RT = {old['human_mean_rt']:.3f} s

## Main interpretation
These outputs are now based on the saved `partial_best` checkpoints rather than stale log parsing. At the current best-so-far points, both age groups still show near-ceiling model accuracy, but the congruency RT gap is much closer to the human target than under the original target-label supervision. The remaining mismatch is now easier to localize: the model captures conflict structure better than the full human temporal regime.
"""
    (OUT_DIR / 'response_supervision_interim_memo.md').write_text(memo)


def main():
    ensure_dir()
    rows = [
        load_partial_best('20-29', 'checkpoints_age_groups_matched', 0.6180),
        load_partial_best('80-89', 'checkpoints_age_groups', 0.9393),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'response_supervision_current_comparison.csv', index=False)
    make_summary_plot(df)
    write_memo(df)
    print(f'Saved response-supervision interim outputs to {OUT_DIR}')


if __name__ == '__main__':
    main()
