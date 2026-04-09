from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path('results/age_groups_response_supervision_frozen')


def extract_best(log_path, age_group, human_mean):
    text = Path(log_path).read_text(errors='replace')
    best = None
    for line in text.splitlines():
        if f'[{age_group} scale ' not in line or 'Eval epoch ' not in line:
            continue
        scale = float(line.split('scale ')[1].split('/5]')[0]) / 10.0
        epoch = int(line.split('Eval epoch ')[1].split(':')[0])
        payload = line.split(': ', 1)[1]
        fields = dict(item.split('=') for item in payload.split())
        model_acc, hum_acc = map(float, fields['acc'].split('/'))
        model_gap, hum_gap = map(float, fields['cong_gap'].split('/'))
        row = {
            'age_group': age_group,
            'scale': scale,
            'epoch': epoch,
            'score': float(fields['score']),
            'rt_score': float(fields['rt_score']),
            'model_accuracy': model_acc,
            'human_accuracy': hum_acc,
            'model_congruency_rt_gap': model_gap,
            'human_congruency_rt_gap': hum_gap,
            'human_mean_rt': human_mean,
            'model_mean_rt': np.nan,
            'checkpoint': f'scale {scale:.1f}, epoch {epoch}',
        }
        if best is None or row['score'] > best['score']:
            best = row
    for line in text.splitlines():
        if best and f'[{age_group} scale {int(best["scale"] * 10)}/5]' in line and 'Finished in ' in line:
            best['model_mean_rt'] = float(line.split('PredMean=')[1].split('s,')[0])
            break
    return best


def make_behavior_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width / 2, df['human_accuracy'], width, color='#A0A0A0', label='Human')
    axes[0].bar(x + width / 2, df['model_accuracy'], width, color='#4C78A8', label='Model')
    axes[0].set_xticks(x); axes[0].set_xticklabels(df['age_group']); axes[0].set_title('Accuracy')
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, df['human_congruency_rt_gap'], width, color='#A0A0A0', label='Human')
    axes[1].bar(x + width / 2, df['model_congruency_rt_gap'], width, color='#E45756', label='Model')
    axes[1].set_xticks(x); axes[1].set_xticklabels(df['age_group']); axes[1].set_title('Congruency RT gap (s)')

    axes[2].bar(x - width / 2, df['human_mean_rt'], width, color='#A0A0A0', label='Human')
    axes[2].bar(x + width / 2, df['model_mean_rt'], width, color='#F58518', label='Model')
    axes[2].set_xticks(x); axes[2].set_xticklabels(df['age_group']); axes[2].set_title('Mean RT (s)')

    fig.suptitle('Figure F1. Frozen current-best response-supervision behavioral summary')
    fig.savefig(OUT_DIR / 'figureF1_frozen_current_best_behavior.png', bbox_inches='tight')
    plt.close(fig)


def write_memo(df):
    young = df[df['age_group'] == '20-29 matched'].iloc[0]
    old = df[df['age_group'] == '80-89'].iloc[0]
    memo = f"""# Frozen current-best response-supervision memo

## Why this snapshot exists
The response-supervision training runs were stopped deliberately before full completion so that the current best-so-far checkpoints could be treated as this phase's working endpoint.

## Frozen best-so-far checkpoints

### 20-29 matched
- checkpoint = {young['checkpoint']}
- score = {young['score']:.4f}
- rt_score = {young['rt_score']:.4f}
- model accuracy = {young['model_accuracy']:.4f}
- human accuracy = {young['human_accuracy']:.4f}
- model congruency RT gap = {young['model_congruency_rt_gap']:.4f}
- human congruency RT gap = {young['human_congruency_rt_gap']:.4f}
- model mean RT = {young['model_mean_rt']:.3f} s
- human mean RT = {young['human_mean_rt']:.3f} s

### 80-89
- checkpoint = {old['checkpoint']}
- score = {old['score']:.4f}
- rt_score = {old['rt_score']:.4f}
- model accuracy = {old['model_accuracy']:.4f}
- human accuracy = {old['human_accuracy']:.4f}
- model congruency RT gap = {old['model_congruency_rt_gap']:.4f}
- human congruency RT gap = {old['human_congruency_rt_gap']:.4f}
- model mean RT = {old['model_mean_rt']:.3f} s
- human mean RT = {old['human_mean_rt']:.3f} s

## Interpretation
Both frozen best-so-far checkpoints occur at scale 0.2, suggesting a stable preferred regime under response-label supervision. Relative to the earlier target-supervision runs, the response-supervision branch initially moved model choice behavior closer to human behavior, but the best-so-far checkpoints still drifted back toward ceiling-level accuracy. The most reliable improvement is that the congruency RT gap became much more human-like, especially in the 80-89 branch.

## Parameter-comparison limitation
Because these runs were stopped before writing new best parameter files, this frozen snapshot supports behavioral comparison but does not yet support a clean parameter-level comparison under response-label supervision. To do that rigorously, a future rerun should save best-so-far parameters whenever a new best checkpoint is found.
"""
    (OUT_DIR / 'frozen_current_best_memo.md').write_text(memo)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        extract_best('logs/train_20_29_matched_response_supervision_safe2.log', '20-29', 0.6180),
        extract_best('logs/train_80_89_response_supervision_safe2.log', '80-89', 0.9393),
    ]
    rows[0]['age_group'] = '20-29 matched'
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'frozen_current_best_comparison.csv', index=False)
    make_behavior_figure(df)
    write_memo(df)
    print(f'Saved frozen current-best outputs to {OUT_DIR}')


if __name__ == '__main__':
    main()
